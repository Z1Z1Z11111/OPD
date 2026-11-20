"""Train the OPD ALFWorld baseline with the bundled HiPER veRL trainer."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, mkdtemp

import yaml

from agent import validate_agent_config
from alfworld_env import project_path
from rl import run_root


def write_slots(path, count, split):
    """The upstream environment supplies the task at reset; parquet rows are batch slots."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        dict(data_source="text", prompt=[dict(role="user", content="")],
             ability="agent", extra_info=dict(split=split, index=i))
        for i in range(count)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def training_settings(base, recipe, smoke=False, single_gpu=False):
    validate_agent_config(base)
    if base.get("benchmark") != "ALFWorld" or base.get("react_style") != "hiper":
        raise ValueError("GRPO requires ALFWorld and react_style: hiper")
    settings = dict(recipe["overrides"])
    settings.update({
        "actor_rollout_ref.model.path": str(project_path(base["model"]).resolve()),
        "data.max_prompt_length": base["max_prompt_tokens"],
        "data.max_response_length": base["max_tokens"],
        "env.max_steps": base["max_steps"],
        "env.history_length": base["context"]["history_length"],
        "env.seed": base["seed"],
    })
    if smoke or single_gpu:
        settings.update({
            "trainer.n_gpus_per_node": 1,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
            "actor_rollout_ref.rollout.name": "hf",
            "actor_rollout_ref.model.attn_implementation": "sdpa",
            "actor_rollout_ref.model.use_remove_padding": False,
            "actor_rollout_ref.rollout.micro_batch_size": 2,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        })
    if smoke:
        settings.update({
            "data.train_batch_size": 1, "data.val_batch_size": 1, "env.rollout.n": 4,
            "env.max_steps": 2, "data.max_response_length": 96,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
            "data.dataloader_num_workers": 0,
            "ray_init.num_cpus": 4,
            "trainer.total_training_steps": 1, "trainer.val_before_train": False,
            "trainer.test_freq": -1, "trainer.save_freq": 1,
        })
    if settings["algorithm.adv_estimator"] != "grpo" or settings["env.rollout.n"] < 2:
        raise ValueError("GRPO requires grouped environment rollouts (env.rollout.n >= 2)")
    return settings


def train(base, recipe, smoke, dry_run, single_gpu=False):
    settings = training_settings(base, recipe, smoke, single_gpu)
    source_dir = Path(__file__).resolve().parent / "hiper"
    trainer_dir = source_dir / "verl/trainer"
    data = project_path(base["data_dir"]).resolve()
    if dry_run:
        print(yaml.safe_dump(dict(source_dir=str(source_dir), ALFWORLD_DATA=str(data),
                                  overrides=settings), sort_keys=False))
        return
    for path in (data / "json_2.1.1/train", data / "json_2.1.1/valid_seen",
                 Path(settings["actor_rollout_ref.model.path"]) / "config.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(source_dir))
    # Ray workers start fresh Python processes and also need the matching veRL fork.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source_dir), os.getenv("PYTHONPATH"))))
    os.environ["ALFWORLD_DATA"] = str(data)
    if settings["actor_rollout_ref.rollout.name"] == "vllm":
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo

    import torch
    gpus = settings["trainer.n_gpus_per_node"]
    if torch.cuda.device_count() < gpus:
        raise RuntimeError(f"Recipe needs {gpus} visible GPUs; found {torch.cuda.device_count()}")
    root = run_root(base)
    root.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=root))
    for role, split in (("train", "train"), ("val", "test")):
        path = run / f"{split}.parquet"
        write_slots(path, settings[f"data.{role}_batch_size"], split)
        settings[f"data.{role}_files"] = str(path)
    settings["trainer.default_local_dir"] = str(run)
    config = OmegaConf.load(trainer_dir / "config/ppo_trainer.yaml")
    OmegaConf.set_struct(config, True)
    # These optional fields are read with .get() by this HiPER checkout.
    optional = {"actor_rollout_ref.model.attn_implementation",
                    "actor_rollout_ref.rollout.micro_batch_size", "data.dataloader_num_workers"}
    for key, value in settings.items():
        OmegaConf.update(config, key, value, merge=False, force_add=key in optional)
    OmegaConf.save(config, run / "config.yaml", resolve=True)
    print(f"GRPO training run: {run}", flush=True)
    run_ppo(config)

    step = int((run / "latest_checkpointed_iteration.txt").read_text())
    export = run / f"global_step_{step}/actor/huggingface"
    if not (export / "config.json").is_file() or not any(export.glob("*.safetensors")):
        raise RuntimeError(f"Incomplete Hugging Face export: {export}")
    manifest = dict(base_model=settings["actor_rollout_ref.model.path"],
                    checkpoint=str(export.relative_to(root)), step=step)
    # Smoke runs use a reduced horizon; keep the production evaluation pointer separate.
    target = root / ("smoke.json" if smoke else "latest.json")
    with NamedTemporaryFile("w", dir=root, delete=False) as fp:
        json.dump(manifest, fp)
    Path(fp.name).replace(target)
    print(f"Evaluation checkpoint: {export}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alfworld.yaml")
    parser.add_argument("--recipe", default="configs/grpo.yaml")
    gpu_options = parser.add_mutually_exclusive_group()
    gpu_options.add_argument("--gpus", type=int, choices=(1, 2), help="Use the PRO 6000 recipe with this many GPUs")
    gpu_options.add_argument("--single-gpu", action="store_true", help="Use the local HiPER HF/SDPA single-GPU backend")
    parser.add_argument("--smoke", action="store_true", help="One update on one GPU with four rollouts")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved overrides without training")
    parser.add_argument("--model", help="Override the base model for this training run")
    args = parser.parse_args()
    base = yaml.safe_load(project_path(args.config).read_text())
    recipe = yaml.safe_load(project_path(args.recipe).read_text())
    if args.gpus:
        profile = recipe["pro6000"]
        base["model"] = profile["model"]
        recipe["overrides"].update(profile["overrides"])
        recipe["overrides"]["trainer.n_gpus_per_node"] = args.gpus
    if args.model:
        base["model"] = args.model
    train(base, recipe, args.smoke, args.dry_run, args.single_gpu)


if __name__ == "__main__":
    main()
