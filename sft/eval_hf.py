"""Standalone Hugging Face Transformers evaluator (no vLLM / Triton / gcc required).

Drop-in ModelCaller replacement for evaluate.py / batched_evaluate.py so any CUDA GPU
(including minimal RTX 3090 nodes without a C compiler) can evaluate exported HF checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfworld_env import project_path
from evaluate import evaluate
from report import EVAL_SPLITS, generate_report

SPLIT_MAP = {
    "seen": "eval_in_distribution",
    "unseen": "eval_out_of_distribution",
}


class HFModelCaller:
    """Batch-capable HF Transformers caller matching ModelCaller's exact interface."""

    def __init__(self, config: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        model_path = project_path(config["model"]).resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"HF model directory not found: {model_path}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()

        self.prompt_limit = int(config.get("max_prompt_tokens", 2048))
        self.max_new_tokens = int(config.get("max_tokens", 512))
        self.temperature = float(config.get("temperature", 0.4))
        self.top_p = float(config.get("top_p", 1.0))
        self.top_k = int(config.get("top_k", 0))
        self.repetition_penalty = float(config.get("repetition_penalty", 1.0))
        self.usage = dict(input_tokens=0, output_tokens=0, total_tokens=0)

        eos = None
        gen_file = model_path / "generation_config.json"
        if gen_file.is_file():
            eos = json.loads(gen_file.read_text(encoding="utf-8")).get("eos_token_id")
        self.eos_token_id = eos if eos is not None else self.tokenizer.eos_token_id

    def __call__(self, messages):
        return self.generate_batch([messages])[0][0]

    @torch.no_grad()
    def generate_batch(self, conversations):
        if not conversations:
            return []
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        padded_len = input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=self.repetition_penalty,
            use_cache=True,
        )
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                gen_kwargs["top_k"] = self.top_k
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        results = []
        eos_set = set(self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id])
        eos_set.add(self.tokenizer.pad_token_id)

        for i in range(len(conversations)):
            prompt_tokens = int(attention_mask[i].sum().item())
            gen_ids = outputs[i, padded_len:].tolist()
            # Trim trailing pad/eos tokens after first eos.
            trimmed = []
            for tid in gen_ids:
                if tid in eos_set:
                    break
                trimmed.append(tid)
            text = self.tokenizer.decode(trimmed, skip_special_tokens=True)
            usage = dict(
                input_tokens=prompt_tokens,
                output_tokens=len(trimmed),
                total_tokens=prompt_tokens + len(trimmed),
            )
            for k in self.usage:
                self.usage[k] += usage[k]
            results.append((text, usage))
        return results

    def token_usage(self):
        return self.usage.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alfworld.yaml")
    parser.add_argument("--model", required=True, help="HF model directory to evaluate")
    parser.add_argument("--split", choices=["seen", "unseen"], default="unseen")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    cfg["model"] = str(project_path(args.model).resolve())
    cfg["split"] = SPLIT_MAP[args.split]
    cfg["batch_size"] = args.batch_size
    cfg["no_progress"] = args.no_progress
    cfg["method"] = "sft"

    torch.manual_seed(int(cfg.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 0)))

    out_root = project_path(cfg.get("output_dir", "outputs")) / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-hf-{args.split}"
    )
    split_dir = out_root / EVAL_SPLITS[args.split]
    print(f"Loading HF model: {cfg['model']}", flush=True)
    caller = HFModelCaller(cfg)
    print(f"Evaluating split={cfg['split']} (batch_size={args.batch_size}) -> {out_root}", flush=True)
    results = evaluate(cfg, caller, split_dir)
    summary = generate_report(out_root, project_path(cfg.get("data_dir", "data/alfworld")))
    split_stats = summary["splits"][EVAL_SPLITS[args.split]]
    print(
        f"Done {args.split}: {split_stats['successes']}/{split_stats['completed_cases']} "
        f"= {split_stats['success_rate'] * 100:.2f}% | summary: {out_root / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
