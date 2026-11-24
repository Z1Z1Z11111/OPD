"""Regression tests for react/grpo mode switching, GRPO config, and pruned rl/hiper integrity."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from alfworld_env import ROOT
from rl import evaluation_config, run_root
from rl.train import training_settings


class GRPOBaselineTest(unittest.TestCase):
    def setUp(self):
        self.base = yaml.safe_load((ROOT / "configs/alfworld.yaml").read_text(encoding="utf-8"))
        self.recipe = yaml.safe_load((ROOT / "configs/grpo.yaml").read_text(encoding="utf-8"))

    def test_react_and_grpo_method_switching(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            self.base.update(model=str(tmp_path / "base"), rl_output_dir=str(tmp_path / "runs"))
            self.assertEqual(evaluation_config(self.base, "react")["model"], self.base["model"])
            with self.assertRaises(ValueError):
                evaluation_config(self.base, "unknown")
            with self.assertRaises(FileNotFoundError):
                evaluation_config(self.base, "grpo")

            root = run_root(self.base)
            export = root / "run/global_step_1/actor/huggingface"
            export.mkdir(parents=True)
            (export / "config.json").write_text("{}", encoding="utf-8")
            (root / "latest.json").write_text(
                json.dumps(dict(base_model=self.base["model"], checkpoint=str(export.relative_to(root)))),
                encoding="utf-8",
            )
            evaluated = evaluation_config(self.base, "grpo")
            self.assertEqual(evaluated["method"], "grpo")
            self.assertEqual(Path(evaluated["model"]), export)
            self.assertEqual(evaluated["base_model"], self.base["model"])

    def test_training_settings_and_smoke_overrides_match_ppo_trainer_schema(self):
        formal = training_settings(self.base, self.recipe)
        self.assertEqual(formal["trainer.n_gpus_per_node"], 2)
        self.assertEqual(formal["actor_rollout_ref.rollout.name"], "vllm")
        self.assertEqual(formal["actor_rollout_ref.rollout.tensor_model_parallel_size"], 1)
        self.assertEqual(formal["data.train_batch_size"], 16)
        self.assertEqual(formal["actor_rollout_ref.rollout.n"], 1)
        self.assertEqual(formal["env.rollout.n"], 8)
        self.assertEqual(formal["algorithm.adv_estimator"], "grpo")
        self.assertEqual(formal["env.env_name"], "alfworld/AlfredTWEnv")

        small = training_settings(self.base, self.recipe, smoke=True)
        self.assertEqual(small["trainer.total_training_steps"], 1)
        self.assertGreater(small["env.rollout.n"], 1)

        # Verify every formal and smoke override key exists in ppo_trainer.yaml (or optional set).
        trainer_cfg = yaml.safe_load(
            (ROOT / "rl/hiper/verl/trainer/config/ppo_trainer.yaml").read_text(encoding="utf-8")
        )
        optional = {
            "actor_rollout_ref.model.attn_implementation",
            "actor_rollout_ref.rollout.micro_batch_size",
            "data.dataloader_num_workers",
        }
        for mode_settings in (formal, small):
            for dotted_key in mode_settings:
                if dotted_key in optional:
                    continue
                node = trainer_cfg
                for part in dotted_key.split("."):
                    self.assertIn(part, node, f"Missing key {dotted_key!r} in ppo_trainer.yaml")
                    node = node[part]

    def test_pruned_hiper_has_no_plan_execute_hae_gigpo_or_broken_imports(self):
        hiper_root = ROOT / "rl/hiper"
        self.assertFalse((hiper_root / "gigpo").exists())
        self.assertFalse((hiper_root / "hiper_hae").exists())

        py_files = sorted(p for p in hiper_root.rglob("*.py") if "__pycache__" not in p.parts)
        self.assertGreater(len(py_files), 0)

        def mod_name(path: Path) -> str:
            parts = list(path.relative_to(hiper_root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)

        mods = {mod_name(p): p for p in py_files}
        local_roots = {"verl", "agent_system", "gigpo", "hiper_hae"}
        broken = []
        for name, path in mods.items():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            cur = name.split(".")
            base = cur if path.name == "__init__.py" else cur[:-1]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in local_roots and alias.name not in mods:
                            broken.append((str(path), node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        prefix = base[: len(base) - node.level + 1]
                        target = ".".join(prefix + ([node.module] if node.module else []))
                    else:
                        target = node.module or ""
                    if target and target.split(".")[0] in local_roots and target not in mods:
                        broken.append((str(path), node.lineno, target))
        self.assertEqual(broken, [], f"Broken internal imports in rl/hiper: {broken}")


if __name__ == "__main__":
    unittest.main()
