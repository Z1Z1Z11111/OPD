"""Regression coverage for the baseline switch and grouped environment sampling."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from alfworld_env import ROOT
from rl import evaluation_config, run_root
from rl.train import training_settings


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.base = yaml.safe_load((ROOT / "configs/alfworld.yaml").read_text())
        self.recipe = yaml.safe_load((ROOT / "configs/grpo.yaml").read_text())

    def test_checkpoint_selection(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            self.base.update(model=str(tmp_path / "base"), rl_output_dir=str(tmp_path / "runs"))
            self.assertEqual(evaluation_config(self.base)["model"], self.base["model"])
            with self.assertRaises(FileNotFoundError):
                evaluation_config(self.base, "rl")
            root = run_root(self.base)
            export = root / "run/global_step_1/actor/huggingface"
            export.mkdir(parents=True)
            (export / "config.json").write_text("{}")
            (root / "latest.json").write_text(json.dumps(dict(
                base_model=self.base["model"], checkpoint=str(export.relative_to(root)))))
            evaluated = evaluation_config(self.base, "rl")
            self.assertEqual(Path(evaluated["model"]), export)
            self.assertEqual(evaluated["base_model"], self.base["model"])

    def test_environment_groups(self):
        formal = training_settings(self.base, self.recipe)
        self.assertEqual(formal["trainer.n_gpus_per_node"], 2)
        self.assertEqual(formal["actor_rollout_ref.rollout.name"], "vllm")
        self.assertEqual(formal["actor_rollout_ref.rollout.tensor_model_parallel_size"], 2)
        self.assertEqual(formal["data.train_batch_size"], 16)
        self.assertEqual(formal["actor_rollout_ref.rollout.n"], 1)
        self.assertEqual(formal["env.rollout.n"], 8)
        self.assertEqual(formal["algorithm.adv_estimator"], "grpo")
        small = training_settings(self.base, self.recipe, smoke=True)
        self.assertEqual(small["trainer.total_training_steps"], 1)
        self.assertGreater(small["env.rollout.n"], 1)
