"""Mocked offline vLLM caller regression tests; no GPU or weights required."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ModelCaller, sampling_options


class ModelCallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        (self.path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 2]}), encoding="utf-8"
        )
        self.config = dict(
            model=str(self.path),
            device="cuda:0",
            temperature=0.4,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=32,
            max_prompt_tokens=64,
            seed=0,
        )
        self.tokenizer = MagicMock()
        self.tokenizer.apply_chat_template.side_effect = lambda messages, **kw: messages[0]["content"]
        self.tokenizer.encode.side_effect = lambda text, **kw: [ord(c) for c in text]
        self.tokenizer.decode.side_effect = lambda ids, **kw: "".join(chr(i) for i in ids)
        self.engine = MagicMock()
        self.engine.get_tokenizer.return_value = self.tokenizer
        self.engine.generate.side_effect = lambda requests, params, **kw: [
            SimpleNamespace(
                prompt_token_ids=r["prompt_token_ids"],
                outputs=[SimpleNamespace(token_ids=[111, 107])],
            )
            for r in requests
        ]
        self.llm = MagicMock(return_value=self.engine)
        self.modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)),
            "vllm": SimpleNamespace(LLM=self.llm, SamplingParams=lambda **kw: SimpleNamespace(**kw)),
        }

    def make_caller(self, **overrides):
        cfg = dict(self.config, **overrides)
        with patch.dict(sys.modules, self.modules):
            return ModelCaller(cfg)

    def test_batch_tokens_and_empty_batch(self):
        caller = self.make_caller()
        self.assertEqual(caller.generate_batch([]), [])
        self.engine.generate.assert_not_called()
        results = caller.generate_batch([
            [dict(role="user", content="abc")],
            [dict(role="user", content="defgh")],
        ])
        self.assertEqual([r[0] for r in results], ["ok", "ok"])
        self.assertEqual(results[0][1], dict(input_tokens=3, output_tokens=2, total_tokens=5))
        self.assertEqual(caller.token_usage(), dict(input_tokens=8, output_tokens=4, total_tokens=12))
        self.assertEqual(caller([dict(role="user", content="a")]), "ok")
        self.assertEqual(caller.token_usage()["total_tokens"], 15)
        self.assertEqual(self.llm.call_args.kwargs["generation_config"], "vllm")
        self.assertEqual(caller.sampling_params.stop_token_ids, [1, 2])

    def test_overlong_prompt_fails_without_generation(self):
        caller = self.make_caller()
        with self.assertRaises(ValueError):
            caller([dict(role="user", content="x" * 65)])
        self.engine.generate.assert_not_called()
        self.assertEqual(caller.token_usage()["total_tokens"], 0)

    def test_batch_reordering_is_rejected(self):
        caller = self.make_caller()
        self.engine.generate.side_effect = lambda *a, **kw: [
            SimpleNamespace(prompt_token_ids=[999], outputs=[SimpleNamespace(token_ids=[111])])
        ]
        with self.assertRaises(RuntimeError):
            caller([dict(role="user", content="abc")])

    def test_disabled_top_k_and_explicit_sampling(self):
        options = sampling_options(self.config)
        self.assertEqual(options["top_k"], -1)
        self.assertEqual(options["temperature"], 0.4)
        self.assertEqual(options["repetition_penalty"], 1.0)
        for invalid in (-2, 1.5, True):
            with self.subTest(top_k=invalid), self.assertRaises(ValueError):
                sampling_options(dict(self.config, top_k=invalid))

    def test_invalid_device_and_tensor_parallelism(self):
        with self.assertRaises(ValueError):
            self.make_caller(device="cpu")
        with self.assertRaises(ValueError):
            self.make_caller(vllm=dict(tensor_parallel_size=2))
        self.llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
