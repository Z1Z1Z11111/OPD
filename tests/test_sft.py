"""Unit tests for the standalone SFT data encoding and OPC-SFT loss."""
import math
import unittest

import torch

from agent import project_response
from sft.prepare_data import _format_assistant, _split_by_episode, _stratified_subsample
from sft.train import PadCollator, compute_sft_loss, encode_sft_example


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class SFTTests(unittest.TestCase):
    def test_format_assistant_passes_agent_projection(self):
        reply = _format_assistant("I should look at the fridge first.", "go to fridge 1")
        decision = project_response(reply)
        self.assertTrue(decision["is_action_valid"])
        self.assertEqual(decision["thought"], "I should look at the fridge first.")
        self.assertEqual(decision["action"], "go to fridge 1")

    def test_encode_sft_example_masks_prompt_only(self):
        tok = _DummyTokenizer()
        messages = [
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<think>t</think>\n<action>look</action>"},
        ]
        encoded = encode_sft_example(messages, tok)
        prompt_len = len(tok.encode(tok.apply_chat_template([messages[0]], add_generation_prompt=True)))
        self.assertEqual(encoded["labels"][:prompt_len], [-100] * prompt_len)
        self.assertEqual(encoded["labels"][prompt_len:], encoded["input_ids"][prompt_len:])

    def test_pad_collator_pads_to_multiple_of_8(self):
        collator = PadCollator(pad_token_id=0, pad_to_multiple_of=8)
        batch = collator([
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5, 6, 7, 8, 9], "labels": [-100, -100, 6, 7, 8, 9]},
        ])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(batch["labels"][0].tolist(), [-100, 2, 3, -100, -100, -100, -100, -100])

    def test_opc_clip_stops_gradient_on_low_prob_tokens(self):
        logits = torch.tensor(
            [[[-5.0, 5.0, 0.0], [-5.0, 5.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor([[-100, 0, 1]], dtype=torch.long)

        loss, metrics = compute_sft_loss(logits, labels, opc_clip_eps=0.05)
        loss.backward()

        self.assertAlmostEqual(metrics["clip_ratio"], 0.5, places=6)
        self.assertTrue(torch.allclose(logits.grad[0, 0], torch.zeros(3), atol=1e-6))
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0.0)
        self.assertGreater(metrics["raw_nll"], metrics["loss"])
        self.assertLessEqual(metrics["loss"], 0.5 * (-math.log(0.05) + 0.1))

    def test_split_by_episode_no_leakage(self):
        steps = [
            {"episode_id": f"ep_{i}", "task_type": "t1", "step": s}
            for i in range(20)
            for s in range(3)
        ]
        train_steps, val_steps = _split_by_episode(steps, val_ratio=0.1, seed=0)
        train_eps = {s["episode_id"] for s in train_steps}
        val_eps = {s["episode_id"] for s in val_steps}
        self.assertTrue(train_eps.isdisjoint(val_eps))
        self.assertEqual(len(val_eps), 2)
        self.assertEqual(len(train_steps) + len(val_steps), len(steps))

    def test_stratified_subsample_preserves_counts(self):
        records = [{"id": f"a_{i}", "task_type": "A"} for i in range(60)] + [
            {"id": f"b_{i}", "task_type": "B"} for i in range(40)
        ]
        sub = _stratified_subsample(records, max_episodes=10, seed=7)
        self.assertEqual(len(sub), 10)
        self.assertEqual(sum(1 for r in sub if r["task_type"] == "A"), 6)
        self.assertEqual(sum(1 for r in sub if r["task_type"] == "B"), 4)


if __name__ == "__main__":
    unittest.main()
