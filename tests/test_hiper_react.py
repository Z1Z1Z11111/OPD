"""Regression tests for the ordinary HiPER ReAct port; no model weights needed.

Run explicitly with: python -m unittest discover -s tests -v
Upstream parity checks are optional when the reference source tree is absent.
"""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Agent, create_agent, validate_agent_config
from evaluate import evaluate
from hiper_react import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HiPERReActAgent,
    HiPERReActContext,
    project_response,
)
from model import generation_kwargs


def config(**overrides):
    settings = dict(
        benchmark="ALFWorld", react_style="hiper", split="eval_out_of_distribution",
        start_case=0, end_case=3, max_steps=4, max_json_retries=0,
        include_actions=True, context=dict(type="recent", history_length=2),
        no_progress=True,
    )
    settings.update(overrides)
    return settings


class MemoryLogger:
    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append(dict(event=event, **deepcopy(fields)))


class FakeEnv:
    """Disjoint lanes, distinct observations, and different episode lengths."""

    def __init__(self, settings):
        self.case_indices = list(range(settings["start_case"], settings["end_case"]))
        self.case_indices = self.case_indices[
            settings.get("shard_rank", 0)::settings.get("shard_count", 1)
        ][settings.get("lane_rank", 0)::settings.get("lane_count", 1)]
        self.cursor = -1
        self.closed = False
        self.actions = []

    def __len__(self):
        return len(self.case_indices)

    def reset(self):
        self.cursor += 1
        self.index = self.case_indices[self.cursor]
        self.turn = 0
        return self.state()

    def state(self):
        done = self.turn >= self.index + 1
        return dict(
            case_id=f"json_2.1.1/valid_unseen/pick_and_place_simple-Object-None-Table-1/trial_{self.index}",
            task=f"Complete case {self.index}", observation=f"case{self.index}:obs{self.turn}",
            actions=["look", "help"], done=done, success=done, score=float(done),
        )

    def step(self, action):
        self.actions.append(action)
        self.turn += 1
        return self.state()

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self):
        self.prompts = []
        self.batch_sizes = []
        self.usage = dict(input_tokens=0, output_tokens=0, total_tokens=0)

    def __call__(self, messages):
        return self.generate_batch([messages])[0][0]

    def generate_batch(self, conversations):
        self.batch_sizes.append(len(conversations))
        results = []
        for messages in conversations:
            self.prompts.append(deepcopy(messages))
            usage = dict(input_tokens=11, output_tokens=7, total_tokens=18)
            for key in self.usage:
                self.usage[key] += usage[key]
            results.append(("<think>diagnostic-only</think><action>LOOK</action>", usage))
        return results

    def token_usage(self):
        return self.usage.copy()


class HiPERPolicyTests(unittest.TestCase):
    def test_first_prompt_matches_template_and_excludes_help(self):
        env = FakeEnv(config())
        state = env.reset()
        policy = create_agent(config(), None, MemoryLogger())
        self.assertIsInstance(policy, HiPERReActAgent)
        policy.context_provider.reset(state)
        self.assertEqual(policy._build_messages(state), [{
            "role": "user",
            "content": ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation="case0:obs0", admissible_actions="'look'",
            ),
        }])

    def test_history_is_pre_action_recent_and_has_no_thoughts(self):
        context = HiPERReActContext(2)
        state = dict(observation="obs0")
        context.reset(state)
        for step in range(3):
            state = dict(observation=f"obs{step + 1}")
            context.update(dict(action=f"act{step}", thought="do-not-replay"), state, step)
        self.assertEqual(context.get_context(), dict(
            type="recent", step_count=3, history_length=2,
            action_history="[Observation 2: 'obs1', Action 2: 'act1']\n"
                           "[Observation 3: 'obs2', Action 3: 'act2']",
        ))
        context.reset(dict(observation="new episode"))
        self.assertEqual(context.get_context()["step_count"], 0)
        self.assertEqual(context.get_context()["action_history"], "")

    def test_multi_turn_prompt_matches_history_numbering(self):
        policy = create_agent(config(), None, MemoryLogger())
        state = dict(task="find bowl", observation="before", actions=["look", "help"])
        policy.context_provider.reset(state)
        next_state = dict(state, observation="after")
        policy.context_provider.update(dict(action="look"), next_state, 0)
        self.assertEqual(policy._build_messages(next_state), [{
            "role": "user",
            "content": ALFWORLD_TEMPLATE.format(
                task_description="find bowl", step_count=1, history_length=1,
                action_history="[Observation 1: 'before', Action 1: 'look']",
                current_step=2, current_observation="after", admissible_actions="'look'",
            ),
        }])

    def test_zero_history_always_uses_no_history_template(self):
        settings = config(context=dict(type="recent", history_length=0))
        env = FakeEnv(settings)
        policy = create_agent(settings, None, MemoryLogger())
        state = env.reset()
        policy.context_provider.reset(state)
        after = env.step("look")
        policy.context_provider.update(dict(action="look"), after, 0)
        self.assertEqual(policy._build_messages(after)[0]["content"],
                         ALFWORLD_TEMPLATE_NO_HIS.format(
                             current_observation=after["observation"], admissible_actions="'look'"))

    def test_projection_preserves_upstream_edge_cases(self):
        cases = [
            ("<think>inspect</think><action> LOOK </action>", "look", True),
            ("<action>LOOK</action>", "look", False),
            ("<THINK>inspect</THINK><ACTION>LOOK</ACTION>", "look", False),
            ("<think>中文</think><action>LOOK</action>", "look", False),
            ("<think>x</think><action></action>", "", True),
            ("<think>x</think><action>not admissible</action>", "not admissible", True),
            ("<think>x</think><action>LOOK", "<think>x</think><action>look", False),
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "ghijklmnopqrstuvwxyz0123456789", False),
            ("", "", False),
        ]
        for raw, action, valid in cases:
            with self.subTest(raw=raw):
                projected = project_response(raw)
                self.assertEqual(projected["action"], action)
                self.assertEqual(projected["is_action_valid"], valid)

    def test_invalid_outputs_execute_once_without_retries(self):
        settings = config(max_steps=3, max_json_retries=99)
        env = FakeEnv(settings)
        # Select the third case so all three turns execute.
        env.case_indices = [2]
        responses = iter(["MALFORMED", "<action></action>", "<think>x</think><action>LOOK</action>"])
        prompts = []

        def call(messages):
            prompts.append(deepcopy(messages))
            return next(responses)

        logger = MemoryLogger()
        result = create_agent(settings, call, logger).run(env)
        self.assertEqual(env.actions, ["malformed", "", "look"])
        self.assertEqual(len(prompts), 3)
        self.assertEqual(result["steps"], 3)
        self.assertTrue(result["success"])
        self.assertEqual([t["is_action_valid"] for t in result["trajectory"]], [False, False, True])
        self.assertFalse(any(e["event"] == "react_missing_action" for e in logger.events))

    def test_budget_counts_invalid_environment_steps(self):
        settings = config(max_steps=2)
        env = FakeEnv(settings)
        env.case_indices = [2]
        result = create_agent(settings, lambda _: "invalid", MemoryLogger()).run(env)
        self.assertEqual(result["steps"], 2)
        self.assertFalse(result["success"])

    def test_legacy_raw_policy_still_repairs_json(self):
        settings = config(react_style="raw", context=dict(type="raw"), max_json_retries=2)
        env = FakeEnv(settings)
        responses = iter(["invalid JSON", '{"thought": "inspect", "action": "look"}'])
        prompts = []

        def call(messages):
            prompts.append(deepcopy(messages))
            return next(responses)

        policy = create_agent(settings, call, MemoryLogger())
        self.assertIs(type(policy), Agent)
        result = policy.run(env)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0][0]["role"], "system")
        self.assertIn("required schema", prompts[1][-1]["content"])

    def test_bad_policy_configurations_fail_early(self):
        bad = [
            config(react_style="unknown"),
            config(context=dict(type="raw")),
            config(context=dict(type="recent", history_length=-1)),
            config(context=dict(type="recent", history_length=True)),
            config(context=dict(type="recent", history_length=1.5)),
            config(include_actions=False),
            config(max_actions_per_turn=2),
            config(react_style="raw", context=dict(type="recent")),
        ]
        for settings in bad:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_agent_config(settings)

    def test_sampling_overrides_checkpoint_defaults(self):
        settings = dict(temperature=0.4, max_tokens=512, top_p=1.0,
                        top_k=0, repetition_penalty=1.0)
        self.assertEqual(generation_kwargs(settings, 151643), dict(
            temperature=0.4, do_sample=True, max_new_tokens=512,
            top_p=1.0, top_k=0, repetition_penalty=1.0, pad_token_id=151643,
        ))
        legacy = generation_kwargs(dict(temperature=0, max_tokens=512), 151643)
        self.assertFalse(legacy["do_sample"])
        self.assertNotIn("repetition_penalty", legacy)


class EvaluationIntegrationTests(unittest.TestCase):
    def run_evaluation(self, batch_size):
        settings = config(batch_size=batch_size)
        model = FakeModel()
        environments = []

        def env_factory(cfg):
            env = FakeEnv(cfg)
            environments.append(env)
            return env

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            results = evaluate(settings, model, output, env_factory=env_factory)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["completed_cases"], 3)
            self.assertEqual(summary["successes"], 3)
            self.assertEqual(summary["total_steps"], 6)
            self.assertEqual(summary["token_usage"]["total_tokens"], 6 * 18)
            self.assertEqual(len(list(output.glob("cases/*/result.json"))), 3)
        self.assertTrue(all(env.closed for env in environments))
        self.assertEqual(len({r["case_id"] for r in results}), 3)
        self.assertEqual(sorted(r["steps"] for r in results), [1, 2, 3])
        for result in results:
            self.assertEqual(result["react_style"], "hiper")
            self.assertTrue(all(t["is_action_valid"] for t in result["trajectory"]))
        for messages in model.prompts:
            text = messages[0]["content"]
            # Each prompt belongs to one task, even after a lane starts a new task.
            self.assertEqual(len(set(re.findall(r"case(\d+):obs", text))), 1)
            self.assertNotIn("diagnostic-only", text)
            self.assertEqual(len(messages), 1)
        return model

    def test_serial_evaluation(self):
        model = self.run_evaluation(1)
        self.assertTrue(all(size == 1 for size in model.batch_sizes))

    def test_batched_evaluation_and_lane_refill(self):
        model = self.run_evaluation(2)
        self.assertIn(2, model.batch_sizes)
        self.assertIn(1, model.batch_sizes)


class OptionalUpstreamParityTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(os.environ.get("HIPER_SOURCE", ROOT.parent / "HiPER-agent-main"))
        if not self.source.is_dir():
            self.skipTest("Set HIPER_SOURCE to the reference HiPER-agent source tree")

    def test_prompt_constants_are_verbatim(self):
        path = self.source / "agent_system/environments/prompts/alfworld.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        upstream = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(ALFWORLD_TEMPLATE_NO_HIS, upstream["ALFWORLD_TEMPLATE_NO_HIS"])
        self.assertEqual(ALFWORLD_TEMPLATE, upstream["ALFWORLD_TEMPLATE"])

    def test_action_projection_matches_upstream(self):
        path = self.source / "agent_system/environments/env_package/alfworld/projection.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "alfworld_projection")
        namespace = {"List": list, "re": re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        samples = [
            "<think>x</think><action> LOOK </action>",
            "<think>x</think><ACTION>LOOK</ACTION>",
            "<THINK>x</THINK><action>LOOK</action>",
            "<think>中文</think><action>LOOK</action>",
            "<action>look</action>", "<think>x</think>",
            "<think>x</think><action></action>",
            "</action><think>x</think><action>",
            "<think>x</think><action>not admissible</action>",
            "<think>x</think><action>look</action><action>help</action>",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "",
        ]
        actions, valids = namespace["alfworld_projection"](samples.copy(), [[] for _ in samples])
        for raw, action, valid in zip(samples, actions, valids):
            with self.subTest(raw=raw):
                actual = project_response(raw)
                self.assertEqual(actual["action"], action)
                self.assertEqual(actual["is_action_valid"], bool(valid))


if __name__ == "__main__":
    unittest.main()
