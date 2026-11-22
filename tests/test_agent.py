"""Regression tests for the ALFWorld ReAct policy and serial/batched evaluation."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    Agent,
    RecentHistoryContext,
    create_agent,
    project_response,
    validate_agent_config,
)
from evaluate import evaluate


def make_config(**overrides):
    settings = dict(
        benchmark="ALFWorld",
        method="react",
        split="eval_out_of_distribution",
        start_case=0,
        end_case=3,
        max_steps=4,
        history_length=2,
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
    """Disjoint lanes, distinct observations, and varying episode lengths."""

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
            task=f"Complete case {self.index}",
            observation=f"case{self.index}:obs{self.turn}",
            actions=["look", "help"],
            done=done,
            success=done,
            score=float(done),
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


class ReActAgentTests(unittest.TestCase):
    def test_first_prompt_matches_template_and_excludes_help(self):
        env = FakeEnv(make_config())
        state = env.reset()
        policy = create_agent(make_config(), None, MemoryLogger())
        self.assertIsInstance(policy, Agent)
        policy.context_provider.reset(state)
        self.assertEqual(
            policy._build_messages(state),
            [{
                "role": "user",
                "content": ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation="case0:obs0",
                    admissible_actions="'look'",
                ),
            }],
        )

    def test_history_is_pre_action_recent_and_has_no_thoughts(self):
        context = RecentHistoryContext(2)
        state = dict(observation="obs0")
        context.reset(state)
        for step in range(3):
            state = dict(observation=f"obs{step + 1}")
            context.update(dict(action=f"act{step}", thought="do-not-replay"), state)
        self.assertEqual(
            context.get_context(),
            dict(
                step_count=3,
                history_length=2,
                action_history="[Observation 2: 'obs1', Action 2: 'act1']\n"
                               "[Observation 3: 'obs2', Action 3: 'act2']",
            ),
        )
        context.reset(dict(observation="new episode"))
        self.assertEqual(context.get_context()["step_count"], 0)
        self.assertEqual(context.get_context()["action_history"], "")

    def test_multi_turn_prompt_matches_history_numbering(self):
        policy = create_agent(make_config(), None, MemoryLogger())
        state = dict(task="find bowl", observation="before", actions=["look", "help"])
        policy.context_provider.reset(state)
        next_state = dict(state, observation="after")
        policy.context_provider.update(dict(action="look"), next_state)
        self.assertEqual(
            policy._build_messages(next_state),
            [{
                "role": "user",
                "content": ALFWORLD_TEMPLATE.format(
                    task_description="find bowl",
                    step_count=1,
                    history_length=1,
                    action_history="[Observation 1: 'before', Action 1: 'look']",
                    current_step=2,
                    current_observation="after",
                    admissible_actions="'look'",
                ),
            }],
        )

    def test_zero_history_always_uses_no_history_template(self):
        settings = make_config(history_length=0)
        env = FakeEnv(settings)
        policy = create_agent(settings, None, MemoryLogger())
        state = env.reset()
        policy.context_provider.reset(state)
        after = env.step("look")
        policy.context_provider.update(dict(action="look"), after)
        self.assertEqual(
            policy._build_messages(after)[0]["content"],
            ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=after["observation"],
                admissible_actions="'look'",
            ),
        )

    def test_projection_matches_training_edge_cases(self):
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
        settings = make_config(max_steps=3)
        env = FakeEnv(settings)
        env.case_indices = [2]
        responses = iter(["MALFORMED", "<action></action>", "<think>x</think><action>LOOK</action>"])
        prompts = []

        def call(messages):
            prompts.append(deepcopy(messages))
            return next(responses)

        result = create_agent(settings, call, MemoryLogger()).run(env)
        self.assertEqual(env.actions, ["malformed", "", "look"])
        self.assertEqual(len(prompts), 3)
        self.assertEqual(result["steps"], 3)
        self.assertTrue(result["success"])
        self.assertEqual([t["is_action_valid"] for t in result["trajectory"]], [False, False, True])

    def test_bad_history_length_fails_early(self):
        for bad in (-1, True, 1.5):
            with self.subTest(history_length=bad), self.assertRaises(ValueError):
                validate_agent_config(make_config(history_length=bad))

    def test_eval_matches_bundled_training_prompts_and_projection(self):
        prompt_path = ROOT / "rl/hiper/agent_system/environments/prompts/alfworld.py"
        tree = ast.parse(prompt_path.read_text(encoding="utf-8"))
        exported = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(ALFWORLD_TEMPLATE_NO_HIS, exported["ALFWORLD_TEMPLATE_NO_HIS"])
        self.assertEqual(ALFWORLD_TEMPLATE, exported["ALFWORLD_TEMPLATE"])
        self.assertEqual(set(exported), {"ALFWORLD_TEMPLATE_NO_HIS", "ALFWORLD_TEMPLATE"})

        proj_path = ROOT / "rl/hiper/agent_system/environments/env_package/alfworld/projection.py"
        proj_tree = ast.parse(proj_path.read_text(encoding="utf-8"))
        fn_names = [n.name for n in proj_tree.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(fn_names, ["alfworld_projection"])
        namespace = {"List": list, "re": re}
        exec(compile(proj_tree, str(proj_path), "exec"), namespace)
        samples = [
            "<think>x</think><action> LOOK </action>",
            "<think>x</think><ACTION>LOOK</ACTION>",
            "<THINK>x</THINK><action>LOOK</action>",
            "<think>中文</think><action>LOOK</action>",
            "<action>look</action>",
            "<think>x</think>",
            "<think>x</think><action></action>",
            "</action><think>x</think><action>",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "",
        ]
        actions, valids = namespace["alfworld_projection"](samples.copy(), [[] for _ in samples])
        for raw, action, valid in zip(samples, actions, valids):
            actual = project_response(raw)
            self.assertEqual(actual["action"], action)
            self.assertEqual(actual["is_action_valid"], bool(valid))


class EvaluationIntegrationTests(unittest.TestCase):
    def run_evaluation(self, batch_size):
        settings = make_config(batch_size=batch_size)
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
        for messages in model.prompts:
            text = messages[0]["content"]
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


if __name__ == "__main__":
    unittest.main()
