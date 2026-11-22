# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ALFWorld ReAct policy aligned with the bundled GRPO training prompt/projection."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
import re


# Preserve upstream ALFWorld ReAct templates verbatim (including whitespace).
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def get_history_length(config: dict) -> int:
    if "history_length" in config:
        return config["history_length"]
    return config.get("context", {}).get("history_length", 2)


class RecentHistoryContext:
    """Keep recent pre-action observations/actions; thoughts stay in logs only."""

    def __init__(self, history_length: int = 2):
        self.history_length = history_length
        self.history = deque(maxlen=history_length)
        self.step_count = 0
        self.previous_observation = ""

    def reset(self, state: dict) -> None:
        self.history.clear()
        self.step_count = 0
        self.previous_observation = state["observation"]

    def update(self, transition: dict, state: dict) -> None:
        self.history.append({
            "observation": self.previous_observation,
            "action": transition["action"],
        })
        self.previous_observation = state["observation"]
        self.step_count += 1

    def get_context(self) -> dict:
        start = self.step_count - len(self.history) + 1
        lines = [
            f"[Observation {number}: '{record['observation']}', Action {number}: '{record['action']}']"
            for number, record in enumerate(self.history, start=start)
        ]
        return {
            "action_history": "\n".join(lines),
            "history_length": len(self.history),
            "step_count": self.step_count,
        }


def project_response(raw_response: str) -> dict:
    """Match training alfworld_projection, including malformed-output fallbacks.

    is_action_valid checks <think>/<action> tag format and absence of Chinese
    characters (not membership in admissible actions). Every generation still
    consumes one environment step without retries.
    """
    normalized = raw_response.lower()
    start = normalized.find("<action>")
    end = normalized.find("</action>")
    has_action_tags = start != -1 and end != -1
    action = (
        normalized[start + len("<action>"):end].strip()
        if has_action_tags
        else normalized[-30:]
    )
    think_start = raw_response.find("<think>")
    think_end = raw_response.find("</think>")
    has_think_tags = think_start != -1 and think_end != -1
    thought = (
        raw_response[think_start + len("<think>"):think_end].strip()
        if has_think_tags
        else ""
    )
    valid = (
        has_action_tags
        and has_think_tags
        and re.search(r"[\u4e00-\u9fff]", raw_response) is None
    )
    return {"thought": thought, "action": action, "is_action_valid": bool(valid)}


class Agent:
    """Run flat <think>/<action> ReAct episodes via a serial/batched generator."""

    def __init__(
        self,
        config: dict,
        call_model: Callable[[list[dict]], str],
        logger,
        context_provider: RecentHistoryContext | None = None,
        progress: Callable | None = None,
    ):
        self.config = config
        self.call_model = call_model
        self.logger = logger
        self.context_provider = context_provider or RecentHistoryContext(get_history_length(config))
        self.progress = progress or (lambda **fields: None)

    def run(self, adapter) -> dict:
        """Drive the episode generator with one model call at a time."""
        episode = self.run_iter(adapter)
        try:
            messages = next(episode)
            while True:
                messages = episode.send(self.call_model(messages))
        except StopIteration as finished:
            return finished.value
        finally:
            episode.close()

    def run_iter(self, adapter):
        """Yield model messages per turn and return the full episode outcome."""
        state = adapter.reset()
        trajectory = []
        self.context_provider.reset(state)

        for turn in range(self.config["max_steps"]):
            self.progress(turn=turn, stage="building prompt", actions=turn)
            messages = self._build_messages(state)
            decision = yield from self._request_decision(
                messages, case_id=state["case_id"], step=turn
            )
            action = decision["action"]
            self.progress(stage=f"executing: {str(action)[:60]}")
            state = adapter.step(action)
            transition = {
                "thought": decision["thought"],
                "action": action,
                "observation": state["observation"],
                "is_action_valid": decision["is_action_valid"],
            }
            trajectory.append(transition)
            self.context_provider.update(transition, state)
            self.progress(turn=turn + 1, stage="observed", actions=turn + 1)
            if state["done"]:
                break

        result = {
            "case_id": state["case_id"],
            "task": state["task"],
            "success": state["success"],
            "score": state["score"],
            "steps": len(trajectory),
            "final_observation": state["observation"],
            "trajectory": trajectory,
            "history_length": self.context_provider.history_length,
        }
        self.progress(stage="success" if result["success"] else "finished", actions=len(trajectory))
        return result

    def _build_messages(self, state: dict) -> list[dict]:
        context = self.context_provider.get_context()
        actions = "\n ".join(f"'{action}'" for action in state["actions"] if action != "help")
        if context["step_count"] == 0 or self.context_provider.history_length <= 0:
            prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=state["observation"],
                admissible_actions=actions,
            )
        else:
            prompt = ALFWORLD_TEMPLATE.format(
                task_description=state["task"],
                step_count=context["step_count"],
                history_length=context["history_length"],
                action_history=context["action_history"],
                current_step=context["step_count"] + 1,
                current_observation=state["observation"],
                admissible_actions=actions,
            )
        return [{"role": "user", "content": prompt}]

    def _request_decision(self, messages: list[dict], case_id: str, step: int):
        self.progress(stage="generating")
        self.logger.log("react_model_input", case_id=case_id, step=step, messages=messages)
        raw_response = yield messages
        self.logger.log("react_model_response", case_id=case_id, step=step, raw_response=raw_response)
        decision = project_response(raw_response)
        self.logger.log(
            "react_model_decision",
            case_id=case_id,
            step=step,
            action=decision["action"],
            is_action_valid=decision["is_action_valid"],
        )
        return decision


def validate_agent_config(config: dict) -> None:
    """Validate ALFWorld ReAct policy configuration before loading models."""
    length = get_history_length(config)
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("history_length must be a nonnegative integer")


def create_agent(config, call_model, logger, **kwargs) -> Agent:
    validate_agent_config(config)
    return Agent(config, call_model, logger, **kwargs)
