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
"""Ordinary ALFWorld ReAct adapted from HiPER-agent, without its RL stack.

Modified for this evaluator: single-episode history, generator-based model calls,
and diagnostics in the existing JSONL/result schema. See THIRD_PARTY_NOTICES.md.
"""
from collections import deque
import re

from agent import Agent, RawTrajectoryContext


# Preserve the upstream templates verbatim, including whitespace.
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


class HiPERReActContext(RawTrajectoryContext):
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

    def update(self, transition: dict, state: dict, step: int) -> None:
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
        return dict(type="recent", action_history="\n".join(lines),
                    history_length=len(self.history), step_count=self.step_count)

    def get_result(self) -> dict:
        return dict(react_style="hiper", history_length=self.history_length)


def project_response(raw_response: str) -> dict:
    """Match upstream alfworld_projection, including malformed-output fallbacks.

    is_action_valid indicates upstream format validity, not membership in the
    admissible action list. Invalid outputs still reach the environment once.
    """
    normalized = raw_response.lower()
    start = normalized.find("<action>")
    end = normalized.find("</action>")
    has_action_tags = start != -1 and end != -1
    action = (normalized[start + len("<action>"):end].strip()
              if has_action_tags else normalized[-30:])
    think_start = raw_response.find("<think>")
    think_end = raw_response.find("</think>")
    has_think_tags = think_start != -1 and think_end != -1
    thought = (raw_response[think_start + len("<think>"):think_end].strip()
               if has_think_tags else "")
    valid = (has_action_tags and has_think_tags
             and re.search(r"[\u4e00-\u9fff]", raw_response) is None)
    return dict(thought=thought, action=action, is_action_valid=bool(valid))


class HiPERReActAgent(Agent):
    """Use HiPER's flat <think>/<action> policy with the common episode loop."""

    def __init__(self, config, call_model, logger, context_provider=None, progress=None):
        context = context_provider or HiPERReActContext(
            config.get("context", {}).get("history_length", 2)
        )
        super().__init__(config, call_model, logger, context, progress)

    def _build_messages(self, state: dict, *, turn: int = 0) -> list[dict]:
        context = self.context_provider.get_context()
        actions = "\n ".join(f"'{action}'" for action in state["actions"] if action != "help")
        if context["step_count"] == 0 or self.context_provider.history_length <= 0:
            prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=state["observation"], admissible_actions=actions,
            )
        else:
            prompt = ALFWORLD_TEMPLATE.format(
                task_description=state["task"], step_count=context["step_count"],
                history_length=context["history_length"], action_history=context["action_history"],
                current_step=context["step_count"] + 1,
                current_observation=state["observation"], admissible_actions=actions,
            )
        # The upstream collector passes one user message to apply_chat_template.
        return [{"role": "user", "content": prompt}]

    def _request_decision(self, messages: list[dict], case_id: str, step: int):
        # No JSON repair or missing-action retry: one generation per env step.
        self.progress(stage="generating (attempt 1)")
        self.logger.log("react_model_input", case_id=case_id, step=step,
                        attempt=0, messages=messages)
        raw_response = yield messages
        self.logger.log("react_model_response", case_id=case_id, step=step,
                        attempt=0, raw_response=raw_response)
        decision = project_response(raw_response)
        self.logger.log("hiper_react_projection", case_id=case_id, step=step,
                        action=decision["action"], is_action_valid=decision["is_action_valid"])
        return decision
