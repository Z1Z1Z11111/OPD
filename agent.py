"""ALFWorld Raw/ReAct adapted from PoS (see LICENSE).

Edit Agent to develop your method. The environment interface stays reset/step.
The default policy keeps raw history and repairs JSON. HiPER's ordinary ReAct
policy is implemented separately in hiper_react.py and shares the episode loop.
"""
from __future__ import annotations
import json
from collections.abc import Callable


SYSTEM_PROMPT = """You are an agent that solves interactive tasks with ReAct.
At each step, reason about the current task and observation, then choose exactly one action.
Use the thought to state what the latest observation established and what you still need to resolve.
The action will be sent directly to the benchmark environment, so copy its syntax exactly.
Return only one JSON object with this format:
{"thought": "brief reasoning", "action": "one action"}
Do not add Markdown fences or any text outside the JSON object.
"""


REPAIR_PROMPT = (
    "Return the same decision as one complete JSON object in the required schema."
)


ALFWORLD_EXAMPLE = """Example:
Task: put a clean bowl in a cabinet
Observation: You are in a kitchen. A countertop 1 and cabinet 1 are visible.
Available actions: ["go to countertop 1", "go to cabinet 1", "look"]
Response: {"thought": "I should inspect likely locations for the bowl first.", "action": "go to countertop 1"}
"""


def build_prompt(
    state: dict,
    trajectory: list[dict],
    config: dict,
) -> list[dict[str, str]]:
    """Build prompt."""

    example = ALFWORLD_EXAMPLE

    history = json.dumps(trajectory, ensure_ascii=False, indent=2)
    actions = state["actions"] if config["include_actions"] else []
    user_prompt = f"""{example}
Task:
{state["task"]}

Current observation:
{state["observation"]}

Available actions:
{json.dumps(actions, ensure_ascii=False)}

Previous steps:
{history}

Choose the next action and return the required JSON object.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_prompt(
    messages: list[dict[str, str]],
    raw_response: str,
) -> list[dict[str, str]]:
    """Build repair prompt."""

    return [
        *messages,
        {"role": "assistant", "content": raw_response},
        {"role": "user", "content": REPAIR_PROMPT},
    ]


class RawTrajectoryContext:
    """Keep the complete uncompressed interaction history within one episode."""

    def __init__(self):
        """Initialize configuration and episode-local runtime state."""

        self.trajectory: list[dict] = []

    def reset(self, state: dict) -> None:
        """Clear the previous episode history."""

        self.trajectory = []

    def update(
        self,
        transition: dict,
        state: dict,
        step: int,
    ) -> None:
        """Append the action-observation pair in execution order."""

        self.trajectory.append(transition)

    def get_context(self) -> dict:
        """Return the uncompressed trajectory for the Raw policy prompt."""

        return {
            "type": "raw",
            "trajectory": self.trajectory,
        }

    def get_action_override(self, state: dict) -> object | None:
        """Return no direct action override; actions are chosen by the task policy."""

        return None

    def get_result(self) -> dict:
        """Return no duplicate trajectory; the policy loop already exports it."""

        return {}


MISSING_ACTION_DIRECTIVE = 'Your previous JSON did not contain an executable environment action.\nThe environment is still running, so an "answer" field or a natural-language completion claim cannot finish the case.\nReturn one new decision that follows the system prompt\'s exact action schema and selects an action verbatim from Available actions.\nUse operation "act" when the current context protocol requires an operation field.\nTreat object and entity names as exact: a similarly named object does not satisfy the requested target.\nDo not return an answer, final response, or another completion claim without an executable action.\n'


class Agent:
    """Execute thought-action-observation decisions until termination or the turn budget."""

    def __init__(
        self,
        config: dict,
        call_model: Callable[[list[dict]], str],
        logger,
        context_provider: RawTrajectoryContext | None = None,
        progress: Callable | None = None,
    ):
        """Initialize configuration and episode-local runtime state."""
        self.config = config
        self.call_model = call_model
        self.logger = logger
        self.context_provider = context_provider or RawTrajectoryContext()
        self.progress = progress or (lambda **fields: None)

    def run(self, adapter) -> dict:
        """Drive the same episode generator with one model request at a time."""
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
        """Run one episode and return the full interaction trajectory and benchmark outcome."""
        state = adapter.reset()
        trajectory = []
        context_errors = []
        try:
            self.context_provider.reset(state)
            step = 0
            turn = 0
            note_turn = getattr(adapter, "note_turn", None)
            while turn < self.config["max_steps"]:
                self.progress(turn=turn, stage="building prompt", actions=step)
                if note_turn is not None:
                    note_turn(turn, self.config["max_steps"])
                messages = self._build_messages(state, turn=turn)
                decision = yield from self._request_decision(
                    messages, case_id=state["case_id"], step=step
                )
                if decision is None:
                    turn += 1
                    self.progress(turn=turn, stage="invalid decision", actions=step)
                    continue
                self.logger.log(
                    "react_model_decision",
                    case_id=state["case_id"],
                    step=step,
                    decision_source="model",
                    decision=decision,
                )
                if not self._has_executable_action(decision):
                    decision = yield from self._decide_after_rejected_operation(
                        messages=messages,
                        decision=decision,
                        state=state,
                        step=step,
                        event="react_missing_action",
                        directive=MISSING_ACTION_DIRECTIVE,
                    )
                    if decision is None:
                        turn += 1
                        self.progress(turn=turn, stage="missing action", actions=step)
                        continue
                for action in self._decision_actions(decision):
                    self.progress(stage=f"executing: {str(action)[:60]}")
                    state = adapter.step(action)
                    transition = {
                        "thought": decision["thought"],
                        "action": action,
                        "observation": state["observation"],
                    }
                    if "is_action_valid" in decision:
                        transition["is_action_valid"] = decision["is_action_valid"]
                    trajectory.append(transition)
                    try:
                        self.context_provider.update(transition, state, step)
                    except Exception as error:
                        if not state["done"]:
                            raise
                        context_errors.append(
                            self._record_terminal_context_error(
                                state, step, "update", error
                            )
                        )
                    step += 1
                    if state["done"]:
                        break
                turn += 1
                self.progress(turn=turn, stage="observed", actions=step)
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
            }
            if "benchmark_info" in state:
                result["benchmark_info"] = state["benchmark_info"]
            try:
                result.update(self.context_provider.get_result())
            except Exception as error:
                if not state["done"]:
                    raise
                context_errors.append(
                    self._record_terminal_context_error(
                        state, step - 1, "export", error
                    )
                )
            if context_errors:
                result["context_errors"] = context_errors
            self.progress(stage="success" if result["success"] else "finished", actions=step)
            return result
        except Exception:
            close_context = getattr(self.context_provider, "close", None)
            if close_context is not None:
                close_context()
            raise

    def _record_terminal_context_error(
        self, state: dict, step: int, phase: str, error: Exception
    ) -> dict:
        """Record a context failure without changing an authoritative terminal result."""
        detail = {
            "step": step,
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        self.logger.log(
            "terminal_context_error",
            case_id=state["case_id"],
            **detail,
        )
        return detail

    def _decide_after_rejected_operation(
        self,
        messages: list[dict],
        decision: dict,
        state: dict,
        step: int,
        event: str,
        directive: str,
        **log_fields,
    ) -> dict | None:
        """Request one executable replacement decision; charge a turn if still invalid."""
        self.logger.log(
            event,
            case_id=state["case_id"],
            step=step,
            rejected_decision=decision,
            **log_fields,
        )
        retry_messages = [
            *messages,
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
            {"role": "user", "content": directive},
        ]
        retried = yield from self._request_decision(
            retry_messages, case_id=state["case_id"], step=step
        )
        if retried is None:
            return None
        operation = str(retried.get("operation", "act")).strip().lower()
        if operation != "act" or not self._has_executable_action(retried):
            return None
        return retried

    def _has_executable_action(self, decision: dict) -> bool:
        """Check whether the current protocol can execute at least one proposed action."""
        if "action" in decision:
            return True
        actions = decision.get("actions")
        return (
            int(self.config.get("max_actions_per_turn", 1)) > 1
            and isinstance(actions, list)
            and bool(actions)
        )

    def _decision_actions(self, decision: dict) -> list:
        """Return at most max_actions_per_turn actions from a model decision."""
        limit = int(self.config.get("max_actions_per_turn", 1))
        actions = decision.get("actions")
        if limit > 1 and isinstance(actions, list) and actions:
            return actions[:limit]
        if "action" in decision:
            return [decision["action"]]
        raise ValueError("Agent decision is missing action")

    def _build_messages(self, state: dict, *, turn: int = 0) -> list[dict]:
        """Build the ALFWorld policy input using raw history."""
        context = self.context_provider.get_context()
        if context["type"] == "raw":
            return build_prompt(state, context["trajectory"], self.config)
        raise ValueError(f"Unsupported context: {context['type']}")

    def _request_decision(self, messages: list[dict], case_id: str, step: int) -> dict | None:
        """Request JSON, logging all attempts and applying bounded format repair."""
        for attempt in range(self.config["max_json_retries"] + 1):
            self.progress(stage=f"generating (attempt {attempt + 1})")
            self.logger.log(
                "react_model_input",
                case_id=case_id,
                step=step,
                attempt=attempt,
                messages=messages,
            )
            raw_response = yield messages
            self.logger.log(
                "react_model_response",
                case_id=case_id,
                step=step,
                attempt=attempt,
                raw_response=raw_response,
            )
            try:
                decision = json.loads(raw_response)
                if not isinstance(decision, dict) or not isinstance(decision.get("thought"), str):
                    raise ValueError("Decision must be a JSON object with a string thought")
                if "action" in decision and (
                    not isinstance(decision["action"], str) or not decision["action"].strip()
                ):
                    raise ValueError("Decision action must be a nonempty string")
                if "actions" in decision and (
                    not isinstance(decision["actions"], list)
                    or not all(isinstance(action, str) and action.strip() for action in decision["actions"])
                ):
                    raise ValueError("Decision actions must be a list of nonempty strings")
                return decision
            except ValueError as error:
                self.logger.log(
                    "react_model_parse_error",
                    case_id=case_id,
                    step=step,
                    attempt=attempt,
                    error=str(error),
                )
                if attempt == self.config["max_json_retries"]:
                    self.logger.log(
                        "react_model_retries_exhausted",
                        case_id=case_id,
                        step=step,
                        attempts=attempt + 1,
                    )
                    return None
                messages = build_repair_prompt(messages, raw_response)


def validate_agent_config(config: dict) -> None:
    """Reject unsupported policy/context combinations before starting inference."""
    style = config.get("react_style", "raw")
    context = config.get("context", {})
    if style == "raw":
        if context.get("type", "raw") != "raw":
            raise ValueError("Raw ReAct requires context.type=raw")
    elif style == "hiper":
        if context.get("type", "recent") != "recent":
            raise ValueError("HiPER ReAct requires context.type=recent")
        length = context.get("history_length", 2)
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("context.history_length must be a nonnegative integer")
        if not config.get("include_actions", True):
            raise ValueError("HiPER ReAct always includes admissible actions")
        if config.get("max_actions_per_turn", 1) != 1:
            raise ValueError("HiPER ReAct executes exactly one action per turn")
    else:
        raise ValueError(f"Unsupported react_style: {style}")


def create_agent(config, call_model, logger, **kwargs) -> Agent:
    """Select a policy while retaining the shared serial/batched episode API."""
    validate_agent_config(config)
    if config.get("react_style", "raw") == "hiper":
        from hiper_react import HiPERReActAgent
        return HiPERReActAgent(config, call_model, logger, **kwargs)
    return Agent(config, call_model, logger, **kwargs)
