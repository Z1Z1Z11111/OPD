"""Standalone task runner; decisions and interactions belong to Agent."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from tqdm.auto import tqdm

from agent import create_agent, validate_agent_config
from alfworld_env import ALFWorldEnv, ROOT, project_path
from typing import Any
from rl import evaluation_config
from report import EVAL_SPLITS, generate_report, task_statistics


class Logger:
    """Logger."""

    def __init__(self, path: Path):
        """Initialize configuration and episode-local runtime state."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def log(self, event: str, **data: Any) -> None:
        """Log."""

        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        """Close."""

        self.file.close()



def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_episode(env, config, call_model, logger, progress_callback=None):
    with tqdm(total=config["max_steps"], desc="Current task", unit="turn",
              position=1, leave=False, dynamic_ncols=True,
              disable=config.get("no_progress", False)) as progress:
        status = {"stage": "resetting", "actions": 0}
        completed_turns = 0
        progress.set_postfix(status)

        def update_progress(turn=None, **fields):
            nonlocal completed_turns
            status.update(fields)
            if turn is not None:
                completed_turns = turn
                progress.update(turn - progress.n)
            progress.set_postfix(status)
            if progress_callback:
                progress_callback(turn=completed_turns, **status)

        return create_agent(config, call_model, logger, progress=update_progress).run(env)


def evaluate(config, call_model, output, env_factory=ALFWorldEnv, progress_callback=None):
    if config.get("benchmark") != "ALFWorld":
        raise ValueError("This runner supports only ALFWorld")
    validate_agent_config(config)
    if config["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    if int(config.get("batch_size", 1)) > 1:
        from batched_evaluate import evaluate_batched
        return evaluate_batched(config, call_model, output, env_factory, progress_callback)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    results = []
    tqdm.write(f"Initializing environment: {config['split']}")
    env = env_factory(config)
    progress = tqdm(total=len(env), desc=config["split"], unit="task",
                    position=0, dynamic_ncols=True,
                    disable=config.get("no_progress", False))
    try:
        if len(env) == 0:
            raise ValueError("No tasks selected; check data and case range")
        for local_index in range(len(env)):
            index = (env.case_indices[local_index] if hasattr(env, "case_indices")
                     else config["start_case"] + local_index)
            if progress_callback:
                progress_callback(completed=len(results), successes=sum(r["success"] for r in results),
                                  case_index=index, turn=0, actions=0, stage="resetting")
            case_path = output / "cases" / f"case_{index:04d}"
            logger = Logger(case_path / "events.jsonl")
            try:
                before = call_model.token_usage()
                started = perf_counter()
                result = run_episode(env, config, call_model, logger, progress_callback)
                result["case_seconds"] = round(perf_counter() - started, 3)
                after = call_model.token_usage()
                result["token_usage"] = {name: after[name] - before[name] for name in after}
            except Exception as error:
                write_json(case_path / "error.json", dict(
                    case_index=index, error_type=type(error).__name__, error=str(error)))
                raise
            finally:
                logger.close()
            write_json(case_path / "result.json", result)
            results.append(result)
            write_json(output / "summary.json", dict(
                completed_cases=len(results), selected_cases=len(env),
                by_task_type=task_statistics(results),
                successes=sum(r["success"] for r in results),
                success_rate=sum(r["success"] for r in results) / len(results),
                total_steps=sum(r["steps"] for r in results),
                token_usage={name: sum(r["token_usage"][name] for r in results) for name in after},
                case_seconds=sum(r["case_seconds"] for r in results)))
            successes = sum(r["success"] for r in results)
            if progress_callback:
                progress_callback(completed=len(results), successes=successes, stage="task finished")
            progress.update(1)
            progress.set_postfix(success=f"{successes}/{len(results)}",
                                 rate=f"{successes / len(results):.1%}",
                                 last_seconds=result["case_seconds"])
            tqdm.write(f"{local_index + 1}/{len(env)} {result['case_id']} "
                       f"success={result['success']} actions={result['steps']} "
                       f"seconds={result['case_seconds']}")
    finally:
        progress.close()
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/alfworld.yaml")
    parser.add_argument("--split", choices=["seen", "unseen", "both"],
                        help="Run all six task types and the full selected split(s)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--method", choices=["react", "grpo"],
                        help="Select evaluation mode: react (base model) or grpo (GRPO checkpoint)")
    parser.add_argument("--checkpoint", help="Evaluate a specific exported GRPO Hugging Face model")
    args = parser.parse_args()
    import yaml
    from model import ModelCaller
    # Fail before expensive inference if the plotting dependency is missing.
    import matplotlib

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    config = evaluation_config(config, args.method)
    validate_agent_config(config)
    config["no_progress"] = args.no_progress
    output = project_path(config.get("output_dir", "outputs")) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tqdm.write(f"Loading model: {config['model']}")
    call_model = ModelCaller(config)
    tqdm.write("Model ready. Starting evaluation.")
    if args.split:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", dict(config, requested_split=args.split))
        selections = list(EVAL_SPLITS) if args.split == "both" else [args.split]
        for selection in selections:
            split_config = dict(config, start_case=0, end_case=None,
                                task_types=[1, 2, 3, 4, 5, 6],
                                split={"seen": "eval_in_distribution",
                                       "unseen": "eval_out_of_distribution"}[selection])
            try:
                evaluate(split_config, call_model, output / EVAL_SPLITS[selection])
            finally:
                tqdm.write("Generating charts...")
                generate_report(output, project_path(config.get("data_dir", "data/alfworld")))
    else:
        try:
            evaluate(config, call_model, output)
        finally:
            if output.exists():
                tqdm.write("Generating charts...")
                generate_report(output, project_path(config.get("data_dir", "data/alfworld")))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
