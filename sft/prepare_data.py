"""Replay Agent-G2 ALFWorld expert trajectories in the local ALFWorld 0.4.2 env
and render step-level SFT pairs strictly aligned with agent.py and rl/hiper GRPO.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    RecentHistoryContext,
    get_history_length,
    project_response,
)
from alfworld_env import project_path


def _extract_task(observation: str) -> str:
    marker = "Your task is to:"
    if marker not in observation:
        raise ValueError(f"Task marker missing from observation: {observation[:120]!r}")
    return observation.split(marker, maxsplit=1)[1].strip()


def _build_user_prompt(
    context_provider: RecentHistoryContext,
    observation: str,
    task: str,
    admissible_commands: list[str],
) -> str:
    context = context_provider.get_context()
    actions = "\n ".join(f"'{action}'" for action in admissible_commands if action != "help")
    if context["step_count"] == 0 or context_provider.history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=observation,
            admissible_actions=actions,
        )
    return ALFWORLD_TEMPLATE.format(
        task_description=task,
        step_count=context["step_count"],
        history_length=context["history_length"],
        action_history=context["action_history"],
        current_step=context["step_count"] + 1,
        current_observation=observation,
        admissible_actions=actions,
    )


def _format_assistant(thought: str, action: str) -> str:
    thought_clean = thought.strip()
    action_clean = action.strip()
    return f"<think>{thought_clean}</think>\n<action>{action_clean}</action>"


def _env_config(data_dir: Path, task_types: list[int], max_steps: int) -> dict:
    return {
        "dataset": {
            "data_path": str(data_dir / "json_2.1.1" / "train"),
            "eval_id_data_path": str(data_dir / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(data_dir / "json_2.1.1" / "valid_unseen"),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": str(data_dir / "logic" / "alfred.pddl"),
            "grammar": str(data_dir / "logic" / "alfred.twl2"),
        },
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False,
            "task_types": task_types,
            "expert_type": "handcoded",
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


def _episode_key_from_gamefile(game_file: str | Path) -> str:
    path = Path(game_file)
    return f"{path.parent.parent.name}_{path.parent.name}"


def _replay_shard(
    shard_id: int,
    items: list[tuple[dict, str]],
    max_steps: int,
    history_length: int,
    require_admissible: bool,
    require_won: bool,
) -> dict:
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    # Identical to rl/hiper's AlfredTWEnv.init_env (use_expert=False, domain_randomization=False),
    # without scanning all 8,810 files per worker or running the slow HandCodedExpert on every step.
    game_files = [game_file for _, game_file in items]
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(
        game_files,
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=wrappers,
    )

    # Map game_file -> raw record so we stay immune to any internal ordering.
    record_by_game = {str(Path(game_file).resolve()): record for record, game_file in items}

    env = textworld.gym.make(env_id)
    kept_steps: list[dict] = []
    reasons: Counter = Counter()
    kept_episodes = 0
    seen_games: set[str] = set()

    try:
        total = len(items)
        for idx in range(1, total + 1):
            observations, infos = env.reset()
            game_file = str(Path(infos["extra.gamefile"][0]).resolve())
            if game_file in seen_games:
                reasons["duplicate_reset"] += 1
                continue
            seen_games.add(game_file)

            record = record_by_game.get(game_file)
            if record is None:
                reasons["unmatched_gamefile"] += 1
                continue

            actions = record.get("actions") or []
            thoughts = record.get("think") or []
            if not actions or len(actions) != len(thoughts):
                reasons["malformed_trajectory"] += 1
                continue

            obs = observations[0]
            task = _extract_task(obs)
            admissible = list(infos["admissible_commands"][0])
            ctx = RecentHistoryContext(history_length=history_length)
            ctx.reset({"observation": obs})

            episode_steps: list[dict] = []
            failed_reason: str | None = None
            won = False

            for step_idx, (gold_action, gold_thought) in enumerate(zip(actions, thoughts)):
                action_norm = gold_action.strip()
                if require_admissible and action_norm not in admissible:
                    # Try case-insensitive exact match against admissible commands.
                    lower_map = {cmd.lower(): cmd for cmd in admissible}
                    if action_norm.lower() in lower_map:
                        action_norm = lower_map[action_norm.lower()]
                    else:
                        failed_reason = "inadmissible_action"
                        break

                user_prompt = _build_user_prompt(ctx, obs, task, admissible)
                assistant_reply = _format_assistant(gold_thought, action_norm)
                decision = project_response(assistant_reply)
                if not decision["is_action_valid"] or decision["action"] != action_norm.lower():
                    failed_reason = "invalid_projection"
                    break

                episode_steps.append(
                    {
                        "episode_id": record["id"],
                        "task_type": record["task_type"],
                        "step": step_idx,
                        "action": action_norm,
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_reply},
                        ],
                    }
                )

                next_obs_list, _, dones, next_infos = env.step([action_norm])
                next_obs = next_obs_list[0]
                won = bool(next_infos["won"][0])
                done = bool(dones[0])
                ctx.update({"action": action_norm}, {"observation": next_obs})
                obs = next_obs
                admissible = list(next_infos["admissible_commands"][0])

                if done and step_idx < len(actions) - 1:
                    if not won:
                        failed_reason = "ended_early_without_win"
                    break

            if failed_reason is not None:
                reasons[failed_reason] += 1
                continue
            if require_won and not won:
                reasons["episode_not_won"] += 1
                continue

            kept_episodes += 1
            kept_steps.extend(episode_steps)

            if idx % 100 == 0 or idx == total:
                print(
                    f"[worker {shard_id}] {idx}/{total} episodes (kept={kept_episodes}, steps={len(kept_steps)})",
                    flush=True,
                )
    finally:
        env.close()

    return {
        "steps": kept_steps,
        "kept_episodes": kept_episodes,
        "reasons": dict(reasons),
    }


def _filter_by_token_budget(
    steps: list[dict],
    model_path: Path,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    filtered: list[dict] = []
    dropped_prompt = 0
    dropped_response = 0
    dropped_prefix = 0
    prompt_lens: list[int] = []
    response_lens: list[int] = []

    for row in steps:
        messages = row["messages"]
        prompt_text = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            dropped_prefix += 1
            continue
        resp_len = len(full_ids) - len(prompt_ids)
        if len(prompt_ids) > max_prompt_tokens:
            dropped_prompt += 1
            continue
        if resp_len > max_response_tokens or resp_len <= 0:
            dropped_response += 1
            continue

        row["prompt_tokens"] = len(prompt_ids)
        row["response_tokens"] = resp_len
        prompt_lens.append(len(prompt_ids))
        response_lens.append(resp_len)
        filtered.append(row)

    def _quantiles(values: list[int]) -> dict:
        if not values:
            return {"min": 0, "p50": 0, "p95": 0, "max": 0}
        ordered = sorted(values)
        n = len(ordered)
        return {
            "min": ordered[0],
            "p50": ordered[int(0.50 * (n - 1))],
            "p95": ordered[int(0.95 * (n - 1))],
            "max": ordered[-1],
        }

    return filtered, {
        "dropped_overlong_prompt": dropped_prompt,
        "dropped_overlong_response": dropped_response,
        "dropped_chat_template_prefix_mismatch": dropped_prefix,
        "prompt_tokens": _quantiles(prompt_lens),
        "response_tokens": _quantiles(response_lens),
    }


def _stratified_subsample(records: list[dict], max_episodes: int, seed: int) -> list[dict]:
    if max_episodes <= 0 or max_episodes >= len(records):
        return records
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["task_type"]].append(r)
    for group in by_type.values():
        rng.shuffle(group)

    total = len(records)
    selected: list[dict] = []
    remainders: list[tuple[float, str]] = []
    allocated: dict[str, int] = {}
    for task_type, group in sorted(by_type.items()):
        exact = max_episodes * len(group) / total
        count = int(exact)
        allocated[task_type] = min(count, len(group))
        remainders.append((exact - count, task_type))

    shortfall = max_episodes - sum(allocated.values())
    for _, task_type in sorted(remainders, reverse=True):
        if shortfall <= 0:
            break
        if allocated[task_type] < len(by_type[task_type]):
            allocated[task_type] += 1
            shortfall -= 1

    for task_type, count in sorted(allocated.items()):
        selected.extend(by_type[task_type][:count])
    rng.shuffle(selected)
    return selected


def _split_by_episode(steps: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    episode_type: dict[str, str] = {}
    for step in steps:
        ep = step["episode_id"]
        by_episode[ep].append(step)
        episode_type[ep] = step["task_type"]

    if val_ratio <= 0 or len(by_episode) < 2:
        return steps, []

    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for ep, ttype in episode_type.items():
        by_type[ttype].append(ep)

    val_episodes: set[str] = set()
    for ttype, eps in sorted(by_type.items()):
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * val_ratio))) if len(eps) > 1 else 0
        val_episodes.update(eps[:n_val])

    train_steps: list[dict] = []
    val_steps: list[dict] = []
    for step in steps:
        if step["episode_id"] in val_episodes:
            val_steps.append(step)
        else:
            train_steps.append(step)
    return train_steps, val_steps


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/sft/Agent-G2-ALFWorld-Webshop-sft-data/alfworld_sft_data.jsonl",
        help="Path to raw Agent-G2 alfworld_sft_data.jsonl",
    )
    parser.add_argument(
        "--config",
        default="configs/alfworld.yaml",
        help="Base ALFWorld config (for data_dir, model, history_length, token limits)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sft/alfworld_hiper_h2",
        help="Directory to write train_steps.jsonl, val_steps.jsonl, and stats.json",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel replay worker processes")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fraction of episodes for validation")
    parser.add_argument("--max-episodes", type=int, default=0, help="Optional stratified cap on episodes (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-inadmissible", action="store_true", help="Do not drop episodes with inadmissible actions")
    parser.add_argument("--allow-unwon", action="store_true", help="Do not drop episodes that do not end in won=True")
    args = parser.parse_args()

    base = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    source_path = project_path(args.source).resolve()
    data_dir = project_path(base.get("data_dir", "data/alfworld")).resolve()
    model_path = project_path(base["model"]).resolve()
    output_dir = project_path(args.output_dir)
    history_length = get_history_length(base)
    task_types = list(base.get("task_types", [1, 2, 3, 4, 5, 6]))
    max_steps = int(base.get("max_steps", 50))
    max_prompt_tokens = int(base.get("max_prompt_tokens", 2048))
    max_response_tokens = int(base.get("max_tokens", 512))

    with source_path.open("r", encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp if line.strip()]
    if args.max_episodes > 0:
        records = _stratified_subsample(records, args.max_episodes, args.seed)

    # Index all local ALFWorld train game files.
    train_root = data_dir / "json_2.1.1" / "train"
    game_index: dict[str, str] = {}
    for game_file in sorted(train_root.rglob("game.tw-pddl")):
        game_index[_episode_key_from_gamefile(game_file)] = str(game_file)

    matched: list[tuple[dict, str]] = []
    missing_ids: list[str] = []
    for rec in records:
        gf = game_index.get(rec["id"])
        if gf is None:
            missing_ids.append(rec["id"])
        else:
            matched.append((rec, gf))

    print(
        f"Loaded {len(records)} records; matched {len(matched)} local train games "
        f"(missing {len(missing_ids)}). Replaying with {args.workers} workers, history_length={history_length}...",
        flush=True,
    )

    workers = max(1, min(args.workers, len(matched))) if matched else 1
    shards: list[list[tuple[dict, str]]] = [matched[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]

    all_steps: list[dict] = []
    kept_episodes = 0
    drop_reasons: Counter = Counter()
    if missing_ids:
        drop_reasons["missing_local_game"] = len(missing_ids)

    if workers == 1:
        res = _replay_shard(
            0,
            shards[0],
            max_steps,
            history_length,
            not args.allow_inadmissible,
            not args.allow_unwon,
        )
        all_steps.extend(res["steps"])
        kept_episodes += res["kept_episodes"]
        drop_reasons.update(res["reasons"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _replay_shard,
                    shard_id,
                    shard,
                    max_steps,
                    history_length,
                    not args.allow_inadmissible,
                    not args.allow_unwon,
                )
                for shard_id, shard in enumerate(shards)
            ]
            for fut in as_completed(futures):
                res = fut.result()
                all_steps.extend(res["steps"])
                kept_episodes += res["kept_episodes"]
                drop_reasons.update(res["reasons"])

    # Deterministic ordering before token filtering & split.
    all_steps.sort(key=lambda r: (r["episode_id"], r["step"]))
    print(
        f"Replay complete: kept {kept_episodes}/{len(matched)} episodes ({len(all_steps)} steps). "
        f"Drop reasons: {dict(drop_reasons)}",
        flush=True,
    )

    filtered_steps, token_stats = _filter_by_token_budget(
        all_steps, model_path, max_prompt_tokens, max_response_tokens
    )
    train_steps, val_steps = _split_by_episode(filtered_steps, args.val_ratio, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train_steps.jsonl", train_steps)
    _write_jsonl(output_dir / "val_steps.jsonl", val_steps)

    train_eps = {s["episode_id"] for s in train_steps}
    val_eps = {s["episode_id"] for s in val_steps}
    task_type_counts = Counter(s["task_type"] for s in filtered_steps)
    verb_counts = Counter(s["action"].split()[0].lower() for s in filtered_steps if s["action"])

    stats = {
        "source": str(source_path),
        "config": str(project_path(args.config)),
        "model_tokenizer": str(model_path),
        "history_length": history_length,
        "max_prompt_tokens": max_prompt_tokens,
        "max_response_tokens": max_response_tokens,
        "raw_episodes": len(records),
        "matched_episodes": len(matched),
        "replayed_kept_episodes": kept_episodes,
        "drop_reasons": dict(drop_reasons),
        "token_filter": token_stats,
        "final_steps": len(filtered_steps),
        "train_episodes": len(train_eps),
        "train_steps": len(train_steps),
        "val_episodes": len(val_eps),
        "val_steps": len(val_steps),
        "task_type_steps": dict(task_type_counts),
        "action_verbs": dict(verb_counts.most_common()),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(train_steps)} train steps ({len(train_eps)} episodes) and "
        f"{len(val_steps)} val steps ({len(val_eps)} episodes) to {output_dir}",
        flush=True,
    )
    print("Token stats:", json.dumps(token_stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
