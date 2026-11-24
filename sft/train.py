"""Standalone PyTorch/Transformers SFT trainer aligned with ALFWorld ReAct & GRPO.

Supports:
- Standard full-parameter SFT (assistant-only loss, exact Qwen2.5 chat template)
- Optional OPC-SFT (ICML 2026) low-probability token clipping (`opc_clip_eps > 0`)
- Single-GPU or multi-GPU DDP via `torchrun`
- Standard Hugging Face checkpoint export (`hf_model/`) ready for `./train-rl --model`
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from tempfile import NamedTemporaryFile, mkdtemp
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfworld_env import project_path


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_sft_example(
    messages: list[dict],
    tokenizer,
    max_seq_len: int | None = None,
) -> dict[str, list[int]]:
    """Encode [user, assistant] with exact inference chat template and assistant-only labels."""
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError("Each SFT sample must contain exactly one user message and one assistant message")

    prompt_text = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt prefix mismatch between prompt_ids and full_ids")
    if max_seq_len is not None and len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    if all(x == -100 for x in labels):
        raise ValueError("Example has zero supervised assistant tokens after truncation")
    return {"input_ids": full_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int | None = None, limit: int = 0):
        self.samples: list[dict[str, list[int]]] = []
        with path.open("r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                if limit > 0 and idx >= limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                self.samples.append(encode_sft_example(row["messages"], tokenizer, max_seq_len))
        if not self.samples:
            raise ValueError(f"No valid SFT examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


class PadCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of > 1:
            rem = max_len % self.pad_to_multiple_of
            if rem:
                max_len += self.pad_to_multiple_of - rem

        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    opc_clip_eps: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute assistant-only NLL loss with optional OPC-SFT low-probability token clipping.

    When `opc_clip_eps > 0`, tokens with current model probability `p_t < opc_clip_eps`
    have their loss clamped at `-log(opc_clip_eps)` (equivalently zero gradient on those
    low-probability tokens), while normalizing by total valid assistant tokens to preserve
    the gradient scale on unclipped tokens.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    valid_count = valid_mask.sum().clamp(min=1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp(min=0)
    target_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    raw_nll = -target_log_probs

    with torch.no_grad():
        probs_all = torch.exp(log_probs)
        token_entropy = -(probs_all * log_probs).sum(dim=-1)
        mean_entropy = (token_entropy * valid_mask).sum() / valid_count
        mean_raw_nll = (raw_nll * valid_mask).sum() / valid_count

    if opc_clip_eps > 0.0:
        max_nll = -math.log(opc_clip_eps)
        clipped_mask = valid_mask & (raw_nll.detach() > max_nll)
        clipped_ratio = clipped_mask.sum().float() / valid_count.float()
        # Clamping at max_nll stops gradients where p_t < opc_clip_eps (OPC-SFT).
        clamped_nll = torch.clamp(raw_nll, max=max_nll)
        loss = (clamped_nll * valid_mask).sum() / valid_count
    else:
        clipped_ratio = torch.zeros((), device=logits.device)
        loss = (raw_nll * valid_mask).sum() / valid_count

    metrics = {
        "loss": float(loss.detach().item()),
        "raw_nll": float(mean_raw_nll.item()),
        "entropy": float(mean_entropy.item()),
        "clip_ratio": float(clipped_ratio.item()),
        "valid_tokens": float(valid_mask.sum().item()),
    }
    return loss, metrics


def _all_reduce_mean(value: float, device: torch.device) -> float:
    if not _is_dist():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / _world_size()).item())


def _all_reduce_sum(value: float, device: torch.device) -> float:
    if not _is_dist():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


@torch.no_grad()
def evaluate(
    model,
    val_loader: DataLoader | None,
    device: torch.device,
    opc_clip_eps: float,
) -> dict[str, float]:
    if val_loader is None:
        return {}
    model.eval()
    total_tokens = 0.0
    weighted_loss = 0.0
    weighted_nll = 0.0
    weighted_ent = 0.0
    weighted_clip = 0.0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            _, m = compute_sft_loss(outputs.logits, labels, opc_clip_eps=opc_clip_eps)
        n = m["valid_tokens"]
        total_tokens += n
        weighted_loss += m["loss"] * n
        weighted_nll += m["raw_nll"] * n
        weighted_ent += m["entropy"] * n
        weighted_clip += m["clip_ratio"] * n

    total_tokens = _all_reduce_sum(total_tokens, device)
    weighted_loss = _all_reduce_sum(weighted_loss, device)
    weighted_nll = _all_reduce_sum(weighted_nll, device)
    weighted_ent = _all_reduce_sum(weighted_ent, device)
    weighted_clip = _all_reduce_sum(weighted_clip, device)
    denom = max(total_tokens, 1.0)
    model.train()
    return {
        "val_loss": weighted_loss / denom,
        "val_nll": weighted_nll / denom,
        "val_entropy": weighted_ent / denom,
        "val_clip_ratio": weighted_clip / denom,
        "val_tokens": total_tokens,
    }


def save_hf_checkpoint(model, tokenizer, base_model_path: Path, export_dir: Path) -> None:
    """Save a complete Hugging Face directory directly consumable by vLLM and ./train-rl."""
    if not _is_main():
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.save_pretrained(str(export_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(export_dir))

    # Preserve generation_config.json from the base model if not written.
    src_gen = base_model_path / "generation_config.json"
    dst_gen = export_dir / "generation_config.json"
    if src_gen.is_file() and not dst_gen.is_file():
        shutil.copy2(src_gen, dst_gen)

    if not (export_dir / "config.json").is_file() or not any(export_dir.glob("*.safetensors")):
        raise RuntimeError(f"Incomplete Hugging Face export at {export_dir}")


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return max_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _setup_distributed() -> tuple[torch.device, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("SFT training requires CUDA GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return device, local_rank


def _load_settings(base_cfg_path: str, recipe_path: str, args) -> dict:
    base = yaml.safe_load(project_path(base_cfg_path).read_text(encoding="utf-8"))
    recipe = yaml.safe_load(project_path(recipe_path).read_text(encoding="utf-8"))
    cfg = dict(recipe)
    cfg.setdefault("model", base["model"])
    cfg.setdefault("max_seq_len", int(base.get("max_prompt_tokens", 2048)) + int(base.get("max_tokens", 512)))
    cfg.setdefault("seed", int(base.get("seed", 42)))

    if args.model:
        cfg["model"] = args.model
    if args.train_file:
        cfg["train_file"] = args.train_file
    if args.val_file:
        cfg["val_file"] = args.val_file
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.opc_clip_eps is not None:
        cfg["opc_clip_eps"] = args.opc_clip_eps
    if args.experiment_name:
        cfg["experiment_name"] = args.experiment_name

    if args.smoke:
        cfg["epochs"] = 1
        cfg["max_steps"] = 4
        cfg["train_limit"] = 32
        cfg["val_limit"] = 16
        cfg["micro_batch_size"] = 2
        cfg["global_batch_size"] = max(2 * _world_size(), 4)
        cfg["log_every"] = 1
        cfg["use_swanlab"] = False
    return cfg


def train(cfg: dict, smoke: bool, dry_run: bool) -> None:
    model_path = project_path(cfg["model"]).resolve()
    train_file = project_path(cfg["train_file"]).resolve()
    val_file = project_path(cfg["val_file"]).resolve() if cfg.get("val_file") else None
    output_base = project_path(cfg.get("output_dir", "outputs/sft"))

    if dry_run:
        if _is_main():
            resolved = dict(cfg)
            resolved["model"] = str(model_path)
            resolved["train_file"] = str(train_file)
            resolved["val_file"] = str(val_file) if val_file else None
            print(yaml.safe_dump(resolved, sort_keys=False))
        return

    for req in (model_path / "config.json", train_file):
        if not req.exists():
            raise FileNotFoundError(req)
    if val_file is not None and not val_file.exists():
        val_file = None

    device, local_rank = _setup_distributed()
    world_size = _world_size()
    _seed_everything(int(cfg["seed"]) + _rank())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = int(cfg["max_seq_len"])
    train_ds = SFTDataset(train_file, tokenizer, max_seq_len=max_seq_len, limit=int(cfg.get("train_limit", 0)))
    val_ds = (
        SFTDataset(val_file, tokenizer, max_seq_len=max_seq_len, limit=int(cfg.get("val_limit", 0)))
        if val_file is not None
        else None
    )

    micro_bs = int(cfg.get("micro_batch_size", 4))
    global_bs = int(cfg.get("global_batch_size", 64))
    if global_bs < micro_bs * world_size or global_bs % (micro_bs * world_size) != 0:
        raise ValueError(
            f"global_batch_size ({global_bs}) must be a positive multiple of "
            f"micro_batch_size ({micro_bs}) * world_size ({world_size})"
        )
    grad_accum = global_bs // (micro_bs * world_size)

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=int(cfg["seed"])) if _is_dist() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=True,
        drop_last=True if len(train_ds) >= micro_bs * world_size else False,
    )

    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False) if _is_dist() else None
        val_loader = DataLoader(
            val_ds,
            batch_size=micro_bs,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=int(cfg.get("num_workers", 2)),
            pin_memory=True,
            drop_last=False,
        )

    # Create run directory on rank 0 and broadcast path.
    model_root = output_base / model_path.name
    run_dir_str = ""
    if _is_main():
        model_root.mkdir(parents=True, exist_ok=True)
        exp_name = cfg.get("experiment_name", "sft")
        prefix = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{exp_name}-"
        run_dir = Path(mkdtemp(prefix=prefix, dir=model_root))
        run_dir_str = str(run_dir)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(dict(cfg, world_size=world_size, grad_accum=grad_accum), sort_keys=False),
            encoding="utf-8",
        )
    if _is_dist():
        holder = [run_dir_str]
        dist.broadcast_object_list(holder, src=0)
        run_dir_str = holder[0]
    run_dir = Path(run_dir_str)

    if _is_main():
        print(
            f"SFT run dir: {run_dir}\n"
            f"Model: {model_path}\n"
            f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}\n"
            f"World size: {world_size}, micro_bs: {micro_bs}, grad_accum: {grad_accum}, global_bs: {global_bs}\n"
            f"LR: {cfg['lr']}, epochs: {cfg['epochs']}, opc_clip_eps: {cfg.get('opc_clip_eps', 0.0)}",
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=False,
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    if _is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Standard AdamW with no weight decay on bias / LayerNorm.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    weight_decay = float(cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    epochs = int(cfg["epochs"])
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = epochs * steps_per_epoch
    max_steps = int(cfg.get("max_steps", 0))
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(round(total_steps * float(cfg.get("warmup_ratio", 0.05))))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    opc_clip_eps = float(cfg.get("opc_clip_eps") or 0.0)
    log_every = max(1, int(cfg.get("log_every", 10)))

    swanlab_run = None
    if _is_main() and cfg.get("use_swanlab", False):
        try:
            import swanlab

            swanlab_run = swanlab.init(
                project=cfg.get("project_name", "opd_alfworld"),
                experiment_name=cfg.get("experiment_name", "sft_qwen2.5_1.5b"),
                config=dict(cfg, world_size=world_size, grad_accum=grad_accum),
                mode=cfg.get("swanlab_mode", "local"),
            )
        except Exception as exc:
            print(f"[warn] SwanLab disabled ({exc})", flush=True)

    metrics_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8") if _is_main() else None

    def _log_metrics(record: dict) -> None:
        if not _is_main():
            return
        if metrics_file is not None:
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_file.flush()
        if swanlab_run is not None:
            try:
                import swanlab

                swanlab.log({k: v for k, v in record.items() if isinstance(v, (int, float))})
            except Exception:
                pass

    # Initial validation before training (baseline reference).
    if val_loader is not None and cfg.get("val_before_train", True):
        val_m = evaluate(model, val_loader, device, opc_clip_eps)
        if _is_main():
            print(
                f"[eval step=0] val_nll={val_m['val_nll']:.4f} val_loss={val_m['val_loss']:.4f} "
                f"val_entropy={val_m['val_entropy']:.4f} val_clip={val_m['val_clip_ratio']:.4f}",
                flush=True,
            )
            _log_metrics({"step": 0, "epoch": 0, **val_m})

    global_step = 0
    best_val_nll = float("inf")
    latest_export: Path | None = None
    best_export: Path | None = None
    stop_early = False

    model.train()
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_nll = 0.0
        accum_ent = 0.0
        accum_clip = 0.0
        accum_tokens = 0.0
        micro_in_window = 0
        t_step = time.perf_counter()

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            is_update_step = (batch_idx % grad_accum == 0) or (batch_idx == num_batches)
            micro_in_window += 1

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            sync_ctx = model.no_sync() if (_is_dist() and not is_update_step) else _nullcontext()
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    loss, m = compute_sft_loss(outputs.logits, labels, opc_clip_eps=opc_clip_eps)
                # Scale by actual window length (handles final partial accumulation cleanly).
                window_denom = grad_accum if (batch_idx <= (num_batches // grad_accum) * grad_accum) else (num_batches % grad_accum or grad_accum)
                (loss / window_denom).backward()

            accum_loss += m["loss"]
            accum_nll += m["raw_nll"]
            accum_ent += m["entropy"]
            accum_clip += m["clip_ratio"]
            accum_tokens += float(input_ids.numel())

            if not is_update_step:
                continue

            lr = _cosine_lr(global_step, total_steps, warmup_steps, float(cfg["lr"]), min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            elapsed = max(time.perf_counter() - t_step, 1e-6)
            step_loss = _all_reduce_mean(accum_loss / micro_in_window, device)
            step_nll = _all_reduce_mean(accum_nll / micro_in_window, device)
            step_ent = _all_reduce_mean(accum_ent / micro_in_window, device)
            step_clip = _all_reduce_mean(accum_clip / micro_in_window, device)
            step_tokens = _all_reduce_sum(accum_tokens, device)
            tok_per_s = step_tokens / elapsed

            if global_step == 1 or global_step % log_every == 0 or global_step >= total_steps:
                if _is_main():
                    print(
                        f"[epoch {epoch}/{epochs} step {global_step}/{total_steps}] "
                        f"loss={step_loss:.4f} nll={step_nll:.4f} ent={step_ent:.4f} "
                        f"clip={step_clip:.4f} grad={grad_norm:.3f} lr={lr:.2e} tok/s={tok_per_s:.0f}",
                        flush=True,
                    )
                    _log_metrics(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "train_loss": step_loss,
                            "train_nll": step_nll,
                            "train_entropy": step_ent,
                            "train_clip_ratio": step_clip,
                            "grad_norm": grad_norm,
                            "lr": lr,
                            "tokens_per_sec": tok_per_s,
                        }
                    )

            accum_loss = accum_nll = accum_ent = accum_clip = accum_tokens = 0.0
            micro_in_window = 0
            t_step = time.perf_counter()

            if max_steps > 0 and global_step >= max_steps:
                stop_early = True
                break

        # End-of-epoch evaluation & checkpoint export.
        val_m = evaluate(model, val_loader, device, opc_clip_eps) if val_loader is not None else {}
        if val_m and _is_main():
            print(
                f"[eval epoch={epoch} step={global_step}] val_nll={val_m['val_nll']:.4f} "
                f"val_loss={val_m['val_loss']:.4f} val_entropy={val_m['val_entropy']:.4f} "
                f"val_clip={val_m['val_clip_ratio']:.4f}",
                flush=True,
            )
            _log_metrics({"step": global_step, "epoch": epoch, **val_m})

        epoch_export = run_dir / f"epoch_{epoch}" / "hf_model"
        if _is_dist():
            dist.barrier()
        save_hf_checkpoint(model, tokenizer, model_path, epoch_export)
        if _is_dist():
            dist.barrier()
        latest_export = epoch_export
        if val_m and val_m["val_nll"] < best_val_nll:
            best_val_nll = val_m["val_nll"]
            best_export = epoch_export

        if stop_early:
            break

    if _is_main() and latest_export is not None:
        manifest = {
            "base_model": str(model_path),
            "checkpoint": str(latest_export.relative_to(model_root)),
            "best_checkpoint": str((best_export or latest_export).relative_to(model_root)),
            "step": global_step,
            "opc_clip_eps": opc_clip_eps,
            "experiment_name": cfg.get("experiment_name", "sft"),
        }
        target = model_root / ("smoke.json" if smoke else "latest.json")
        with NamedTemporaryFile("w", dir=model_root, delete=False, encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)
        Path(fp.name).replace(target)
        print(f"Latest HF checkpoint: {latest_export}", flush=True)
        if best_export is not None:
            print(f"Best-val HF checkpoint: {best_export}", flush=True)
        print(f"Next step (GRPO): ./train-rl --recipe configs/grpo_from_sft.yaml --model {latest_export}", flush=True)

    if metrics_file is not None:
        metrics_file.close()
    if swanlab_run is not None:
        try:
            import swanlab

            swanlab.finish()
        except Exception:
            pass
    if _is_dist():
        dist.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alfworld.yaml", help="Base ALFWorld config")
    parser.add_argument("--recipe", default="configs/sft.yaml", help="SFT training recipe YAML")
    parser.add_argument("--model", help="Override base model directory")
    parser.add_argument("--train-file", help="Override train_steps.jsonl path")
    parser.add_argument("--val-file", help="Override val_steps.jsonl path")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--opc-clip-eps", type=float, help="Override OPC-SFT low-prob clipping epsilon")
    parser.add_argument("--experiment-name", help="Override experiment name")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny 4-step smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without training")
    args = parser.parse_args()

    cfg = _load_settings(args.config, args.recipe, args)
    train(cfg, smoke=args.smoke, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
