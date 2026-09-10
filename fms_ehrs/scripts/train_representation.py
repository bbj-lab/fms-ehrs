#!/usr/bin/env python3

"""
Unified training script for Experiment 2 representation mechanics.

This script supports Exp2 configurations:
- representation ∈ {discrete, soft, xval, xval_affine}
- temporal ∈ {time_tokens, time_rope, event_order}

The script uses padded collation (one hospitalization per row) to preserve
per-admission temporal structure needed for:
- Time-Aware RoPE: Requires relative time in hours since admission
- Soft/xVal: Requires numeric_values aligned to token positions

Usage:
    # Discrete + time tokens (baseline)
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation discrete \\
        --temporal time_tokens

    # Soft discretization + Time-Aware RoPE
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation soft \\
        --temporal time_rope

    # xVal + Time-Aware RoPE
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation xval \\
        --temporal time_rope

Note:
    Soft and xVal representations require unfused tokenization
    (fused_category_values=false in the tokenizer config).
"""

import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import signal
import subprocess
import time
import typing
from datetime import datetime, timezone

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    LlamaConfig,
    Qwen3Config,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.model_wrapper import create_representation_model
from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.training_telemetry import LiveProgressCallback

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


def _parse_bool(x: typing.Any, *, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return bool(default)


def _normalize_attn_impl(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return s


def _safe_artifact_name(name: str) -> str:
    cleaned = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    out = "".join(cleaned).strip("-")
    return out or "artifact"


def _unwrap_model(model: t.nn.Module) -> t.nn.Module:
    """Return the underlying model when Trainer/Accelerate wraps it."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _has_weight_file(path: pathlib.Path) -> bool:
    return (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()


def _perplexity_from_loss(value: typing.Any) -> float | None:
    try:
        loss = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(loss):
        return None
    # Avoid overflow while preserving the useful range for CE curves.
    return float(np.exp(min(loss, 50.0)))


def _add_perplexity_metrics(logs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    enriched = dict(logs)
    metric_pairs = (
        ("loss", "train_perplexity"),
        ("train_loss", "final_train_perplexity"),
        ("eval_loss", "eval_perplexity"),
    )
    for loss_key, ppl_key in metric_pairs:
        if loss_key in enriched and ppl_key not in enriched:
            ppl = _perplexity_from_loss(enriched[loss_key])
            if ppl is not None:
                enriched[ppl_key] = ppl
    return enriched


def _write_loss_perplexity_curve(
    *,
    output_dir: pathlib.Path,
    log_history: list[dict[str, typing.Any]],
) -> None:
    rows = []
    keys = [
        "step",
        "epoch",
        "loss",
        "train_loss",
        "eval_loss",
        "train_perplexity",
        "final_train_perplexity",
        "eval_perplexity",
        "learning_rate",
        "grad_norm",
        "grad_norm_clipped",
        "token_loss",
        "numeric_loss",
        "eval_token_loss",
        "eval_numeric_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_tokens_per_second",
    ]
    for entry in log_history:
        enriched = _add_perplexity_metrics(entry)
        if not any(k in enriched for k in ("loss", "train_loss", "eval_loss")):
            continue
        rows.append({k: enriched.get(k, "") for k in keys})

    if not rows:
        logger.warning("No loss/perplexity rows found in Trainer log history.")
        return

    output_dir.mkdir(exist_ok=True, parents=True)
    csv_path = output_dir / "loss_perplexity_curve.csv"
    jsonl_path = output_dir / "loss_perplexity_curve.jsonl"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote loss/perplexity curves to %s and %s", csv_path, jsonl_path)


def _sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_dir: pathlib.Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _validate_checkpoint_contract(
    *,
    save_strategy: str,
    eval_strategy: str,
    save_steps: int,
    eval_steps: int | None,
    load_best_model_at_end: bool,
) -> None:
    if not load_best_model_at_end:
        return

    if save_strategy == "no" or eval_strategy == "no":
        raise ValueError(
            "load_best_model_at_end requires both save_strategy and eval_strategy."
        )
    if save_strategy != eval_strategy:
        raise ValueError(
            "load_best_model_at_end requires matching save_strategy and eval_strategy "
            f"(got save={save_strategy!r}, eval={eval_strategy!r})."
        )
    if save_strategy == "steps":
        if eval_steps is None or eval_steps <= 0:
            raise ValueError(
                "Step-based best-checkpoint selection requires a positive eval_steps."
            )
        if save_steps <= 0 or save_steps % eval_steps != 0:
            raise ValueError(
                "Step-based save_steps must be a positive multiple of eval_steps "
                f"(got save_steps={save_steps}, eval_steps={eval_steps})."
            )


def _write_run_record(
    *,
    output_dir: pathlib.Path,
    trainer: Trainer,
    training_args: TrainingArguments,
    model_name: str,
    model_version: str,
    rope_theta: float,
    seconds_per_position: float,
    data_version: str,
    representation: str,
    temporal: str,
    seed: int,
    jid: str,
    dataset: Datasets,
    optimizer: str,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    checkpoint_interval_seconds: float,
    exported_model_path: pathlib.Path,
) -> None:
    model_config_path = exported_model_path / "config.json"
    vocab_path = exported_model_path / "vocab.gzip"
    log_history = list(trainer.state.log_history)
    grad_norms = [
        float(entry["grad_norm"])
        for entry in log_history
        if entry.get("grad_norm") is not None
    ]
    clipped = [
        value
        for value in grad_norms
        if value > float(training_args.max_grad_norm)
    ]
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": training_args.run_name,
        "model": {
            "name": model_name,
            "version": model_version,
            "rope_theta": float(rope_theta),
            "seconds_per_position": float(seconds_per_position),
            "parameter_count": int(sum(p.numel() for p in trainer.model.parameters())),
            "exported_path": str(exported_model_path),
            "config_sha256": _sha256_file(model_config_path),
            "vocab_sha256": _sha256_file(vocab_path),
        },
        "data": {
            "version": data_version,
            "n_train": int(dataset.n_train),
            "n_val": int(dataset.n_val),
        },
        "representation": representation,
        "temporal": temporal,
        "seed": int(seed),
        "slurm": {
            "jid": jid,
            "job_id": os.getenv("SLURM_JOB_ID"),
            "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
            "partition": os.getenv("SLURM_JOB_PARTITION"),
            "nproc_per_node": os.getenv("IRB_NPROC_PER_NODE"),
            "world_size": os.getenv("WORLD_SIZE"),
        },
        "optimizer": {
            "name": optimizer,
            "learning_rate": float(learning_rate),
            "adam_beta1": float(adam_beta1),
            "adam_beta2": float(adam_beta2),
            "weight_decay": float(training_args.weight_decay),
            "max_grad_norm": float(training_args.max_grad_norm),
            "lr_scheduler_type": str(training_args.lr_scheduler_type),
            "warmup_ratio": float(training_args.warmup_ratio),
        },
        "checkpoint_selection": {
            "metric_for_best_model": training_args.metric_for_best_model,
            "greater_is_better": training_args.greater_is_better,
            "load_best_model_at_end": training_args.load_best_model_at_end,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "latest_checkpoint": _latest_complete_checkpoint(output_dir),
            "periodic_checkpoint_seconds": float(checkpoint_interval_seconds),
        },
        "training": {
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
            "per_device_train_batch_size": int(
                training_args.per_device_train_batch_size
            ),
            "gradient_accumulation_steps": int(
                training_args.gradient_accumulation_steps
            ),
            "effective_batch_windows": int(
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * max(1, int(os.getenv("WORLD_SIZE", "1")))
            ),
            "max_sequence_length": int(os.getenv("IRB_MAX_SEQ_LENGTH", "4096")),
            "max_input_tokens_seen": max(
                (
                    int(entry["num_input_tokens_seen"])
                    for entry in log_history
                    if entry.get("num_input_tokens_seen") is not None
                ),
                default=None,
            ),
            "gpu_max_memory_allocated_bytes": (
                int(t.cuda.max_memory_allocated()) if t.cuda.is_available() else None
            ),
            "gpu_max_memory_reserved_bytes": (
                int(t.cuda.max_memory_reserved()) if t.cuda.is_available() else None
            ),
            "gradient_norm_observations": len(grad_norms),
            "gradient_norm_clipped_observations": len(clipped),
            "gradient_norm_clipped_fraction": (
                len(clipped) / len(grad_norms) if grad_norms else None
            ),
            "arguments": training_args.to_dict(),
        },
        "source": {
            "fms_ehrs_commit": _git_commit(pathlib.Path(__file__).resolve().parents[2]),
            "benchmark_commit": os.getenv("IRB_BENCHMARK_COMMIT"),
        },
    }
    record_path = output_dir / "run_record.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote run record to %s", record_path)


def _normalize_saved_config(path: pathlib.Path) -> None:
    """Keep HF config invariants valid after layer-count overrides."""
    config_path = path / "config.json"
    if not config_path.exists():
        return

    with config_path.open() as f:
        config = json.load(f)

    num_layers = config.get("num_hidden_layers")
    layer_types = config.get("layer_types")
    if isinstance(num_layers, int) and isinstance(layer_types, list) and len(layer_types) != num_layers:
        if len(layer_types) < num_layers:
            raise ValueError(
                f"Cannot extend layer_types in {config_path}: "
                f"{len(layer_types)} entries for {num_layers} layers"
            )
        config["layer_types"] = layer_types[:num_layers]
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        logger.info("Normalized layer_types in %s to %s entries", config_path, num_layers)


def _representation_state(
    *,
    model: t.nn.Module,
    representation: str,
    temporal: str,
    num_bins: int,
    seconds_per_position: float,
) -> dict[str, typing.Any]:
    from fms_ehrs.framework.xval import XValModelWrapper, persist_xval_mechanics

    state = {
        "representation": representation,
        "temporal": temporal,
        "num_bins": num_bins,
        "time_unit": "seconds",
        "seconds_per_position": float(seconds_per_position),
        "value_encoder_state": (
            model.value_encoder.state_dict()
            if hasattr(model, "value_encoder") and model.value_encoder is not None
            else None
        ),
    }
    if isinstance(model, XValModelWrapper):
        state.update(persist_xval_mechanics(model))
    return state


_CHECKPOINT_COMPLETE_FILE = "checkpoint_complete.json"


def _latest_complete_checkpoint(output_dir: pathlib.Path) -> str | None:
    """Return the newest fully-written Trainer checkpoint, if one exists."""
    checkpoints: list[tuple[int, pathlib.Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if (
            path.is_dir()
            and (path / "trainer_state.json").is_file()
            and (path / _CHECKPOINT_COMPLETE_FILE).is_file()
        ):
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def _mark_checkpoint_complete(checkpoint_dir: pathlib.Path, state) -> None:
    """Write a completion marker only after every checkpoint payload is durable."""
    marker = checkpoint_dir / _CHECKPOINT_COMPLETE_FILE
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(
        json.dumps(
            {
                "global_step": int(state.global_step),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(marker)


def _append_checkpoint_event(args, state) -> None:
    event = {
        "event": "checkpoint_saved",
        "checkpoint": str(pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}"),
        "global_step": int(state.global_step),
        "epoch": float(state.epoch) if state.epoch is not None else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (pathlib.Path(args.output_dir) / "checkpoint_events.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


class WrapperCheckpointCallback(TrainerCallback):
    """Write wrapper model weights into Trainer checkpoints on the main process."""

    def __init__(
        self,
        *,
        representation: str,
        temporal: str,
        num_bins: int,
        seconds_per_position: float,
    ):
        super().__init__()
        self.representation = representation
        self.temporal = temporal
        self.num_bins = int(num_bins)
        self.seconds_per_position = float(seconds_per_position)

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        _append_checkpoint_event(args, state)
        checkpoint_dir = pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if model is not None:
            from fms_ehrs.framework.xval import XValModelWrapper

            unwrapped = _unwrap_model(model)
            if isinstance(unwrapped, (RepresentationModelWrapper, XValModelWrapper)):
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                set_perms(lambda f, obj: t.save(obj, f))(
                    str(checkpoint_dir / "pytorch_model.bin"),
                    unwrapped.state_dict(),
                )
                set_perms(lambda f, obj: t.save(obj, f))(
                    str(checkpoint_dir / "representation_mechanics.pt"),
                    _representation_state(
                        model=unwrapped,
                        representation=self.representation,
                        temporal=self.temporal,
                        num_bins=self.num_bins,
                        seconds_per_position=self.seconds_per_position,
                    ),
                )
        _mark_checkpoint_complete(checkpoint_dir, state)
        return control


class PeriodicCheckpointCallback(TrainerCallback):
    """Request a full Trainer checkpoint at a wall-clock interval."""

    def __init__(self, interval_seconds: float):
        super().__init__()
        self.interval_seconds = float(interval_seconds)
        self._last_checkpoint_at: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_checkpoint_at = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._last_checkpoint_at is None:
            self._last_checkpoint_at = time.monotonic()
        if time.monotonic() - self._last_checkpoint_at >= self.interval_seconds:
            logger.info(
                "Requesting defensive checkpoint at global step %s after %.0f seconds.",
                state.global_step,
                self.interval_seconds,
            )
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self._last_checkpoint_at = time.monotonic()
        return control


class SignalCheckpointCallback(TrainerCallback):
    """Request a checkpoint at the next safe step after a Slurm checkpoint signal."""

    def __init__(self):
        super().__init__()
        self._requested_at: float | None = None
        self._requested_signal: int | None = None
        self._previous_handlers: dict[int, typing.Any] = {}

    def _request_checkpoint(self, signum, frame) -> None:
        self._requested_at = time.monotonic()
        self._requested_signal = signum

    def on_train_begin(self, args, state, control, **kwargs):
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request_checkpoint)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._requested_at is not None:
            logger.warning(
                "Received Slurm checkpoint signal %s; saving at safe global step %s.",
                self._requested_signal,
                state.global_step,
            )
            self._requested_at = None
            self._requested_signal = None
            control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        return control


def _log_wandb_directory_artifact(
    *,
    directory: pathlib.Path,
    artifact_name: str,
    artifact_type: str,
    metadata: dict[str, typing.Any],
    project: str,
    run_name: str,
    require_wandb: bool,
) -> None:
    if not directory.exists():
        msg = f"Artifact directory does not exist: {directory}"
        if require_wandb:
            raise RuntimeError(msg)
        logger.warning(msg)
        return

    if str(os.getenv("WANDB_MODE", "")).strip().lower() == "offline":
        logger.info(
            "WANDB_MODE=offline; retaining local artifact %s at %s.",
            artifact_name,
            directory,
        )
        return

    try:
        import wandb

        run = wandb.run
        created_run = False
        if run is None:
            run = wandb.init(
                project=project,
                name=f"{run_name}-artifact",
                job_type="artifact-export",
                reinit=False,
            )
            created_run = True

        artifact = wandb.Artifact(
            name=_safe_artifact_name(artifact_name),
            type=artifact_type,
            metadata=metadata,
        )
        artifact.add_dir(str(directory))
        run.log_artifact(artifact, aliases=["latest"])
        logger.info("Logged W&B artifact '%s' from %s", artifact_name, str(directory))

        if created_run and run is not None:
            run.finish()
    except Exception as e:
        if require_wandb:
            raise RuntimeError(
                f"Failed to log required W&B artifact {artifact_name} from {directory}: {e}"
            ) from e
        logger.warning(
            "W&B artifact upload skipped for %s (%s): %s",
            artifact_name,
            str(directory),
            e,
        )


class NanStoppingCallback(TrainerCallback):
    """Stop training on encountering a NaN objective."""

    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    if state.is_world_process_zero:
                        logger.warning(f"Encountered non-finite metric {k} ({v}).")
                    control.should_training_stop = True


class RepresentationDataCollator:
    """Data collator that handles numeric_values and relative_times_seconds.

    For Exp2, we need to pass additional tensors beyond input_ids:
    - numeric_values: For soft discretization and xVal
    - relative_times_seconds: For Time-Aware RoPE temporal encoding

    """

    def __init__(
        self,
        pad_token_id: int,
        include_numeric_values: bool = False,
        include_times: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.include_numeric_values = include_numeric_values
        self.include_times = include_times

    def __call__(self, features: list[dict]) -> dict:
        """Collate batch of features."""
        labels = t.stack([f["input_ids"] for f in features])
        labels = labels.clone()
        labels[labels == self.pad_token_id] = -100

        batch = {
            "input_ids": t.stack([f["input_ids"] for f in features]),
            "attention_mask": t.stack(
                [
                    (f["input_ids"] != self.pad_token_id).long()
                    for f in features
                ]
            ),
            "labels": labels,
        }

        if self.include_numeric_values and "numeric_values" in features[0]:
            batch["numeric_values"] = t.stack(
                [f["numeric_values"] for f in features]
            )

        if self.include_times and "relative_times_seconds" in features[0]:
            batch["relative_times_seconds"] = t.stack(
                [f["relative_times_seconds"] for f in features]
            )

        return batch


class IRBTrainer(Trainer):
    """Trainer with representation aux-loss logging and AdamW via HF defaults."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_aux_losses: dict[str, float] = {}

    def compute_loss(self, *args, return_outputs: bool = False, **kwargs):
        loss, outputs = super().compute_loss(
            *args,
            return_outputs=True,
            **kwargs,
        )
        losses = {}
        for name in ("token_loss", "numeric_loss"):
            value = outputs.get(name) if isinstance(outputs, dict) else getattr(outputs, name, None)
            if value is not None:
                losses[name] = float(value.detach().float().mean().cpu())
        if losses:
            prefix = "" if self.model.training else "eval_"
            self._last_aux_losses = {f"{prefix}{name}": value for name, value in losses.items()}
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args, **kwargs):
        # HF logs cross-entropy losses by default. Add perplexity before the
        # integrations run so W&B and trainer_state.json carry the curves.
        enriched = _add_perplexity_metrics(logs)
        if "eval_loss" in enriched:
            enriched.update(
                {
                    key: value
                    for key, value in self._last_aux_losses.items()
                    if key.startswith("eval_")
                }
            )
        elif "loss" in enriched:
            enriched.update(
                {
                    key: value
                    for key, value in self._last_aux_losses.items()
                    if not key.startswith("eval_")
                }
            )
        if "grad_norm" in enriched:
            enriched["grad_norm_clipped"] = float(
                float(enriched["grad_norm"]) > float(self.args.max_grad_norm)
            )
        return super().log(enriched, *args, **kwargs)


@logger.log_calls
def main(
    *,
    # Data parameters
    data_dir: os.PathLike = None,
    data_version: str = "day_stays",
    # Model parameters
    model_dir: os.PathLike = None,
    model_name: str = "meta-llama/Llama-3.2-1B",
    model_version: str = "llama1b",
    rope_theta: float = float(os.getenv("IRB_ROPE_THETA", "10000.0")),
    # Optional performance settings (objective-preserving):
    use_bf16: bool = _parse_bool(os.getenv("IRB_USE_BF16", "true"), default=True),
    attn_implementation: str | None = _normalize_attn_impl(os.getenv("IRB_ATTN_IMPL", "sdpa")),
    # Representation parameters
    representation: typing.Literal["discrete", "soft", "xval", "xval_affine"] = "discrete",
    temporal: typing.Literal[
        "time_tokens", "time_rope", "event_order"
    ] = "time_tokens",
    num_bins: int = 20,
    # Time-Aware RoPE setting: admission-relative seconds per integer position.
    seconds_per_position: float = float(
        os.getenv("IRB_SECONDS_PER_POSITION", "60.0")
    ),
    # xVal tuning setting
    numeric_loss_weight: float = float(os.getenv("IRB_XVAL_NUMERIC_LOSS_WEIGHT", "1.0")),
    clip_sigma: float = float(os.getenv("IRB_XVAL_CLIP_SIGMA", "5.0")),
    # Training parameters
    n_epochs: float = float(os.getenv("IRB_EXP23_STAGE1_EPOCHS", os.getenv("IRB_STAGE1_EPOCHS", "1"))),
    per_device_train_batch_size: int = int(
        os.getenv("IRB_PER_DEVICE_TRAIN_BATCH_SIZE", "1")
    ),
    per_device_eval_batch_size: int = int(
        os.getenv("IRB_PER_DEVICE_EVAL_BATCH_SIZE", "1")
    ),
    gradient_accumulation_steps: int = int(
        os.getenv("IRB_GRADIENT_ACCUMULATION_STEPS", "1")
    ),
    # Base AdamW learning rate for all Stage-1 representation runs.
    learning_rate: float = float(os.getenv("IRB_STAGE1_LR", "1e-4")),
    weight_decay: float = float(os.getenv("IRB_STAGE1_WEIGHT_DECAY", "0.01")),
    adam_beta1: float = float(os.getenv("IRB_ADAM_BETA1", "0.9")),
    adam_beta2: float = float(os.getenv("IRB_ADAM_BETA2", "0.999")),
    max_grad_norm: float = float(os.getenv("IRB_STAGE1_MAX_GRAD_NORM", "1.0")),
    lr_scheduler_type: str = os.getenv("IRB_STAGE1_LR_SCHEDULER", "linear"),
    warmup_ratio: float = float(os.getenv("IRB_STAGE1_WARMUP_RATIO", "0.0")),
    logging_steps: int = int(os.getenv("IRB_STAGE1_LOGGING_STEPS", "100")),
    max_seq_length: int = int(os.getenv("IRB_MAX_SEQ_LENGTH", "4096")),
    # Full-timeline padded-mode training via windowing (Exp2/Exp3):
    # When enabled, we train on *all* tokens in a hospitalization by slicing the
    # variable-length `tokens` timeline into overlapping windows of length max_seq_length.
    windowed_padded: bool = False,
    window_stride: int | None = None,
    max_windows_per_admission: int | None = None,
    add_cont_token: bool = True,
    # Checkpoint/resume controls
    resume_from_checkpoint: str | None = os.getenv("IRB_RESUME_FROM_CHECKPOINT", None),
    save_strategy: str = os.getenv("IRB_STAGE1_SAVE_STRATEGY", "epoch"),
    save_steps: int = int(os.getenv("IRB_STAGE1_SAVE_STEPS", "2000")),
    save_total_limit: int = int(os.getenv("IRB_STAGE1_SAVE_TOTAL_LIMIT", "2")),
    checkpoint_interval_seconds: float = float(
        os.getenv("IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS", "600")
    ),
    progress_interval_seconds: float = float(
        os.getenv("IRB_PROGRESS_INTERVAL_SECONDS", "60")
    ),
    eval_strategy: str = os.getenv("IRB_STAGE1_EVAL_STRATEGY", "epoch"),
    eval_steps: int | None = (
        int(os.environ["IRB_STAGE1_EVAL_STEPS"])
        if os.getenv("IRB_STAGE1_EVAL_STEPS")
        else None
    ),
    evaluations_per_epoch: int = int(
        os.getenv("IRB_STAGE1_EVALUATIONS_PER_EPOCH", "1")
    ),
    max_steps: int = int(os.getenv("IRB_STAGE1_MAX_STEPS", "-1")),
    load_best_model_at_end: bool = _parse_bool(
        os.getenv("IRB_LOAD_BEST_MODEL_AT_END", "true"),
        default=True,
    ),
    early_stopping_patience: int = int(
        os.getenv("IRB_STAGE1_EARLY_STOPPING_PATIENCE", "3")
    ),
    # Experiment tracking
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = "mimic-representation",
    seed: int = 42,
    **model_kwargs,
):
    """Train a model with specified representation mechanics.

    Parameters
    ----------
    data_dir : PathLike
        Root directory containing tokenized data
    data_version : str
        Data version name (e.g., "day_stays")
    model_dir : PathLike
        Directory to save trained models
    model_name : str
        HuggingFace model name for config
    model_version : str
        Version tag for saved model
    representation : {"discrete", "soft", "xval", "xval_affine"}
        Value representation method
    temporal : {"time_tokens", "time_rope", "event_order"}
        Temporal encoding method
    num_bins : int
        Number of quantile bins for soft discretization
    n_epochs : float
        Number of training epochs
    per_device_train_batch_size : int
        Training batch size per device
    per_device_eval_batch_size : int
        Evaluation batch size per device
    gradient_accumulation_steps : int
        Gradient accumulation steps
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay (decoupled; AdamW-style).
    max_seq_length : int
        Maximum sequence length
    jid : str
        SLURM job ID for logging
    wandb_project : str
        Weights & Biases project name
    seed : int
        Random seed
    **model_kwargs
        Additional model configuration parameters
    """
    # Set random seeds
    t.manual_seed(seed)
    np.random.seed(seed)

    # Validate configuration
    if int(gradient_accumulation_steps) <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive "
            f"(got {gradient_accumulation_steps})."
        )
    attn_implementation = _normalize_attn_impl(attn_implementation)
    if attn_implementation == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "attn_implementation=flash_attention_2 requires the optional `flash-attn` package "
                "(and a compatible GPU/CUDA build)."
            )
    if representation in ("soft", "xval", "xval_affine") and temporal == "time_tokens":
        logger.warning(
            "This is valid but typically paired with time_rope for Exp2."
        )
    if rope_theta <= 0:
        raise ValueError(f"rope_theta must be positive (got {rope_theta}).")

    # Determine what additional data to load
    needs_numeric_values = representation in ("soft", "xval", "xval_affine")
    needs_times = temporal in ("time_rope",)

    if needs_numeric_values or needs_times:
        logger.info(
            f"Loading extended data: numeric_values={needs_numeric_values}, "
            f"times={needs_times}"
        )

    # Setup paths
    data_dir, model_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_dir)
    )

    # Setup experiment tracking
    run_name = f"{model_version}-{representation}-{temporal}-{jid}"
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    require_wandb = _parse_bool(os.getenv("IRB_REQUIRE_WANDB", "false"), default=False)

    output_dir = model_dir / run_name
    output_dir.mkdir(exist_ok=True, parents=True)

    use_bf16 = _parse_bool(use_bf16, default=True)
    windowed_padded = _parse_bool(windowed_padded, default=False)
    add_cont_token = _parse_bool(add_cont_token, default=True)
    load_best_model_at_end = _parse_bool(load_best_model_at_end, default=True)
    save_strategy = str(save_strategy).strip().lower()
    eval_strategy = str(eval_strategy).strip().lower()
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive (got {max_grad_norm}).")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1) (got {warmup_ratio}).")
    if logging_steps <= 0:
        raise ValueError(f"logging_steps must be positive (got {logging_steps}).")
    if evaluations_per_epoch <= 0:
        raise ValueError(
            f"evaluations_per_epoch must be positive (got {evaluations_per_epoch})."
        )
    if max_steps == 0 or max_steps < -1:
        raise ValueError(f"max_steps must be -1 or positive (got {max_steps}).")
    if checkpoint_interval_seconds <= 0:
        raise ValueError(
            "checkpoint_interval_seconds must be positive "
            f"(got {checkpoint_interval_seconds})."
        )
    if progress_interval_seconds <= 0:
        raise ValueError(
            "progress_interval_seconds must be positive "
            f"(got {progress_interval_seconds})."
        )
    if seconds_per_position <= 0:
        raise ValueError(
            f"seconds_per_position must be positive (got {seconds_per_position})."
        )
    resolved_resume_from_checkpoint: str | None = None
    if resume_from_checkpoint is not None:
        resume_value = str(resume_from_checkpoint).strip()
        if resume_value and resume_value.lower() not in ("0", "false", "none", "null", "no"):
            if resume_value.lower() == "auto":
                last_checkpoint = _latest_complete_checkpoint(output_dir)
                if last_checkpoint is not None:
                    resolved_resume_from_checkpoint = last_checkpoint
                    logger.info("Auto-resuming from checkpoint: %s", last_checkpoint)
                else:
                    logger.info(
                        "No completed checkpoint found under %s; starting fresh.",
                        output_dir,
                    )
            else:
                resolved_resume_from_checkpoint = str(
                    pathlib.Path(resume_value).expanduser().resolve()
                )
                logger.info("Resuming from checkpoint: %s", resolved_resume_from_checkpoint)

    # Windowed padded guardrail for extreme-length admissions (compute stability).
    #
    # If max_windows_per_admission is not explicitly provided, allow an env override.
    # Convention: <=0 disables the cap.
    resolved_max_windows_per_admission: int | None = max_windows_per_admission
    if resolved_max_windows_per_admission is None:
        env = os.getenv("IRB_MAX_WINDOWS_PER_ADMISSION")
        if env is not None:
            s = str(env).strip().lower()
            if s not in ("", "none", "null"):
                try:
                    v = int(s)
                except ValueError:
                    raise ValueError(
                        f"Invalid IRB_MAX_WINDOWS_PER_ADMISSION={env!r}; expected int or unset."
                    )
                resolved_max_windows_per_admission = v if v > 0 else None

    # Load dataset with extended features if needed
    # Use padded collation for Exp2 (preserves per-admission structure)
    dataset = Datasets(
        data_version=data_version,
        data_dir=data_dir,
        collation="padded",  # Required for representation mode
        max_seq_length=max_seq_length,
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
        windowed_padded=windowed_padded,
        window_stride=window_stride,
        add_cont_token=add_cont_token,
        max_windows_per_admission=resolved_max_windows_per_admission,
    )

    if evaluations_per_epoch > 1:
        if eval_strategy != "epoch" or save_strategy != "epoch":
            raise ValueError(
                "evaluations_per_epoch > 1 owns the evaluation schedule; "
                "set save_strategy and eval_strategy to 'epoch' before invocation."
            )
        updates_per_epoch = math.ceil(
            dataset.n_train
            / (int(per_device_train_batch_size) * int(gradient_accumulation_steps))
        )
        eval_steps = max(1, math.ceil(updates_per_epoch / evaluations_per_epoch))
        save_steps = eval_steps
        eval_strategy = "steps"
        save_strategy = "steps"

    _validate_checkpoint_contract(
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        save_steps=int(save_steps),
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
    )

    logger.info(f"Loaded {dataset.n_train} train, {dataset.n_val} val samples")
    logger.info(f"Vocabulary size: {len(dataset.vocab)}")

    # Load quantizer/anchoring-independent numeric stats (if available).
    #
    # These are produced by tokenization as:
    #   <data_dir>/<data_version>-tokenized/train/numeric_stats.json
    #
    # If present, we use them to define (median_c, IQR-scale_c) for the continuous
    # encoder, decoupling continuous scaling from discretization choices
    # (e.g., 5-10-5 anchoring).
    numeric_stats: dict[str, dict[str, float]] | None = None
    stats_path = data_dir / f"{data_version}-tokenized" / "train" / "numeric_stats.json"
    if stats_path.exists():
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            numeric_stats = payload.get("stats", None)
            logger.info("Loaded numeric_stats.json for continuous scaling: %s", str(stats_path))
        except Exception as e:
            logger.warning(
                "Failed to load numeric_stats.json (falling back to vocab-aux-derived stats): %s",
                e,
            )

    def log_model_hyperparameters(
        model,
        *,
        selected_num_bins: int,
        selected_seconds_per_position: float,
        selected_numeric_loss_weight: float,
    ):
        if isinstance(model, RepresentationModelWrapper):
            base_params = sum(p.numel() for p in model.base_model.parameters())
            value_params = (
                sum(p.numel() for p in model.value_encoder.parameters())
                if model.value_encoder is not None
                else 0
            )
            total_params = base_params + value_params
            logger.info(
                "params: base=%s value=%s total=%s | settings: num_bins=%s seconds_per_position=%s numeric_loss_weight=%s",
                f"{base_params:,}",
                f"{value_params:,}",
                f"{total_params:,}",
                selected_num_bins,
                selected_seconds_per_position,
                selected_numeric_loss_weight,
            )
        else:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "params: base=%s | settings: bins=%s seconds_per_position=%.1f w=%.1f",
                f"{total_params:,}",
                selected_num_bins,
                selected_seconds_per_position,
                selected_numeric_loss_weight,
            )

    def _build_model(
        selected_num_bins: int,
        selected_numeric_loss_weight: float,
    ):
        # Build model once (no HPO in Exp2/Exp3).
        cfg_kwargs = dict(model_kwargs)
        if attn_implementation is not None:
            cfg_kwargs["attn_implementation"] = attn_implementation
        cfg_kwargs.setdefault("rope_theta", float(rope_theta))
        model_name_lower = str(model_name).lower()
        if "llama" in model_name_lower:
            config_type = LlamaConfig
            # Llama keeps separate input and output embedding matrices.
            cfg_kwargs.setdefault("tie_word_embeddings", False)
        elif "qwen3" in model_name_lower:
            config_type = Qwen3Config
            # Match the Qwen3-0.6B configuration after reducing attention
            # width: multi-head attention and tied input/output embeddings.
            cfg_kwargs.setdefault(
                "num_key_value_heads",
                int(cfg_kwargs.get("num_attention_heads", 32)),
            )
            cfg_kwargs.setdefault("tie_word_embeddings", True)
        else:
            raise ValueError(
                "ML4H supports only the Llama and Qwen3 backbones; "
                f"received model_name={model_name!r}."
            )
        # These experiments train from random initialization, so instantiating
        # the selected local config avoids a network dependency on compute nodes.
        config = config_type(
            vocab_size=len(dataset.vocab),
            bos_token_id=dataset.vocab("TL_START"),
            eos_token_id=dataset.vocab("TL_END"),
            pad_token_id=dataset.vocab("PAD"),
            **cfg_kwargs,
        )
        config._name_or_path = str(model_name)
        logger.info(
            "tie_word_embeddings=%s (True for Qwen, False for Llama)",
            bool(getattr(config, "tie_word_embeddings", False)),
        )
        base_model = AutoModelForCausalLM.from_config(config)
        return create_representation_model(
            base_model=base_model,
            vocab=dataset.vocab,
            representation=representation,
            temporal=temporal,
            num_bins=selected_num_bins,
            numeric_stats=numeric_stats if representation in ("xval", "xval_affine") else None,
            clip_sigma=float(clip_sigma),
            seconds_per_position=float(seconds_per_position),
            numeric_loss_weight=selected_numeric_loss_weight,
        )

    final_num_bins = num_bins
    final_numeric_loss_weight = float(numeric_loss_weight)

    model = _build_model(final_num_bins, final_numeric_loss_weight)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {n_params:,} parameters")
    logger.info(f"Representation: {representation}, Temporal: {temporal}")
    logger.info(
        "Representation settings: num_bins=%s",
        num_bins,
    )

    # Create data collator
    data_collator = RepresentationDataCollator(
        pad_token_id=dataset.vocab("PAD"),
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
    )

    # Training arguments
    # IMPORTANT (DDP correctness):
    # For soft/continuous/xval representations, the value encoder is only exercised on
    # batches that contain at least one numeric quantile token with a non-NaN value.
    # Some admissions contain no numeric events, so some steps legitimately do not
    # use the value-encoder parameters. In DDP, this requires find_unused_parameters,
    # otherwise PyTorch raises:
    #   RuntimeError: Expected to have finished reduction ... parameters not used ...
    ddp_find_unused = representation in ("soft", "xval", "xval_affine")
    use_bf16 = bool(use_bf16) and t.cuda.is_available()

    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=float(adam_beta1),
        adam_beta2=float(adam_beta2),
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        tf32=True,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen="all",
        num_train_epochs=n_epochs,
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=load_best_model_at_end,
        greater_is_better=False,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        # HF Trainer defaults to safetensors. With tied embeddings (common for causal LMs),
        # safetensors errors because multiple state_dict entries share the same storage:
        #   RuntimeError: Some tensors share memory ... {'model.embed_tokens.weight', 'lm_head.weight'}
        # Use standard torch serialization for checkpoints instead.
        save_safetensors=False,
        ddp_find_unused_parameters=ddp_find_unused,
        seed=seed,
        data_seed=seed,
    )

    log_model_hyperparameters(
        model,
        selected_num_bins=num_bins,
        selected_seconds_per_position=seconds_per_position,
        selected_numeric_loss_weight=numeric_loss_weight,
    )

    trainer = IRBTrainer(
        model=model,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["val"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[
            NanStoppingCallback(),
            LiveProgressCallback(
                output_dir,
                interval_seconds=progress_interval_seconds,
            ),
            PeriodicCheckpointCallback(checkpoint_interval_seconds),
            SignalCheckpointCallback(),
            WrapperCheckpointCallback(
                representation=representation,
                temporal=temporal,
                num_bins=final_num_bins,
                seconds_per_position=float(seconds_per_position),
            ),
            # Matches the Exp1 contract in tune_model.py: stop after three
            # consecutive evaluations without a strictly lower eval_loss.
            *(
                [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
                if early_stopping_patience > 0
                else []
            ),
        ],
    )

    logger.info("Starting training...")
    if t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()
        t.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=resolved_resume_from_checkpoint)

    if trainer.is_world_process_zero():
        if load_best_model_at_end and trainer.state.best_model_checkpoint is None:
            raise RuntimeError(
                "Training completed without a best_model_checkpoint despite "
                "load_best_model_at_end=true."
            )
        _write_loss_perplexity_curve(
            output_dir=output_dir,
            log_history=list(trainer.state.log_history),
        )

    # Save final model from the Trainer's main process.
    if trainer.is_world_process_zero():
        final_model_path = output_dir / f"model-{representation}-{temporal}"
        final_model_path.mkdir(exist_ok=True, parents=True)
        unwrapped_model = _unwrap_model(model)

        # IMPORTANT: for wrapper models (soft/time_rope/xval), we must save:
        # 1) the underlying HF model in standard `save_pretrained` format (config + weights)
        # 2) the representation-mechanics parameters (value encoder) separately
        #
        # This allows downstream scripts (e.g., sequence classification) to reload the same
        # representation mechanics and apply them using numeric_values / relative_times.
        from fms_ehrs.framework.xval import XValModelWrapper
        if isinstance(unwrapped_model, (RepresentationModelWrapper, XValModelWrapper)):
            # Save the wrapped HF model (config + weights)
            set_perms(unwrapped_model.base_model.save_pretrained)(str(final_model_path))
            _normalize_saved_config(final_model_path)
            if not _has_weight_file(final_model_path):
                set_perms(lambda f, obj: t.save(obj, f))(
                    str(final_model_path / "pytorch_model.bin"),
                    unwrapped_model.base_model.state_dict(),
                )

            # Save representation-mechanics parameters
            # NOTE: `set_perms` expects a saver with signature saver(file, *args),
            # but torch.save is torch.save(obj, file). Wrap to avoid arg order bugs.
            set_perms(lambda f, obj: t.save(obj, f))(
                str(final_model_path / "representation_mechanics.pt"),
                _representation_state(
                    model=unwrapped_model,
                    representation=representation,
                    temporal=temporal,
                    num_bins=final_num_bins,
                    seconds_per_position=float(seconds_per_position),
                ),
            )
        else:
            # Discrete + time_tokens returns a standard HF model; Trainer can save normally.
            set_perms(trainer.save_model)(str(final_model_path))
            _normalize_saved_config(final_model_path)

        if not _has_weight_file(final_model_path):
            raise RuntimeError(
                f"No model weight file found after export under {final_model_path} "
                "(expected pytorch_model.bin or model.safetensors)."
            )

        logger.info(f"Saved model to {final_model_path}")

        # Also save vocabulary
        dataset.vocab.save(final_model_path / "vocab.gzip")
        _write_run_record(
            output_dir=output_dir,
            trainer=trainer,
            training_args=training_args,
            model_name=model_name,
            model_version=model_version,
            rope_theta=rope_theta,
            seconds_per_position=seconds_per_position,
            data_version=data_version,
            representation=representation,
            temporal=temporal,
            seed=seed,
            jid=jid,
            dataset=dataset,
            optimizer="adamw",
            learning_rate=float(learning_rate),
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            exported_model_path=final_model_path,
        )

        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        latest_ckpt = _latest_complete_checkpoint(output_dir)
        _log_wandb_directory_artifact(
            directory=final_model_path,
            artifact_name=f"{run_name}-exported-model",
            artifact_type="model-export",
            metadata={
                "model_version": model_version,
                "data_version": data_version,
                "representation": representation,
                "temporal": temporal,
                "rope_theta": rope_theta,
                "seconds_per_position": seconds_per_position,
                "seed": seed,
                "slurm_jid": jid,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
                "irb_nproc_per_node": os.getenv("IRB_NPROC_PER_NODE"),
                "world_size": os.getenv("WORLD_SIZE"),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "n_epochs": n_epochs,
                "save_strategy": save_strategy,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "eval_strategy": eval_strategy,
                "eval_steps": eval_steps,
                "load_best_model_at_end": load_best_model_at_end,
                "resume_from_checkpoint": resume_from_checkpoint,
                "resolved_resume_from_checkpoint": resolved_resume_from_checkpoint,
                "best_model_checkpoint": str(best_ckpt) if best_ckpt else None,
                "latest_checkpoint": str(latest_ckpt) if latest_ckpt else None,
            },
            project=wandb_project,
            run_name=run_name,
            require_wandb=require_wandb,
        )

        artifact_ckpt = best_ckpt or latest_ckpt
        if artifact_ckpt:
            best_ckpt_path = pathlib.Path(artifact_ckpt)
            if best_ckpt_path.exists():
                _log_wandb_directory_artifact(
                    directory=best_ckpt_path,
                    artifact_name=f"{run_name}-best-checkpoint",
                    artifact_type="model-checkpoint",
                    metadata={
                        "model_version": model_version,
                        "data_version": data_version,
                        "representation": representation,
                        "temporal": temporal,
                        "rope_theta": rope_theta,
                        "seconds_per_position": seconds_per_position,
                        "seed": seed,
                        "slurm_jid": jid,
                        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                        "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
                        "irb_nproc_per_node": os.getenv("IRB_NPROC_PER_NODE"),
                        "world_size": os.getenv("WORLD_SIZE"),
                        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                        "n_epochs": n_epochs,
                        "save_strategy": save_strategy,
                        "save_steps": save_steps,
                        "save_total_limit": save_total_limit,
                        "eval_strategy": eval_strategy,
                        "eval_steps": eval_steps,
                        "load_best_model_at_end": load_best_model_at_end,
                        "resume_from_checkpoint": resume_from_checkpoint,
                        "resolved_resume_from_checkpoint": resolved_resume_from_checkpoint,
                        "best_model_checkpoint": str(best_ckpt) if best_ckpt else None,
                        "latest_checkpoint": str(latest_ckpt) if latest_ckpt else None,
                    },
                    project=wandb_project,
                    run_name=run_name,
                    require_wandb=require_wandb,
                )
            elif require_wandb:
                raise RuntimeError(
                    f"Required best checkpoint path was not found on disk: {best_ckpt_path}"
                )
            else:
                logger.warning(
                    "Best checkpoint path not found on disk; skipping artifact upload: %s",
                    str(best_ckpt_path),
                )

        return final_model_path

    return None


if __name__ == "__main__":
    fi.Fire(main)
