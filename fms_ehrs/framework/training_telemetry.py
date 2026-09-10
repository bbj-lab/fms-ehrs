"""Durable, local progress telemetry for long-running Trainer jobs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _json_value(value: Any) -> Any:
    """Convert scalar metric values to JSON-safe primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class LiveProgressCallback(TrainerCallback):
    """Emit stdout and JSONL heartbeats without depending on W&B connectivity."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        interval_seconds: float = 60.0,
    ):
        super().__init__()
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive (got {interval_seconds})."
            )
        self.output_path = Path(output_dir) / "progress.jsonl"
        self.interval_seconds = float(interval_seconds)
        self._started_at: float | None = None
        self._last_heartbeat_at: float | None = None
        self._last_metrics: dict[str, Any] = {}

    @staticmethod
    def _is_main_process(state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _emit(self, event: str, state, *, metrics: dict[str, Any] | None = None) -> None:
        if not self._is_main_process(state):
            return
        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now
        elapsed_seconds = now - self._started_at
        global_step = int(getattr(state, "global_step", 0))
        max_steps = int(getattr(state, "max_steps", 0))
        steps_per_second = global_step / elapsed_seconds if elapsed_seconds else None
        remaining_seconds = (
            (max_steps - global_step) / steps_per_second
            if steps_per_second and max_steps >= global_step
            else None
        )
        if metrics:
            self._last_metrics.update(
                {str(key): _json_value(value) for key, value in metrics.items()}
            )

        payload = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
            "global_step": global_step,
            "max_steps": max_steps or None,
            "progress_fraction": global_step / max_steps if max_steps else None,
            "epoch": _json_value(getattr(state, "epoch", None)),
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": steps_per_second,
            "estimated_remaining_seconds": remaining_seconds,
            "metrics": self._last_metrics,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
        print(f"[IRB_PROGRESS] {json.dumps(payload, sort_keys=True)}", flush=True)
        self._last_heartbeat_at = now

    def on_train_begin(self, args, state, control, **kwargs):
        self._started_at = time.monotonic()
        self._emit("train_begin", state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        if (
            self._last_heartbeat_at is None
            or now - self._last_heartbeat_at >= self.interval_seconds
        ):
            self._emit("heartbeat", state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._emit("metrics", state, metrics=logs or {})
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._emit("evaluation", state, metrics=metrics or {})
        return control

    def on_save(self, args, state, control, **kwargs):
        self._emit("checkpoint", state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._emit("train_end", state)
        return control
