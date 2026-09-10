"""Admission-relative temporal position helpers."""

from __future__ import annotations

import torch


def admission_relative_position_ids(
    relative_times_seconds: torch.Tensor,
    *,
    seconds_per_position: float,
) -> torch.Tensor:
    """Convert admission-relative seconds to non-negative RoPE positions."""
    if seconds_per_position <= 0:
        raise ValueError(
            f"seconds_per_position must be positive (got {seconds_per_position})."
        )
    seconds = relative_times_seconds.nan_to_num(0.0).clamp_min(0.0)
    return torch.div(
        seconds,
        float(seconds_per_position),
        rounding_mode="floor",
    ).to(dtype=torch.long)
