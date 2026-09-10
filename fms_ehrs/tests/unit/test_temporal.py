import pytest
import torch

from fms_ehrs.framework.dataset import (
    compute_relative_times_hours,
    compute_relative_times_seconds,
)
from fms_ehrs.framework.temporal import admission_relative_position_ids


def test_relative_times_seconds_preserve_sub_hour_resolution():
    timestamps_ms = [1_000, 31_000, 91_000, None]
    assert compute_relative_times_seconds(timestamps_ms) == [0.0, 30.0, 90.0, 0.0]
    assert compute_relative_times_hours(timestamps_ms) == [0.0, 30.0 / 3600.0, 0.025, 0.0]


def test_admission_relative_position_ids_uses_seconds_per_position():
    positions = admission_relative_position_ids(
        torch.tensor([[0.0, 29.9, 30.0, float("nan"), -1.0]]),
        seconds_per_position=30.0,
    )
    assert positions.tolist() == [[0, 0, 1, 0, 0]]


def test_admission_relative_position_ids_rejects_invalid_bin_width():
    with pytest.raises(ValueError, match="must be positive"):
        admission_relative_position_ids(torch.zeros((1, 1)), seconds_per_position=0)
