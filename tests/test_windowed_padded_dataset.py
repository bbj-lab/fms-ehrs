import math

import pytest

from fms_ehrs.framework.dataset import _windowed_padded_examples


def test_windowed_padded_examples_overlap_and_cont_token():
    # 10 tokens, window_len=6, stride=4:
    # - window0: tokens[0:6] = 0..5
    # - window1: TL_CONT + tokens[4:9] = CONT,4,5,6,7,8
    # - window2: TL_CONT + tokens[8:10] + PADs = CONT,8,9,PAD,PAD,PAD
    tokens = list(range(10))
    times = [i * 3600_000 for i in range(10)]  # ms since "admission"; monotone
    nums = [float(i) if i % 2 == 0 else None for i in range(10)]

    out = _windowed_padded_examples(
        tokens=tokens,
        times=times,
        numeric_values=nums,
        window_len=6,
        window_stride=4,
        pad_id=999,
        cont_id=111,
    )

    assert list(out.keys()) == ["input_ids", "numeric_values", "relative_times"]
    assert len(out["input_ids"]) == 3
    assert len(out["numeric_values"]) == 3
    assert len(out["relative_times"]) == 3

    assert out["input_ids"][0] == [0, 1, 2, 3, 4, 5]
    assert out["input_ids"][1] == [111, 4, 5, 6, 7, 8]
    assert out["input_ids"][2] == [111, 8, 9, 999, 999, 999]

    # Relative time should always be referenced to the full-admission start time (t0=0),
    # not re-zeroed for later windows.
    assert out["relative_times"][0] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert out["relative_times"][1][0] == 0.0  # TL_CONT has time=None -> 0.0
    assert out["relative_times"][1][1:] == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert out["relative_times"][2][1:3] == [8.0, 9.0]

    # Numeric values: even indices have values, odd are NaN, TL_CONT is NaN, PAD is NaN.
    w1 = out["numeric_values"][1]
    assert math.isnan(w1[0])  # TL_CONT
    assert w1[1] == 4.0
    assert math.isnan(w1[2])  # 5 -> None

    w2 = out["numeric_values"][2]
    assert math.isnan(w2[0])  # TL_CONT
    assert w2[1] == 8.0
    assert math.isnan(w2[2])  # 9 -> None
    assert all(math.isnan(x) for x in w2[3:])  # PADs


def test_windowed_padded_examples_no_cont_token():
    tokens = list(range(7))
    out = _windowed_padded_examples(
        tokens=tokens,
        times=None,
        numeric_values=None,
        window_len=4,
        window_stride=2,
        pad_id=0,
        cont_id=None,
    )
    # Without cont_id, windows should not insert a marker; they should just slice+pad.
    assert out["input_ids"] == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 0],
    ]


def test_windowed_padded_examples_non_overlapping_stride_with_cont_token():
    # Non-overlapping mode with TL_CONT:
    # When cont_id is provided and window_stride == window_len, window0 uses L tokens,
    # and subsequent windows cover (L-1) raw tokens with a TL_CONT prefix, advancing
    # by (L-1) after the first window to avoid gaps/overlap.
    tokens = list(range(12))
    out = _windowed_padded_examples(
        tokens=tokens,
        times=None,
        numeric_values=None,
        window_len=6,
        window_stride=6,
        pad_id=999,
        cont_id=111,
    )
    assert out["input_ids"] == [
        [0, 1, 2, 3, 4, 5],
        [111, 6, 7, 8, 9, 10],
        [111, 11, 999, 999, 999, 999],
    ]


def test_windowed_padded_examples_max_windows_caps_and_includes_first_last():
    tokens = list(range(100))
    out = _windowed_padded_examples(
        tokens=tokens,
        times=None,
        numeric_values=None,
        window_len=10,
        window_stride=10,
        pad_id=0,
        cont_id=None,
        max_windows=4,
    )
    assert len(out["input_ids"]) == 4
    # First and last windows are always included.
    assert out["input_ids"][0] == list(range(0, 10))
    assert out["input_ids"][-1] == list(range(90, 100))


def test_windowed_padded_examples_empty_sequence():
    out = _windowed_padded_examples(
        tokens=[],
        times=[],
        numeric_values=[],
        window_len=5,
        window_stride=3,
        pad_id=7,
        cont_id=1,
    )
    assert out["input_ids"] == [[7, 7, 7, 7, 7]]
    # times empty -> all zeros
    assert out["relative_times"] == [[0.0, 0.0, 0.0, 0.0, 0.0]]


def test_windowed_padded_examples_validates_alignment():
    with pytest.raises(ValueError, match="times must align"):
        _windowed_padded_examples(
            tokens=[1, 2, 3],
            times=[0],
            numeric_values=None,
            window_len=4,
            window_stride=2,
            pad_id=0,
            cont_id=None,
        )
    with pytest.raises(ValueError, match="numeric_values must align"):
        _windowed_padded_examples(
            tokens=[1, 2, 3],
            times=None,
            numeric_values=[1.0, 2.0],
            window_len=4,
            window_stride=2,
            pad_id=0,
            cont_id=None,
        )

