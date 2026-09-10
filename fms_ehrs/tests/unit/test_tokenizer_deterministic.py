from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from fms_ehrs.framework.tokenizer import Tokenizer21
from fms_ehrs.framework.vocabulary import Vocabulary

ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_CONFIG = ROOT / "fms_ehrs/config/mimic-meds-exp3-full.yaml"
MEDS_SCHEMA = {
    "subject_id": pl.Int64,
    "hadm_id": pl.Int64,
    "time": pl.Datetime,
    "code": pl.String,
    "numeric_value": pl.Float32,
    "text_value": pl.String,
}


def _write_meds(data_dir: Path, *, reverse_rows: bool = False) -> None:
    start = datetime(2020, 1, 1)
    rows = []
    for subject_id, hadm_id, sodium in ((1, 101, 140.0), (2, 202, 142.0)):
        rows.extend(
            [
                (
                    subject_id,
                    hadm_id,
                    start,
                    "HOSPITAL_ADMISSION//ed",
                    None,
                    None,
                ),
                (
                    subject_id,
                    hadm_id,
                    start + timedelta(hours=1),
                    "LAB//sodium",
                    sodium,
                    None,
                ),
                (
                    subject_id,
                    hadm_id,
                    start + timedelta(hours=2),
                    "NOTE//alpha",
                    None,
                    None,
                ),
                (
                    subject_id,
                    hadm_id,
                    start + timedelta(hours=2),
                    "NOTE//zeta",
                    None,
                    None,
                ),
                (
                    subject_id,
                    hadm_id,
                    start + timedelta(hours=3),
                    "NOTE//status",
                    None,
                    "good",
                ),
                (
                    subject_id,
                    hadm_id,
                    start + timedelta(hours=25),
                    "HOSPITAL_DISCHARGE//home",
                    None,
                    None,
                ),
            ]
        )
    if reverse_rows:
        rows.reverse()
    data_dir.mkdir(parents=True)
    pl.DataFrame(rows, schema=MEDS_SCHEMA, orient="row").write_parquet(
        data_dir / "meds.parquet"
    )


def _write_breakpoint_meds(data_dir: Path) -> None:
    start = datetime(2020, 1, 1)
    rows = [(1, 101, start, "HOSPITAL_ADMISSION//ed", None, None)]
    for index, minutes in enumerate(
        (5, 15, 60, 120, 360, 720, 1440, 4320),
        start=1,
    ):
        rows.append(
            (
                1,
                101,
                start + timedelta(minutes=minutes),
                f"NOTE//event_{index}",
                None,
                None,
            )
        )
    rows.append(
        (
            1,
            101,
            start + timedelta(days=4),
            "HOSPITAL_DISCHARGE//home",
            None,
            None,
        )
    )
    data_dir.mkdir(parents=True)
    pl.DataFrame(rows, schema=MEDS_SCHEMA, orient="row").write_parquet(
        data_dir / "meds.parquet"
    )


def _decoded(timelines: pl.DataFrame, tokenizer: Tokenizer21) -> list[dict]:
    return [
        {
            "subject_id": row["hadm_id"],
            "tokens": [tokenizer.vocab.reverse[token] for token in row["tokens"]],
            "times": [str(time) for time in row["times"]],
            "numeric_values": row["numeric_values"],
        }
        for row in timelines.to_dicts()
    ]


def _deterministic_result(data_dir: Path) -> dict:
    tokenizer = Tokenizer21(
        data_dir=data_dir,
        config_file=TOKENIZER_CONFIG,
        deterministic_vocab=True,
    )
    timelines = tokenizer.get_tokens_timelines()
    return {
        "lookup": tokenizer.vocab.lookup,
        "timelines": _decoded(timelines, tokenizer),
    }


def test_deterministic_vocab_matches_legacy_decoded_timelines(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_meds(data_dir)

    legacy = Tokenizer21(data_dir=data_dir, config_file=TOKENIZER_CONFIG)
    legacy_timelines = legacy.get_tokens_timelines()
    deterministic = Tokenizer21(
        data_dir=data_dir,
        config_file=TOKENIZER_CONFIG,
        deterministic_vocab=True,
    )
    deterministic_timelines = deterministic.get_tokens_timelines()

    assert _decoded(legacy_timelines, legacy) == _decoded(
        deterministic_timelines, deterministic
    )
    assert deterministic.vocab.is_training is False

    lookup_before_repeat = dict(deterministic.vocab.lookup)
    deterministic.get_tokens_timelines()
    assert deterministic.vocab.lookup == lookup_before_repeat


def test_deterministic_vocab_matches_legacy_fused_numeric_tokens(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _write_meds(data_dir)

    legacy = Tokenizer21(
        data_dir=data_dir,
        config_file=TOKENIZER_CONFIG,
        fused_category_values=True,
    )
    legacy_timelines = legacy.get_tokens_timelines()
    deterministic = Tokenizer21(
        data_dir=data_dir,
        config_file=TOKENIZER_CONFIG,
        fused_category_values=True,
        deterministic_vocab=True,
    )
    deterministic_timelines = deterministic.get_tokens_timelines()

    assert _decoded(legacy_timelines, legacy) == _decoded(
        deterministic_timelines, deterministic
    )


def test_native_time_spacing_matches_legacy_at_breakpoints(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_breakpoint_meds(data_dir)

    legacy = Tokenizer21(data_dir=data_dir, config_file=TOKENIZER_CONFIG)
    deterministic = Tokenizer21(
        data_dir=data_dir,
        config_file=TOKENIZER_CONFIG,
        deterministic_vocab=True,
    )

    assert _decoded(legacy.get_tokens_timelines(), legacy) == _decoded(
        deterministic.get_tokens_timelines(), deterministic
    )


def test_deterministic_vocab_is_invariant_to_input_row_order(tmp_path: Path) -> None:
    ordered = tmp_path / "ordered"
    reversed_rows = tmp_path / "reversed"
    _write_meds(ordered)
    _write_meds(reversed_rows, reverse_rows=True)

    ordered_result = _deterministic_result(ordered)
    reversed_result = _deterministic_result(reversed_rows)

    assert ordered_result == reversed_result
    assert ordered_result["timelines"][0]["tokens"].index("EVENT_NOTE//alpha") < (
        ordered_result["timelines"][0]["tokens"].index("EVENT_NOTE//zeta")
    )


def test_deterministic_frozen_vocab_maps_unknown_tokens_to_none(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    _write_meds(train_dir)
    trainer = Tokenizer21(
        data_dir=train_dir,
        config_file=TOKENIZER_CONFIG,
        deterministic_vocab=True,
    )
    trainer.get_tokens_timelines()
    vocab_path = tmp_path / "vocab.gzip"
    trainer.vocab.save(vocab_path)

    val_dir = tmp_path / "val"
    _write_meds(val_dir)
    events = pl.read_parquet(val_dir / "meds.parquet")
    events = events.with_columns(
        pl.when(pl.col("code") == "NOTE//alpha")
        .then(pl.lit("NOTE//unseen"))
        .otherwise(pl.col("code"))
        .alias("code")
    )
    events.write_parquet(val_dir / "meds.parquet")

    tokenizer = Tokenizer21(
        data_dir=val_dir,
        config_file=TOKENIZER_CONFIG,
        vocab_path=vocab_path,
        deterministic_vocab=True,
    )
    vocabulary_size = len(tokenizer.vocab)
    timelines = tokenizer.get_tokens_timelines()
    words = _decoded(timelines, tokenizer)

    assert tokenizer.vocab.is_training is False
    assert len(tokenizer.vocab) == vocabulary_size
    assert all("EVENT_NOTE//unseen" not in row["tokens"] for row in words)
    assert all("EVENT_NOTE//zeta" in row["tokens"] for row in words)


def test_deterministic_vocab_is_thread_count_invariant(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_meds(data_dir)
    script = """
import json
import sys
from pathlib import Path
from fms_ehrs.framework.tokenizer import Tokenizer21

tokenizer = Tokenizer21(
    data_dir=Path(sys.argv[1]),
    config_file=Path(sys.argv[2]),
    deterministic_vocab=True,
)
timelines = tokenizer.get_tokens_timelines()
print(json.dumps({
    "lookup": {str(key): value for key, value in tokenizer.vocab.lookup.items()},
    "tokens": timelines.get_column("tokens").to_list(),
    "times": [[str(value) for value in row] for row in timelines.get_column("times").to_list()],
    "numeric_values": timelines.get_column("numeric_values").to_list(),
}, sort_keys=True))
"""

    def run(thread_count: int) -> dict:
        environment = os.environ | {
            "POLARS_MAX_THREADS": str(thread_count),
            "PYTHONPATH": str(ROOT),
        }
        result = subprocess.run(
            [sys.executable, "-c", script, str(data_dir), str(TOKENIZER_CONFIG)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(result.stdout)

    assert run(1) == run(4)


def test_split_cli_matches_monolithic_cli(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_meds(tmp_path / split)

    script = ROOT / "fms_ehrs/scripts/tokenize_w_config.py"
    common = [
        sys.executable,
        str(script),
        "--data_dir",
        str(tmp_path),
        "--config_loc",
        str(TOKENIZER_CONFIG),
        "--include_24h_cut",
        "--skip_summary",
        "--deterministic_vocab",
    ]
    for split in ("train", "val", "test"):
        subprocess.run(
            common
            + [
                "--data_version_out",
                "split",
                "--splits",
                split,
            ],
            check=True,
        )
    subprocess.run(
        common + ["--data_version_out", "monolithic"],
        check=True,
    )

    split_vocab = Vocabulary().load(tmp_path / "split-tokenized/train/vocab.gzip")
    monolithic_vocab = Vocabulary().load(
        tmp_path / "monolithic-tokenized/train/vocab.gzip"
    )
    assert split_vocab.lookup == monolithic_vocab.lookup
    for suffix in ("-tokenized", "_first_24h-tokenized"):
        for split in ("train", "val", "test"):
            split_frame = pl.read_parquet(
                tmp_path / f"split{suffix}" / split / "tokens_timelines.parquet"
            )
            monolithic_frame = pl.read_parquet(
                tmp_path
                / f"monolithic{suffix}"
                / split
                / "tokens_timelines.parquet"
            )
            assert split_frame.equals(monolithic_frame)


def test_numeric_stats_store_iqr_scale_not_sample_std() -> None:
    import numpy as np

    from fms_ehrs.framework.tokenizer_base import BaseTokenizer

    tokenizer = BaseTokenizer(data_dir=".")
    tokenizer.set_quants(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "sodium", prefix="LAB")
    stats = tokenizer.numeric_stats["LAB_sodium"]

    assert "std" not in stats
    assert stats["median"] == 3.0
    assert stats["scale"] == max((stats["q75"] - stats["q25"]) / 1.35, 1e-8)
