from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fms_ehrs.framework.cocoa_meds import (
    ADMISSION_PREFIX,
    DISCHARGE_PREFIX,
    EVENT_FAMILY_PREFIXES,
    audit_code_status_assignment,
    freeze_exp3_splits,
    materialize_clif_meds,
    validate_collation_config,
    validate_exp3_meds_contract,
)
from fms_ehrs.framework.tokenizer import Tokenizer21

ROOT = Path(__file__).resolve().parents[3]
COLLATION_CONFIG = ROOT / "fms_ehrs/config/cocoa-exp3-clif-full.yaml"
TOKENIZER_CONFIG = ROOT / "fms_ehrs/config/mimic-meds-exp3-full.yaml"
MEDS_SCHEMA = {
    "subject_id": pl.Int64,
    "hadm_id": pl.Int64,
    "time": pl.Datetime,
    "code": pl.String,
    "numeric_value": pl.Float32,
    "text_value": pl.String,
}
SPECS = {
    "train": (101, 1001),
    "val": (102, 1002),
    "test": (103, 1003),
}


def _write_cohort_and_raw(tmp_path: Path) -> tuple[Path, Path]:
    cohort = tmp_path / "cohort"
    raw = tmp_path / "raw"
    cohort.mkdir()
    all_hadm = []
    all_patients = []
    base = datetime(2020, 1, 1)
    for index, (split, (subject_id, hadm_id)) in enumerate(SPECS.items()):
        all_hadm.append(hadm_id)
        all_patients.append(subject_id)
        pl.DataFrame({"hadm_id": [hadm_id]}).write_csv(
            cohort / f"{split}_hadm_ids.csv"
        )
        pl.DataFrame({"subject_id": [subject_id]}).write_csv(
            cohort / f"{split}_patient_ids.csv"
        )
        start = base + timedelta(days=index * 3)
        rows = [
            (subject_id, hadm_id, start, f"{ADMISSION_PREFIX}ed", None, None),
            (
                subject_id,
                hadm_id,
                start + timedelta(hours=1),
                "RAW//event",
                1.0,
                None,
            ),
            (
                subject_id,
                hadm_id,
                start + timedelta(hours=24),
                f"{DISCHARGE_PREFIX}home",
                None,
                None,
            ),
        ]
        frame = pl.DataFrame(rows, schema=MEDS_SCHEMA, orient="row")
        path = raw / split / "meds.parquet"
        path.parent.mkdir(parents=True)
        frame.write_parquet(path)
    pl.DataFrame({"hadm_id": all_hadm}).write_csv(cohort / "cohort_hadm_ids.csv")
    pl.DataFrame({"subject_id": all_patients}).write_csv(
        cohort / "cohort_patient_ids.csv"
    )
    return cohort, raw


def _write_contract_meds(source: Path, output: Path) -> None:
    for index, (split, (subject_id, hadm_id)) in enumerate(SPECS.items()):
        start = datetime(2020, 1, 1) + timedelta(days=index * 3)
        rows = [
            (subject_id, hadm_id, start, f"{ADMISSION_PREFIX}ed", None, None),
            (
                subject_id,
                hadm_id,
                start + timedelta(hours=24),
                f"{DISCHARGE_PREFIX}home",
                None,
                None,
            ),
        ]
        for family, prefixes in EVENT_FAMILY_PREFIXES.items():
            prefix = prefixes[0]
            rows.append(
                (
                    subject_id,
                    hadm_id,
                    start + timedelta(hours=1),
                    f"{prefix}{family}",
                    1.0 if family in {"labs", "vitals"} else None,
                    None,
                )
            )
        frame = pl.DataFrame(rows, schema=MEDS_SCHEMA, orient="row")
        path = output / split / "meds.parquet"
        path.parent.mkdir(parents=True)
        frame.write_parquet(path)


def _write_small_clif(raw_data_home: Path) -> None:
    starts = [datetime(2020, 1, 1) + timedelta(days=index * 3) for index in range(3)]
    subject_ids = [str(spec[0]) for spec in SPECS.values()]
    hadm_ids = [str(spec[1]) for spec in SPECS.values()]
    ends = [start + timedelta(hours=24) for start in starts]
    raw_data_home.mkdir()
    pl.DataFrame(
        {
            "patient_id": subject_ids,
            "hospitalization_id": hadm_ids,
            "admission_dttm": starts,
            "discharge_dttm": ends,
            "admission_type_category": ["ed"] * 3,
            "discharge_category": ["home"] * 3,
        }
    ).write_parquet(raw_data_home / "clif_hospitalization.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": hadm_ids,
            "in_dttm": starts,
            "out_dttm": [start + timedelta(hours=2) for start in starts],
            "location_category": ["icu"] * 3,
        }
    ).write_parquet(raw_data_home / "clif_adt.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": hadm_ids,
            "lab_order_dttm": starts,
            "lab_result_dttm": [start + timedelta(hours=1) for start in starts],
            "lab_category": ["sodium"] * 3,
            "lab_value_numeric": [140.0] * 3,
        }
    ).write_parquet(raw_data_home / "clif_labs.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": hadm_ids,
            "recorded_dttm": [start + timedelta(hours=1) for start in starts],
            "vital_category": ["heart_rate"] * 3,
            "vital_value": [80.0] * 3,
        }
    ).write_parquet(raw_data_home / "clif_vitals.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": hadm_ids,
            "recorded_dttm": [start + timedelta(hours=1) for start in starts],
            "assessment_category": ["gcs_total"] * 3,
            "numerical_value": [15.0] * 3,
            "categorical_value": pl.Series([None] * 3, dtype=pl.String),
        }
    ).write_parquet(raw_data_home / "clif_patient_assessments.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": hadm_ids,
            "recorded_dttm": [start + timedelta(hours=1) for start in starts],
            "position_category": ["prone"] * 3,
        }
    ).write_parquet(raw_data_home / "clif_position.parquet")
    respiratory = {
        "hospitalization_id": hadm_ids,
        "recorded_dttm": [start + timedelta(hours=1) for start in starts],
        "device_category": ["IMV"] * 3,
        "mode_category": ["Pressure Control"] * 3,
        "tracheostomy": [False] * 3,
    }
    for column in (
        "fio2_set",
        "lpm_set",
        "tidal_volume_set",
        "resp_rate_set",
        "pressure_control_set",
        "pressure_support_set",
        "flow_rate_set",
        "peak_inspiratory_pressure_set",
        "inspiratory_time_set",
        "peep_set",
        "tidal_volume_obs",
        "resp_rate_obs",
        "plateau_pressure_obs",
        "peak_inspiratory_pressure_obs",
        "peep_obs",
        "minute_vent_obs",
        "mean_airway_pressure_obs",
    ):
        respiratory[column] = [1.0] * 3
    pl.DataFrame(respiratory).write_parquet(
        raw_data_home / "clif_respiratory_support.parquet"
    )
    for table, category in (
        ("clif_medication_admin_continuous", "norepinephrine"),
        ("clif_medication_admin_intermittent", "ceftriaxone"),
    ):
        pl.DataFrame(
            {
                "hospitalization_id": hadm_ids,
                "admin_dttm": [start + timedelta(hours=1) for start in starts],
                "med_category": [category] * 3,
                "mar_action_category": ["given"] * 3,
                "med_dose": [1.0] * 3,
            }
        ).write_parquet(raw_data_home / f"{table}.parquet")
    crrt = {
        "hospitalization_id": hadm_ids,
        "recorded_dttm": [start + timedelta(hours=1) for start in starts],
        "crrt_mode_category": ["cvvh"] * 3,
    }
    for column in (
        "blood_flow_rate",
        "pre_filter_replacement_fluid_rate",
        "post_filter_replacement_fluid_rate",
        "dialysate_flow_rate",
        "ultrafiltration_out",
    ):
        crrt[column] = [1.0] * 3
    pl.DataFrame(crrt).write_parquet(raw_data_home / "clif_crrt_therapy.parquet")
    pl.DataFrame(
        {
            "patient_id": subject_ids,
            "start_dttm": [start + timedelta(hours=1) for start in starts],
            "code_status_category": ["Full"] * 3,
        }
    ).write_parquet(raw_data_home / "clif_code_status.parquet")


def test_locked_collation_config_has_only_the_declared_tables() -> None:
    config = validate_collation_config(COLLATION_CONFIG)
    tables = {
        entry["table"] for entry in config["entries"] if entry["table"] != "REFERENCE"
    }
    assert tables == {
        "clif_adt",
        "clif_code_status",
        "clif_crrt_therapy",
        "clif_labs",
        "clif_medication_admin_continuous",
        "clif_medication_admin_intermittent",
        "clif_patient_assessments",
        "clif_position",
        "clif_respiratory_support",
        "clif_vitals",
    }
    assert "subject_splits" not in config


def test_freeze_and_validate_preserve_exact_patient_split(tmp_path: Path) -> None:
    cohort, raw = _write_cohort_and_raw(tmp_path)
    frozen = cohort / "frozen_split_manifest.json"
    manifest = freeze_exp3_splits(
        cohort_dir=cohort, raw_meds_dir=raw, output_path=frozen
    )
    assert manifest["splits"]["train"]["hadm_count"] == 1
    output = tmp_path / "clif_meds"
    _write_contract_meds(raw, output)
    validation = validate_exp3_meds_contract(
        meds_dir=output, frozen_manifest_path=frozen
    )
    assert validation["splits"]["test"]["event_family_counts"]["code_status"] == 1


def test_code_status_audit_counts_out_of_window_rows(tmp_path: Path) -> None:
    source = tmp_path / "clif"
    source.mkdir()
    start = datetime(2020, 1, 1)
    pl.DataFrame(
        {
            "patient_id": ["1", "1", "9"],
            "start_dttm": [
                start + timedelta(hours=1),
                start + timedelta(hours=26),
                start + timedelta(hours=1),
            ],
        }
    ).write_parquet(source / "clif_code_status.parquet")
    reference = pl.DataFrame(
        {
            "hospitalization_id": ["100"],
            "patient_id": ["1"],
            "admission_dttm": [start],
            "discharge_dttm": [start + timedelta(hours=24)],
        }
    )
    audit = audit_code_status_assignment(raw_data_home=source, reference=reference)
    assert audit == {
        "source_rows": 3,
        "eligible_patient_rows": 2,
        "matched_rows": 1,
        "out_of_window_rows": 1,
        "ambiguous_rows": 0,
    }


def test_full_tokenizer_discards_null_codes(tmp_path: Path) -> None:
    start = datetime(2020, 1, 1)
    data_dir = tmp_path / "native"
    data_dir.mkdir()
    pl.DataFrame(
        [
            (101, 1001, start, f"{ADMISSION_PREFIX}ed", None, None),
            (101, 1001, start + timedelta(hours=1), None, None, None),
            (
                101,
                1001,
                start + timedelta(hours=2),
                "RAW//string_placeholder",
                None,
                "None",
            ),
            (
                101,
                1001,
                start + timedelta(hours=24),
                f"{DISCHARGE_PREFIX}home",
                None,
                None,
            ),
        ],
        schema=MEDS_SCHEMA,
        orient="row",
    ).write_parquet(data_dir / "meds.parquet")
    tokenizer = Tokenizer21(
        data_dir=data_dir,
        config_file=TOKENIZER_CONFIG,
        cut_at_24h=True,
    )
    timelines = tokenizer.get_tokens_timelines()
    assert timelines.height == 1
    assert not tokenizer.vocab.in_lookup("EVENT_None")


def test_cocoa_adapter_and_fms_tokenizer_smoke(tmp_path: Path) -> None:
    pytest.importorskip("cocoa.collator")
    cohort, raw = _write_cohort_and_raw(tmp_path)
    frozen = cohort / "frozen_split_manifest.json"
    freeze_exp3_splits(cohort_dir=cohort, raw_meds_dir=raw, output_path=frozen)
    clif = tmp_path / "clif"
    _write_small_clif(clif)
    output = tmp_path / "clif_meds"
    materialize_clif_meds(
        raw_data_home=clif,
        collation_config=COLLATION_CONFIG,
        frozen_manifest_path=frozen,
        output_meds_dir=output,
        work_dir=tmp_path / "cocoa_work",
    )
    tokenizer = Tokenizer21(
        data_dir=output / "train",
        config_file=TOKENIZER_CONFIG,
        cut_at_24h=True,
    )
    timelines = tokenizer.get_tokens_timelines()
    assert timelines.height == 1
    assert timelines.get_column("tokens").list.len().item() > 2
    assert not tokenizer.vocab.in_lookup("EVENT_None")
