"""Adapt Cocoa-collated CLIF events to the benchmark MEDS split contract.

This module deliberately keeps Cocoa optional.  Importing it is required only
for materialization; split-manifest and output-contract checks stay usable in
ordinary FMS environments.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any, Iterable

import polars as pl
from ruamel.yaml import YAML

SPLITS = ("train", "val", "test")
ALLOWED_CLIF_TABLES = frozenset(
    {
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
)
EVENT_FAMILY_PREFIXES = {
    "adt": ("ADT-IN//", "ADT-OUT//"),
    "labs": ("LAB-ORDER//", "LAB-RESULT//"),
    "vitals": ("VITAL//",),
    "assessments": ("ASSESSMENT//",),
    "position": ("POSITION//",),
    "respiratory_support": ("RESP//",),
    "medication_continuous": ("MED-CONT//",),
    "medication_intermittent": ("MED-INT//",),
    "crrt": ("CRRT//",),
    "code_status": ("CODE-STATUS//",),
}
ADMISSION_PREFIX = "HOSPITAL_ADMISSION//"
DISCHARGE_PREFIX = "HOSPITAL_DISCHARGE//"
MEDS_COLUMNS = (
    "subject_id",
    "hadm_id",
    "time",
    "code",
    "numeric_value",
    "text_value",
)


def _hash_values(values: Iterable[object]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_from_csv(path: Path, column: str) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    frame = pl.read_csv(path)
    if column not in frame.columns:
        raise ValueError(f"Missing column {column!r} in {path}")
    ids = (
        frame.select(pl.col(column).cast(pl.Int64, strict=True).drop_nulls())
        .to_series()
        .to_list()
    )
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {column} values in {path}")
    return ids


def _assert_disjoint(values_by_split: dict[str, set[int]], label: str) -> None:
    for index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[index + 1 :]:
            overlap = values_by_split[left_split] & values_by_split[right_split]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"{label} leakage between {left_split} and {right_split}: {sample}"
                )


def _collect_raw_split_assignments(raw_meds_dir: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for split in SPLITS:
        path = raw_meds_dir / split / "meds.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw MEDS split: {path}")
        schema = pl.scan_parquet(path).collect_schema()
        missing = {"subject_id", "hadm_id"} - set(schema.names())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        assignments = (
            pl.scan_parquet(path)
            .select(
                pl.col("subject_id").cast(pl.Int64, strict=True),
                pl.col("hadm_id").cast(pl.Int64, strict=True),
            )
            .drop_nulls()
            .unique()
            .collect(engine="streaming")
        )
        duplicated_hadm = (
            assignments.group_by("hadm_id")
            .len()
            .filter(pl.col("len") != 1)
            .height
        )
        if duplicated_hadm:
            raise ValueError(
                f"Raw MEDS {split} contains hospitalizations mapped to multiple patients"
            )
        frames.append(assignments.with_columns(pl.lit(split).alias("split")))
    assignments = pl.concat(frames)
    _assert_disjoint(
        {
            split: set(
                assignments.filter(pl.col("split") == split)
                .get_column("hadm_id")
                .to_list()
            )
            for split in SPLITS
        },
        "hospitalization",
    )
    _assert_disjoint(
        {
            split: set(
                assignments.filter(pl.col("split") == split)
                .get_column("subject_id")
                .to_list()
            )
            for split in SPLITS
        },
        "patient",
    )
    return assignments


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen split manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported frozen split manifest: {path}")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError(f"Frozen split manifest has invalid split keys: {path}")
    return manifest


def _validate_raw_assignments_against_manifest(
    assignments: pl.DataFrame, manifest: dict[str, Any]
) -> None:
    for split in SPLITS:
        actual = assignments.filter(pl.col("split") == split)
        expected = manifest["splits"][split]
        hadm_ids = actual.get_column("hadm_id").to_list()
        patient_ids = actual.get_column("subject_id").unique().to_list()
        if len(hadm_ids) != expected["hadm_count"]:
            raise ValueError(
                f"Raw {split} hospitalization count drifted: "
                f"{len(hadm_ids)} != {expected['hadm_count']}"
            )
        if len(patient_ids) != expected["patient_count"]:
            raise ValueError(
                f"Raw {split} patient count drifted: "
                f"{len(patient_ids)} != {expected['patient_count']}"
            )
        if _hash_values(hadm_ids) != expected["hadm_sha256"]:
            raise ValueError(f"Raw {split} hospitalization IDs drifted from frozen manifest")
        if _hash_values(patient_ids) != expected["patient_sha256"]:
            raise ValueError(f"Raw {split} patient IDs drifted from frozen manifest")


def freeze_exp3_splits(
    *,
    cohort_dir: Path,
    raw_meds_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze split counts and hashes after verifying the native H_ICU arm."""
    cohort_dir = cohort_dir.expanduser().resolve()
    raw_meds_dir = raw_meds_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    assignments = _collect_raw_split_assignments(raw_meds_dir)

    cohort_hadm = set(_ids_from_csv(cohort_dir / "cohort_hadm_ids.csv", "hadm_id"))
    cohort_patients = set(
        _ids_from_csv(cohort_dir / "cohort_patient_ids.csv", "subject_id")
    )
    split_hadm: dict[str, set[int]] = {}
    split_patients: dict[str, set[int]] = {}
    manifest_splits: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        split_hadm[split] = set(
            _ids_from_csv(cohort_dir / f"{split}_hadm_ids.csv", "hadm_id")
        )
        split_patients[split] = set(
            _ids_from_csv(cohort_dir / f"{split}_patient_ids.csv", "subject_id")
        )
        actual = assignments.filter(pl.col("split") == split)
        actual_hadm = set(actual.get_column("hadm_id").to_list())
        actual_patients = set(actual.get_column("subject_id").to_list())
        if actual_hadm != split_hadm[split]:
            raise ValueError(
                f"Native {split} admission membership does not equal frozen cohort list"
            )
        if actual_patients != split_patients[split]:
            raise ValueError(
                f"Native {split} patient membership does not equal frozen cohort list"
            )
        manifest_splits[split] = {
            "hadm_count": len(actual_hadm),
            "patient_count": len(actual_patients),
            "hadm_sha256": _hash_values(actual_hadm),
            "patient_sha256": _hash_values(actual_patients),
        }

    _assert_disjoint(split_hadm, "cohort hospitalization")
    _assert_disjoint(split_patients, "cohort patient")
    if set().union(*split_hadm.values()) != cohort_hadm:
        raise ValueError("Split hospitalization lists do not cover H_ICU exactly")
    if set().union(*split_patients.values()) != cohort_patients:
        raise ValueError("Split patient lists do not cover H_ICU exactly")

    manifest = {
        "schema_version": 1,
        "cohort_dir": str(cohort_dir),
        "raw_meds_dir": str(raw_meds_dir),
        "splits": manifest_splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_collation_config(config_path: Path) -> dict[str, Any]:
    """Reject a Cocoa config that can silently widen the locked Exp3 policy."""
    config_path = config_path.expanduser().resolve()
    config = YAML(typ="safe").load(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Cocoa config: {config_path}")
    if config.get("subject_id") != "hospitalization_id":
        raise ValueError("Cocoa config must use hospitalization_id as its subject ID")
    if config.get("group_id") != "patient_id":
        raise ValueError("Cocoa config must use patient_id as its group ID")
    if "subject_splits" in config:
        raise ValueError("Cocoa config must not define its own subject_splits")
    reference = config.get("reference", {})
    if reference.get("table") != "clif_hospitalization":
        raise ValueError("Cocoa config must use clif_hospitalization as reference")

    entries = config.get("entries", [])
    if not entries:
        raise ValueError("Cocoa config has no entries")
    reference_entries = [entry for entry in entries if entry.get("table") == "REFERENCE"]
    if len(reference_entries) != 2:
        raise ValueError("Cocoa config must emit one admission and one discharge event")
    if any(entry.get("table") != "REFERENCE" for entry in entries[:2]):
        raise ValueError("Cocoa REFERENCE entries must precede clinical entries")

    event_tables = {
        entry.get("table") for entry in entries if entry.get("table") != "REFERENCE"
    }
    if event_tables != ALLOWED_CLIF_TABLES:
        raise ValueError(
            "Cocoa config event-table policy drifted: "
            f"expected {sorted(ALLOWED_CLIF_TABLES)}, found {sorted(event_tables)}"
        )
    for entry in entries:
        table = entry.get("table")
        if table == "REFERENCE":
            continue
        expected_key = "patient_id" if table == "clif_code_status" else "hospitalization_id"
        if entry.get("reference_key") != expected_key:
            raise ValueError(
                f"{table} must use reference_key={expected_key!r}, got "
                f"{entry.get('reference_key')!r}"
            )
        lowered = json.dumps(entry).lower()
        if "_converted" in lowered or "_processed" in lowered:
            raise ValueError(f"Derived table leaked into locked Cocoa policy: {entry}")
    return config


def _require_cocoa_collator():
    try:
        from cocoa.collator import Collator
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Cocoa is required to materialize the CLIF arm. Install the "
            "fms-ehrs[clif-cocoa] extra or the adjacent cocoa checkout."
        ) from error
    return Collator


def _starts_with_any(prefixes: tuple[str, ...]) -> pl.Expr:
    return reduce(or_, (pl.col("code").str.starts_with(prefix) for prefix in prefixes))


def _event_family_expression() -> pl.Expr:
    expression = pl.lit("unknown")
    for family, prefixes in reversed(tuple(EVENT_FAMILY_PREFIXES.items())):
        expression = pl.when(_starts_with_any(prefixes)).then(pl.lit(family)).otherwise(
            expression
        )
    return expression


def audit_code_status_assignment(
    *, raw_data_home: Path, reference: pl.DataFrame
) -> dict[str, int]:
    """Count in-window, dropped, and ambiguous patient-keyed code-status rows."""
    source_path = raw_data_home / "clif_code_status.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing required CLIF table: {source_path}")
    required_reference = {
        "hospitalization_id",
        "patient_id",
        "admission_dttm",
        "discharge_dttm",
    }
    missing_reference = required_reference - set(reference.columns)
    if missing_reference:
        raise ValueError(
            f"Cocoa reference frame is missing code-status columns: "
            f"{sorted(missing_reference)}"
        )

    source = pl.read_parquet(source_path)
    required_source = {"patient_id", "start_dttm"}
    missing_source = required_source - set(source.columns)
    if missing_source:
        raise ValueError(
            f"CLIF code-status table is missing columns: {sorted(missing_source)}"
        )
    source = (
        source.select("patient_id", "start_dttm")
        .drop_nulls()
        .with_row_index("_source_row")
        .with_columns(pl.col("patient_id").cast(pl.String))
    )
    windows = (
        reference.select(
            pl.col("hospitalization_id").cast(pl.String),
            pl.col("patient_id").cast(pl.String),
            "admission_dttm",
            "discharge_dttm",
        )
        .unique()
        .with_columns(
            pl.col("admission_dttm").cast(pl.Datetime, strict=False),
            pl.col("discharge_dttm").cast(pl.Datetime, strict=False),
        )
    )
    eligible_patients = windows.select("patient_id").unique()
    candidates = source.join(eligible_patients, on="patient_id", how="inner")
    if not candidates.height:
        return {
            "source_rows": source.height,
            "eligible_patient_rows": 0,
            "matched_rows": 0,
            "out_of_window_rows": 0,
            "ambiguous_rows": 0,
        }

    per_row = (
        candidates.join(windows, on="patient_id", how="left")
        .with_columns(
            pl.col("start_dttm")
            .cast(pl.Datetime, strict=False)
            .is_between(pl.col("admission_dttm"), pl.col("discharge_dttm"))
            .fill_null(False)
            .alias("_in_window")
        )
        .group_by("_source_row")
        .agg(pl.col("_in_window").sum().alias("_matches"))
    )
    matched_rows = per_row.filter(pl.col("_matches") == 1).height
    out_of_window_rows = per_row.filter(pl.col("_matches") == 0).height
    ambiguous_rows = per_row.filter(pl.col("_matches") > 1).height
    if ambiguous_rows:
        raise ValueError(
            f"Ambiguous code-status assignment for {ambiguous_rows} source rows"
        )
    return {
        "source_rows": source.height,
        "eligible_patient_rows": candidates.height,
        "matched_rows": matched_rows,
        "out_of_window_rows": out_of_window_rows,
        "ambiguous_rows": ambiguous_rows,
    }


def _assert_reference_matches_native(
    *, reference: pl.DataFrame, assignments: pl.DataFrame
) -> pl.DataFrame:
    expected = assignments.with_columns(
        pl.col("hadm_id").cast(pl.String).alias("_hadm_key"),
        pl.col("subject_id").cast(pl.String).alias("_subject_key"),
    )
    actual = (
        reference.select(
            pl.col("hospitalization_id").cast(pl.String).alias("_hadm_key"),
            pl.col("patient_id").cast(pl.String).alias("_subject_key_clif"),
        )
        .drop_nulls()
        .unique()
    )
    duplicate_hadm = actual.group_by("_hadm_key").len().filter(pl.col("len") != 1)
    if duplicate_hadm.height:
        raise ValueError("CLIF hospitalization reference has duplicate hospitalization IDs")
    joined = expected.join(actual, on="_hadm_key", how="left")
    if joined.filter(pl.col("_subject_key_clif").is_null()).height:
        raise ValueError("CLIF reference is missing one or more frozen H_ICU hospitalizations")
    mismatch = joined.filter(pl.col("_subject_key") != pl.col("_subject_key_clif"))
    if mismatch.height:
        raise ValueError(
            "CLIF and native MEDS disagree on patient identity for frozen hospitalizations"
        )
    return expected


def _sink_parquet(frame: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    try:
        frame.sink_parquet(path, engine="streaming")
    except Exception:
        path.unlink(missing_ok=True)
        frame.sink_parquet(path, engine="in-memory")


def _clear_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def validate_exp3_meds_contract(
    *,
    meds_dir: Path,
    frozen_manifest_path: Path,
    require_all_families: bool = True,
) -> dict[str, Any]:
    """Validate split parity and the FMS MEDS schema before Stage 0."""
    meds_dir = meds_dir.expanduser().resolve()
    manifest_path = frozen_manifest_path.expanduser().resolve()
    manifest = _read_manifest(manifest_path)
    raw_assignments = _collect_raw_split_assignments(Path(manifest["raw_meds_dir"]))
    _validate_raw_assignments_against_manifest(raw_assignments, manifest)

    split_frames: list[pl.DataFrame] = []
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        path = meds_dir / split / "meds.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing materialized MEDS split: {path}")
        schema = pl.scan_parquet(path).collect_schema()
        missing = set(MEDS_COLUMNS) - set(schema.names())
        if missing:
            raise ValueError(f"{path} is missing MEDS columns: {sorted(missing)}")
        observed = (
            pl.scan_parquet(path)
            .select(
                pl.col("subject_id").cast(pl.Int64, strict=True),
                pl.col("hadm_id").cast(pl.Int64, strict=True),
            )
            .drop_nulls()
            .unique()
            .collect(engine="streaming")
        )
        duplicate_hadm = observed.group_by("hadm_id").len().filter(pl.col("len") != 1)
        if duplicate_hadm.height:
            raise ValueError(
                f"{path} maps one or more hospitalizations to multiple patients"
            )
        expected = raw_assignments.filter(pl.col("split") == split).select(
            "subject_id", "hadm_id"
        )
        if observed.join(expected, on=["subject_id", "hadm_id"], how="anti").height:
            raise ValueError(f"{split} output includes an unexpected patient/hospitalization")
        if expected.join(observed, on=["subject_id", "hadm_id"], how="anti").height:
            raise ValueError(f"{split} output is missing a frozen patient/hospitalization")

        events = pl.scan_parquet(path)
        admissions = (
            events.filter(pl.col("code").str.starts_with(ADMISSION_PREFIX))
            .group_by("hadm_id")
            .agg(
                pl.len().alias("admission_rows"),
                pl.col("time").min().alias("admission_time"),
            )
            .collect(engine="streaming")
        )
        discharges = (
            events.filter(pl.col("code").str.starts_with(DISCHARGE_PREFIX))
            .group_by("hadm_id")
            .agg(
                pl.len().alias("discharge_rows"),
                pl.col("time").max().alias("discharge_time"),
            )
            .collect(engine="streaming")
        )
        markers = expected.select("hadm_id").join(admissions, on="hadm_id", how="left").join(
            discharges, on="hadm_id", how="left"
        )
        invalid_markers = markers.filter(
            (pl.col("admission_rows") != 1)
            | (pl.col("discharge_rows") != 1)
            | (pl.col("admission_time") > pl.col("discharge_time"))
        )
        if invalid_markers.height:
            raise ValueError(
                f"{split} has missing, duplicate, or inverted admission/discharge markers"
            )

        allowed_expr = reduce(
            or_,
            (
                pl.col("code").str.starts_with(prefix)
                for prefixes in EVENT_FAMILY_PREFIXES.values()
                for prefix in prefixes
            ),
        ) | pl.col("code").str.starts_with(ADMISSION_PREFIX) | pl.col("code").str.starts_with(
            DISCHARGE_PREFIX
        )
        unexpected_codes = (
            events.filter(~allowed_expr)
            .select("code")
            .unique()
            .limit(10)
            .collect(engine="streaming")
        )
        if unexpected_codes.height:
            raise ValueError(
                f"{split} contains prohibited code families: "
                f"{unexpected_codes.get_column('code').to_list()}"
            )
        family_counts = (
            events.filter(
                ~pl.col("code").str.starts_with(ADMISSION_PREFIX)
                & ~pl.col("code").str.starts_with(DISCHARGE_PREFIX)
            )
            .with_columns(_family=_event_family_expression())
            .group_by("_family")
            .len()
            .collect(engine="streaming")
        )
        counts = {
            family: int(
                family_counts.filter(pl.col("_family") == family)
                .get_column("len")
                .item()
            )
            if family_counts.filter(pl.col("_family") == family).height
            else 0
            for family in EVENT_FAMILY_PREFIXES
        }
        if require_all_families:
            missing_families = [
                family for family, count in counts.items() if count == 0
            ]
            if missing_families:
                raise ValueError(
                    f"{split} has no events for required CLIF families: {missing_families}"
                )
        split_frames.append(observed.with_columns(pl.lit(split).alias("split")))
        split_summary[split] = {
            "hospitalizations": observed.height,
            "patients": observed.get_column("subject_id").n_unique(),
            "event_family_counts": counts,
        }

    assignments = pl.concat(split_frames)
    _assert_disjoint(
        {
            split: set(
                assignments.filter(pl.col("split") == split)
                .get_column("subject_id")
                .to_list()
            )
            for split in SPLITS
        },
        "materialized patient",
    )
    return {"splits": split_summary}


def materialize_clif_meds(
    *,
    raw_data_home: Path,
    collation_config: Path,
    frozen_manifest_path: Path,
    output_meds_dir: Path,
    work_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Cocoa collation once, restore frozen IDs, and write benchmark splits."""
    raw_data_home = raw_data_home.expanduser().resolve()
    collation_config = collation_config.expanduser().resolve()
    frozen_manifest_path = frozen_manifest_path.expanduser().resolve()
    output_meds_dir = output_meds_dir.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if not raw_data_home.is_dir():
        raise FileNotFoundError(f"Missing CLIF source directory: {raw_data_home}")
    validate_collation_config(collation_config)
    manifest = _read_manifest(frozen_manifest_path)
    raw_assignments = _collect_raw_split_assignments(Path(manifest["raw_meds_dir"]))
    _validate_raw_assignments_against_manifest(raw_assignments, manifest)

    existing_outputs = [output_meds_dir / split / "meds.parquet" for split in SPLITS]
    if any(path.exists() for path in existing_outputs):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing CLIF MEDS output: {output_meds_dir}; "
                "pass --overwrite only after reviewing it."
            )
        _clear_path(output_meds_dir)
    if work_dir.exists() and overwrite:
        _clear_path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    Collator = _require_cocoa_collator()
    collator = Collator(
        collation_cfg=collation_config,
        raw_data_home=raw_data_home,
        processed_data_home=work_dir,
    )
    reference = collator.get_reference_frame().collect(engine="streaming")
    expected = _assert_reference_matches_native(
        reference=reference, assignments=raw_assignments
    )
    selected_reference = reference.join(
        expected.select("_hadm_key"),
        left_on="hospitalization_id",
        right_on="_hadm_key",
        how="inner",
    )
    code_status_audit = audit_code_status_assignment(
        raw_data_home=raw_data_home, reference=selected_reference
    )

    collated_path = work_dir / "cocoa_events.parquet"
    _sink_parquet(collator.get_all(), collated_path)
    events = (
        pl.scan_parquet(collated_path)
        .rename({"subject_id": "_hadm_key"})
        .with_columns(pl.col("_hadm_key").cast(pl.String))
        .join(expected.lazy(), on="_hadm_key", how="inner")
        .select(
            pl.col("subject_id").cast(pl.Int64, strict=True),
            pl.col("hadm_id").cast(pl.Int64, strict=True),
            pl.col("time").cast(pl.Datetime, strict=False),
            pl.col("code").cast(pl.String),
            pl.col("numeric_value").cast(pl.Float32, strict=False),
            pl.col("text_value").cast(pl.String),
            "split",
        )
    )
    for split in SPLITS:
        _sink_parquet(
            events.filter(pl.col("split") == split).select(*MEDS_COLUMNS),
            output_meds_dir / split / "meds.parquet",
        )

    validation = validate_exp3_meds_contract(
        meds_dir=output_meds_dir,
        frozen_manifest_path=frozen_manifest_path,
    )
    try:
        cocoa_version = importlib.metadata.version("cocoa")
    except importlib.metadata.PackageNotFoundError:
        cocoa_version = "unknown"
    artifact_manifest = {
        "schema_version": 1,
        "collation_config": str(collation_config),
        "collation_config_sha256": _hash_file(collation_config),
        "frozen_split_manifest": str(frozen_manifest_path),
        "frozen_split_manifest_sha256": _hash_file(frozen_manifest_path),
        "raw_data_home": str(raw_data_home),
        "cocoa_version": cocoa_version,
        "code_status_audit": code_status_audit,
        "validation": validation,
    }
    (output_meds_dir / "cocoa_adapter_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_manifest
