# Adding an Event Stream or Downstream Outcome

This guide explains how to extend Stage 0 tokenization in `fms-ehrs` or add a downstream prediction target in `input-representation-benchmark`.

## 1. Choose the Target Layer

### Input Event Stream (Stage 0)

Modify the Stage 0 tokenizer configuration when the model must observe the signal in the input sequence during pretraining and feature extraction. Typical cases include:

- A new timed MEDS event table.
- A new categorical prefix or suffix field.
- A new numeric measurement that uses quantile binning or continuous value encoding.

### Downstream Prediction Label (Stage 3)

Keep the signal outside `fms-ehrs` when the model does not observe the variable as an input token. Stage 3 linear probes then predict the variable as a downstream target.

Downstream outcomes are extracted and joined in [`../input-representation-benchmark`](../input-representation-benchmark).

- Primary binary and length-of-stay outcomes: `../input-representation-benchmark/pipeline/scripts/extract_outcomes_meds.py`.
- 24-to-48-hour clinical outcomes (37 tasks across 8 families): `../input-representation-benchmark/pipeline/scripts/extract_extended_outcomes.py`.

## 2. Add a Timed Event Table to Stage 0

Format the new event table as a MEDS parquet file with these columns.

- `subject_id`.
- `time`.
- `code`.
- `numeric_value` (optional).
- `text_value` (optional).

Add the table to the `events` block of the YAML configuration passed to `fms_ehrs/scripts/tokenize_w_config.py`.

```yaml
events:
  - table: my_event_table
    prefix: MY_EVENT
    code: code
    time: time
```

For numeric events, retain `code` and `numeric_value` (plus optional reference-range columns for anchored partitioning). See the configurations in `fms_ehrs/config/` for complete examples.

- `mimic-meds-ed.yaml`.
- `mimic-meds.yaml`.
- `mimic-meds-exp3-full.yaml`.
- `cocoa-exp3-clif-full.yaml`.

## 3. Add Admission-Level Metadata

To attach admission-level metadata rather than a timed event, declare an entry under `reference.augmentation_tables` and expose the column under `prefix` or `suffix`.

```yaml
reference:
  augmentation_tables:
    - table: my_admission_table
      key: hospitalization_id
      agg_expr: pl.col("group_code").sort().alias("group_codes")
      validation: "1:1"

suffix:
  - column: group_codes
    prefix: GROUP
    is_list: true
```

## 4. Verification Checklist

1. Decide if the variable belongs in the Stage 0 token sequence or only in the Stage 3 outcome table.
2. For an input token variable, update the YAML configuration in `fms_ehrs/config/` and re-run Stage 0 tokenization and Stage 1 training.
3. For a downstream outcome variable, extract the label column in `input-representation-benchmark`, join it onto the outcome parquet, and run Stage 3 probes.
4. Add a unit test under `fms_ehrs/tests/unit/` or `../input-representation-benchmark/pipeline/tests/unit/`.
