# Adding an outcome

This note covers the active path for extending the tokenizer and label surface.

## Pick the right layer

There are two common cases.

### 1. The signal belongs in the token sequence

Use this when the model should see the event during Stage 0 tokenization.

Examples:

- a new MEDS event stream
- a new categorical suffix field
- a new numeric measurement that should be discretized or passed through the wrapper path

### 2. The signal is only a downstream label

Use this when the model does not need the event as an input token, but Stage 3 should predict it.

For the current benchmark, downstream labels are handled in the sibling repo:

- base outcomes: `../input-representation-benchmark/scripts/extract_outcomes_meds.py`
- extended outcomes: `../input-representation-benchmark/scripts/extract_extended_outcomes.py`

## Adding a new tokenized event table

Create a MEDS-style parquet table with:

- `subject_id`
- `time`
- `code`
- optional `numeric_value`
- optional `text_value`

Then reference that table in the tokenizer config used by `tokenize_w_config.py`.

## Minimal event example

```yaml
events:
  - table: my_event_table
    prefix: MY_EVENT
    code: code
    time: time
```

This inserts tokens such as `MY_EVENT_xxx` where `xxx` comes from the `code` column.

## Adding numeric events

If the new table has a numeric measurement, keep the MEDS columns:

- `code`
- `numeric_value`
- optional reference-range columns if you need anchored binning

The active MEDS configs in `fms_ehrs/config/` show the pattern used by the benchmark:

- `mimic-meds.yaml`
- `mimic-meds-ed.yaml`
- `mimic-meds-exp3-icu.yaml`

## Adding joined columns to the reference frame

If the signal is admission-level metadata rather than a timed event, add it through `reference.augmentation_tables` and then expose it in the prefix or suffix sections of the config.

Typical use cases:

- grouped diagnosis labels
- admission-level flags
- demographic or cohort metadata

### Example

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

This appends tokens such as `GROUP_A` and `GROUP_B` after the discharge portion of the sequence.

## When the label should stay outside the sequence

If you only need a prediction target, keep it out of `fms-ehrs`.

For the benchmark path:

1. leave Stage 0 tokenization unchanged
2. derive the label parquet in `input-representation-benchmark`
3. join that label onto `tokens_timelines_outcomes.parquet` or `tokens_timelines_extended_outcomes.parquet`
4. rerun the affected Stage 3 and stats families

This is the right path for most benchmark-only labels.

## Checklist

1. Decide whether the signal belongs in the token sequence or only in the label parquet.
2. If it belongs in the sequence, add the table or joined column to the tokenizer config.
3. Retokenize the affected data version.
4. Retrain any models whose Stage 0 inputs changed.
5. If it is only a downstream label, update the benchmark-side outcome extraction scripts instead.
6. Add a small regression test near the code you changed.

Older CLIF/UCMC examples were archived under `../deprecated/`.
