# fms-ehrs

Modeling codebase for [`../input-representation-benchmark`](../input-representation-benchmark):
tokenization, training, feature extraction, downstream probes, and prediction
aggregation. The benchmark repo schedules jobs and assembles final statistics.

Read both README files when reproducing a run.

## Environment

Clone this repo next to `input-representation-benchmark`, then:

```bash
conda env create -f ../input-representation-benchmark/environment.yml
conda activate input-rep
```

## Scripts by stage

| Stage | Script | Notes |
| --- | --- | --- |
| 0 | `fms_ehrs/scripts/tokenize_w_config.py` | YAML configs in `fms_ehrs/config/` |
| 1 (Exp1) | `fms_ehrs/scripts/tune_model.py` | packed training |
| 1 (Exp2/3) | `fms_ehrs/scripts/train_representation.py` | windowed training; resume/save controls for long runs |
| 2 | `fms_ehrs/scripts/extract_hidden_states.py` | final-valid-token features; multi-GPU sharded extraction |
| 3 | `fms_ehrs/scripts/transfer_rep_based_preds.py` | probes on extracted features |
| stats | `fms_ehrs/scripts/aggregate_version_preds.py` | per-family metrics backend |
| analysis | `fms_ehrs/scripts/eval_token_ce.py` | token CE for mechanistic figures |

Archived scripts live under `deprecated/`. Script list:
[`fms_ehrs/scripts/README.md`](fms_ehrs/scripts/README.md).

## Benchmark snapshot

The paper trains 28 model settings for one epoch each:

- **Exp1:** bin size, reference-range anchoring, fused vs unfused code/value tokens
- **Exp2:** value methods (`discrete`, `soft`, `xval`, `xval_affine`) and time
  methods (`none`, `age`, `rope`)
- **Exp3:** mapping arms (`native`, `clif_mapped`, `rand_mapped`, `freq_mapped`)
  with `discrete + rope`

There are 30 outcomes; each experiment evaluates 29 because the ICU outcome
differs between Exp1–2 and Exp3.

Exp3 tokenization reads only `LAB` and `VITAL` blocks
(`fms_ehrs/config/mimic-meds-exp3-icu.yaml`).

## Output contract

| Output | Produced by | Used by |
| --- | --- | --- |
| `<data_version>-tokenized/train/vocab.gzip` | `tokenize_w_config.py` | training, extraction |
| `<data_version>-tokenized/train/numeric_stats.json` | `tokenize_w_config.py` | `xval` / `xval_affine` |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines.parquet` | tokenization | extraction |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines_outcomes.parquet` | benchmark outcome joiners | Stage 3 |
| `<model_dir>/checkpoint-*` | training scripts | extraction |
| `<model_dir>/representation_mechanics.pt` | `train_representation.py` | wrapper rebuild at extraction |
| `<split>/features-<model_stem>.npy` | `extract_hidden_states.py` | Stage 3 |
| `test/*-preds-*.pkl` | `transfer_rep_based_preds.py` | stats refresh |

Qwen3 and Llama10ep runs use longer feature stems such as
`features-<run_dir>-model-discrete-time_rope.npy`. Stage 3 must load the same
stem. See the benchmark README, "Additional runs" section.

Long windowed jobs may set `IRB_USE_MAPPED_DATASET_CACHE=true` and
`IRB_MAPPED_DATASET_CACHE_DIR` so mapped HuggingFace datasets are built once and
reused after restarts.

## Assumptions

- Stages 2–3 read first-24-hour tokenized timelines.
- Default extraction uses the final valid token before `TRUNC`, `TL_END`, or
  `PAD`. For causal LMs, that position has the widest left-context view of the
  window.
- `xval` / `xval_affine` need both `numeric_stats.json` and
  `representation_mechanics.pt`.

Paper stats, additional-run stats, and training-stability plots are assembled in
the benchmark repo. Start with `Where to look first` there.

## Directory map

| Path | Role |
| --- | --- |
| `fms_ehrs/framework/` | library modules |
| `fms_ehrs/config/` | active MEDS configs |
| `fms_ehrs/scripts/` | CLI entrypoints |
| `fms_ehrs/tests/` | unit and dry-run checks |
| `notes/`, `docs/` | short reference notes |
| `deprecated/` | archived material |

`slurm/` is a pointer; live launchers are in the benchmark repo.

## Related docs

| Doc | Contents |
| --- | --- |
| [`../input-representation-benchmark/README.md`](../input-representation-benchmark/README.md) | full run path and output locations |
| [`../input-representation-benchmark/PIPELINE.md`](../input-representation-benchmark/PIPELINE.md) | stage-by-stage walkthrough |
| [`fms_ehrs/scripts/README.md`](fms_ehrs/scripts/README.md) | compact script inventory |
| [`fms_ehrs/tests/README.md`](fms_ehrs/tests/README.md) | test commands |
