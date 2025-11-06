# Generic Tokenizer

See[https://github.com/bbj-lab/generic-ehr-tokenizer](https://github.com/bbj-lab/generic-ehr-tokenizer)
for a generic tokenizer that builds tokenized timelines from EHR data. It
decouples tokenization logic from data pre-processing by requiring a data
processor class that implements three methods: `get_prefix_query`,
`get_event_query`, and `get_suffix_query`.

## Required Data Processor Interface

`get_prefix_query() -> pl.LazyFrame`

Returns prefix data (demographics and admission info) for each hospitalization.

-   subject_id (Utf8): patient/subject id

-   hadm_id (Utf8): hospitalization/admission id

-   admission_time (Datetime[ms]): admission timestamp

-   race (Utf8): patient race

-   sex (Utf8): patient sex

-   age_at_admission (Float64): age

-   admission_type (Utf8): admission type

`get_event_query(event_config: dict) -> pl.LazyFrame`

Returns event data (e.g., labs, medications).

-   subject_id (Utf8): patient/subject id

-   hadm_id (Utf8): hospitalization/admission id

-   time (Datetime[ms]): event timestamp

-   code (Utf8): event code (normalized to lowercase with underscores)

-   numeric_value (Float64, optional): numeric event value

-   text_value (Utf8, optional): text event value

> The structure of the event data object required by the tokenizer is based on
> the
> [Medical Event Data Standard](https://github.com/Medical-Event-Data-Standard/meds)
> DataSchema with the addition of a hadm_id.

`get_suffix_query() -> pl.LazyFrame`

Returns suffix data (discharge info) for each hospitalization.

-   subject_id (Utf8): patient/subject id

-   hadm_id (Utf8): hospitalization id

-   discharge_time (Datetime[ms]): discharge timestamp

-   discharge_category (Utf8): discharge category

## Usage

### Environment Setup

Create conda environment.

```
conda create --name fms_ehrs python=3.12
conda activate fms_ehrs
```

Install dependencies from pyproject.toml

```
pip install -e
```

### Command Line Arguments

-   `--config-path (string, default: cfg_mimic_default.yaml)`

Path to the tokenizer configuration YAML defining prefix, events, suffix, and
tokenizer options.

-   `--data-dir (string, default: MIMIC-IV parquet directory)`

Directory containing the source data files in parquet format to tokenize.

-   `--timelines-output (string, default: mimiciv_timelines.parquet)`

Path where tokenized timelines are saved as Parquet.

-   `--vocab-output (string, default: mimiciv_vocabulary.gzip)`

Path where the vocabulary is saved as a compressed file.

-   `--limit (integer, default: 500)`

Number of admission records to process. Use -1 to process all records.

-   ` --debug (flag)`

Enables timestamp logging for each token in the summary.

### Example Commands

Process the first 100 admissions:

`python tokenizer2.py --limit=100`

Run with custom paths:

```
python tokenizer2.py \
  --config-path /path/to/config.yaml \
  --data-dir /path/to/data \
  --timelines-output /path/to/output/timelines.parquet \
  --vocab-output /path/to/output/vocabulary.gzip
```

Show timestamps for example timelines `python tokenizer2.py --limit=100 --debug`

### Configuration

The tokenizer uses a YAML config to control tokenization and which data tables to
include. Specify the config file path with the --config-path command line
argument.

### Config File Structure

1. `tokenizer` - Core settings

```
tokenizer:
	subject_id: subject_id           		# Specify patient/subject identifier column
	max_padded_len: 1024             		# Maximum length for padding/truncation
	quantizer: deciles               		# Quantization method: "deciles" or "sigmas"
	include_time_spacing_tokens: false      # Add time spacing tokens (T_5m-15m, etc.)
	cut_at_24h: false                		# Cut timelines at 24 hours
	day_stay_filter: true            		# Filter patients with <24h stays
	max_text_value_length: 10       		# Max characters for text values
```

2. `prefix` - Admission/demographic tokens settings

```
prefix:
  - column: race                   # Column name from data
    prefix: RACE                   # Token prefix (e.g., "RACE_asian")
  - column: sex
    prefix: SEX
  - column: age_at_admission
    quantize: true                 # Convert to quantile tokens (Q0-Q9)
  - column: admission_type
    prefix: ADMN
```

3. `events` - Event tokens (labs, vitals, procedures, etc.) settings

```
events:
  - table: labevents               # Source table name
    prefix: LAB                    # Token prefix
    code: itemid                   # Event code column
    numeric_value: ~               # Numeric value column (~ = auto-handle)
    text_value: ~                  # Text value column (~ = auto-handle)
    time: charttime                # Timestamp column
```

### Example Configurations

`cfg_mimic_default.yaml`

-   Vitals (ICU chartevents)

-   Lab events

-   ICD procedures

-   Text values enabled (max 10 chars)

`cfg_mimic_add_omr.yaml`

-   Everything in default

-   OMR (Observation Medical Record) events

-   Text values enabled

`cfg_mimic_no_text.yaml`

-   Default events without OMR

-   Text values disabled (max_text_value_length: 0)
