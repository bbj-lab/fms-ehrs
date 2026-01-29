# FMs-EHRs

> This repo contains code to tokenize electronic health records, train generative
> event models on those tokenized records, and then perform various downstream
> analyses. [^1] [^2]

## Requirements

You can use [uv](https://docs.astral.sh/uv/pip/) to create an environment for
running this code (with Python >= 3.12) as follows:

```sh
uv venv --python=$(which python3)
. .venv/bin/activate
uv sync
```

<!-- pip install --torch-backend=cu128 --link-mode=copy -e . -->

<details>

<summary>For plots to render correctly, you may need to place Computer Modern fonts on your system.</summary>

These fonts appear in most PMLR-styled publications and can be installed as
follows:

```sh
mkdir -p ~/.local/share/fonts/CMU
cd ~/.local/share/fonts/CMU
wget https://mirrors.ctan.org/fonts/cm-unicode.zip
unzip cm-unicode.zip
find . -type f \( -iname "*.ttf" -o -iname "*.otf" \) -exec mv {} ~/.local/share/fonts/CMU/ \;
fc-cache -f -v
```

</details>

## Typical tokenization workflow

In a typical workflow, we first split our data into training, validation, and
test portions and then learn the bins and vocabulary on the training portion of
the data.

1.  [partition_w_config.py](fms_ehrs/scripts/partition_w_config.py) is our
    partitioning script; see [01_create_splits.sh](slurm/01_create_splits.sh) for
    example usage.

2.  [tokenize_w_config.py](fms_ehrs/scripts/tokenize_w_config.py) is our
    tokenization script; see [02_tokenize_splits.sh](slurm/02_tokenize_splits.sh)
    for example usage.

This means that deciles are training-set deciles, and tokens must appear in the
training set in order to be registered in the vocabulary (and have a learned
embedding in the model). This prevents tokens from appearing for the first time
in the test set (because a model is trained on a specific, fixed vocabulary).

## tl;dr

To run tokenization, place all data tables (in parquet format) into some
directory `data_dir`, define a yaml configuration file `config.yaml` (see
[clif-21.yaml](fms_ehrs/config/clif-21.yaml) for an example on the CLIF-2.1
standard), and then run:

```py
from fms_ehrs.framework.tokenizer import Tokenizer21

tkzr = Tokenizer21(config_file=config.yaml, data_dir=data_dir)
tt = tkzr.get_tokens_timelines()
```

Then `tt` will be a polars dataframe with columns "hospitalization_id", "tokens",
and "times" in the schema as follows:

```
┌────────────────────┬─────────────────┬─────────────────────────────────┐
│ hospitalization_id ┆ tokens          ┆ times                           │
│ ---                ┆ ---             ┆ ---                             │
│ str                ┆ list[i64]       ┆ list[datetime[ms]]              │
╞════════════════════╪═════════════════╪═════════════════════════════════╡
│ 20002103           ┆ [20, 350, … 21] ┆ [2116-05-08 02:45:00, 2116-05-… │
│ 20008372           ┆ [20, 350, … 21] ┆ [2110-10-30 13:03:00, 2110-10-… │
│ …                  ┆ …               ┆ …                               │
│ 29994865           ┆ [20, 364, … 21] ┆ [2111-01-28 21:49:00, 2111-01-… │
└────────────────────┴─────────────────┴─────────────────────────────────┘
```

The tokens can be converted back to strings using the vocabulary object inside
the tokenizer, in this case `tkzr.vocab` with something like this:

```py
tt.sample(1).select(
    pl.col("tokens").list.eval(pl.element().replace_strict(tkzr.vocab.reverse))
).item()
```

The tokenizer object also holds the auxiliary data used by the tokenizer. This
can be seen with:

```py
tkzr.print_aux()
```

For some summary statistics of the results, you might try:

```py
from fms_ehrs.framework.tokenizer_base import summarize

summarize(tkzr, tt)
```

## Configuring tokenization

We use a yaml file to configure the tokenization process. It's organized as
follows.

### Global options

-   We first define the `subject_id` (required) and `group_id` (optional):
    ```yaml
    subject_id: hospitalization_id # designator for what timelines represent
    group_id: patient_id # multiple subjects can belong to a group
    ```
-   We then define global options for tokenization, with descriptions as in the
    comments:

    ```yaml
    options:
        day_stay_filter: !!bool true # keep only hospitalizations >=24h in duration
        max_padded_len: !!int 2048 # length of sequences used for supervised fine-tuning
        quantizer: ventiles # 20 bins
        include_time_spacing_tokens: !!bool true # chronicle the passage of time
        fused_category_values: !!bool false # combine categories and values into single tokens
        detect_discrete: !!bool true # simple strategy for categories that take fewer values than no. Q tokens
        max_event_days: !!int 7 # truncate events at 7 days with an ellipsis token
    ```

### Reference table

-   Next we define a reference table that contains static data at the
    `subject_id` level. Columns from this table will be used to create the
    prefixes and suffixes of our timelines. The `table` field indicates the name
    of the parquet file in the `data_dir` folder to load. That table should
    contain a column `subject_id` and columns corresponding to the `start_time`
    and `end_time` for the `subject_id`.

    ```yaml
    reference:
        table: clif_hospitalization
        start_time: admission_dttm
        end_time: discharge_dttm
    ```

    Sometimes there are additional tables at different levels of granularity that
    need to be added to the reference table. This can be done with, e.g.:

    ```yaml
    reference:
        ...
        # join other tables to the reference table
        augmentation_tables:
            - table: clif_patient
              key: patient_id
              validation: "m:1"
    ```

    The key is used for joining onto the original reference table. Tables are
    loaded first, then filtered with an optional `filter_expr` (string or list of
    strings), then columns may be added with a `with_columns_expr` expression
    (string or list of strings), and finally aggregation is done where grouping
    is done by key if available or else the `subject_id` and then with an
    `agg_expr` expression. In all cases, expressions should evaluate to something
    that can be interpreted as a `pl.Expr`, for example:

    ```yaml
    reference:
        ...
        augmentation_tables:
            ...
            - table: clif_hospital_diagnosis
              key: hospitalization_id
              filter_expr: pl.col("diagnosis_primary") == 1
              with_col_expr: pl.col("diagnosis_code").str.split(".").list.first().alias("primary_dx_type")
              agg_expr: pl.col("primary_dx_type").sort().alias("primary_dx_types")
              validation: "1:1"
    ```

### Prefix

Now, we are prepared to configure the actual token columns. We form timelines as
the concatenation of prefixes, events, and suffixes.

-   We specify prefix tokens using columns that should exist in the reference
    frame (joining/postprocessing will have been completed prior to selecting
    these columns). For example:

    ```yaml
    prefix:
        - column: sex_category
        prefix: SEX
    ```

    inserts "SEX_Female" and "SEX_Male" tokens. Here "Female" and "Male"
    correspond to the entries in the `sex_category` column in the reference
    table. The `prefix` designation causes "SEX\_" to be inserted as a prefix for
    the tokens and helps us to recognize the type of these tokens. The prefix
    columns are inserted in the order in which they appear in this list.

### Events

-   We next specify events that are inserted into respective timelines according
    to the `time` designation for each event listed. For example:

    ```yaml
    events:
        - table: clif_adt
          prefix: XFR-IN
          code: location_category
          time: in_dttm
    ```

    pulls from the `location_category` column in the `clif_adt` table (that needs
    to exist as `clif_adt.parquet` in the `data_dir` discussed above.) To handle
    numeric values, we typically perform category-value tokenization that bins
    values according to the quantiles for a category as learned on training data:

    ![Category-value tokenization](./img/schematic.svg)

    For example:

    ```yaml
    events:
        ...
        - table: clif_labs
          prefix: LAB-RES
          code: lab_category
          numeric_value: lab_value_numeric
          time: lab_result_dttm
    ```

    inserts pairs of tokens like `('LAB-RES_potassium', 'Q14')`,
    `('LAB-RES_sodium', 'Q10')`, and `('LAB-RES_bun', 'Q2')`, using data from the
    table `clif_labs`. The table might look like this, with potentially
    additional comes (columns not used are ignored):

    ```
    ┌────────────────────┬─────────────────────────┬──────────────────────┬───────────────────┐
    │ hospitalization_id ┆ lab_result_dttm         ┆ lab_category         ┆ lab_value_numeric │
    │ ---                ┆ ---                     ┆ ---                  ┆ ---               │
    │ str                ┆ datetime[μs, UTC]       ┆ str                  ┆ f64               │
    ╞════════════════════╪═════════════════════════╪══════════════════════╪═══════════════════╡
    │ 23888643           ┆ 2111-01-06 05:16:00 UTC ┆ alt                  ┆ 21.0              │
    │ 23888643           ┆ 2111-01-06 05:16:00 UTC ┆ alkaline_phosphatase ┆ 42.0              │
    │ …                  ┆ …                       ┆ …                    ┆ …                 │
    │ 25401731           ┆ 2110-09-18 22:12:00 UTC ┆ pco2_venous          ┆ 46.0              │
    └────────────────────┴─────────────────────────┴──────────────────────┴───────────────────┘
    ```

    The tokenizer knows to use the `hospitalization_id` column, because of the
    specification `subject_id: hospitalization_id` at the beginning of the
    configuration file. To process the first row, the tokenizer looks up the
    quantiles for `lab_category = "alt"` and assigns `lab_value_numeric` to token
    `Q3` where `3` corresponds to the bin number. It then inserts
    `('LAB-RES_alt', 'Q3')` into the timeline for `hospitalization_id=23888643`
    at time `lab_result_dttm=2111-01-06 05:16:00 UTC`.

    There are certain variations on this scheme that our tokenizer can
    accommodate; we've been building out functionality as required by our use
    cases. We can filter and add columns in this section just as we can for the
    augmentation tables. For example, the follow processes categorical patient
    assessments:

    ```yaml
    events:
        ...
        - table: clif_patient_assessments
          prefix: ASMT
          filter_expr: pl.col("numerical_value").is_null()
          code: assessment_category
          text_value: categorical_value
          time: recorded_dttm
    ```

### Suffix

-   Finally we specify suffix tokens. These work in the same way as the prefix
    tokens. For example:

    ```yaml
    suffix:
        - column: discharge_category
          prefix: DSCG

        - column: primary_dx_types
          prefix: DX
          is_list: !!bool true
    ```

    The `is_list: !!bool true` processes `primary_dx_types` as a list.

## Post-tokenization

Once tokenization has completed, the next step is typically to train a model on
the training portion of the data.

3. [tune_model.py](fms_ehrs/scripts/tune_model.py) is a model tuning script. (We
   typically run hyperparameter tuning with
   [optuna](https://optuna.readthedocs.io) as part of the training process.) See
   [03_tune_model.sh](slurm/03_tune_model.sh) for example usage.

## Usage notes

-   We've started experimenting with [apptainer](https://apptainer.org)-based
    containerization, a successor to
    [singularity](https://singularityware.github.io/index.html). In an
    environment with apptainer available (e.g.
    `/gpfs/data/bbj-lab/.envs/apptainer`), you can define something like

    ```sh
    export hm="/gpfs/data/bbj-lab/users/$(whoami)"
    python3() {
        apptainer exec --bind $hm:$hm --nv /gpfs/data/bbj-lab/users/burkh4rt/env.sif python3 "$@"
    }
    ```

    and then your calls to python3 will be using it. This is considered
    experimental; any feedback is welcome.

    You can can also create your own version of this container with:

    ```sh
    micromamba activate apptainer
    export TMPDIR="/scratch/$(whoami)/cache"
    export APPTAINER_TMPDIR="/scratch/$(whoami)/cache"
    export APPTAINER_CACHEDIR="/scratch/$(whoami)/cache"

    apptainer build env.sif env.def
    ```

-   The number of model parameters depends on the size of the vocabulary (because
    we're learning a token embedding).

[^1]:
    M. Burkhart, B. Ramadan, Z. Liao, K. Chhikara, J. Rojas, W. Parker, & B.
    Beaulieu-Jones, Foundation models for electronic health records:
    representation dynamics and transferability,
    [arXiv:2504.10422](https://doi.org/10.48550/arXiv.2504.10422)

[^2]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188.

<!--

Format:
````
ruff format .
ruff check .
shfmt -w slurm/
```

Send to randi:
```
rsync -avht \
 --delete \
 --exclude "slurm/output/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/fms-ehrs-reps \
 bbj-lab2:~
```

Run on randi:
```
systemd-run --scope --user tmux new -s t3q || tmux a -t t3q
srun -p tier3q \
 --mem=100GB \
 --time=8:00:00 \
 --job-name=adhoc \
 --pty bash -i
source .venv/bin/activate
```

Troubleshoot:
```
systemd-run --scope --user tmux new -s gpuq || tmux a -t gpuq
srun -p gpudev \
 --gres=gpu:1 \
 --time=8:00:00 \
 --job-name=adhoc \
 --pty bash -i
source .venv/bin/activate
jupyter notebook --no-browser --ip=0.0.0.0 --port=8088 ssh -L 8088:localhost:8088 cri22cn401
```

Grab generated plots:
```
rsync -avht \
 randi:/gpfs/data/bbj-lab/users/burkh4rt/figs \
 ~/Downloads
```

Grab dev sample:
```
rsync -avht \
 --delete \
 randi:/gpfs/data/bbj-lab/users/burkh4rt/development-sample-21 \
 ~/Downloads
```

Save environment:
```
uv pip compile --torch-backend=cu128 pyproject.toml -o requirements.txt
```

Install directly from github:

```sh
pip install -e "git+https://github.com/bbj-lab/clif-tokenizer.git@main#egg=fms-ehrs"
````

Fix permissions:

```sh
chgrp -R cri-bbj_lab . && chmod -R +770 .
```

Send to bbj-lab:

```
rsync -avht \
  --delete \
  --exclude "slurm/output/" \
  --exclude ".venv/" \
  --exclude ".idea/" \
  ~/Documents/chicago/fms-ehrs-reps \
  bbj-lab2:~
```

-->
