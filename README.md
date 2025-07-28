# Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models

> We present a foundation model-derived method to identify highly informative
> tokens and events in a patient's electronic healthcare record. Our approach
> considers incoming data in the entire context of a patient's hospitalization
> and so can flag anomalous events that rule-based approaches would consider
> within a normal range. We demonstrate that the events our model flags are
> significant for predicting downstream patient outcomes, and that events
> identified as carrying little information can safely be dropped. Finally, we
> show how informativeness can help to interpret the predictions of prognostic
> models trained on FM-derived representations.

## Requirements & structure

The bash scripts can be run in a [slurm](https://slurm.schedmd.com) environment
with the specified resource requirements. (We used compute nodes with 8×A100
40GB-PCIe GPUs, connected with 2×16-core 3.0-GHz AMD Milan processors for
GPU-based work.) Each bash script calls one or more python scripts that depend on
an environment as described in the `requirements.txt` file. You can set up an
environment with [pytorch](https://pytorch.org/get-started/locally/) configured
for CUDA 12.8 with [uv](https://docs.astral.sh/uv/pip/) as follows:

```sh
uv venv --python=$(which python3) venv
. venv/bin/activate
uv pip install --torch-backend=cu128 --link-mode=copy -e .
```

For plots to render correctly, you may need to install a working version of
[tex](https://www.tug.org/texlive/) on your system. The code is structured
logically as follows, where the numerical prefixes correspond to the prefixes in
the slurm files (located in the [slurm folder](./slurm)):

```mermaid
---
config:
  theme: neutral
  look: handDrawn
  layout: elk
  themeCSS: "* { overflow: visible; }"
---
flowchart TD
 subgraph s1["Data processing"]
        N1["01_create_data_splits"]
        N2["02_tokenize_data_splits"]
        N3["03_extract_outcomes"]
        N16["16_aggregate_stats"]
        N17["17_pull_mimic_info"]
  end
 subgraph s2["Information estimation"]
        N4["04_tune_model"]
        N5["05_examine_model"]
        N6["06_extract_information"]
        N7["07_process_info"]
  end
 subgraph s3["Redaction experiment"]
        N8["08_redact_timelines"]
        N9["09_extract_reps"]
        N10["10_xfer_rep_based_preds"]
        N11["11_version_comparison"]
        N12["12_run_stats"]
  end
 subgraph s4["Reps vs info"]
        N13["13_extract_all_reps"]
        N14["14_process_trajectories"]
        N15["15_jumps_vs_info"]
  end
    N1 --> N2
    N2 --> N3 & N4
    N3 --> N16 & N10
    N4 --> N5 & N6 & N8 & N13
    N6 --> N7 & N15
    N7 --> N8
    N8 --> N9
    N4 --> N9 --> N10
    N10 --> N11
    N11 --> N12
    N13 --> N14
    N14 --> N15
    N16 --> N17
```

## What the code does

### Data processing (and tokenization)

We consider 422k hospitalization events for adults (age 18 or older) from the
Beth Israel Deaconess Medical Center between 2008–2019
([MIMIC-IV-3.1](https://physionet.org/content/mimiciv/3.1/)) and 50k
hospitalization events from [UCMC](https://www.uchicagomedicine.org) between
March 2020 and March 2022. We restricted to patients with stays of at least 24
hours. We formatted EHR data from each health system into the
[CLIF-2.0.0 format](https://web.archive.org/web/20250711203935/https://clif-consortium.github.io/website/data-dictionary/data-dictionary-2.0.0.html).
The MIMIC patients were partitioned intro training, validation, and test sets at
a 70\%-10\%-20\% rate, according to the randomized time of their first
hospitalization event, with training patients coming first, followed by
validation and then test. We then collected each hospitalization event for
patients in a given set. In this way, hospitalization records in the test set
corresponded to patients with no hospitalization events in the training or
validation sets. UCMC data was primarily used as a held-out test set. For this
reason, we partitioned UCMC hospitalizations into training, validation, and test
sets at a 5\%-5\%-90\% rate in the same manner as used for MIMIC.

We convert each hospitalization event into a sequence of integers corresponding
to the stay. For a given sequence, the first token always corresponds to timeline
start token. The next three tokens contain patient-level demographic information
on race, ethnicity, and sex. The following two tokens correspond to
admission-specific information, namely patient age converted to a decile and
admission type. Taken together, we refer to the 5 tokens occurring immediately
after the timelines start token as the _admission prefix_. Tokens corresponding
to a variety of events for a hospitalization are then inserted in the same order
in which these events occurred. Transfers are encoded with their CLIF location
category. Labs are encoded with two tokens and inserted at the time results
become available: one for the lab category, and a second corresponding to the
deciled lab value in the training data within that category. We call this
strategy, of tokenizing categories and binning their corresponding values
according to the training value of the deciles, category-value tokenization:

![Category-value tokenization](./img/schematic.svg)

A handful of other tables receive this type of tokenization: vitals and results
according to vital category, medication and dosage by medication category,
assessment and results by assessment category. Respiratory information is
recorded at the beginning of respiratory support; the encoded information is mode
category and device category. We include a token indicating if a patient is
placed into a prone position. All hospitalization-related data is encoded this
way and inserted in chronological order. Tokens that arrive synchronously
correspond to an event and always appear coterminously in a sequence. Timelines
then end with a token for discharge category and a dedicated timeline end token.

### Information estimation

Consider the set $V^T$ of length $T$ sequences of tokens drawn from some
vocabulary $V$. For a given sequence $x=(x_1,\dotsc, x_T)$ and indices
$1\leq u \leq v \leq T$, we let $x_{u:v}=(x_u, x_{u+1}, \dotsc, x_v)$ correspond
to the subsequence and $x_{<u}=x_{1:u-1}$ to the context at $u$ for $u>1$. If $p$
is a probability distribution on $V^T$, we let
$p(x_{u:v})=P_{X\sim p}(X_{u:v}=x_{u:v})$ denote the marginal distribution and
$p(x_{u:v}|x_{y:z})=P_{X\sim p}(X_{u:v}=x_{u:v}|X_{y:z}=x_{y:z})$ denote the
conditional for indices $u,v,y,z$. We adopt the convention that
$p(x_{u:v} | x_{<1}) = p(x_{u:v})$. In this work, we focus on the context-aware
information given by

$\boxed{I_p(x_t | x_{<t}) = - \log_{2} p(x_t | x_{<t})}$

for tokens $x_t$ and by

$\boxed{I_p(x_{u:v} | x_{<u}) = - \log_{2} p(x_{u:v} | x_{<u})}$

for subsequences $x_{u:v}$. As
$p(x_{u:v} | x_{<t})=\textstyle\prod\nolimits_{t=u}^v p(x_t | x_{<t})$, it
follows that
$I_{p}(x_{u:v} | x_{<u})=\textstyle\sum\nolimits_{t=u}^v I_{p}(x_t | x_{<t})$.
Thus, the context-aware information for subsequences can be obtained by adding
over that of the individual tokens. In our case, we focus on subsequences of
tokens that are added to our timelines contemporaneously. We call these "events."

We train a foundation model (FM) from scratch using a variation of the Llama-3.2
1B-parameter architecture on our tokenized MIMIC training and validation sets.
This model then takes the place of $p$ in the above equations. We can take
example timelines and use the model-determined measure of information to
highlight them (first 102 tokens shown here, reading from left to right in
row-major order):

![Example highlighted timeline](./img/example_tl.svg)

### Redaction experiment

We drop events (corresponding to subsequences $x_{u:v}$) according to their
model-determined informativeness $I_{p}(x_{u:v} | x_{<u})$. For details, please
see our manuscript.

### Representations vs. information

We relate movements in model-derived representation space to the informativeness
of the corresponding tokens or events. Again, further details are available in
the manuscript.

## Usage notes

-   Credentialed users may obtain the
    [MIMIC-IV-3.1 dataset](https://physionet.org/content/mimiciv/3.1/) from
    Physionet. [This repo](https://github.com/bbj-lab/CLIF-MIMIC) contains
    instructions and code for converting it to the
    [CLIF-2.0.0 format](https://web.archive.org/web/20250711203935/https://clif-consortium.github.io/website/data-dictionary/data-dictionary-2.0.0.html).
    (Use the [v0.1.0](https://github.com/bbj-lab/CLIF-MIMIC/releases/tag/v0.1.0)
    release.) The `rclif-2.0` folder location is then passed as `--data_dir_in`
    to the [first slurm script](./slurm/01_create_train_val_test_split.sh).

-   Many of the slurm scripts assume a folder structure as follows, where
    `tree ${hm}` (_cf_
    [tree](https://manpages.ubuntu.com/manpages/noble/man1/tree.1.html)) looks
    something like this:

    ```sh
    .
    ├── data-mimic # MIMIC datasets
    │   ├── raw
    │   │   ├── test
    │   │   │   ├── clif_adt.parquet
    │   │   │   ├── ...
    │   │   │   └── clif_vitals.parquet
    │   │   ├── train
    │   │   │   ├── clif_adt.parquet
    │   │   │   ├── ...
    │   │   │   └── clif_vitals.parquet
    │   │   └── val
    │   │       ├── clif_adt.parquet
    │   │       ├── ...
    │   │       └── clif_vitals.parquet
    │   ├── ...
    │   └── W++_first_24h-tokenized
    │       ├── test
    │       │   └── tokens_timelines.parquet
    │       ├── train
    │       │   ├── tokens_timelines.parquet
    │       │   └── vocab.gzip
    │       └── val
    │           └── tokens_timelines.parquet
    ├── data-ucmc  # UCMC datasets
    │   └── ...
    ├── mdls  # to hold all models generated
    │   └── ...
    ├── mdls-archive  # models for long-term storage
    │   └── llama-med-60358922_1-hp-W++
    │       ├── config.json
    │       ├── generation_config.json
    │       └── model.safetensors
    ├── fms-ehrs-info  # THIS REPO
    │   └── ...
    └── figs  # for generated figures
    ```

    Tokenized datasets are deposited into the `data-mimic` or `data-ucmc` folder,
    depending on data provenance. Trained models are stored in `mdls`. Many
    models are generated and these take up significant amounts of space. Models
    to be kept are copied into `mdls-archive`. Generated figures are placed in
    the `figs` folder.

-   Slurm jobs can be queued in sequence as follows:

    ```sh
    j01=$(sbatch --parsable 01_create_train_val_test_split.sh)
    j02=$(sbatch --parsable --depend=afterok:${j01} 02_tokenize_train_val_test_split.sh)
    j03=$(sbatch --parsable --depend=afterok:${j02} 03_extract_outcomes.sh)
    ...
    ```

-   If you find yourself manually running python scripts from an interactive
    slurm job afer running `preamble.sh`, you can append:

    ```sh
    2>&1 | tee -a output/$SLURM_JOBID-$jname.stdout
    ```

    to keep logs.

<!--

Format:
```
isort fms_ehrs/
black fms_ehrs/
shfmt -w slurm/
prettier --write *.md
```

Send to randi:
```
rsync -avht \
  --delete \
  --exclude "slurm/output/" \
  --exclude "venv/" \
  --exclude ".idea/" \
  ~/Documents/chicago/fms-ehrs-reps \
  randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Run on randi:
```
systemd-run --scope --user tmux new -s t2q
srun -p tier2q \
  --mem=25GB \
  --time=8:00:00 \
  --job-name=adhoc \
  --pty bash -i
source venv/bin/activate
```

Troubleshoot:
```
systemd-run --scope --user tmux new -s gpuq
srun -p gpuq \
  --reservation=gpudev \
  --gres=gpu:1 \
  --time=8:00:00 \
  --job-name=adhoc \
  --pty bash -i
. venv/bin/activate
jupyter notebook --no-browser --ip=0.0.0.0 --port=8088
ssh -L 8088:localhost:8088 cri22cn401
```

Grab generated plots:
```
rsync -avht \
    randi:/gpfs/data/bbj-lab/users/burkh4rt/figs \
    ~/Downloads
```

Save environment:
```
uv pip compile --torch-backend=cu128 pyproject.toml -o requirements.txt
```

Get fonts on randi:
```
mkdir -p ~/.local/share/fonts/CMU
cd ~/.local/share/fonts/CMU
wget https://mirrors.ctan.org/fonts/cm-unicode.zip
unzip cm-unicode.zip
find . -type f \( -iname "*.ttf" -o -iname "*.otf" \) -exec mv {} ~/.local/share/fonts/CMU/ \;
fc-cache -f -v
fc-list | grep -i cmu
```

Install directly from github:

```sh
pip install -e "git+https://github.com/bbj-lab/clif-tokenizer.git@main#egg=fms-ehrs"
```

-->
