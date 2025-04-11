# FMs for EHRs

> This workflow can be used to reproduce the results in the accompanying
> manuscript.

## Requirements & structure

The bash scripts can be run in a slurm environment with the specified resource
requirements. (We used compute nodes with 8xA100 GPUs, connected with 2x 16-core
3.0-GHz AMD Milan processors for GPU-based work.) Each bash script calls one or
more python scripts that depend on an environment as described in the
`requirements.txt` file:

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

The code is structured logically as follows, where the numerical prefixes
correspond to the prefixes in the bash (`.sh`) files:

![Diagram for running the code](img/code-schematic.svg "Schematic")

## What the code does

### Data wrangling & tokenization

The code operates on MIMIC tabular data converted to the
[CLIF-2.0.0 format](https://clif-consortium.github.io/website/data-dictionary.html).
It gathers data associated to a given `hospitalization_id` and generates a
sequence of integers corresponding to the stay. Each sequence begins with a start
token, information about the patient, information about the stay itself, and then
encoded category-value pairs corresponding to, inter alia, lab records, vitals,
and medication. The sequences end with information on discharge and an end token,
like so:

<img src="./img/eg_timeline.png" 
     alt="Example timeline" 
     style="max-width:500px;width:100%">

Category-value tokenization iterates over all categories present in a table and
learns deciles for the values within each category. For example, the vital
corresponding to temperature in Celsius may be assigned the integer label ‘33.’
All measurements of temperature in the training set are used to determine deciles
for measurements within this category. For hospitalization 42, the tokens ‘33’
for this category and then ‘0’ for the corresponding deciled measurement would be
inserted into the timeline at ‘E1’:

<img src="./img/category-value-tokenization.png" 
     alt="CatVal tokenization" 
     style="max-width:500px;width:100%">

### Self-supervised training

Our training process packs sequences together, allowing one sequence to bleed
into the next example within a batch. The dark goldenrod boundary outlines tokens
corresponding to two individual hospitalization events:

<img src="./img/training.png" 
     alt="Training" 
     style="max-width:500px;width:100%">

We insert a variable number of padding tokens between sequences to expose the
model to padding. For the initial training, the model attempted to predict the
next token in a sequence given the previous tokens (‘context’).

### Objective-specific finetuning

We perform supervised fine-tuning with left-padded sequences. Each
hospitalization event (truncated at 24 hours) occupies a single training instance
and is paired with its associated subsequent outcome. In this way, fine-tuning is
outcome-specific.

<img src="./img/sft.png" 
     alt="Supervised Finetuning" 
     style="max-width:500px;width:100%">

### Representation extraction and analysis

Our pipeline extracts model-specific representations for each hospitalization
event that our useful for predicting a number of subsequent outcomes.

<!--

Format:
```
isort *.py
black *.py
shfmt -w *.sh
prettier --write --print-width 81 --prose-wrap always *.md
```

-->
