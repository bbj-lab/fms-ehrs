# CLIF Tokenizer

This workflow tokenizes CLIF data
(https://clif-consortium.github.io/website/data-dictionary.html) into
`hospitalization_id`-level timelines and trains a small instance of Mamba on the
resulting data.

The CLIF-2.0 specification provides the following schemas:
<img src="./img/ERD-2.png" 
       alt="schematic for CLIF-2.0" 
       style="max-width:700px;width:100%">

The conversion script for
[MIMIC-IV-3.1](https://physionet.org/content/mimiciv/3.1/) data creates 9 of
these tables:

- patient
- hospitalization
- adt
- vitals
- patient assessments
- respiratory support
- labs
- medication admin continuous
- position

The tables created using the conversion can be found at
`/gpfs/data/bbj-lab/users/burkh4rt/CLIF-MIMIC/rclif`.

The python scripts can be run in an environment as described in the
`requirements.txt` file:

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

Dataset creation and tokenization was tested on a cluster with a large amount of
memory. (All datasets could be fully loaded in memory.) To run on randi:

```sh
sbatch 1_2_make_tokenized_datasets.sh
```

Training is currently run with 8x Nvidia A100's:

```sh
sbatch 3_train_small_mamba.sh
```

Monitoring statistics and logs are collected at:
[https://wandb.ai/burkhart/clif_mamba](https://wandb.ai/burkhart/clif_mamba)

<!--

Send code:
```sh
rsync -avht \
      --cvs-exclude \
      --exclude ".venv/*" \
      --exclude ".idea/*" \
      ~/Documents/chicago/clif-tokenizer \
      randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Update venv:
```sh
pip3 list --format=freeze > requirements.txt
```

Grab development sample:
```sh
export hm=/gpfs/data/bbj-lab/users/burkh4rt
rsync -avht \
    --delete \
    randi:${hm}/clif-development-sample \
    ~/Documents/chicago/CLIF/
```

Format:
```
isort *.py
black *.py
prettier --write --print-width 81 --prose-wrap always *.md
```

Run on randi:
```
systemd-run --scope --user tmux new -s t3q
srun -p tier3q \
  --nodes=1 \
  --mem=1TB \
  --time=8:00:00 \
  --job-name=adhoc \
  --pty bash -i
source venv/bin/activate
```

-->
