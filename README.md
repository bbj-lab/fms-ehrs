# CLIF Tokenizer

This workflow tokenizes CLIF data
(https://clif-consortium.github.io/website/data-dictionary.html) into
`hospitalization_id`-level timelines and trains some small LLM's/FM's on the
result.

The conversion script for
[MIMIC-IV-3.1](https://physionet.org/content/mimiciv/3.1/) data creates 9 of
these tables. The tables created using the conversion can be found at
`/gpfs/data/bbj-lab/users/burkh4rt/CLIF-MIMIC/rclif`.

The python scripts can be run in an environment as described in the
`requirements.txt` file:

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

<!--

Send code:
```sh
rsync -avht \
      --exclude "venv/" \
      --exclude ".idea/" \
      --exclude "output/" \
      --exclude "wandb/" \
      --exclude "results/" \
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
shfmt -w -i 4 *.sh
prettier --write --print-width 81 --prose-wrap always *.md
```

Run on randi:
```
systemd-run --scope --user tmux new -s t2q
srun -p tier2q \
  --mem=100GB \
  --time=8:00:00 \
  --job-name=adhoc \
  --pty bash -i
source venv/bin/activate
```

Troubleshoot:
```
systemd-run --scope --user tmux new -s gpuq
srun -p gpuq \
  --gres=gpu:1 \
  --time=1:00:00 \
  --job-name=adhoc \
  --pty bash -i
```

Grab features and outcomes:
```
export hm=/gpfs/data/bbj-lab/users/burkh4rt
rsync -avht \
    --exclude "**/tokens_timelines.parquet" \
    randi:${hm}/clif-data/first-24h-tokenized \
    ~/Documents/chicago/clif-tokenizer/results
```

-->
