# CLIF Tokenizer

This workflow tokenizes CLIF data
(https://clif-consortium.github.io/website/data-dictionary.html) into
`hospitalization_id`-level timelines.

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

Provided files are as follows:

1. [`create_development_sample.py`](./create_development_sample.py) pulls out a
   table corresponding to the first 10,000 patients to have a hospitalization
   event. This dataset can be used for quick prototyping.

2. [`create_train_val_test_split.py`](./create_train_val_test_split.py)
   partitions the full dataset into training, validation, and test sets of
   `hospital_id`-related events at the `patient_id` level.

3. [`vocabulary.py`](./vocabulary.py) demonstrates the `Vocabulary` class that
   powers tokenization.

4. [`tokenizer.py`](./tokenizer.py) demonstrates the `ClifTokenizer` class that
   operates on a directory containing CLIF tables in parquet format.

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
