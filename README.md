# CLIF Tokenizer

This workflow tokenizes CLIF data
(https://clif-consortium.github.io/website/data-dictionary.html) into
`hospitalization_id`-level timelines.

The CLIF-2.0 specification provides the following schemas:
<img src="./img/ERD-2.png" 
       alt="schematic for CLIF-2.0" 
       style="max-width:700px;width:100%">

The conversion script for
[MIMIC-IV-3.1](https://physionet.org/content/mimiciv/3.1/) data creates 8 of
these tables:

- patient
- hospitalization
- adt
- vitals
- patient assessments
- respiratory support
- labs
- medication admin continuous

The tables created using the conversion can be found at
`/gpfs/data/bbj-lab/users/burkh4rt/CLIF-MIMIC/rclif`.

The python scripts can be run in an environment as described in the
`requirements.txt` file:

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

The scripts are as follows:

1. [`create_development_sample.py`](./create_development_sample.py) pulls out a
   table corresponding to the first 10,000 patients to have a hospitalization
   event. This dataset can be used for quick prototyping.

2. [`create_train_val_test_split.py`](./create_train_val_test_split.py)
   partitions the full dataset into training, validation, and test sets at the
   `patient_id` level.

3. [`tokenizer_development.py`](./tokenizer_development.py) provides a demo
   script that creates sample tokenized timelines at the `hospitalization_id`
   level.

<!--

Send code:
```sh
rsync -avht \
      --cvs-exclude \
      --exclude ".venv/*" \
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
    randi:${hm}/clif-development-sample \
    ~/Documents/chicago/CLIF/
```

Format:
```
isort *.py
black *.py
```

-->
