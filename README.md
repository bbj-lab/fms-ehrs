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
`/gpfs/data/bbj-lab/users/burkh4rt/CLIF-MIMIC/rclif`

<!--

Send code:
```sh
rsync -avht \
      --cvs-exclude \
      --exclude ".venv/*" \
      ~/Documents/chicago/clif-tokenizer \
      randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Environment:
```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Grab development sample:
```sh
export hm=/gpfs/data/bbj-lab/users/burkh4rt
rsync -avht \
    randi:${hm}/clif-development-sample \
    ~/Documents/chicago/CLIF/
```

-->
