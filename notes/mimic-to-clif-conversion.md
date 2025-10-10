To download and process MIMIC data into CLIF-2.1.0 (with
[jq](https://jqlang.org)):

Install `jq` if needed:

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba install jq
```

Run conversion pipeline:

```sh
git clone git@github.com:Common-Longitudinal-ICU-data-Format/CLIF-MIMIC.git \
    CLIF-MIMICv0.2.0 --branch release/0.2.0
cd CLIF-MIMICv0.2.0 || exit
# wget -r -N -c -np --user "$(whoami)" --ask-password https://physionet.org/files/mimiciv/3.1/
export hm=/gpfs/data/bbj-lab/users/$(whoami)
jq --arg path "${hm}/physionet.org/files/mimiciv/3.1/" \
   '.default.mimic_csv_dir = $path | .clif_version = "2.1"' \
   ./config/config_template.json > ./config/config.json
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

This leaves parquet tables in ./output/rclif-2.0
