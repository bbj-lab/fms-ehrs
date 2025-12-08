#!/usr/bin/env python3

"""
Run the tokenizer on various dev datasets and configurations;
this is a good thing to do after making changes to the tokenizer to verify
backwards compatibility
"""

import os
import pathlib
import tempfile

from fms_ehrs.framework.tokenizer import Tokenizer21
from fms_ehrs.framework.tokenizer_base import summarize


dev_dir = (
    pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/development-sample-21")
    if os.uname().nodename.startswith("cri")
    else pathlib.Path("~/Downloads/development-sample-21")
    .expanduser()
    .resolve()  # change if developing locally
)


for name, conf in {
    "raw-mimic": "clif-20"
}.items() | {  # run a clif-20 config on CLIF-2.1 data
    "raw-mimic": "clif-21",
    "raw-ucmc": "clif-21",
    "raw-meds": "mimic-meds",
    "raw-meds-ed": "mimic-meds-ed",
}.items():
    print(f"{name.upper()} with {conf.upper()}".ljust(72, "="))
    tkzr = Tokenizer21(
        config_file=f"../config/{conf}.yaml", data_dir=dev_dir / name / "dev"
    )

    tt = tkzr.get_tokens_timelines()
    summarize(tkzr, tt)
    tkzr.vocab.print_aux()
    print(list(tkzr.vocab.lookup.keys()))


print("transfer vocab test".upper().ljust(72, "="))
tkzr21_pp = Tokenizer21(
    config_file="../config/clif-21.yaml", data_dir=dev_dir.joinpath("raw-mimic/dev")
)
tt21_pp = tkzr21_pp.get_tokens_timelines()
summarize(tkzr21_pp, tt21_pp)
print(f"{len(tkzr21_pp.vocab)=}")
tkzr21_pp.vocab.print_aux()
print(list(tkzr21_pp.vocab.lookup.keys()))

with tempfile.NamedTemporaryFile() as fp:
    tkzr21_pp.vocab.save(fp.name)
    tkzr21_pp_ucmc = Tokenizer21(
        vocab_path=fp.name,
        config_file="../config/clif-21.yaml",
        data_dir=dev_dir.joinpath("raw-ucmc/dev"),
    )
    tt21_pp_ucmc = tkzr21_pp_ucmc.get_tokens_timelines()
    summarize(tkzr21_pp_ucmc, tt21_pp_ucmc)
