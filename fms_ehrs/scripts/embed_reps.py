#!/usr/bin/env python3

"""
embed 24h-representations
"""

import argparse
import pathlib

import numpy as np
import pacmap
import sklearn as skl
import umap

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../data-ucmc")
parser.add_argument("--data_version", type=str, default="QC_noX_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama1b-original-59946215-hp-QC_noX",
)
parser.add_argument("--mapper", choices=["umap", "isomap", "pacmap"], default="umap")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)

versions = ("orig", "new")
data_dirs = {
    v: (data_dir_orig if v == "orig" else data_dir_new).joinpath(
        f"{args.data_version}-tokenized", "test"
    )
    for v in versions
}

features = {
    v: np.load(data_dirs[v].joinpath("features-{m}.npy".format(m=model_loc.stem)))
    for v in versions
}

match args.mapper:
    case "isomap":
        mapper = skl.manifold.Isomap(n_jobs=-1, n_components=2, n_neighbors=100)
    case "umap":
        mapper = umap.UMAP(n_jobs=-1, n_components=2, n_neighbors=100)
    case "pacmap":
        mapper = pacmap.PaCMAP(
            n_components=2, n_neighbors=None, save_tree=True, random_state=42
        )
    case _:
        raise Exception(f"{args.mapper=} unsupported")

set_perms(np.save)(
    data_dirs["orig"].joinpath(
        "features-{typ}-{m}.npy".format(typ=args.mapper, m=model_loc.stem)
    ),
    mapper.fit_transform(features["orig"]),
)
set_perms(np.save)(
    data_dirs["new"].joinpath(
        "features-{typ}-{m}.npy".format(typ=args.mapper, m=model_loc.stem)
    ),
    mapper.transform(features["new"]),
)


logger.info("---fin")
