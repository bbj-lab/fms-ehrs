#!/usr/bin/env python3

"""
Do highly informative tokens correspond to bigger jumps in representation space?
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger

# import plotly.express as px


logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-60358922_1-hp-W++",
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.model_loc),
)

jumps = np.load(
    data_dir.joinpath(
        f"{args.data_version}-tokenized", "test", f"all-jumps-{model_loc.stem}.npy"
    ),
)

infm = np.load(
    data_dir.joinpath(
        f"{args.data_version}-tokenized", "test", f"log_probs-{model_loc.stem}.npy"
    )
) / -np.log(2)

assert jumps.shape == infm[:, 1:].shape

df = pd.DataFrame(
    {"jump_length": jumps.ravel(), "information": infm[:, 1:].ravel()}
).dropna()

# fig = px.scatter(df, x="information", y="jump_length", trendline="ols")

# fig.update_layout(
#     title="Jump length vs. Information (Token-wise)",
#     template="plotly_white",
#     font_family="Computer Modern, CMU Serif",
# )

# fig.write_image(
#     out_dir.joinpath(
#         "jumps-vs-infm-{m}-{d}.pdf".format(m=model_loc.stem, d=data_dir.stem)
#     )
# )

lm = smf.ols(f"jump_length ~ 1 + information", data=df).fit()
logger.info(lm.summary())
