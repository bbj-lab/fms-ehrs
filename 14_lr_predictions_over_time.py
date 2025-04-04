#!/usr/bin/env python3

"""
learn a urt-lr model on training reps and apply it to pred reps
"""

import argparse
import pathlib
import pickle

import joblib
import numpy as np
import polars as pl
import sklearn as skl
import tqdm

from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_train", type=pathlib.Path)
parser.add_argument("--data_dir_pred", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--model_loc_sft", type=pathlib.Path)
parser.add_argument("--model_loc_base", type=pathlib.Path)
parser.add_argument("--big_batch_sz", type=int, default=2**12)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

rng = np.random.default_rng(42)

model_loc_sft, model_loc_base, data_dir_train, data_dir_pred = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.model_loc_sft, args.model_loc_base, args.data_dir_train, args.data_dir_pred),
)
data_version = args.data_version

data_dir_train = data_dir_train.joinpath(f"{data_version}-tokenized", "train")
data_dir_pred = data_dir_pred.joinpath(f"{data_version}-tokenized", "test")

with open(
    data_dir_pred.joinpath("sft_preds-n100_tokenwise-" + model_loc_sft.stem + ".pkl"),
    "rb",
) as fp:
    ml_info = pickle.load(fp)

featfiles = sorted(
    data_dir_train.glob("all-features-{m}-batch*.npy".format(m=model_loc_base.stem)),
    key=lambda s: int(s.stem.split("batch")[-1]),
)


def get_urt_slice_from_shard(f: pathlib.PurePath) -> np.array:
    raw = np.load(f)  # shape n_obs × tl_len × d_rep
    lens = np.argmin(np.isfinite(raw[:, :, 0]), axis=1)
    lens[lens == 0] = raw.shape[1]  # all finite gives 0
    idx = rng.integers(lens)
    return np.stack([raw[i, j] for i, j in enumerate(idx)])


urt_feats = np.concatenate(
    joblib.Parallel(n_jobs=-1, verbose=True)(
        joblib.delayed(get_urt_slice_from_shard)(f)
        for f in tqdm.tqdm(featfiles, desc="shards")
    )
)

labels = (
    pl.scan_parquet(data_dir_train.joinpath("tokens_timelines_outcomes.parquet"))
    .select("same_admission_death")
    .collect()
    .to_numpy()
    .ravel()
)  # n_train

labels = labels[: len(urt_feats)]

estimator = skl.pipeline.make_pipeline(
    skl.preprocessing.StandardScaler(),
    skl.linear_model.LogisticRegression(
        max_iter=10_000,
        n_jobs=-1,
        random_state=42,
        solver="newton-cholesky",
    ),
)
estimator.fit(X=urt_feats, y=labels)

predfiles = sorted(
    data_dir_pred.glob("all-features-{m}-batch*.npy".format(m=model_loc_base.stem)),
    key=lambda s: int(s.stem.split("batch")[-1]),
)


def get_preds_for_id(i: int):
    f = predfiles[i // args.big_batch_sz]
    tl = np.load(f)[i % args.big_batch_sz]  # now shape tl_len × d_rep
    preds = estimator.predict_proba(tl[np.isfinite(tl[:, 0])])[:, 1]
    return preds


mort_preds_lr = joblib.Parallel(n_jobs=-1, verbose=True)(
    joblib.delayed(get_preds_for_id)(k) for k in tqdm.tqdm(ml_info["mort_preds"].keys())
)

live_preds_lr = joblib.Parallel(n_jobs=-1, verbose=True)(
    joblib.delayed(get_preds_for_id)(k) for k in tqdm.tqdm(ml_info["live_preds"].keys())
)

ml_info["mort_preds_lr"] = dict(zip(ml_info["mort_preds"].keys(), mort_preds_lr))
ml_info["live_preds_lr"] = dict(zip(ml_info["live_preds"].keys(), live_preds_lr))

with open(
    data_dir_pred.joinpath(
        "sft_preds-n100_tokenwise-" + model_loc_sft.stem + "_LR.pkl"
    ),
    "wb",
) as fp:
    pickle.dump(ml_info, fp)
