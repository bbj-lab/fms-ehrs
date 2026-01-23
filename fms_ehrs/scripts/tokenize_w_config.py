#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test sets
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import fix_perms, set_perms
from fms_ehrs.framework.tokenizer import Tokenizer21
from fms_ehrs.framework.tokenizer_base import summarize

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../tmp-test/")
parser.add_argument("--data_version_in", type=str, default="raw")
parser.add_argument("--data_version_out", type=str, default="test")
parser.add_argument("--vocab_path", type=pathlib.Path, default=None)
parser.add_argument("--include_24h_cut", action="store_true")
parser.add_argument(
    "--only_24h_cut",
    action="store_true",
    help=(
        "If set, ONLY write <data_version_out>_first_24h-tokenized outputs and skip "
        "writing <data_version_out>-tokenized outputs. This avoids tokenizing the "
        "full-length timelines when downstream stages only consume the 24h-cut data."
    ),
)
parser.add_argument(
    "--config_loc", type=pathlib.Path, default="../fms_ehrs/config/clif-21.yaml"
)
parser.add_argument(
    "--quantizer",
    type=str,
    choices=["deciles", "ventiles", "trentiles", "centiles"],
    default=None,
    help="Override quantizer (bins) used by tokenizer",
)
parser.add_argument(
    "--clinical_anchoring",
    type=str,
    choices=["none", "5-10-5", "10-10-10"],
    default=None,
    help="Override clinically anchored bin allocation (requires include_ref_ranges)",
)
parser.add_argument(
    "--include_ref_ranges",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override whether reference ranges are used for anchored binning",
)
parser.add_argument(
    "--include_time_spacing_tokens",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override whether ETHOS-style time spacing tokens are inserted",
)
parser.add_argument(
    "--fused_category_values",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override whether numeric events are fused (code+quantile) into one token",
)
parser.add_argument(
    "--numeric_encoding",
    type=str,
    choices=["quantile", "xval"],
    default=None,
    help="Override numeric encoding: 'quantile' (default) or 'xval' ([NUM] placeholder token)",
)
parser.add_argument(
    "--detect_discrete",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override discrete-value detection for small-support numeric codes",
)
parser.add_argument(
    "--max_padded_len",
    type=int,
    default=None,
    help="Override maximum padded sequence length",
)

args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

# make output sub-directories
data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
splits = ("train", "val", "test")

dirs_in = dict()
dirs_out = dict()
dirs_out_24h = dict()
for s in splits:
    # Input layout:
    # - CLIF typically uses versioned dirs: <data_dir>/<data_version_in>/<split>/
    # - MEDS extraction pipelines often place shards directly under: <data_dir>/<split>/
    #   (no "raw/" version directory). For convenience and backwards compatibility,
    #   if data_version_in=="raw" and <data_dir>/raw does not exist, we fall back
    #   to the no-version layout.
    base_in = data_dir.joinpath(args.data_version_in)
    if args.data_version_in == "raw" and not base_in.exists():
        base_in = data_dir

    # MEDS pipelines sometimes name validation split `tuning` instead of `val`.
    split_in = s
    if s == "val":
        cand = base_in.joinpath(s)
        if not cand.exists():
            split_in = "tuning"
    dirs_in[s] = base_in.joinpath(split_in)

    if not args.only_24h_cut:
        dirs_out[s] = data_dir.joinpath(args.data_version_out + "-tokenized", s)
        dirs_out[s].mkdir(exist_ok=True, parents=True)
        fix_perms(data_dir.joinpath(args.data_version_out + "-tokenized"))
        fix_perms(dirs_out[s])

    if args.include_24h_cut or args.only_24h_cut:
        dirs_out_24h[s] = data_dir.joinpath(args.data_version_out + "_first_24h-tokenized", s)
        dirs_out_24h[s].mkdir(exist_ok=True, parents=True)
        fix_perms(data_dir.joinpath(args.data_version_out + "_first_24h-tokenized"))
        fix_perms(dirs_out_24h[s])

if args.only_24h_cut:
    # Tokenize directly with a 24h cut to avoid materializing full-length timelines.
    logger.info("train (24h cut only) ...")
    tkzr = Tokenizer21(
        data_dir=dirs_in["train"],
        vocab_path=(
            pathlib.Path(args.vocab_path).expanduser().resolve()
            if args.vocab_path is not None
            else None
        ),
        cut_at_24h=True,
        config_file=args.config_loc,
        max_padded_len=args.max_padded_len,
        quantizer=args.quantizer,
        clinical_anchoring=args.clinical_anchoring,
        numeric_encoding=args.numeric_encoding,
        include_ref_ranges=args.include_ref_ranges,
        include_time_spacing_tokens=args.include_time_spacing_tokens,
        fused_category_values=args.fused_category_values,
        detect_discrete=args.detect_discrete,
    )
    tokens_timelines_24h = tkzr.get_tokens_timelines()
    summarize(tkzr, tokens_timelines_24h, logger=logger)
    # IMPORTANT (Stage0E use-case): do NOT materialize a fully-padded (max_len) column
    # for all timelines at tokenization time. For large cohorts and max_padded_len=4096,
    # writing padded sequences to parquet is prohibitively expensive (I/O + storage).
    #
    # Instead, we only truncate sequences that exceed max_padded_len (by appending TRUNC),
    # and downstream scripts (e.g., extract_hidden_states.py) dynamically pad per-batch.
    if args.max_padded_len is not None:
        max_len = int(args.max_padded_len)
        trunc_id = tkzr.vocab("TRUNC")
        tokens_timelines_24h = (
            tokens_timelines_24h.lazy()
            .with_columns(seq_len=pl.col("tokens").list.len())
            .with_columns(
                tokens=pl.when(pl.col("seq_len") > max_len)
                .then(
                    pl.concat_list(
                        pl.col("tokens").list.slice(offset=0, length=max_len - 1),
                        pl.lit(trunc_id),
                    )
                )
                .otherwise(pl.col("tokens")),
                times=pl.when(pl.col("seq_len") > max_len)
                .then(
                    pl.concat_list(
                        pl.col("times").list.slice(offset=0, length=max_len - 1),
                        pl.lit(None).cast(pl.Datetime(time_unit="ms")),
                    )
                )
                .otherwise(pl.col("times")),
                numeric_values=pl.when(pl.col("seq_len") > max_len)
                .then(
                    pl.concat_list(
                        pl.col("numeric_values").list.slice(offset=0, length=max_len - 1),
                        pl.lit(None).cast(pl.Float32),
                    )
                )
                .otherwise(pl.col("numeric_values")),
            )
            .drop("seq_len")
            .collect()
        )
    set_perms(tokens_timelines_24h.write_parquet)(
        dirs_out_24h["train"].joinpath("tokens_timelines.parquet")
    )
    tkzr.vocab.save(dirs_out_24h["train"].joinpath("vocab.gzip"))
    tkzr.save_numeric_stats(dirs_out_24h["train"].joinpath("numeric_stats.json"))
    fix_perms(dirs_out_24h["train"].joinpath("numeric_stats.json"))

    # take the learned/fixed tokenizer and tokenize the validation and test sets
    train_vocab_path = (
        pathlib.Path(args.vocab_path).expanduser().resolve()
        if args.vocab_path is not None
        else data_dir.joinpath(f"{args.data_version_out}_first_24h-tokenized", "train", "vocab.gzip")
    )
    for s in ("val", "test"):
        logger.info(f"{s} (24h cut only) ...")
        tkzr = Tokenizer21(
            data_dir=dirs_in[s],
            vocab_path=train_vocab_path,
            cut_at_24h=True,
            config_file=args.config_loc,
            max_padded_len=args.max_padded_len,
            quantizer=args.quantizer,
            clinical_anchoring=args.clinical_anchoring,
            numeric_encoding=args.numeric_encoding,
            include_ref_ranges=args.include_ref_ranges,
            include_time_spacing_tokens=args.include_time_spacing_tokens,
            fused_category_values=args.fused_category_values,
            detect_discrete=args.detect_discrete,
        )
        tokens_timelines_24h = tkzr.get_tokens_timelines()
        summarize(tkzr, tokens_timelines_24h, logger=logger)
        if args.max_padded_len is not None:
            max_len = int(args.max_padded_len)
            trunc_id = tkzr.vocab("TRUNC")
            tokens_timelines_24h = (
                tokens_timelines_24h.lazy()
                .with_columns(seq_len=pl.col("tokens").list.len())
                .with_columns(
                    tokens=pl.when(pl.col("seq_len") > max_len)
                    .then(
                        pl.concat_list(
                            pl.col("tokens").list.slice(offset=0, length=max_len - 1),
                            pl.lit(trunc_id),
                        )
                    )
                    .otherwise(pl.col("tokens")),
                    times=pl.when(pl.col("seq_len") > max_len)
                    .then(
                        pl.concat_list(
                            pl.col("times").list.slice(offset=0, length=max_len - 1),
                            pl.lit(None).cast(pl.Datetime(time_unit="ms")),
                        )
                    )
                    .otherwise(pl.col("times")),
                    numeric_values=pl.when(pl.col("seq_len") > max_len)
                    .then(
                        pl.concat_list(
                            pl.col("numeric_values").list.slice(offset=0, length=max_len - 1),
                            pl.lit(None).cast(pl.Float32),
                        )
                    )
                    .otherwise(pl.col("numeric_values")),
                )
                .drop("seq_len")
                .collect()
            )
        set_perms(tokens_timelines_24h.write_parquet)(
            dirs_out_24h[s].joinpath("tokens_timelines.parquet")
        )
else:
    # tokenize training set (full timelines)
    tkzr = Tokenizer21(
        data_dir=dirs_in["train"],
        vocab_path=(
            pathlib.Path(args.vocab_path).expanduser().resolve()
            if args.vocab_path is not None
            else None
        ),
        cut_at_24h=False,
        config_file=args.config_loc,
        max_padded_len=args.max_padded_len,
        quantizer=args.quantizer,
        clinical_anchoring=args.clinical_anchoring,
        numeric_encoding=args.numeric_encoding,
        include_ref_ranges=args.include_ref_ranges,
        include_time_spacing_tokens=args.include_time_spacing_tokens,
        fused_category_values=args.fused_category_values,
        detect_discrete=args.detect_discrete,
    )
    tokens_timelines = tkzr.get_tokens_timelines()
    logger.info("train...")
    summarize(tkzr, tokens_timelines, logger=logger)
    tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
    set_perms(tokens_timelines.write_parquet)(
        dirs_out["train"].joinpath("tokens_timelines.parquet")
    )
    tkzr.vocab.save(dirs_out["train"].joinpath("vocab.gzip"))
    # Save raw-value numeric stats (train split) for quantizer/anchoring-independent scaling.
    tkzr.save_numeric_stats(dirs_out["train"].joinpath("numeric_stats.json"))
    fix_perms(dirs_out["train"].joinpath("numeric_stats.json"))

    if args.include_24h_cut:
        tokens_timelines_24h = tkzr.cut_at_time(tokens_timelines)
        logger.info("24h cut...")
        summarize(tkzr, tokens_timelines_24h, logger=logger)
        tokens_timelines_24h = tkzr.pad_and_truncate(tokens_timelines_24h)
        set_perms(tokens_timelines_24h.write_parquet)(
            dirs_out_24h["train"].joinpath("tokens_timelines.parquet")
        )
        tkzr.vocab.save(dirs_out_24h["train"].joinpath("vocab.gzip"))
        tkzr.save_numeric_stats(dirs_out_24h["train"].joinpath("numeric_stats.json"))
        fix_perms(dirs_out_24h["train"].joinpath("numeric_stats.json"))

    # take the learned tokenizer and tokenize the validation and test sets
    for s in ("val", "test"):
        tkzr = Tokenizer21(
            data_dir=dirs_in[s],
            vocab_path=(
                pathlib.Path(args.vocab_path).expanduser().resolve()
                if args.vocab_path is not None
                else data_dir.joinpath(
                    f"{args.data_version_out}-tokenized", "train", "vocab.gzip"
                )
            ),
            cut_at_24h=False,
            config_file=args.config_loc,
            max_padded_len=args.max_padded_len,
            quantizer=args.quantizer,
            clinical_anchoring=args.clinical_anchoring,
            numeric_encoding=args.numeric_encoding,
            include_ref_ranges=args.include_ref_ranges,
            include_time_spacing_tokens=args.include_time_spacing_tokens,
            fused_category_values=args.fused_category_values,
            detect_discrete=args.detect_discrete,
        )
        tokens_timelines = tkzr.get_tokens_timelines()
        logger.info(f"{s}...")
        summarize(tkzr, tokens_timelines, logger=logger)
        tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
        set_perms(tokens_timelines.write_parquet)(
            dirs_out[s].joinpath("tokens_timelines.parquet")
        )
        if args.include_24h_cut:
            tokens_timelines_24h = tkzr.cut_at_time(tokens_timelines)
            logger.info("24h cut...")
            summarize(tkzr, tokens_timelines_24h, logger=logger)
            tokens_timelines_24h = tkzr.pad_and_truncate(tokens_timelines_24h)
            set_perms(tokens_timelines_24h.write_parquet)(
                dirs_out_24h[s].joinpath("tokens_timelines.parquet")
            )

logger.info("---fin")
