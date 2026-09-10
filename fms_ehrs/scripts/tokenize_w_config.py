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
parser.add_argument(
    "--splits",
    nargs="+",
    choices=("train", "val", "test"),
    default=("train", "val", "test"),
    help="Tokenize only these splits; validation/test require an existing train vocabulary.",
)
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
    "--skip_summary",
    action="store_true",
    help=(
        "Skip descriptive token/timeline reporting. This avoids expanding full-length "
        "timelines solely for logging."
    ),
)
parser.add_argument(
    "--deterministic_vocab",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Build the training vocabulary before bulk tokenization, then use native "
        "Polars lookups and time-spacing operations."
    ),
)
parser.add_argument(
    "--config_loc", type=pathlib.Path, default="../fms_ehrs/config/mimic-meds.yaml"
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


def summarize_if_enabled(tokenizer: Tokenizer21, timelines: pl.DataFrame) -> None:
    if args.skip_summary:
        logger.info("Skipping timeline summary.")
    else:
        summarize(tokenizer, timelines, logger=logger)


# make output sub-directories
data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
all_splits = ("train", "val", "test")
requested_splits = tuple(split for split in all_splits if split in args.splits)

dirs_in: dict[str, pathlib.Path] = {}
dirs_out: dict[str, pathlib.Path] = {}
dirs_out_24h: dict[str, pathlib.Path] = {}
base_in = data_dir.joinpath(args.data_version_in)
if args.data_version_in == "raw" and not base_in.exists():
    base_in = data_dir

for split in all_splits:
    split_in = "tuning" if split == "val" and not base_in.joinpath("val").exists() else split
    dirs_in[split] = base_in.joinpath(split_in)
    if split not in requested_splits:
        continue
    if not args.only_24h_cut:
        dirs_out[split] = data_dir.joinpath(f"{args.data_version_out}-tokenized", split)
        dirs_out[split].mkdir(exist_ok=True, parents=True)
        fix_perms(data_dir.joinpath(f"{args.data_version_out}-tokenized"))
        fix_perms(dirs_out[split])
    if args.include_24h_cut or args.only_24h_cut:
        dirs_out_24h[split] = data_dir.joinpath(
            f"{args.data_version_out}_first_24h-tokenized", split
        )
        dirs_out_24h[split].mkdir(exist_ok=True, parents=True)
        fix_perms(data_dir.joinpath(f"{args.data_version_out}_first_24h-tokenized"))
        fix_perms(dirs_out_24h[split])


def tokenizer_for(
    split: str,
    *,
    vocab_path: pathlib.Path | None,
    cut_at_24h: bool,
) -> Tokenizer21:
    return Tokenizer21(
        data_dir=dirs_in[split],
        vocab_path=vocab_path,
        cut_at_24h=cut_at_24h,
        config_file=args.config_loc,
        max_padded_len=args.max_padded_len,
        quantizer=args.quantizer,
        clinical_anchoring=args.clinical_anchoring,
        numeric_encoding=args.numeric_encoding,
        include_ref_ranges=args.include_ref_ranges,
        include_time_spacing_tokens=args.include_time_spacing_tokens,
        fused_category_values=args.fused_category_values,
        detect_discrete=args.detect_discrete,
        deterministic_vocab=args.deterministic_vocab,
    )


def truncate_only(timelines: pl.DataFrame, tokenizer: Tokenizer21) -> pl.DataFrame:
    if args.max_padded_len is None:
        return timelines
    max_len = int(args.max_padded_len)
    trunc_id = tokenizer.vocab("TRUNC")
    return (
        timelines.lazy()
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


def configured_vocab_path(*, first_24h: bool) -> pathlib.Path | None:
    if args.vocab_path is not None:
        return pathlib.Path(args.vocab_path).expanduser().resolve()
    suffix = "_first_24h-tokenized" if first_24h else "-tokenized"
    path = data_dir / f"{args.data_version_out}{suffix}" / "train" / "vocab.gzip"
    return path if path.exists() else None


provided_vocab_path = (
    pathlib.Path(args.vocab_path).expanduser().resolve()
    if args.vocab_path is not None
    else None
)


def save_training_metadata(tokenizer: Tokenizer21, output_dir: pathlib.Path) -> None:
    tokenizer.vocab.save(output_dir / "vocab.gzip")
    tokenizer.save_numeric_stats(output_dir / "numeric_stats.json")
    fix_perms(output_dir / "numeric_stats.json")


if args.only_24h_cut:
    for split in requested_splits:
        is_train = split == "train"
        vocab_path = (
            provided_vocab_path
            if is_train
            else configured_vocab_path(first_24h=True)
        )
        if not is_train and vocab_path is None:
            raise FileNotFoundError(
                "Validation/test tokenization requires the existing train 24h vocabulary."
            )
        logger.info(f"{split} (24h cut only) ...")
        tokenizer = tokenizer_for(
            split,
            vocab_path=vocab_path,
            cut_at_24h=True,
        )
        timelines = truncate_only(tokenizer.get_tokens_timelines(), tokenizer)
        summarize_if_enabled(tokenizer, timelines)
        set_perms(timelines.write_parquet)(
            dirs_out_24h[split] / "tokens_timelines.parquet"
        )
        if is_train:
            save_training_metadata(tokenizer, dirs_out_24h[split])
else:
    for split in requested_splits:
        is_train = split == "train"
        vocab_path = (
            provided_vocab_path
            if is_train
            else configured_vocab_path(first_24h=False)
        )
        if not is_train and vocab_path is None:
            raise FileNotFoundError(
                "Validation/test tokenization requires the existing train full vocabulary."
            )
        tokenizer = tokenizer_for(
            split,
            vocab_path=vocab_path,
            cut_at_24h=False,
        )
        timelines = tokenizer.get_tokens_timelines()
        logger.info(f"{split}...")
        summarize_if_enabled(tokenizer, timelines)
        timelines = tokenizer.pad_and_truncate(timelines)
        set_perms(timelines.write_parquet)(
            dirs_out[split] / "tokens_timelines.parquet"
        )
        if is_train:
            save_training_metadata(tokenizer, dirs_out[split])

        if args.include_24h_cut:
            timelines_24h = tokenizer.cut_at_time(timelines)
            logger.info("24h cut...")
            summarize_if_enabled(tokenizer, timelines_24h)
            timelines_24h = tokenizer.pad_and_truncate(timelines_24h)
            set_perms(timelines_24h.write_parquet)(
                dirs_out_24h[split] / "tokens_timelines.parquet"
            )
            if is_train:
                save_training_metadata(tokenizer, dirs_out_24h[split])

logger.info("---fin")
