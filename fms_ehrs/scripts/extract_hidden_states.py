#!/usr/bin/env python3

"""
Extract the final hidden state (at just under 24h) from each provided sequence.

Supports both standard models (Exp1) and Exp2 wrapper models (soft discretization,
xVal, Time-Aware RoPE) by auto-detecting representation_mechanics.pt in the model directory.
"""

import json
import os
import pathlib
import re
import shutil

import fire as fi
import numpy as np
import torch as t
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.dataset import compute_relative_times_hours
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.model_wrapper import create_representation_model
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


def _sanitize_model_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def _feature_model_stem(model_loc: pathlib.Path) -> str:
    if model_loc.name.startswith("model-") and model_loc.parent.name:
        return _sanitize_model_stem(f"{model_loc.parent.name}-{model_loc.name}")
    return _sanitize_model_stem(model_loc.stem)


def _dist_barrier(world_size: int) -> None:
    if world_size > 1:
        t.distributed.barrier()


def _feature_path(data_dir: pathlib.Path, *, all_layers: bool, model_stem: str) -> pathlib.Path:
    return data_dir.joinpath(
        "features{x}-{m}.npy".format(
            x="-all-layers" if all_layers else "",
            m=model_stem,
        )
    )


def _shard_dir_for(feature_path: pathlib.Path) -> pathlib.Path:
    return feature_path.parent / f".{feature_path.stem}.shards"


def _prepare_shard_dir(shard_dir: pathlib.Path, *, rank: int, world_size: int) -> None:
    if world_size == 1:
        return
    if rank == 0 and shard_dir.exists():
        shutil.rmtree(shard_dir)
    _dist_barrier(world_size)
    shard_dir.mkdir(parents=True, exist_ok=True)
    _dist_barrier(world_size)


def _merge_feature_shards(
    *,
    output_path: pathlib.Path,
    shard_dir: pathlib.Path,
    n_rows: int,
    feature_shape: tuple[int, ...],
    world_size: int,
) -> None:
    features = np.empty((n_rows, *feature_shape), dtype=np.float16)
    seen = np.zeros(n_rows, dtype=bool)

    for shard_rank in range(world_size):
        shard_path = shard_dir / f"rank-{shard_rank:05d}-of-{world_size:05d}.npz"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing extraction shard: {shard_path}")
        with np.load(shard_path) as shard:
            indices = shard["indices"].astype(np.int64, copy=False)
            shard_features = shard["features"].astype(np.float16, copy=False)
        if indices.shape[0] != shard_features.shape[0]:
            raise ValueError(
                f"Shard {shard_path} has {indices.shape[0]} indices but "
                f"{shard_features.shape[0]} feature rows."
            )
        if indices.size:
            if indices.min() < 0 or indices.max() >= n_rows:
                raise ValueError(f"Shard {shard_path} contains out-of-range row indices.")
            if seen[indices].any():
                raise ValueError(f"Shard {shard_path} overlaps a previously merged shard.")
            features[indices] = shard_features
            seen[indices] = True

    if not seen.all():
        missing = np.flatnonzero(~seen)
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"Merged extraction is missing {missing.size} row(s); first missing: {preview}"
        )

    set_perms(np.save)(output_path, features)


def _load_representation_meta(model_loc: pathlib.Path) -> dict | None:
    """Load representation_mechanics.pt from model directory if present.

    Returns
    -------
    dict with keys 'representation', 'temporal', 'num_bins', 'time_rope_scaling',
    'value_encoder_state', or None if not found.
    """
    rep_path = model_loc / "representation_mechanics.pt"
    if rep_path.exists():
        meta = t.load(rep_path, map_location="cpu", weights_only=False)
        logger.info("Loaded representation mechanics from %s", rep_path)
        logger.info(
            "  representation=%s  temporal=%s  num_bins=%s",
            meta.get("representation"),
            meta.get("temporal"),
            meta.get("num_bins"),
        )
        return meta
    return None


def _infer_representation_from_path(model_loc: pathlib.Path) -> tuple[str, str]:
    """Infer representation and temporal from the model directory name.

    Model directories follow the naming convention:
        model-<representation>-<temporal>
    e.g., model-xval-time_rope, model-discrete-time_tokens
    """
    stem = model_loc.name  # e.g., "model-xval-time_rope"
    parts = stem.split("-", 1)  # ["model", "xval-time_rope"]
    if len(parts) < 2:
        return "discrete", "time_tokens"
    remainder = parts[1]  # "xval-time_rope"
    # Split representation from temporal
    # IMPORTANT: order matters (e.g., "xval_affine" must match before "xval").
    for rep in ("xval_affine", "soft", "xval", "discrete"):
        if remainder.startswith(rep):
            rest = remainder[len(rep):]
            if rest.startswith("-"):
                temporal = rest[1:]  # "time_rope" or "time_tokens"
            else:
                temporal = "time_tokens"
            return rep, temporal
    return "discrete", "time_tokens"




@logger.log_calls
def main(
    *,
    data_dir: os.PathLike = "../../data-mimic",
    data_version: str = "QC_day_stays_first_24h",
    model_loc: os.PathLike = "../../mdls-archive/llama1b-57928921-run1",
    batch_sz: int = 2**5,
    all_layers: bool = False,
):
    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_loc)
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not t.distributed.is_initialized():
        dist_backend = os.environ.get("IRB_EXTRACT_DIST_BACKEND", "gloo")
        t.distributed.init_process_group(backend=dist_backend, init_method="env://")
        rank = t.distributed.get_rank()
        world_size = t.distributed.get_world_size()
    if not t.cuda.is_available():
        raise RuntimeError("extract_hidden_states.py requires a CUDA-visible GPU.")
    if local_rank >= t.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} but only {t.cuda.device_count()} CUDA device(s) are visible."
        )
    device = t.device(f"cuda:{local_rank}")
    t.cuda.set_device(device)
    is_main_process = rank == 0
    logger.info(
        "Extraction rank setup: rank=%s world_size=%s local_rank=%s device=%s",
        rank,
        world_size,
        local_rank,
        device,
    )

    # load and prep data
    splits = ("train", "val", "test")
    data_dirs = dict()
    for s in splits:
        data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)

    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

    # -------------------------------------------------------------------------
    # Detect representation mechanics (Exp2 wrapper models)
    # -------------------------------------------------------------------------
    rep_meta = _load_representation_meta(model_loc)
    if rep_meta is not None:
        representation = rep_meta["representation"]
        temporal = rep_meta["temporal"]
    else:
        # Infer from directory name (covers xVal models that lack
        # representation_mechanics.pt, and discrete models that don't need it)
        representation, temporal = _infer_representation_from_path(model_loc)

    needs_wrapper = not (representation == "discrete" and temporal == "time_tokens")
    needs_numeric = representation in ("soft", "xval", "xval_affine")
    needs_times = temporal == "time_rope"

    if needs_wrapper:
        logger.info(
            "Wrapper model detected: representation=%s, temporal=%s",
            representation,
            temporal,
        )

    # -------------------------------------------------------------------------
    # Load dataset with extra columns for wrapper models
    # -------------------------------------------------------------------------
    dataset_raw = load_dataset(
        "parquet",
        data_files={s: str(data_dirs[s].joinpath("tokens_timelines.parquet")) for s in splits},
    )

    # Determine available columns
    col_names = dataset_raw["train"].column_names
    use_padded = "padded" in col_names
    has_numeric_values = "numeric_values" in col_names
    has_padded_numeric = "padded_numeric_values" in col_names
    has_times = "times" in col_names
    has_padded_times = "padded_times" in col_names

    if needs_numeric and not (has_numeric_values or has_padded_numeric):
        raise ValueError(
            "Wrapper model needs numeric_values but the tokenized parquet lacks that column. "
            "Re-run Stage 0 with numeric values preserved for this representation."
        )

    if needs_times and not (has_times or has_padded_times):
        raise ValueError(
            "Wrapper model uses time_rope but the tokenized parquet lacks times columns. "
            "Re-run Stage 0 with relative-time inputs preserved for this representation."
        )

    def process_batch(batch):
        """Map batch to include input_ids and optional numeric/time features."""
        if use_padded:
            result = {"input_ids": batch["padded"]}
        else:
            result = {"input_ids": batch["tokens"]}

        if needs_numeric:
            if use_padded and has_padded_numeric:
                result["numeric_values"] = [
                    [float(v) if v is not None else float("nan") for v in seq]
                    for seq in batch["padded_numeric_values"]
                ]
            elif has_numeric_values:
                result["numeric_values"] = [
                    [float(v) if v is not None else float("nan") for v in seq]
                    for seq in batch["numeric_values"]
                ]

        if needs_times:
            if use_padded and has_padded_times:
                result["relative_times"] = [
                    compute_relative_times_hours(seq)
                    for seq in batch["padded_times"]
                ]
            elif has_times:
                result["relative_times"] = [
                    compute_relative_times_hours(seq)
                    for seq in batch["times"]
                ]

        return result

    dataset = dataset_raw.map(process_batch, batched=True)
    if not use_padded:
        # Variable-length sequences — keep as lists for dynamic padding later
        pass
    else:
        dataset = dataset.with_format("torch")

    # -------------------------------------------------------------------------
    # Load and prep model
    # -------------------------------------------------------------------------

    # Load the base HF model. For xVal models saved with the legacy format
    # (trainer.save_model()), the pytorch_model.bin contains keys with
    # 'base_model.' prefix. We handle this by trying from_pretrained()
    # first, then falling back to manual loading from config + weights.
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_loc, torch_dtype=t.float16
        )
        logger.info("Loaded base model via from_pretrained()")
    except (ValueError, RuntimeError) as e:
        # Legacy xVal format: pytorch_model.bin has wrapper-prefixed keys.
        logger.warning(
            "from_pretrained() failed (%s). Trying legacy wrapper format...", e
        )
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_loc)
        from transformers import AutoModelForCausalLM as AutoCausal
        base_model = AutoCausal.from_config(config).to(t.float16)

        # Load weights, stripping 'base_model.' prefix
        weights_path = model_loc / "pytorch_model.bin"
        if not weights_path.exists():
            # Try safetensors
            weights_path = model_loc / "model.safetensors"
        if weights_path.exists():
            state = t.load(weights_path, map_location="cpu", weights_only=True)
            base_state = {}
            for k, v in state.items():
                if k.startswith("base_model."):
                    base_state[k[len("base_model."):]] = v
                elif not k.startswith("number_head.") and not k.endswith("_by_id"):
                    base_state[k] = v
            base_model.load_state_dict(base_state, strict=False)
            logger.info("Loaded base model weights from legacy wrapper format")
        else:
            raise FileNotFoundError(f"No weights file found in {model_loc}")

    d = base_model.config.hidden_size
    h = base_model.config.num_hidden_layers

    if needs_wrapper:
        # Reconstruct wrapper model
        wrapper_kwargs = {}
        if rep_meta is not None:
            wrapper_kwargs["num_bins"] = rep_meta.get("num_bins", 10)
            wrapper_kwargs["time_rope_scaling"] = rep_meta.get("time_rope_scaling", 60.0)

        # Load numeric_stats for xVal if available
        if representation in ("xval", "xval_affine"):
            stats_path = data_dir / f"{data_version}-tokenized" / "train" / "numeric_stats.json"
            if stats_path.exists():
                try:
                    payload = json.loads(stats_path.read_text(encoding="utf-8"))
                    wrapper_kwargs["numeric_stats"] = payload.get("stats", None)
                    logger.info("Loaded numeric_stats.json for xVal: %s", stats_path)
                except Exception as e:
                    logger.warning("Failed to load numeric_stats.json: %s", e)

        model = create_representation_model(
            base_model=base_model,
            vocab=vocab,
            representation=representation,
            temporal=temporal,
            **wrapper_kwargs,
        )

        # Load saved value encoder weights (soft discretization)
        if rep_meta is not None and rep_meta.get("value_encoder_state") is not None:
            from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
            if isinstance(model, RepresentationModelWrapper) and model.value_encoder is not None:
                model.value_encoder.load_state_dict(rep_meta["value_encoder_state"])
                logger.info("Loaded value_encoder weights from representation_mechanics.pt")

        # Load xVal number_head weights
        if representation in ("xval", "xval_affine"):
            from fms_ehrs.framework.xval import XValModelWrapper
            if isinstance(model, XValModelWrapper):
                # Try representation_mechanics.pt first (new format)
                if rep_meta is not None and rep_meta.get("number_head_state") is not None:
                    model.number_head.load_state_dict(rep_meta["number_head_state"])
                    logger.info("Loaded number_head weights from representation_mechanics.pt")
                else:
                    # Fall back to pytorch_model.bin (legacy format)
                    weights_path = model_loc / "pytorch_model.bin"
                    if weights_path.exists():
                        state = t.load(weights_path, map_location="cpu", weights_only=True)
                        head_keys = {
                            k.replace("number_head.", ""): v
                            for k, v in state.items()
                            if k.startswith("number_head.")
                        }
                        if head_keys:
                            model.number_head.load_state_dict(head_keys)
                            logger.info("Loaded number_head from legacy pytorch_model.bin")
                        else:
                            logger.warning(
                                "No number_head weights found. "
                                "Numeric head will use random initialization."
                            )
    else:
        model = base_model

    model = model.to(device)
    model.eval()

    # iterate over splits and run inference using model
    pad_id = vocab("PAD")
    stop_tokens = t.tensor([pad_id, vocab("TRUNC"), vocab("TL_END")]).to(device)

    def _as_padded_long(seqs, *, fill_value: int) -> tuple[t.Tensor, int]:
        max_len = max((len(x) for x in seqs), default=0)
        out = t.full(
            (len(seqs), max_len),
            fill_value=fill_value,
            dtype=t.long,
            device=device,
        )
        for i, seq in enumerate(seqs):
            if len(seq) > 0:
                out[i, : len(seq)] = t.tensor(seq, dtype=t.long, device=device)
        return out, max_len

    def _as_padded_float(seqs, *, max_len: int, fill_value: float) -> t.Tensor:
        out = t.full(
            (len(seqs), max_len),
            fill_value=fill_value,
            dtype=t.float32,
            device=device,
        )
        for i, seq in enumerate(seqs):
            if seq is not None and len(seq) > 0:
                out[i, : len(seq)] = t.tensor(seq, dtype=t.float32, device=device)
        return out

    for s in splits:
        n = dataset[s].num_rows
        feature_shape = (d, h + 1) if all_layers else (d,)
        output_path = _feature_path(
            data_dirs[s],
            all_layers=all_layers,
            model_stem=_feature_model_stem(model_loc),
        )
        shard_dir = _shard_dir_for(output_path)
        _prepare_shard_dir(shard_dir, rank=rank, world_size=world_size)

        local_indices = t.arange(rank, n, world_size, dtype=t.long)
        local_features = np.empty((local_indices.numel(), *feature_shape), dtype=np.float16)
        local_offset = 0
        logger.info(
            "Split %s: rank %s/%s extracting %s of %s row(s).",
            s,
            rank,
            world_size,
            local_indices.numel(),
            n,
        )

        for batch_idx in tqdm(
            t.split(local_indices, batch_sz),
            disable=not is_main_process,
        ):
            # Build input_ids batch
            if use_padded:
                batch_raw = dataset[s]["input_ids"][batch_idx]
                if hasattr(batch_raw, "to"):
                    batch = batch_raw.to(device)
                    max_len = batch.size(1)
                else:
                    batch, max_len = _as_padded_long(batch_raw, fill_value=pad_id)
            else:
                seqs = dataset[s]["input_ids"][batch_idx.tolist()]
                batch, max_len = _as_padded_long(seqs, fill_value=pad_id)

            # Build forward kwargs
            fwd_kwargs = {"input_ids": batch, "output_hidden_states": True}

            # Add numeric_values for soft/xVal models
            if needs_numeric and "numeric_values" in dataset[s].column_names:
                if use_padded:
                    nv_raw = dataset[s]["numeric_values"][batch_idx]
                    nv = (
                        nv_raw.to(device)
                        if hasattr(nv_raw, "to")
                        else _as_padded_float(nv_raw, max_len=max_len, fill_value=float("nan"))
                    )
                else:
                    nv_seqs = dataset[s]["numeric_values"][batch_idx.tolist()]
                    nv = _as_padded_float(
                        nv_seqs,
                        max_len=max_len,
                        fill_value=float("nan"),
                    )
                fwd_kwargs["numeric_values"] = nv

            # Add relative_times for time_rope models
            if needs_times and "relative_times" in dataset[s].column_names:
                if use_padded:
                    rt_raw = dataset[s]["relative_times"][batch_idx]
                    rt = (
                        rt_raw.to(device)
                        if hasattr(rt_raw, "to")
                        else _as_padded_float(rt_raw, max_len=max_len, fill_value=float("nan"))
                    )
                else:
                    rt_seqs = dataset[s]["relative_times"][batch_idx.tolist()]
                    rt = _as_padded_float(
                        rt_seqs,
                        max_len=max_len,
                        fill_value=float("nan"),
                    )
                fwd_kwargs["relative_times"] = rt

            stop_mask = t.isin(batch, stop_tokens)
            has_stop = stop_mask.any(dim=1, keepdim=True)
            first_stop = t.argmax(stop_mask.int(), dim=1, keepdim=True)
            final_nonpadding_idx = t.where(
                has_stop,
                first_stop - 1,
                t.full_like(first_stop, batch.size(1) - 1),
            )
            with t.inference_mode():
                x = model.forward(**fwd_kwargs)
            # Handle both dict-like and ModelOutput returns
            if isinstance(x, dict):
                hidden_states = x.get("hidden_states")
                if hidden_states is None:
                    # Wrapper models may not return hidden_states in dict
                    raise ValueError(
                        "Model did not return hidden_states. "
                        "Ensure output_hidden_states=True is passed."
                    )
            else:
                hidden_states = x.hidden_states

            ret = t.empty(
                size=(
                    (final_nonpadding_idx.size(dim=0), d, h + 1)
                    if all_layers
                    else (final_nonpadding_idx.size(dim=0), d)
                ),
                dtype=hidden_states[-1].dtype,
                device=device,
            )
            x = t.stack(hidden_states, dim=-1) if all_layers else hidden_states[-1]
            for i, j in enumerate(final_nonpadding_idx):
                ret[i] = x[i, j]
            batch_n = int(batch_idx.numel())
            local_features[local_offset : local_offset + batch_n] = ret.detach().to(
                "cpu"
            )
            local_offset += batch_n
            t.cuda.empty_cache()

        if local_offset != local_features.shape[0]:
            raise RuntimeError(
                f"Rank {rank} wrote {local_offset} rows but allocated "
                f"{local_features.shape[0]} rows for split {s}."
            )

        if world_size == 1:
            set_perms(np.save)(output_path, local_features)
            continue

        shard_path = shard_dir / f"rank-{rank:05d}-of-{world_size:05d}.npz"
        set_perms(np.savez)(
            shard_path,
            indices=local_indices.numpy().astype(np.int64, copy=False),
            features=local_features,
        )
        _dist_barrier(world_size)

        if is_main_process:
            logger.info("Merging %s shard(s) for split %s into %s", world_size, s, output_path)
            _merge_feature_shards(
                output_path=output_path,
                shard_dir=shard_dir,
                n_rows=n,
                feature_shape=feature_shape,
                world_size=world_size,
            )
            shutil.rmtree(shard_dir)

        _dist_barrier(world_size)

    if world_size > 1 and t.distributed.is_initialized():
        t.distributed.destroy_process_group()


if __name__ == "__main__":
    fi.Fire(main)
