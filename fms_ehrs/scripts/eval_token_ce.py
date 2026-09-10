#!/usr/bin/env python3

"""
Token-level cross-entropy (CE) evaluation for apples-to-apples loss comparison.

Why this exists
---------------
Pretraining losses across Exp2 configs are NOT directly comparable because:
  1. Vocabulary sizes differ (discrete models have quantile tokens; xVal does not)
  2. Soft discretization uses KL-divergence against soft targets (mixed loss)
  3. xVal models report CE + MSE jointly (number head loss)

This script provides a **standard hard-target CE** on the held-out test set,
applied uniformly to the LM head logits across ALL representations. This strips
out representation-specific loss terms and gives a fair comparison of how well
the model predicts the *next token* (hard argmax ground truth).

Additionally, it reports CE **stratified by token type**:
  - **overall**: all non-padding positions
  - **numeric**: positions where the target token is a numeric/quantile token
  - **non_numeric**: positions where the target token is a clinical code, time, etc.

This decomposition reveals whether representations differ in their ability to
predict numeric values vs clinical events.

Output
------
Prints a JSON summary per split with:
  - overall_ce, numeric_ce, non_numeric_ce (mean CE per position)
  - overall_n, numeric_n, non_numeric_n (counts)
  - per_position_ce (optional, saved as .npz if --save_per_position)

Usage
-----
python fms_ehrs/scripts/eval_token_ce.py \\
    --data_dir /path/to/data \\
    --data_version deciles_none_unfused_time_tokens_first_24h \\
    --model_loc /path/to/model-discrete-time_tokens \\
    --splits test \\
    --batch_sz 16
"""

import json
import os
import pathlib
import sys

import fire as fi
import numpy as np
import torch as t
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.artifacts import model_artifact_stem
from fms_ehrs.framework.dataset import compute_relative_times_seconds
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.model_wrapper import create_representation_model
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


# ---------------------------------------------------------------------------
# Model loading helpers (shared with extract_hidden_states.py)
# ---------------------------------------------------------------------------


def _load_representation_meta(model_loc: pathlib.Path) -> dict | None:
    """Load representation_mechanics.pt from model directory if present."""
    rep_path = model_loc / "representation_mechanics.pt"
    if rep_path.exists():
        meta = t.load(rep_path, map_location="cpu", weights_only=False)
        logger.info(
            "Loaded representation mechanics: representation=%s temporal=%s",
            meta.get("representation"),
            meta.get("temporal"),
        )
        return meta
    return None


def _infer_representation_from_path(model_loc: pathlib.Path) -> tuple[str, str]:
    """Infer representation and temporal from the model directory name."""
    stem = model_loc.name
    parts = stem.split("-", 1)
    if len(parts) < 2:
        return "discrete", "time_tokens"
    remainder = parts[1]
    # IMPORTANT: order matters (e.g., "xval_affine" must match before "xval").
    for rep in ("xval_affine", "soft", "xval", "discrete"):
        if remainder.startswith(rep):
            rest = remainder[len(rep):]
            temporal = rest[1:] if rest.startswith("-") else "time_tokens"
            return rep, temporal
    return "discrete", "time_tokens"


# ---------------------------------------------------------------------------
# Numeric token detection
# ---------------------------------------------------------------------------


def _build_numeric_token_mask(vocab: Vocabulary) -> set[int]:
    """Return the set of token IDs that represent numeric/quantile values.

    Numeric tokens include:
    - Quantile tokens (Q0, Q1, ..., Q9, or similar patterns)
    - Any token whose string representation is a bare number
    - Special numeric marker tokens (NUM, NUMERIC, etc.)

    This allows stratifying CE loss by numeric vs non-numeric positions.
    """
    numeric_ids = set()
    # vocab.lookup maps word (str) -> token_id (int)
    for token_str, token_id in vocab.lookup.items():
        if token_str is None or token_id is None or token_id < 0:
            continue
        # Convert to string for pattern matching
        token_s = str(token_str)
        # Quantile tokens: Q0, Q1, ..., Q9 (or q0, q1, etc.)
        if token_s.startswith("Q") and len(token_s) >= 2 and token_s[1:].isdigit():
            numeric_ids.add(token_id)
        # Bare numeric strings
        elif _is_numeric_token(token_s):
            numeric_ids.add(token_id)
    return numeric_ids


def _is_numeric_token(s: str) -> bool:
    """Check if a token string represents a numeric value."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    *,
    data_dir: os.PathLike = "../../data-mimic",
    data_version: str = "QC_day_stays_first_24h",
    model_loc: os.PathLike = "../../mdls-archive/llama1b-57928921-run1",
    batch_sz: int = 16,
    splits: str = "test",
    save_per_position: bool = False,
):
    """Evaluate token-level CE stratified by numeric vs non-numeric positions.

    Parameters
    ----------
    data_dir : path
        Root data directory containing tokenized and MEDS data.
    data_version : str
        Data version string (should end with _first_24h).
    model_loc : path
        Path to the trained model directory.
    batch_sz : int
        Batch size for inference (default 16).
    splits : str
        Comma-separated split names to evaluate (default: test).
    save_per_position : bool
        If True, save per-sequence CE arrays as .npz files.
    """
    data_dir = pathlib.Path(data_dir).expanduser().resolve()
    # Artifact identity follows the requested per-run path, not its symlink target.
    model_loc_given = pathlib.Path(model_loc).expanduser().absolute()
    model_stem = model_artifact_stem(model_loc_given)
    model_loc = model_loc_given.resolve()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    if device.type == "cuda":
        t.cuda.set_device(0)

    split_list = [s.strip() for s in splits.split(",") if s.strip()]

    # Prepare data directories
    data_dirs = {s: data_dir / f"{data_version}-tokenized" / s for s in ("train",) + tuple(split_list)}
    # Always need train for vocab
    vocab = Vocabulary().load(data_dirs["train"] / "vocab.gzip")

    # Build the set of numeric token IDs for stratification
    numeric_token_ids = _build_numeric_token_mask(vocab)
    logger.info("Detected %d numeric token IDs in vocabulary", len(numeric_token_ids))

    # Special tokens
    pad_id = vocab("PAD")
    stop_tokens_set = {pad_id, vocab("TRUNC"), vocab("TL_END")}

    # -------------------------------------------------------------------------
    # Detect representation mechanics
    # -------------------------------------------------------------------------
    rep_meta = _load_representation_meta(model_loc)
    if rep_meta is not None:
        representation = rep_meta["representation"]
        temporal = rep_meta["temporal"]
    else:
        representation, temporal = _infer_representation_from_path(model_loc)

    needs_wrapper = not (representation == "discrete" and temporal == "time_tokens")
    needs_numeric = representation in ("soft", "xval", "xval_affine")
    needs_times = temporal == "time_rope"

    logger.info(
        "Model config: representation=%s, temporal=%s, wrapper=%s",
        representation, temporal, needs_wrapper,
    )

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------
    data_files = {
        s: str(data_dirs[s] / "tokens_timelines.parquet")
        for s in split_list
        if data_dirs[s].exists()
    }
    if not data_files:
        logger.error("No valid split directories found!")
        return 1

    dataset_raw = load_dataset("parquet", data_files=data_files)

    col_names = dataset_raw[split_list[0]].column_names
    use_padded = "padded" in col_names
    has_numeric_values = "numeric_values" in col_names
    has_padded_numeric = "padded_numeric_values" in col_names
    has_times = "times" in col_names
    has_padded_times = "padded_times" in col_names

    if needs_numeric and not (has_numeric_values or has_padded_numeric):
        logger.warning("No numeric_values column. Falling back to bare model.")
        needs_numeric = False
        if representation in ("soft", "xval", "xval_affine") and not needs_times:
            needs_wrapper = False

    if needs_times and not (has_times or has_padded_times):
        logger.warning("No times column. Falling back to sequential positions.")
        needs_times = False
        if not needs_numeric:
            needs_wrapper = False

    def process_batch(batch):
        result = {"input_ids": batch["padded"] if use_padded else batch["tokens"]}
        if needs_numeric:
            raw_key = "padded_numeric_values" if (use_padded and has_padded_numeric) else "numeric_values"
            if raw_key in batch:
                result["numeric_values"] = [
                    [float(v) if v is not None else float("nan") for v in seq]
                    for seq in batch[raw_key]
                ]
        if needs_times:
            raw_key = "padded_times" if (use_padded and has_padded_times) else "times"
            if raw_key in batch:
                result["relative_times_seconds"] = [
                    compute_relative_times_seconds(seq) for seq in batch[raw_key]
                ]
        return result

    dataset = dataset_raw.map(process_batch, batched=True)
    if use_padded:
        dataset = dataset.with_format("torch")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_loc, torch_dtype=t.float16
        )
        logger.info("Loaded base model via from_pretrained()")
    except (ValueError, RuntimeError) as e:
        logger.warning("from_pretrained() failed (%s). Trying legacy format...", e)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_loc)
        base_model = AutoModelForCausalLM.from_config(config).to(t.float16)
        weights_path = model_loc / "pytorch_model.bin"
        if not weights_path.exists():
            weights_path = model_loc / "model.safetensors"
        if weights_path.exists():
            state = t.load(weights_path, map_location="cpu", weights_only=True)
            base_state = {}
            for k, v in state.items():
                if k.startswith("base_model."):
                    base_state[k[len("base_model."):]] = v
                elif (
                    not k.startswith("number_head.")
                    and k != "num_bias"
                    and not k.endswith("_by_id")
                ):
                    base_state[k] = v
            base_model.load_state_dict(base_state, strict=False)

    if needs_wrapper:
        wrapper_kwargs = {}
        if rep_meta is not None:
            wrapper_kwargs["num_bins"] = rep_meta.get("num_bins", 10)
            wrapper_kwargs["seconds_per_position"] = rep_meta.get(
                "seconds_per_position",
                3600.0 / float(rep_meta.get("time_rope_scaling", 60.0)),
            )
        if representation in ("xval", "xval_affine"):
            stats_path = data_dir / f"{data_version}-tokenized" / "train" / "numeric_stats.json"
            if stats_path.exists():
                try:
                    payload = json.loads(stats_path.read_text(encoding="utf-8"))
                    wrapper_kwargs["numeric_stats"] = payload.get("stats", None)
                except Exception as e:
                    logger.warning("Failed to load numeric_stats.json: %s", e)

        model = create_representation_model(
            base_model=base_model,
            vocab=vocab,
            representation=representation,
            temporal=temporal,
            **wrapper_kwargs,
        )

        # Load value encoder weights
        if rep_meta is not None and rep_meta.get("value_encoder_state") is not None:
            from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
            if isinstance(model, RepresentationModelWrapper) and model.value_encoder is not None:
                model.value_encoder.load_state_dict(rep_meta["value_encoder_state"])
                logger.info("Loaded value_encoder weights")

        # Load xVal number_head weights, and trained affine bias when present.
        if representation in ("xval", "xval_affine"):
            from fms_ehrs.framework.xval import XValModelWrapper, apply_xval_mechanics
            if isinstance(model, XValModelWrapper):
                if representation == "xval_affine":
                    if rep_meta is None:
                        raise RuntimeError(
                            "xval_affine evaluation requires representation_mechanics.pt "
                            "with trained num_bias."
                        )
                    apply_xval_mechanics(model, rep_meta)
                    logger.info("Loaded xVal-affine mechanics from representation_mechanics.pt")
                elif rep_meta is not None and rep_meta.get("number_head_state") is not None:
                    apply_xval_mechanics(model, rep_meta)
                    logger.info("Loaded xVal mechanics from representation_mechanics.pt")
                else:
                    wp = model_loc / "pytorch_model.bin"
                    if wp.exists():
                        state = t.load(wp, map_location="cpu", weights_only=True)
                        head_keys = {
                            k.replace("number_head.", ""): v
                            for k, v in state.items()
                            if k.startswith("number_head.")
                        }
                        if head_keys:
                            model.number_head.load_state_dict(head_keys)
    else:
        model = base_model

    model = model.to(device)
    model.eval()

    vocab_size = base_model.config.vocab_size

    # Build numeric token mask tensor for fast indexing
    numeric_mask_np = np.zeros(vocab_size, dtype=bool)
    for tid in numeric_token_ids:
        if tid < vocab_size:
            numeric_mask_np[tid] = True
    numeric_mask_tensor = t.from_numpy(numeric_mask_np).to(device)

    # -------------------------------------------------------------------------
    # Compute token-level CE per split
    # -------------------------------------------------------------------------
    results = {}

    for s in split_list:
        if s not in dataset:
            logger.warning("Split %s not in dataset, skipping", s)
            continue

        n = dataset[s].num_rows
        logger.info("Evaluating split=%s (%d sequences)", s, n)

        # Accumulators
        total_ce = 0.0
        total_n = 0
        numeric_ce = 0.0
        numeric_n = 0
        non_numeric_ce = 0.0
        non_numeric_n = 0

        per_seq_ce = [] if save_per_position else None

        for batch_idx in tqdm(t.split(t.arange(n), batch_sz), desc=f"CE {s}"):
            # Build input_ids
            if use_padded:
                input_ids = dataset[s]["input_ids"][batch_idx].to(device)
            else:
                seqs = dataset[s]["input_ids"][batch_idx.tolist()]
                max_len = max((len(x) for x in seqs), default=0)
                input_ids = t.full(
                    (len(seqs), max_len), fill_value=pad_id,
                    dtype=t.long, device=device,
                )
                for i, seq in enumerate(seqs):
                    if seq:
                        input_ids[i, :len(seq)] = t.tensor(seq, dtype=t.long, device=device)

            # Build forward kwargs (no labels — we compute CE manually)
            fwd_kwargs = {"input_ids": input_ids}

            if needs_numeric and "numeric_values" in dataset[s].column_names:
                if use_padded:
                    nv = dataset[s]["numeric_values"][batch_idx].to(device)
                else:
                    nv_seqs = dataset[s]["numeric_values"][batch_idx.tolist()]
                    nv = t.full(
                        (len(nv_seqs), max_len), fill_value=float("nan"),
                        dtype=t.float32, device=device,
                    )
                    for i, seq in enumerate(nv_seqs):
                        if seq is not None and len(seq) > 0:
                            nv[i, :len(seq)] = t.tensor(seq, dtype=t.float32, device=device)
                fwd_kwargs["numeric_values"] = nv

            if needs_times and "relative_times_seconds" in dataset[s].column_names:
                if use_padded:
                    rt = dataset[s]["relative_times_seconds"][batch_idx].to(device)
                else:
                    rt_seqs = dataset[s]["relative_times_seconds"][batch_idx.tolist()]
                    rt = t.full(
                        (len(rt_seqs), max_len), fill_value=float("nan"),
                        dtype=t.float32, device=device,
                    )
                    for i, seq in enumerate(rt_seqs):
                        if seq is not None and len(seq) > 0:
                            rt[i, :len(seq)] = t.tensor(seq, dtype=t.float32, device=device)
                fwd_kwargs["relative_times_seconds"] = rt

            with t.inference_mode():
                outputs = model.forward(**fwd_kwargs)

            # Get logits
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs.logits

            # Compute standard hard-target CE per position
            # logits: (B, T, V), targets are input_ids shifted by 1
            # logits[:, :-1, :] predicts input_ids[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()

            # Per-position CE (unreduced)
            ce_per_pos = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction="none",
                ignore_index=pad_id,
            ).view(shift_targets.shape)  # (B, T-1)

            # Build masks for valid positions (non-padding)
            valid_mask = shift_targets != pad_id  # (B, T-1)

            # Also exclude positions where target is a stop token
            for st in stop_tokens_set:
                valid_mask = valid_mask & (shift_targets != st)

            # Numeric mask for target positions
            target_is_numeric = numeric_mask_tensor[shift_targets.clamp(0, vocab_size - 1)]  # (B, T-1)

            numeric_pos_mask = valid_mask & target_is_numeric
            non_numeric_pos_mask = valid_mask & ~target_is_numeric

            # Accumulate
            valid_ce_sum = (ce_per_pos * valid_mask.float()).sum().item()
            valid_count = valid_mask.sum().item()
            total_ce += valid_ce_sum
            total_n += valid_count

            num_ce = (ce_per_pos * numeric_pos_mask.float()).sum().item()
            num_count = numeric_pos_mask.sum().item()
            numeric_ce += num_ce
            numeric_n += num_count

            nnum_ce = (ce_per_pos * non_numeric_pos_mask.float()).sum().item()
            nnum_count = non_numeric_pos_mask.sum().item()
            non_numeric_ce += nnum_ce
            non_numeric_n += nnum_count

            if save_per_position:
                # Save per-sequence mean CE
                for i in range(ce_per_pos.size(0)):
                    mask_i = valid_mask[i]
                    if mask_i.any():
                        per_seq_ce.append(ce_per_pos[i][mask_i].mean().item())
                    else:
                        per_seq_ce.append(float("nan"))

            t.cuda.empty_cache()

        # Aggregate results
        split_results = {
            "overall_ce": total_ce / max(total_n, 1),
            "overall_n": total_n,
            "numeric_ce": numeric_ce / max(numeric_n, 1),
            "numeric_n": numeric_n,
            "non_numeric_ce": non_numeric_ce / max(non_numeric_n, 1),
            "non_numeric_n": non_numeric_n,
            "numeric_frac": numeric_n / max(total_n, 1),
        }
        results[s] = split_results

        logger.info("Split=%s results:", s)
        logger.info(
            "  overall: CE=%.4f (n=%d)",
            split_results["overall_ce"], split_results["overall_n"],
        )
        logger.info(
            "  numeric: CE=%.4f (n=%d, %.1f%% of tokens)",
            split_results["numeric_ce"],
            split_results["numeric_n"],
            100 * split_results["numeric_frac"],
        )
        logger.info(
            "  non_numeric: CE=%.4f (n=%d)",
            split_results["non_numeric_ce"], split_results["non_numeric_n"],
        )

        # Save per-sequence CE if requested
        if save_per_position and per_seq_ce:
            out_path = data_dirs[s] / f"token_ce-{model_stem}.npz"
            set_perms(np.savez)(
                out_path,
                per_sequence_ce=np.array(per_seq_ce, dtype=np.float32),
            )
            logger.info("Saved per-sequence CE to %s", out_path)

    # Print final JSON summary (useful for programmatic parsing)
    summary = {
        "model": str(model_loc_given),
        "model_target": str(model_loc),
        "model_stem": model_stem,
        "representation": representation,
        "temporal": temporal,
        "data_version": data_version,
        "results": results,
    }
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL CE SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    # Also save as JSON. Several runs can publish the same checkpoint directory,
    # so the filename carries the run stem.
    out_json = model_loc / f"token_ce_results-{model_stem}.json"
    try:
        out_json.write_text(json.dumps(summary, indent=2))
        logger.info("Saved results to %s", out_json)
    except PermissionError:
        # Model dir may be read-only; try the data dir instead
        for s in split_list:
            alt_path = data_dirs[s] / f"token_ce_results-{model_stem}.json"
            alt_path.write_text(json.dumps(summary, indent=2))
            logger.info("Saved results to %s (fallback)", alt_path)
            break

    return 0


if __name__ == "__main__":
    fi.Fire(main)
