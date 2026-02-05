#!/usr/bin/env python3

"""
Unified training script for Experiment 2 representation mechanics.

This script supports all 6 Exp2 configurations:
- representation ∈ {discrete, soft, xval}
- temporal ∈ {time_tokens, time2vec}

The script uses padded collation (one hospitalization per row) to preserve
per-admission temporal structure needed for:
- Time2Vec: Requires relative time in hours since admission
- Soft/xVal: Requires numeric_values aligned to token positions

Usage:
    # Discrete + time tokens (baseline)
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation discrete \\
        --temporal time_tokens

    # Soft discretization + Time2Vec
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation soft \\
        --temporal time2vec

    # xVal + Time2Vec
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation xval \\
        --temporal time2vec

Note:
    Soft and xVal representations require unfused tokenization
    (fused_category_values=false in the tokenizer config).
"""

import json
import os
import pathlib
import typing
import importlib.util

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.model_wrapper import create_representation_model
from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
from fms_ehrs.framework.optim import AdamWConfig, MuonConfig, build_muon_with_aux_adamw
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


def _parse_bool(x: typing.Any, *, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return bool(default)


def _normalize_attn_impl(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return s


class NanStoppingCallback(TrainerCallback):
    """Stop training on encountering a NaN objective."""

    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    if state.is_world_process_zero:
                        logger.warning(f"Encountered non-finite metric {k} ({v}).")
                    control.should_training_stop = True


class RepresentationDataCollator:
    """Data collator that handles numeric_values and relative_times.

    For Exp2, we need to pass additional tensors beyond input_ids:
    - numeric_values: For soft discretization and xVal
    - relative_times: For Time2Vec temporal encoding
    """

    def __init__(
        self,
        pad_token_id: int,
        include_numeric_values: bool = False,
        include_times: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.include_numeric_values = include_numeric_values
        self.include_times = include_times

    def __call__(self, features: list[dict]) -> dict:
        """Collate batch of features."""
        batch = {
            "input_ids": t.stack([f["input_ids"] for f in features]),
            "attention_mask": t.stack(
                [
                    (f["input_ids"] != self.pad_token_id).long()
                    for f in features
                ]
            ),
            "labels": t.stack([f["input_ids"] for f in features]),
        }

        if self.include_numeric_values and "numeric_values" in features[0]:
            batch["numeric_values"] = t.stack(
                [f["numeric_values"] for f in features]
            )

        if self.include_times and "relative_times" in features[0]:
            batch["relative_times"] = t.stack(
                [f["relative_times"] for f in features]
            )

        return batch


class IRBTrainer(Trainer):
    """Trainer with explicit optimizer selection.

    Rationale: Muon is not one of the built-in HF Trainer optimizer strings, and
    for transformer models it is typically used on 2D hidden-layer matrices while
    keeping embeddings/heads/1D params on AdamW. We implement this explicitly to
    keep Experiment 2/3 training reproducible and free of implicit HPO behavior.
    """

    def __init__(
        self,
        *args,
        optimizer_name: typing.Literal["adamw", "muon"] = "adamw",
        muon_cfg: MuonConfig | None = None,
        aux_adamw_cfg: AdamWConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizer_name = optimizer_name
        self._muon_cfg = muon_cfg
        self._aux_adamw_cfg = aux_adamw_cfg

    def create_optimizer(self):
        if self.optimizer is not None:
            return

        if self._optimizer_name != "muon":
            return super().create_optimizer()

        if self._muon_cfg is None or self._aux_adamw_cfg is None:
            raise ValueError("Muon optimizer requested but optimizer configs were not provided.")

        self.optimizer = build_muon_with_aux_adamw(
            model=self.model,
            muon=self._muon_cfg,
            adamw=self._aux_adamw_cfg,
        )


@logger.log_calls
def main(
    *,
    # Data parameters
    data_dir: os.PathLike = None,
    data_version: str = "day_stays",
    # Model parameters
    model_dir: os.PathLike = None,
    model_name: str = "meta-llama/Llama-3.2-1B",
    model_version: str = "llama1b",
    # Optional performance knobs (objective-preserving):
    use_bf16: bool = _parse_bool(os.getenv("IRB_USE_BF16", "true"), default=True),
    attn_implementation: str | None = _normalize_attn_impl(os.getenv("IRB_ATTN_IMPL", "sdpa")),
    # Representation parameters
    representation: typing.Literal["discrete", "soft", "xval"] = "discrete",
    temporal: typing.Literal["time_tokens", "time2vec"] = "time_tokens",
    num_bins: int = 20,
    time2vec_dim: int = int(os.getenv("IRB_TIME2VEC_DIM", "128")),
    # xVal canonical knob
    numeric_loss_weight: float = float(os.getenv("IRB_XVAL_NUMERIC_LOSS_WEIGHT", "1.0")),
    clip_sigma: float = float(os.getenv("IRB_XVAL_CLIP_SIGMA", "5.0")),
    # Training parameters
    n_epochs: int = int(os.getenv("IRB_EXP23_STAGE1_EPOCHS", os.getenv("IRB_STAGE1_EPOCHS", "1"))),
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    # NOTE (benchmark fairness): fixed to remove token-exposure confounding across arms.
    gradient_accumulation_steps: int = 2,
    # Base LR (used for non-Muon runs, and as a backward-compat default for Muon+aux).
    learning_rate: float = float(os.getenv("IRB_STAGE1_LR", "1e-4")),
    # If optimizer="muon", we allow explicit LR splitting:
    # - Muon LR for 2D hidden-layer matrices
    # - Aux AdamW LR for embeddings/heads/1D params
    #
    # Defaults preserve prior behavior: both fall back to `learning_rate`.
    muon_learning_rate: float | None = None,
    aux_adamw_learning_rate: float | None = None,
    weight_decay: float = float(os.getenv("IRB_STAGE1_WEIGHT_DECAY", "0.01")),
    optimizer: typing.Literal["adamw", "muon"] = typing.cast(
        typing.Literal["adamw", "muon"],
        os.getenv("IRB_STAGE1_OPTIMIZER", "muon"),
    ),
    # Muon knobs (defaults follow torch.optim.Muon defaults where applicable).
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    # Aux AdamW knobs (used when optimizer="muon"; also matches our Exp1 defaults).
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1.0e-8,
    max_seq_length: int = int(os.getenv("IRB_MAX_SEQ_LENGTH", "4096")),
    # Full-timeline padded-mode training via windowing (Exp2/Exp3):
    # When enabled, we train on *all* tokens in a hospitalization by slicing the
    # variable-length `tokens` timeline into overlapping windows of length max_seq_length.
    windowed_padded: bool = False,
    window_stride: int | None = None,
    max_windows_per_admission: int | None = None,
    add_cont_token: bool = True,
    # Experiment tracking
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = "mimic-representation",
    seed: int = 42,
    **model_kwargs,
):
    """Train a model with specified representation mechanics.

    Parameters
    ----------
    data_dir : PathLike
        Root directory containing tokenized data
    data_version : str
        Data version name (e.g., "day_stays")
    model_dir : PathLike
        Directory to save trained models
    model_name : str
        HuggingFace model name for config
    model_version : str
        Version tag for saved model
    representation : {"discrete", "soft", "xval"}
        Value representation method
    temporal : {"time_tokens", "time2vec"}
        Temporal encoding method
    num_bins : int
        Number of quantile bins for soft discretization
    time2vec_dim : int
        Internal dimension for Time2Vec
    n_epochs : int
        Number of training epochs
    per_device_train_batch_size : int
        Training batch size per device
    per_device_eval_batch_size : int
        Evaluation batch size per device
    gradient_accumulation_steps : int
        Gradient accumulation steps
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay (decoupled; AdamW-style).
    max_seq_length : int
        Maximum sequence length
    jid : str
        SLURM job ID for logging
    wandb_project : str
        Weights & Biases project name
    seed : int
        Random seed
    **model_kwargs
        Additional model configuration parameters
    """
    # Set random seeds
    t.manual_seed(seed)
    np.random.seed(seed)

    # Validate configuration
    if int(gradient_accumulation_steps) != 2:
        raise ValueError(
            "For benchmark fairness, gradient_accumulation_steps is fixed to 2 "
            f"(got {gradient_accumulation_steps})."
        )
    attn_implementation = _normalize_attn_impl(attn_implementation)
    if attn_implementation == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "attn_implementation=flash_attention_2 requires the optional `flash-attn` package "
                "(and a compatible GPU/CUDA build)."
            )
    if representation in ("soft", "xval") and temporal == "time_tokens":
        logger.warning(
            f"Using {representation} representation with time_tokens temporal encoding. "
            "This is valid but typically paired with time2vec for Exp2."
        )

    # Determine what additional data to load
    needs_numeric_values = representation in ("soft", "xval")
    needs_times = temporal == "time2vec"

    if needs_numeric_values or needs_times:
        logger.info(
            f"Loading extended data: numeric_values={needs_numeric_values}, "
            f"times={needs_times}"
        )

    # Setup paths
    data_dir, model_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_dir)
    )

    # Setup experiment tracking
    run_name = f"{model_version}-{representation}-{temporal}-{jid}"
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = run_name

    output_dir = model_dir / run_name
    output_dir.mkdir(exist_ok=True, parents=True)

    # Windowed padded guardrail for extreme-length admissions (compute stability).
    #
    # If max_windows_per_admission is not explicitly provided, allow an env override.
    # Convention: <=0 disables the cap.
    resolved_max_windows_per_admission: int | None = max_windows_per_admission
    if resolved_max_windows_per_admission is None:
        env = os.getenv("IRB_MAX_WINDOWS_PER_ADMISSION")
        if env is not None:
            s = str(env).strip().lower()
            if s not in ("", "none", "null"):
                try:
                    v = int(s)
                except ValueError:
                    raise ValueError(
                        f"Invalid IRB_MAX_WINDOWS_PER_ADMISSION={env!r}; expected int or unset."
                    )
                resolved_max_windows_per_admission = v if v > 0 else None

    # Load dataset with extended features if needed
    # Use padded collation for Exp2 (preserves per-admission structure)
    dataset = Datasets(
        data_version=data_version,
        data_dir=data_dir,
        collation="padded",  # Required for representation mode
        max_seq_length=max_seq_length,
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
        windowed_padded=windowed_padded,
        window_stride=window_stride,
        add_cont_token=add_cont_token,
        max_windows_per_admission=resolved_max_windows_per_admission,
    )

    logger.info(f"Loaded {dataset.n_train} train, {dataset.n_val} val samples")
    logger.info(f"Vocabulary size: {len(dataset.vocab)}")

    # Load quantizer/anchoring-independent numeric stats (if available).
    #
    # These are produced by tokenization as:
    #   <data_dir>/<data_version>-tokenized/train/numeric_stats.json
    #
    # If present, we use them to define (\mu_c, \sigma_c) for the continuous encoder,
    # decoupling continuous scaling from discretization choices (e.g., 5-10-5 anchoring).
    numeric_stats: dict[str, dict[str, float]] | None = None
    stats_path = data_dir / f"{data_version}-tokenized" / "train" / "numeric_stats.json"
    if stats_path.exists():
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            numeric_stats = payload.get("stats", None)
            logger.info("Loaded numeric_stats.json for continuous scaling: %s", str(stats_path))
        except Exception as e:
            logger.warning(
                "Failed to load numeric_stats.json (falling back to vocab-aux-derived stats): %s",
                e,
            )

    def _log_param_deltas(
        model,
        *,
        selected_num_bins: int,
        selected_time2vec_dim: int,
        selected_numeric_loss_weight: float,
    ):
        if isinstance(model, RepresentationModelWrapper):
            base_params = sum(p.numel() for p in model.base_model.parameters())
            value_params = (
                sum(p.numel() for p in model.value_encoder.parameters())
                if model.value_encoder is not None
                else 0
            )
            time_params = (
                sum(p.numel() for p in model.time2vec_layer.parameters())
                if model.time2vec_layer is not None
                else 0
            )
            total_params = base_params + value_params + time_params
            logger.info(
                "params: base=%s value=%s time=%s total=%s | knobs: num_bins=%s time2vec_dim=%s numeric_loss_weight=%s",
                f"{base_params:,}",
                f"{value_params:,}",
                f"{time_params:,}",
                f"{total_params:,}",
                selected_num_bins,
                selected_time2vec_dim,
                selected_numeric_loss_weight,
            )
        else:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "params: base=%s | knobs: num_bins=%s time2vec_dim=%s numeric_loss_weight=%s",
                f"{total_params:,}",
                selected_num_bins,
                selected_time2vec_dim,
                selected_numeric_loss_weight,
            )

    def _build_model(
        selected_num_bins: int,
        selected_time2vec_dim: int,
        selected_numeric_loss_weight: float,
    ):
        # Build model once (no HPO in Exp2/Exp3).
        cfg_kwargs = dict(model_kwargs)
        if attn_implementation is not None:
            cfg_kwargs["attn_implementation"] = attn_implementation
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=len(dataset.vocab),
            bos_token_id=dataset.vocab("TL_START"),
            eos_token_id=dataset.vocab("TL_END"),
            pad_token_id=dataset.vocab("PAD"),
            **cfg_kwargs,
        )
        base_model = AutoModelForCausalLM.from_config(config)
        return create_representation_model(
            base_model=base_model,
            vocab=dataset.vocab,
            representation=representation,
            temporal=temporal,
            num_bins=selected_num_bins,
            time2vec_dim=selected_time2vec_dim,
            numeric_stats=numeric_stats if representation == "xval" else None,
            clip_sigma=float(clip_sigma),
            numeric_loss_weight=selected_numeric_loss_weight,
        )

    final_num_bins = num_bins
    final_time2vec_dim = time2vec_dim
    final_numeric_loss_weight = float(numeric_loss_weight)

    model = _build_model(final_num_bins, final_time2vec_dim, final_numeric_loss_weight)
    _log_param_deltas(
        model,
        selected_num_bins=final_num_bins,
        selected_time2vec_dim=final_time2vec_dim,
        selected_numeric_loss_weight=final_numeric_loss_weight,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {n_params:,} parameters")
    logger.info(f"Representation: {representation}, Temporal: {temporal}")
    logger.info(
        "Representation knobs: num_bins=%s time2vec_dim=%s",
        num_bins,
        time2vec_dim,
    )

    # Create data collator
    data_collator = RepresentationDataCollator(
        pad_token_id=dataset.vocab("PAD"),
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
    )

    # Training arguments
    # IMPORTANT (DDP correctness):
    # For soft/continuous/xval representations, the value encoder is only exercised on
    # batches that contain at least one numeric quantile token with a non-NaN value.
    # Some admissions contain no numeric events, so some steps legitimately do not
    # use the value-encoder parameters. In DDP, this requires find_unused_parameters,
    # otherwise PyTorch raises:
    #   RuntimeError: Expected to have finished reduction ... parameters not used ...
    ddp_find_unused = representation in ("soft", "xval")
    use_bf16 = bool(use_bf16) and t.cuda.is_available()

    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        tf32=True,
        max_grad_norm=1.0,
        num_train_epochs=n_epochs,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        # HF Trainer defaults to safetensors. With tied embeddings (common for causal LMs),
        # safetensors errors because multiple state_dict entries share the same storage:
        #   RuntimeError: Some tensors share memory ... {'model.embed_tokens.weight', 'lm_head.weight'}
        # Use standard torch serialization for checkpoints instead.
        save_safetensors=False,
        ddp_find_unused_parameters=ddp_find_unused,
        seed=seed,
        data_seed=seed,
    )

    # Create trainer
    resolved_muon_lr = float(
        muon_learning_rate
        if muon_learning_rate is not None
        else os.getenv("IRB_MUON_LR", str(learning_rate))
    )
    resolved_aux_lr = float(
        aux_adamw_learning_rate
        if aux_adamw_learning_rate is not None
        else os.getenv("IRB_AUX_ADAMW_LR", str(learning_rate))
    )

    muon_cfg = MuonConfig(
        lr=float(resolved_muon_lr),
        weight_decay=float(weight_decay),
        momentum=float(muon_momentum),
        nesterov=bool(muon_nesterov),
        ns_steps=int(muon_ns_steps),
    )
    aux_cfg = AdamWConfig(
        lr=float(resolved_aux_lr),
        weight_decay=float(weight_decay),
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_epsilon),
    )
    trainer = IRBTrainer(
        model=model,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["val"],
        args=training_args,
        data_collator=data_collator,
        optimizer_name=optimizer,
        muon_cfg=muon_cfg if optimizer == "muon" else None,
        aux_adamw_cfg=aux_cfg if optimizer == "muon" else None,
        callbacks=[
            NanStoppingCallback(),
        ],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save best model
    if os.getenv("RANK", "0") == "0":
        final_model_path = output_dir / f"model-{representation}-{temporal}"
        final_model_path.mkdir(exist_ok=True, parents=True)

        # IMPORTANT: for wrapper models (soft/time2vec), we must save:
        # 1) the underlying HF model in standard `save_pretrained` format (config + weights)
        # 2) the representation-mechanics parameters (value encoder / time2vec) separately
        #
        # This allows downstream scripts (e.g., sequence classification) to reload the same
        # representation mechanics and apply them using numeric_values / relative_times.
        if isinstance(model, RepresentationModelWrapper):
            # Save the wrapped HF model (config + weights)
            set_perms(model.base_model.save_pretrained)(str(final_model_path))

            # Save representation-mechanics parameters
            rep_state = {
                "representation": representation,
                "temporal": temporal,
                "num_bins": final_num_bins,
                "time2vec_dim": final_time2vec_dim,
                "value_encoder_state": (
                    model.value_encoder.state_dict() if model.value_encoder is not None else None
                ),
                "time2vec_state": (
                    model.time2vec_layer.state_dict() if model.time2vec_layer is not None else None
                ),
            }
            # NOTE: `set_perms` expects a saver with signature saver(file, *args),
            # but torch.save is torch.save(obj, file). Wrap to avoid arg order bugs.
            set_perms(lambda f, obj: t.save(obj, f))(
                str(final_model_path / "representation_mechanics.pt"),
                rep_state,
            )
        else:
            # Discrete + time_tokens returns a standard HF model; Trainer can save normally.
            set_perms(trainer.save_model)(str(final_model_path))

        logger.info(f"Saved model to {final_model_path}")

        # Also save vocabulary
        dataset.vocab.save(final_model_path / "vocab.gzip")

        return final_model_path

    return None


if __name__ == "__main__":
    fi.Fire(main)
