#!/usr/bin/env python3

"""
Unified training script for Experiment 2 representation mechanics.

This script supports all 6 Exp2 configurations:
- representation ∈ {discrete, soft, continuous}
- temporal ∈ {time_tokens, time2vec}

The script uses padded collation (one hospitalization per row) to preserve
per-admission temporal structure needed for:
- Time2Vec: Requires relative time in hours since admission
- Soft/Continuous: Requires numeric_values aligned to token positions

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

    # Continuous encoder + Time2Vec
    python train_representation.py \\
        --data_dir /path/to/data \\
        --model_dir /path/to/models \\
        --representation continuous \\
        --temporal time2vec

Note:
    Soft and continuous representations require unfused tokenization
    (fused_category_values=false in the tokenizer config).
"""

import json
import os
import pathlib
import typing

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.model_wrapper import create_representation_model
from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


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
    - numeric_values: For soft/continuous encoding
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
    # Representation parameters
    representation: typing.Literal["discrete", "soft", "continuous", "xval"] = "discrete",
    temporal: typing.Literal["time_tokens", "time2vec"] = "time_tokens",
    num_bins: int = 20,
    time2vec_dim: int = 64,
    continuous_num_scales: int = 1,
    # xVal (canonical) knobs
    numeric_loss_weight: float = 1.0,
    numeric_loss_weight_choices: typing.Sequence[float] = (0.1, 1.0, 10.0),
    # Training parameters
    n_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    # HPO (reference-consistent: tune lr + grad accumulation; optional method knobs)
    do_hpo: bool = False,
    n_trials: int = 3,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    gr_acc_min: int = 1,
    gr_acc_max: int = 3,
    # HPO (method-specific essential knobs; matched-budget)
    tune_representation_hparams: bool = False,
    num_bins_choices: typing.Sequence[int] | None = None,
    time2vec_dim_choices: typing.Sequence[int] | None = None,
    continuous_num_scales_choices: typing.Sequence[int] | None = None,
    max_seq_length: int = 1024,
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
    representation : {"discrete", "soft", "continuous"}
        Value representation method
    temporal : {"time_tokens", "time2vec"}
        Temporal encoding method
    num_bins : int
        Number of quantile bins for soft discretization
    time2vec_dim : int
        Internal dimension for Time2Vec
    continuous_num_scales : int
        Number of xVal multiscale embeddings (continuous representation)
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
    max_seq_length : int
        Maximum sequence length
    tune_representation_hparams : bool
        If True, allow Optuna to tune method-specific essential knobs.
    num_bins_choices : Sequence[int], optional
        Candidate num_bins values for soft discretization.
    time2vec_dim_choices : Sequence[int], optional
        Candidate Time2Vec dimensions.
    continuous_num_scales_choices : Sequence[int], optional
        Candidate xVal multiscale counts for continuous encoding.
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
    if representation in ("soft", "continuous", "xval") and temporal == "time_tokens":
        logger.warning(
            f"Using {representation} representation with time_tokens temporal encoding. "
            "This is valid but typically paired with time2vec for Exp2."
        )

    # Default pre-registered grids if tuning is enabled but choices are omitted.
    # These defaults preserve the "no extra knobs" setting unless explicitly set.
    if tune_representation_hparams:
        if representation == "soft" and not num_bins_choices:
            num_bins_choices = [num_bins]
        if temporal == "time2vec" and not time2vec_dim_choices:
            time2vec_dim_choices = [time2vec_dim]
        if representation == "continuous" and not continuous_num_scales_choices:
            continuous_num_scales_choices = [continuous_num_scales]

    def _select_numeric_loss_weight(trial=None, trial_params: dict | None = None) -> float:
        # Matched-budget tuning:
        # - Always keep n_trials identical across conditions.
        # - For xVal only, tune numeric_loss_weight over a tiny fixed grid.
        if representation != "xval":
            return float(numeric_loss_weight)
        choices = [float(x) for x in numeric_loss_weight_choices]
        if trial is not None:
            return float(trial.suggest_categorical("numeric_loss_weight", choices))
        if trial_params and "numeric_loss_weight" in trial_params:
            return float(trial_params["numeric_loss_weight"])
        return float(numeric_loss_weight)

    # Determine what additional data to load
    needs_numeric_values = representation in ("soft", "continuous", "xval")
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

    # Load dataset with extended features if needed
    # Use padded collation for Exp2 (preserves per-admission structure)
    dataset = Datasets(
        data_version=data_version,
        data_dir=data_dir,
        collation="padded",  # Required for representation mode
        max_seq_length=max_seq_length,
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
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

    def _select_representation_knobs(trial=None, trial_params: dict | None = None):
        selected_num_bins = num_bins
        selected_time2vec_dim = time2vec_dim
        selected_num_scales = continuous_num_scales

        if tune_representation_hparams:
            if trial is not None:
                if representation == "soft" and num_bins_choices:
                    selected_num_bins = trial.suggest_categorical(
                        "num_bins", list(num_bins_choices)
                    )
                if temporal == "time2vec" and time2vec_dim_choices:
                    selected_time2vec_dim = trial.suggest_categorical(
                        "time2vec_dim", list(time2vec_dim_choices)
                    )
                if representation == "continuous" and continuous_num_scales_choices:
                    selected_num_scales = trial.suggest_categorical(
                        "continuous_num_scales", list(continuous_num_scales_choices)
                    )
            elif trial_params:
                if representation == "soft" and "num_bins" in trial_params:
                    selected_num_bins = trial_params["num_bins"]
                if temporal == "time2vec" and "time2vec_dim" in trial_params:
                    selected_time2vec_dim = trial_params["time2vec_dim"]
                if representation == "continuous" and "continuous_num_scales" in trial_params:
                    selected_num_scales = trial_params["continuous_num_scales"]

        return selected_num_bins, selected_time2vec_dim, selected_num_scales

    def _log_param_deltas(
        model,
        *,
        trial=None,
        selected_num_bins: int,
        selected_time2vec_dim: int,
        selected_num_scales: int,
        selected_numeric_loss_weight: float,
    ):
        trial_tag = f"trial={trial.number}" if trial is not None else "trial=base"
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
                "[%s] params: base=%s value=%s time=%s total=%s | knobs: num_bins=%s time2vec_dim=%s num_scales=%s numeric_loss_weight=%s",
                trial_tag,
                f"{base_params:,}",
                f"{value_params:,}",
                f"{time_params:,}",
                f"{total_params:,}",
                selected_num_bins,
                selected_time2vec_dim,
                selected_num_scales,
                selected_numeric_loss_weight,
            )
        else:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "[%s] params: base=%s | knobs: num_bins=%s time2vec_dim=%s num_scales=%s numeric_loss_weight=%s",
                trial_tag,
                f"{total_params:,}",
                selected_num_bins,
                selected_time2vec_dim,
                selected_num_scales,
                selected_numeric_loss_weight,
            )

    def _build_model(
        selected_num_bins: int,
        selected_time2vec_dim: int,
        selected_num_scales: int,
        selected_numeric_loss_weight: float,
    ):
        # Rebuild model each time (required for Trainer.hyperparameter_search)
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=len(dataset.vocab),
            bos_token_id=dataset.vocab("TL_START"),
            eos_token_id=dataset.vocab("TL_END"),
            pad_token_id=dataset.vocab("PAD"),
            **model_kwargs,
        )
        base_model = AutoModelForCausalLM.from_config(config)
        return create_representation_model(
            base_model=base_model,
            vocab=dataset.vocab,
            representation=representation,
            temporal=temporal,
            num_bins=selected_num_bins,
            time2vec_dim=selected_time2vec_dim,
            continuous_num_scales=selected_num_scales,
            continuous_numeric_stats=numeric_stats if representation in ("continuous", "xval") else None,
            numeric_loss_weight=selected_numeric_loss_weight,
        )

    def model_init(trial=None):
        selected_num_bins, selected_time2vec_dim, selected_num_scales = _select_representation_knobs(
            trial=trial
        )
        selected_weight = _select_numeric_loss_weight(trial=trial)
        model = _build_model(selected_num_bins, selected_time2vec_dim, selected_num_scales, selected_weight)
        _log_param_deltas(
            model,
            trial=trial,
            selected_num_bins=selected_num_bins,
            selected_time2vec_dim=selected_time2vec_dim,
            selected_num_scales=selected_num_scales,
            selected_numeric_loss_weight=selected_weight,
        )
        return model

    # Build initial model (non-HPO path) and use model_init for HPO path
    model = model_init()
    final_num_bins = num_bins
    final_time2vec_dim = time2vec_dim
    final_num_scales = continuous_num_scales

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {n_params:,} parameters")
    logger.info(f"Representation: {representation}, Temporal: {temporal}")
    logger.info(
        "Representation knobs: num_bins=%s time2vec_dim=%s continuous_num_scales=%s",
        num_bins,
        time2vec_dim,
        continuous_num_scales,
    )

    # Create data collator
    data_collator = RepresentationDataCollator(
        pad_token_id=dataset.vocab("PAD"),
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
    )

    # Training arguments
    # IMPORTANT (DDP correctness):
    # For soft/continuous representations, the value encoder is only exercised on
    # batches that contain at least one numeric quantile token with a non-NaN value.
    # Some admissions contain no numeric events, so some steps legitimately do not
    # use the value-encoder parameters. In DDP, this requires find_unused_parameters,
    # otherwise PyTorch raises:
    #   RuntimeError: Expected to have finished reduction ... parameters not used ...
    ddp_find_unused = representation in ("soft", "continuous", "xval")

    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
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
    trainer = Trainer(
        model=model,
        model_init=model_init if do_hpo else None,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["val"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            NanStoppingCallback(),
        ],
    )

    logger.info("Starting training...")
    if do_hpo:
        logger.info(
            "HPO budget: n_trials=%s | hp_space: lr in [%s,%s] (log), grad_acc in [%s,%s]%s",
            n_trials,
            lr_min,
            lr_max,
            gr_acc_min,
            gr_acc_max,
            f", numeric_loss_weight in {list(map(float, numeric_loss_weight_choices))}" if representation == "xval" else "",
        )

        def optuna_hp_space(trial):
            hp = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", lr_min, lr_max, log=True
                ),
                "gradient_accumulation_steps": trial.suggest_int(
                    "gradient_accumulation_steps", gr_acc_min, gr_acc_max
                ),
            }
            # Matched-budget tuning: include numeric_loss_weight only for xVal.
            if representation == "xval":
                hp["numeric_loss_weight"] = trial.suggest_categorical(
                    "numeric_loss_weight", [float(x) for x in numeric_loss_weight_choices]
                )
            return hp

        best_trial = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=n_trials,
        )
        logger.info(f"Best trial: {best_trial}")
        # IMPORTANT (reference-consistent):
        # Do NOT retrain after HPO. Instead, load the best checkpoint from the best run
        # and then proceed to save it in our standardized output format.
        run_dir = pathlib.Path(training_args.output_dir) / f"run-{best_trial.run_id}"
        state_path = run_dir / "trainer_state.json"
        best_ckpt_dir: pathlib.Path | None = None
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                best_ckpt = state.get("best_model_checkpoint")
                if best_ckpt:
                    best_ckpt_dir = pathlib.Path(best_ckpt)
            except Exception as e:
                logger.warning(f"Failed to parse {state_path}: {e}")
        if best_ckpt_dir is None:
            ckpts = sorted(run_dir.glob("checkpoint-*"))
            if not ckpts:
                raise RuntimeError(f"No checkpoints found under {run_dir} for best trial run_id={best_trial.run_id}")
            best_ckpt_dir = ckpts[-1]

        weights_path = best_ckpt_dir / "pytorch_model.bin"
        if not weights_path.exists():
            raise RuntimeError(f"Expected checkpoint weights not found: {weights_path}")
        logger.info(f"Loading best checkpoint weights from: {best_ckpt_dir}")
        state_dict = t.load(weights_path, map_location="cpu")
        selected_num_bins, selected_time2vec_dim, selected_num_scales = _select_representation_knobs(
            trial_params=best_trial.params
        )
        selected_weight = _select_numeric_loss_weight(trial_params=best_trial.params)
        model = _build_model(selected_num_bins, selected_time2vec_dim, selected_num_scales, selected_weight)
        _log_param_deltas(
            model,
            trial=None,
            selected_num_bins=selected_num_bins,
            selected_time2vec_dim=selected_time2vec_dim,
            selected_num_scales=selected_num_scales,
            selected_numeric_loss_weight=selected_weight,
        )
        model.load_state_dict(state_dict)
        final_num_bins = selected_num_bins
        final_time2vec_dim = selected_time2vec_dim
        final_num_scales = selected_num_scales
    else:
        trainer.train()

    # Save best model
    if os.getenv("RANK", "0") == "0":
        final_model_path = output_dir / f"model-{representation}-{temporal}"
        final_model_path.mkdir(exist_ok=True, parents=True)

        # IMPORTANT: for wrapper models (soft/continuous/time2vec), we must save:
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
                "continuous_num_scales": final_num_scales,
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
