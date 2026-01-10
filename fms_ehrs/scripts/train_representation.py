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
    representation: typing.Literal["discrete", "soft", "continuous"] = "discrete",
    temporal: typing.Literal["time_tokens", "time2vec"] = "time_tokens",
    num_bins: int = 20,
    time2vec_dim: int = 64,
    # Training parameters
    n_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
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
    if representation in ("soft", "continuous") and temporal == "time_tokens":
        logger.warning(
            f"Using {representation} representation with time_tokens temporal encoding. "
            "This is valid but typically paired with time2vec for Exp2."
        )

    # Determine what additional data to load
    needs_numeric_values = representation in ("soft", "continuous")
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

    # Create base model
    config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=len(dataset.vocab),
        bos_token_id=dataset.vocab("TL_START"),
        eos_token_id=dataset.vocab("TL_END"),
        pad_token_id=dataset.vocab("PAD"),
        **model_kwargs,
    )
    base_model = AutoModelForCausalLM.from_config(config)

    # Wrap with representation mechanics
    model = create_representation_model(
        base_model=base_model,
        vocab=dataset.vocab,
        representation=representation,
        temporal=temporal,
        num_bins=num_bins,
        time2vec_dim=time2vec_dim,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {n_params:,} parameters")
    logger.info(f"Representation: {representation}, Temporal: {temporal}")

    # Create data collator
    data_collator = RepresentationDataCollator(
        pad_token_id=dataset.vocab("PAD"),
        include_numeric_values=needs_numeric_values,
        include_times=needs_times,
    )

    # Training arguments
    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
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
        ddp_find_unused_parameters=False,
        seed=seed,
        data_seed=seed,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["val"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            NanStoppingCallback(),
        ],
    )

    # Train
    logger.info("Starting training...")
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
                "num_bins": num_bins,
                "time2vec_dim": time2vec_dim,
                "value_encoder_state": (
                    model.value_encoder.state_dict() if model.value_encoder is not None else None
                ),
                "time2vec_state": (
                    model.time2vec_layer.state_dict() if model.time2vec_layer is not None else None
                ),
            }
            set_perms(t.save)(rep_state, str(final_model_path / "representation_mechanics.pt"))
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
