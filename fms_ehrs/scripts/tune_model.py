#!/usr/bin/env python3

"""
tune a model with a packing strategy
"""

import argparse
import os
import pathlib

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
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


class NanStoppingCallback(TrainerCallback):
    """stop training on encountering a nan objective"""

    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    if state.is_world_process_zero:
                        logger.warning(f"Encountered non-finite metric {k} ({v}).")
                    control.should_training_stop = True


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--data_version", type=str, default="day_stays")
parser.add_argument("--model_version", type=str, default="llama1b")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--lr_min", type=float, default=5e-5)
parser.add_argument("--lr_max", type=float, default=5e-4)
parser.add_argument("--gr_acc_min", type=int, default=1)
parser.add_argument("--gr_acc_max", type=int, default=1)
parser.add_argument("--data_dir", type=pathlib.Path, default=None)
parser.add_argument("--model_dir", type=pathlib.Path, default=None)
parser.add_argument("--jid", type=str, default=os.getenv("SLURM_JOB_ID", ""))
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--n_trials", type=int, default=5)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

os.environ["WANDB_PROJECT"] = args.wandb_project
os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=args.model_version, j=args.jid)

data_dir, model_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_dir)
)

output_dir = model_dir / "{m}-{j}".format(m=args.model_version, j=args.jid)
output_dir.mkdir(exist_ok=True, parents=True)

dataset = Datasets(
    data_version=args.data_version,
    data_dir=data_dir,
    max_seq_length=args.max_seq_length,
)

conf_param = dict(
    vocab_size=len(dataset.vocab),
    bos_token_id=dataset.vocab("TL_START"),
    eos_token_id=dataset.vocab("TL_END"),
    pad_token_id=dataset.vocab("PAD"),
)


def model_init(trial=None):
    t.cuda.empty_cache()
    config = AutoConfig.from_pretrained(args.model_name, **conf_param)
    mdl = AutoModelForCausalLM.from_config(config)
    mdl_params = sum(p.numel() for p in mdl.parameters())
    logger.info("Model initialized, n. param = {}".format(mdl_params))
    return mdl


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True
        ),
        "gradient_accumulation_steps": trial.suggest_int(
            "gradient_accumulation_steps", args.gr_acc_min, args.gr_acc_max
        ),
    }


# train model
training_args = TrainingArguments(
    report_to="wandb",
    run_name="{m}-{j}".format(m=args.model_version, j=args.jid),
    output_dir=str(output_dir),
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # this is handled in our dataset object
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    eval_strategy="steps",
    save_strategy="best",
    ddp_find_unused_parameters=False,
    optim="paged_adamw_8bit",
    bf16=True,
)


def collate_fn(batch):
    input_ids = t.tensor([x["input_ids"] for x in batch])
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


trainer = Trainer(
    model=model_init(),
    model_init=model_init,
    data_collator=collate_fn,
    train_dataset=(tr_ds := dataset.get_train_dataset(n_epochs=args.n_epochs)),
    eval_dataset=dataset.get_val_dataset(),
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), NanStoppingCallback()],
)

logger.info(f"Training on {len(tr_ds)} sequences of length {args.max_seq_length}")

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=args.n_trials,
)

if os.getenv("RANK", "0") == "0":
    best_ckpt = sorted(
        (output_dir / f"run-{best_trial.run_id}").glob("checkpoint-*")
    ).pop()
    best_mdl_loc = model_dir / "{m}-{j}-hp-{d}".format(
        m=args.model_version, j=args.jid, d=args.data_version
    )
    set_perms(AutoModelForCausalLM.from_pretrained(best_ckpt).save_pretrained)(
        best_mdl_loc
    )

logger.info("---fin")
