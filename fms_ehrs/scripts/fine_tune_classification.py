#!/usr/bin/env python3

"""
fine-tune a pretrained model for sequence classification
"""

import argparse
import os
import pathlib
import sys

import datasets as ds
import numpy as np
import scipy as sp
import sklearn.metrics as skl_mets
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.util import rt_padding_to_left
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


parser = argparse.ArgumentParser()
parser.add_argument("--model_loc", type=pathlib.Path, default=None)
parser.add_argument("--data_dir", type=pathlib.Path, default=None)
parser.add_argument("--data_version", type=str, default="day_stays_first_24h")
parser.add_argument("--out_dir", type=pathlib.Path, default=None)
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--jid", type=str, default=os.getenv("SLURM_JOB_ID", ""))
parser.add_argument("--wandb_project", type=str, default="mimic-sft-clsfr")
parser.add_argument("--metric_for_best_model", type=str, default="eval_auc")
parser.add_argument("--greater_is_better", action="store_true")
parser.add_argument(
    "--outcome",
    choices=[
        "same_admission_death",
        "long_length_of_stay",
        "icu_admission",
        "imv_event",
    ],
    default="same_admission_death",
)
parser.add_argument("--unif_rand_trunc", action="store_true")
parser.add_argument("--tune", action="store_true")
parser.add_argument("--training_fraction", type=float, default=1.0)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

model_loc, data_dir, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.model_loc, args.data_dir, args.out_dir),
)

os.environ["WANDB_PROJECT"] = args.wandb_project
os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_loc.stem, j=args.jid)

output_dir = out_dir / "{m}-{j}".format(m=model_loc.stem, j=args.jid)
output_dir.mkdir(exist_ok=True, parents=True)

# load and prep data
splits = ("train", "val")
data_dirs = {s: data_dir / f"{args.data_version}-tokenized" / s for s in splits}
np_rng = np.random.default_rng(42)
vocab = Vocabulary().load(data_dirs["train"] / "vocab.gzip")

dataset = (
    ds.load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s] / "tokens_timelines_outcomes.parquet") for s in splits
        },
        columns=["padded", args.outcome],
    )
    .with_format("torch")
    .map(
        lambda x: {
            "input_ids": rt_padding_to_left(
                x["padded"], vocab("PAD"), unif_rand_trunc=args.unif_rand_trunc
            ),
            "label": x[args.outcome],
        },
        remove_columns=["padded", args.outcome],
    )
)

assert 0 <= args.training_fraction <= 1.0
if args.training_fraction < 1.0 - sys.float_info.epsilon:
    tr = dataset["train"].shuffle(generator=np_rng)
    n_tr = int(len(tr) * args.training_fraction)
    dataset["train"] = dataset["train"].select(range(n_tr))


def model_init(trial=None):
    return AutoModelForSequenceClassification.from_pretrained(model_loc)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
        "gradient_accumulation_steps": trial.suggest_int(
            "gradient_accumulation_steps", 1, 3
        ),
    }


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    probs = sp.special.softmax(logits, axis=1)[:, 1]
    preds = np.argmax(logits, axis=1)
    prec, rec, f1, _ = skl_mets.precision_recall_fscore_support(
        y_true=labels, y_pred=preds, pos_label=1, average="binary"
    )
    auc = skl_mets.roc_auc_score(y_true=labels, y_score=probs)
    return {"prec": prec, "rec": rec, "f1": f1, "auc": auc}


# train model
training_args = TrainingArguments(
    report_to="wandb",
    run_name="{m}-{j}".format(m=model_loc.stem, j=args.jid),
    output_dir=str(output_dir),
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.n_epochs,
    save_total_limit=2,
    metric_for_best_model=args.metric_for_best_model,
    greater_is_better=args.greater_is_better,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="best",
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model_init(),
    model_init=model_init if args.tune else None,
    train_dataset=dataset["train"].shuffle(generator=np_rng),
    eval_dataset=dataset["val"],
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics,
)

if args.tune:
    best_trial = trainer.hyperparameter_search(
        direction="minimize", backend="optuna", hp_space=optuna_hp_space, n_trials=5
    )
    best_ckpt = sorted(
        (output_dir / f"run-{best_trial.run_id}").glob("checkpoint-*")
    ).pop()
    best_mdl_loc = out_dir / "{m}-{j}-hp".format(m=model_loc.stem, j=args.jid)
    AutoModelForSequenceClassification.from_pretrained(best_ckpt).save_pretrained(
        best_mdl_loc
    )

else:
    trainer.train()
    trainer.save_model(
        str(
            best_mdl_loc := output_dir
            / "mdl-{m}-{j}-clsfr-{o}{u}".format(
                m=model_loc.stem,
                j=args.jid,
                o=args.outcome,
                u="-urt" if args.unif_rand_trunc else "",
            )
        )
    )

logger.info("---fin")
