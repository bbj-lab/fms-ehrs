#!/usr/bin/env python3

"""
fine-tune a pretrained model for sequence classification
"""

import os
import pathlib
import sys
import typing

import datasets as ds
import fire as fi
import numpy as np
import scipy as sp
import sklearn.metrics as skl_mets
import torch as t
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from fms_ehrs.framework.dataset import compute_relative_times_hours
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
from fms_ehrs.framework.model_wrapper import create_representation_model
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.util import rt_padding_to_left
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    model_loc: os.PathLike = None,
    data_dir: os.PathLike = None,
    data_version: str = "day_stays_first_24h",
    out_dir: os.PathLike = None,
    n_epochs: int = 5,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = "mimic-sft-clsfr",
    metric_for_best_model: str = "eval_auc",
    greater_is_better: bool = True,
    outcome: typing.Literal[
        "same_admission_death", "long_length_of_stay", "icu_admission", "imv_event"
    ] = "same_admission_death",
    unif_rand_trunc: bool = False,
    tune: bool = False,
    training_fraction: float = 1.0,
) -> pathlib.PurePath | None:
    model_loc, data_dir, out_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (model_loc, data_dir, out_dir)
    )

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_loc.stem, j=jid)

    output_dir = out_dir.joinpath("{m}-{j}".format(m=model_loc.stem, j=jid))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Detect whether this model checkpoint includes Exp2 representation mechanics.
    # If present, we must (a) wrap the sequence-classification model, and (b) load
    # numeric_values / relative_times from the dataset so the wrapper can apply
    # the representation during downstream training/evaluation.
    rep_mech_path = model_loc.joinpath("representation_mechanics.pt")
    rep_mech = None
    if rep_mech_path.exists():
        rep_mech = t.load(rep_mech_path, map_location="cpu")
        logger.info(f"Found representation mechanics at {rep_mech_path}")
        logger.info(f"  representation={rep_mech.get('representation')}")
        logger.info(f"  temporal={rep_mech.get('temporal')}")

    # load and prep data
    splits = ("train", "val")
    data_dirs = {s: data_dir.joinpath(f"{data_version}-tokenized", s) for s in splits}
    np_rng = np.random.default_rng(42)
    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

    needs_numeric_values = rep_mech is not None and rep_mech.get("representation") in (
        "soft",
        "continuous",
    )
    needs_times = rep_mech is not None and rep_mech.get("temporal") == "time2vec"

    columns = ["padded", outcome]
    if needs_numeric_values:
        columns.append("padded_numeric_values")
    if needs_times:
        columns.append("padded_times")

    raw = ds.load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
            for s in splits
        },
        columns=columns,
    )

    # Build processed dataset with optional numeric_values / relative_times aligned to left padding.
    def _first_pad_index(seq: list[int], pad_id: int) -> int:
        for i, tok in enumerate(seq):
            if tok == pad_id:
                return i
        return 0

    def _process_batch(batch):
        pad_id = int(vocab("PAD"))
        input_ids_out = []
        labels_out = []
        numeric_out = [] if needs_numeric_values else None
        rel_times_out = [] if needs_times else None

        for i in range(len(batch["padded"])):
            padded = batch["padded"][i]
            lbl = batch[outcome][i]
            cut = _first_pad_index(padded, pad_id)

            # Shift padding to left (same semantics as rt_padding_to_left)
            if cut > 0:
                n_left = len(padded) - cut
                input_ids = [pad_id] * n_left + padded[:cut]
            else:
                input_ids = padded

            input_ids_out.append(input_ids)
            labels_out.append(int(lbl))

            if needs_numeric_values:
                nv = batch["padded_numeric_values"][i]
                nv = [float(v) if v is not None else float("nan") for v in nv]
                if cut > 0:
                    n_left = len(nv) - cut
                    nv = [float("nan")] * n_left + nv[:cut]
                numeric_out.append(nv)

            if needs_times:
                ts = batch["padded_times"][i]
                rel = compute_relative_times_hours(ts)
                if cut > 0:
                    n_left = len(rel) - cut
                    rel = [0.0] * n_left + rel[:cut]
                rel_times_out.append(rel)

        result = {
            "input_ids": input_ids_out,
            "label": labels_out,
        }
        if needs_numeric_values:
            result["numeric_values"] = numeric_out
        if needs_times:
            result["relative_times"] = rel_times_out
        return result

    features = {
        "input_ids": ds.Sequence(ds.Value("int64")),
        "label": ds.Value("int64"),
    }
    if needs_numeric_values:
        features["numeric_values"] = ds.Sequence(ds.Value("float32"))
    if needs_times:
        features["relative_times"] = ds.Sequence(ds.Value("float32"))

    dataset = raw.map(
        _process_batch,
        batched=True,
        remove_columns=columns,
        features=ds.Features(features),
    ).with_format("torch")

    assert 0 <= training_fraction <= 1.0
    if training_fraction < 1.0 - sys.float_info.epsilon:
        tr = dataset["train"].shuffle(generator=np_rng)
        n_tr = int(len(tr) * training_fraction)
        dataset["train"] = dataset["train"].select(range(n_tr))

    def model_init(trial=None):
        base = AutoModelForSequenceClassification.from_pretrained(model_loc)
        if rep_mech is None:
            return base
        # Hyperparameter tuning with wrapper models is not currently supported.
        return base

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
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

    class _Collator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = pad_token_id

        def __call__(self, features: list[dict]) -> dict:
            batch = {
                "input_ids": t.stack([f["input_ids"] for f in features]),
                "attention_mask": t.stack(
                    [(f["input_ids"] != self.pad_token_id).long() for f in features]
                ),
                "labels": t.tensor([int(f["label"]) for f in features], dtype=t.long),
            }
            if needs_numeric_values:
                batch["numeric_values"] = t.stack([f["numeric_values"] for f in features])
            if needs_times:
                batch["relative_times"] = t.stack([f["relative_times"] for f in features])
            return batch

    # Build model (optionally wrapped with representation mechanics)
    base_cls = AutoModelForSequenceClassification.from_pretrained(model_loc)
    if rep_mech is not None:
        representation = rep_mech.get("representation", "discrete")
        temporal = rep_mech.get("temporal", "time_tokens")
        num_bins = int(rep_mech.get("num_bins", 20))
        time2vec_dim = int(rep_mech.get("time2vec_dim", 64))

        model = create_representation_model(
            base_model=base_cls,
            vocab=vocab,
            representation=representation,
            temporal=temporal,
            num_bins=num_bins,
            time2vec_dim=time2vec_dim,
        )
        if isinstance(model, RepresentationModelWrapper):
            if rep_mech.get("value_encoder_state") is not None and model.value_encoder is not None:
                model.value_encoder.load_state_dict(rep_mech["value_encoder_state"], strict=True)
            if rep_mech.get("time2vec_state") is not None and model.time2vec_layer is not None:
                model.time2vec_layer.load_state_dict(rep_mech["time2vec_state"], strict=True)
    else:
        model = base_cls

    # train model
    training_args = TrainingArguments(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_loc.stem, j=jid),
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=n_epochs,
        save_total_limit=2,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="best",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        model_init=model_init if tune else None,
        train_dataset=dataset["train"].shuffle(generator=np_rng),
        eval_dataset=dataset["val"],
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
        data_collator=_Collator(pad_token_id=vocab("PAD")),
    )

    if tune:
        best_trial = trainer.hyperparameter_search(
            direction="minimize", backend="optuna", hp_space=optuna_hp_space, n_trials=5
        )

        if os.getenv("RANK", "0") == "0":
            best_ckpt = sorted(
                output_dir.joinpath(f"run-{best_trial.run_id}").glob("checkpoint-*")
            ).pop()
            best_mdl_loc = out_dir.joinpath(
                "{m}-{j}-hp".format(m=model_loc.stem, j=jid)
            )
            set_perms(
                AutoModelForSequenceClassification.from_pretrained(
                    best_ckpt
                ).save_pretrained
            )(best_mdl_loc)

            return best_mdl_loc

    else:
        trainer.train()
        best_mdl_loc = output_dir.joinpath(
            "mdl-{m}-{j}-clsfr-{o}{u}".format(
                m=model_loc.stem,
                j=jid,
                o=outcome,
                u="-urt" if unif_rand_trunc else "",
            )
        )

        if isinstance(model, RepresentationModelWrapper):
            # Save base sequence-classification model in HF format + the mechanics state.
            set_perms(model.base_model.save_pretrained)(str(best_mdl_loc))
            rep_state = {
                "representation": rep_mech.get("representation"),
                "temporal": rep_mech.get("temporal"),
                "num_bins": rep_mech.get("num_bins"),
                "time2vec_dim": rep_mech.get("time2vec_dim"),
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
                str(best_mdl_loc / "representation_mechanics.pt"),
                rep_state,
            )
        else:
            set_perms(trainer.save_model)(str(best_mdl_loc))

        return best_mdl_loc


if __name__ == "__main__":
    fi.Fire(main)
