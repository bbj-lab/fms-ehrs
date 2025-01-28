import datetime
import os
import pathlib

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_PROJECT"] = "clif_mamba"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_RUN_NAME"] = "all"

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from vocabulary import Vocabulary

# locate data and vocab
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()
data_dirs = dict()
data_dirs["train"] = hm.joinpath("clif-training-set-tokenized")
data_dirs["val"] = hm.joinpath("clif-validation-set-tokenized")
data_dirs["test"] = hm.joinpath("clif-test-set-tokenized")
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

# grab a small mamba for training
model_name = "state-spaces/mamba-130m-hf"
config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(vocab),
    bos_token_id=vocab("TL_START"),
    eos_token_id=[vocab("TL_END"), vocab("TRUNC")],
    pad_token_id=vocab("PAD"),
)
model = AutoModelForCausalLM.from_config(config)

# load data
dataset = (
    load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))
            for s in ("train", "val")
        },
    )
    .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
    .shuffle(seed=42)
)

# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name=os.environ["WANDB_RUN_NAME"],
    max_seq_length=1024,
    output_dir="./tmp",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    save_total_limit=2,
    load_best_model_at_end=True,
    neftune_noise_alpha=5,
    eval_strategy="epochs",
    save_strategy="epochs",
)
trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    args=training_args,
)
trainer.train()
trainer.save_model(
    "mdl-{}".format(
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .astimezone()
        .isoformat()
    )
)
