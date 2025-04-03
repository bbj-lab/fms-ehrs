#!/usr/bin/env python3

"""
grab all hidden states (at just under 24h) from each provided sequence;
Cf. 05_extract_hidden_states.py
"""

import os
import pathlib

import fire as fi
import numpy as np
import torch as t
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from logger import get_logger
from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    data_dir: os.PathLike = None,
    data_version: str = "day_stays_first_24h",
    model_loc: os.PathLike = None,
    small_batch_sz: int = 2**4,
    big_batch_sz: int = 2**12,
    test_only: bool = True,
):

    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, model_loc),
    )

    # prepare parallelism
    is_parallel = t.cuda.device_count() > 1
    if is_parallel:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
    device = t.device(f"cuda:{rank}")
    t.cuda.set_device(device)

    # load and prep data
    splits = ("train", "val", "test")
    data_dirs = dict()
    for s in splits:
        data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)

    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
    splits = ("test",) if test_only else splits

    dataset = (
        load_dataset(
            "parquet",
            data_files={
                s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))
                for s in splits
            },
        )
        .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
        .with_format("torch")
    )

    # load and prep model
    model = AutoModelForCausalLM.from_pretrained(model_loc)  # in eval mode by default
    d = model.config.hidden_size
    model = model.to(device)
    if is_parallel:
        model = t.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # iterate over splits and run inference using model
    stop_tokens = t.tensor([vocab("PAD"), vocab("TRUNC")]).to(device)
    features = dict()

    for s in splits:
        n = dataset[s].num_rows
        tl_len = dataset[s].select(range(1))["input_ids"].shape[-1]
        for batch_num, big_batch in tqdm(enumerate(t.split(t.arange(n), big_batch_sz))):
            features[s] = np.empty((big_batch.size(0), tl_len, d), dtype=np.float16)
            for small_batch in t.split(big_batch, small_batch_sz):
                batch = dataset[s]["input_ids"][small_batch].to(device)
                final_nonpadding_idx = t.argmax(
                    t.isin(batch, stop_tokens).int(), axis=1, keepdim=True
                )
                with t.inference_mode():
                    x = model.forward(input_ids=batch, output_hidden_states=True)
                feats = x.hidden_states[-1].to("cpu").numpy().astype(np.float16)
                for i, j in enumerate(final_nonpadding_idx.cpu().numpy().ravel()):
                    if j != 0:
                        feats[i, j:] = np.nan
                features[s][small_batch - batch_num * big_batch_sz] = feats
            np.save(
                data_dirs[s].joinpath(
                    "all-features-{m}-batch{n}.npy".format(
                        m=model_loc.stem, n=batch_num
                    )
                ),
                features[s],
            )


if __name__ == "__main__":
    fi.Fire(main)
