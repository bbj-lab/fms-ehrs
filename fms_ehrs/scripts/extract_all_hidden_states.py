#!/usr/bin/env python3

"""
grab all hidden states (at just under 24h) from each provided sequence;
Cf. extract_hidden_states
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

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

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
    splits: tuple[str] = ("train", "val", "test"),
    out_dir: os.PathLike = None,
    all_layers: bool = False,
    batch_num_start: int = None,
    batch_num_end: int = None,
):
    out_dir = out_dir if out_dir else data_dir

    data_dir, model_loc, out_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_loc, out_dir)
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
    all_splits = ("train", "val", "test")
    data_dirs = dict()
    out_dirs = dict()
    for s in all_splits:
        data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)
        out_dirs[s] = out_dir.joinpath(f"{data_version}-tokenized", s)
        out_dirs[s].mkdir(exist_ok=True, parents=True)

    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

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
    model = AutoModelForCausalLM.from_pretrained(
        model_loc, torch_dtype=t.float16
    )  # in eval mode by default
    d = model.config.hidden_size
    h = model.config.num_hidden_layers
    model = model.to(device)
    if is_parallel:
        model = t.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # iterate over splits and run inference using model
    stop_tokens = t.tensor([vocab("PAD"), vocab("TRUNC"), vocab("TL_END")])

    for s in splits:
        n = dataset[s].num_rows
        tl_len = len(dataset[s].select(range(1))["input_ids"][0])
        for batch_num, big_batch in tqdm(enumerate(t.split(t.arange(n), big_batch_sz))):
            if batch_num_start is not None and batch_num < batch_num_start:
                continue
            if batch_num_end is not None and batch_num >= batch_num_end:
                continue
            features = (
                np.empty((big_batch.size(0), tl_len, d, h + 1), dtype=np.float16)
                if all_layers
                else np.empty((big_batch.size(0), tl_len, d), dtype=np.float16)
            )
            for small_batch in t.split(big_batch, small_batch_sz):
                batch = dataset[s]["input_ids"][small_batch]
                with t.inference_mode(), t.amp.autocast("cuda", dtype=t.float16):
                    x = model.forward(
                        input_ids=batch.to(device), output_hidden_states=True
                    )
                feats = (
                    t.stack(
                        tuple(hs.detach().to("cpu") for hs in x.hidden_states), dim=-1
                    )
                    .numpy()
                    .astype(np.float16)
                    if all_layers
                    else x.hidden_states[-1]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float16)
                )
                first_stop_idx = t.argmax(
                    t.isin(batch, stop_tokens).int(), dim=1, keepdim=True
                )  # or 0 if no stop token
                for i, j in enumerate(first_stop_idx.cpu().numpy().ravel()):
                    if j > 0:
                        feats[i, j:] = np.nan
                features[small_batch - batch_num * big_batch_sz] = feats
                # t.cuda.empty_cache()
            set_perms(np.save)(
                out_dirs[s].joinpath(
                    "all-features{x}-{m}-batch{n}.npy".format(
                        x="-all-layers" if all_layers else "",
                        m=model_loc.stem,
                        n=batch_num,
                    )
                ),
                features,
            )


if __name__ == "__main__":
    fi.Fire(main)
