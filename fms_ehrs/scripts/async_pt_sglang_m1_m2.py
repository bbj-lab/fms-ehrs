"""
Generate timeline completions with patient-major ordering and interleaved M1/M2.

Key optimizations:
1. Patient-major ordering for radix cache locality
2. Interleaved generation: M1 and M2 trajectories generated together to share prefix cache
3. Two-pass approach: Generate first (no logprobs), then score (prefill-only)
4. Dynamic batching handled by SGLang

Estimators:
- M0: Simple Monte Carlo (binary: did DSCG_expired appear?)
- M1 (SCOPE): Sum of P(DSCG_expired) at each position in trajectory
- M2 (REACH): P(DSCG_expired would have occurred) on counterfactual trajectory

MODIFICATION: Scoring pass now pulls ALL token logprobs (full vocabulary) instead of
only the target token. This is to benchmark the effect on inference speed.
Only DSCG_expired logprobs are actually used for estimator computation.
"""

import argparse
import asyncio
import os
import pathlib
import time
import typing
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import polars as pl
import sglang as sgl
from tqdm import tqdm

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.stats import bootstrap_ci
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=pathlib.Path,
    default="/gpfs/data/bbj-lab/users/burkh4rt/data-mimic"
    if os.uname().nodename.startswith("cri")
    else "/mnt/bbj-lab/users/burkh4rt/data-mimic",
)
parser.add_argument("--data_version", type=str, default="Y21_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument("--tto_version", type=str, default="Y21_first_24h")
parser.add_argument("--max_len", type=int, default=10_000)
parser.add_argument("--n_samp", type=int, default=20)
parser.add_argument("--test_size", type=int, default=1_000)
args, unknowns = parser.parse_known_args()


def extract_token_logprobs(
    output: dict, 
    target_token_id: int, 
    source: str = "output",
    skip: int = 0,
) -> list[float]:
    """
    Extract logprobs for a target token from SGLang output.
    
    Works with both filtered logprobs (when token_ids_logprob is specified)
    and full-vocabulary logprobs (when it is not).
    
    Args:
        output: SGLang generation output dict
        target_token_id: Token ID to extract logprobs for
        source: "output" for output_token_ids_logprobs, "input" for input_token_ids_logprobs
        skip: Number of initial entries to skip
    
    Returns:
        List of logprobs for the target token at each position
    """
    meta = output.get("meta_info", {})
    key = "output_token_ids_logprobs" if source == "output" else "input_token_ids_logprobs"
    token_ids_logprobs = meta.get(key, [])
    
    logprobs = []
    for position_entry in token_ids_logprobs[skip:]:
        if position_entry is None:
            continue
        for logprob, token_id, _ in position_entry:
            if token_id == target_token_id:
                logprobs.append(logprob)
                break
    
    return logprobs


class TrajectoryType(Enum):
    M1 = "m1"  # DSCG_expired allowed (for M0 and M1)
    M2 = "m2"  # DSCG_expired forbidden (for M2)


@dataclass 
class PatientResults:
    """Aggregated results for a single patient."""
    m0_samples: list[bool] = field(default_factory=list)
    m1_samples: list[float] = field(default_factory=list)
    m2_samples: list[float] = field(default_factory=list)


class InterleavedScheduler:
    """
    Scheduler that interleaves M1 and M2 trajectory generation for better cache utilization.
    
    Each trajectory is scored immediately after generation to maximize cache hits
    on the full sequence (prompt + generated tokens).
    """
    
    def __init__(
        self,
        engine: sgl.Engine,
        n_samp: int,
        max_len: int,
        dscg_expired_id: int,
        tl_end_id: int,
        pad_id: int,
        trunc_id: int,
        vocab_size: int = 256,
    ):
        self.engine = engine
        self.n_samp = n_samp
        self.max_len = max_len
        self.dscg_expired_id = dscg_expired_id
        self.tl_end_id = tl_end_id
        self.pad_id = pad_id
        self.trunc_id = trunc_id
        self.vocab_size = vocab_size

    async def _generate_and_score(
        self,
        prompt_tokens: list[int],
        traj_type: TrajectoryType,
    ) -> tuple[bool, float]:
        """
        Generate a trajectory and immediately score it.
        
        This keeps the full sequence (prompt + trajectory) hot in cache
        for the scoring pass.
        
        Returns: (has_dscg, score)
        """
        # === GENERATION PASS ===
        if traj_type == TrajectoryType.M1:
            gen_output = await self.engine.async_generate(
                input_ids=prompt_tokens,
                sampling_params={
                    "max_new_tokens": self.max_len - len(prompt_tokens) - 1,
                    "temperature": 1.0,
                    "stop_token_ids": [self.tl_end_id, self.dscg_expired_id],
                    "logit_bias": {
                        self.pad_id: -10000,
                        self.trunc_id: -10000
                    },
                },
                return_logprob=False,
            )
        else:
            gen_output = await self.engine.async_generate(
                input_ids=prompt_tokens,
                sampling_params={
                    "max_new_tokens": self.max_len - len(prompt_tokens) - 1,
                    "temperature": 1.0,
                    "stop_token_ids": [self.tl_end_id, self.pad_id, self.trunc_id],
                    "logit_bias": {
                        self.dscg_expired_id: -10000,
                        self.pad_id: -10000,
                        self.trunc_id: -10000
                    },
                },
                return_logprob=False,
            )
        
        meta = gen_output.get("meta_info", {})
        output_ids = meta.get("output_ids", gen_output.get("output_ids", []))
        has_dscg = self.dscg_expired_id in output_ids
        
        if not output_ids:
            return (has_dscg, 0.0)
        
        # === SCORING PASS (immediate) ===
        # Determine which tokens to score
        if traj_type == TrajectoryType.M1 and has_dscg:
            stop_idx = output_ids.index(self.dscg_expired_id)
            scoring_ids = list(output_ids[:stop_idx + 1])
        else:
            scoring_ids = list(output_ids)
        
        max_scoring_len = self.max_len - len(prompt_tokens) - 10
        if len(scoring_ids) > max_scoring_len:
            scoring_ids = scoring_ids[:max_scoring_len]
        
        if not scoring_ids:
            return (has_dscg, 0.0)
        
        full_sequence = prompt_tokens + scoring_ids
        prompt_len = len(prompt_tokens)
        
        # CHANGED: Added top_logprobs_num at request level to pull full vocab logprobs.
        # token_ids_logprob kept to guarantee DSCG_expired is always included.
        score_output = await self.engine.async_generate(
            input_ids=full_sequence,
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 1.0,
            },
            return_logprob=True,
            logprob_start_len=prompt_len - 1,
            top_logprobs_num=self.vocab_size,
            token_ids_logprob=[self.dscg_expired_id],
        )
        
        # Extract logprobs — extract_token_logprobs searches through the full
        # vocabulary logprobs at each position to find our target token
        dscg_logprobs = extract_token_logprobs(
            score_output, self.dscg_expired_id, source="input", skip=2
        )
        output_logprobs = extract_token_logprobs(
            score_output, self.dscg_expired_id, source="output"
        )
        if output_logprobs:
            dscg_logprobs.append(output_logprobs[0])
        
        if not dscg_logprobs:
            return (has_dscg, 0.0)
        
        probs = np.clip(np.exp(dscg_logprobs), 0.0, 1.0)
        
        if traj_type == TrajectoryType.M1:
            score = float(np.sum(probs))
        else:
            score = float(1.0 - np.prod(1.0 - probs))
        
        return (has_dscg, score)

    async def run_all(
        self,
        patient_tokens: list[list[int]],
    ) -> list[PatientResults]:
        """
        Run interleaved generation and immediate scoring for all patients.
        
        For each patient, generates M1 and M2 trajectories interleaved,
        scoring each immediately after generation to maximize cache hits.
        """
        num_patients = len(patient_tokens)
        
        results = {i: PatientResults() for i in range(num_patients)}
        results_lock = asyncio.Lock()
        completed_patients = set()
        
        async def process_sample(patient_idx: int, tokens: list[int], sample_idx: int, pbar: tqdm):
            # Generate and score M1
            has_dscg, m1_score = await self._generate_and_score(tokens, TrajectoryType.M1)
            
            # Generate and score M2 (immediately after M1 to share prompt cache)
            _, m2_score = await self._generate_and_score(tokens, TrajectoryType.M2)
            
            async with results_lock:
                results[patient_idx].m0_samples.append(has_dscg)
                results[patient_idx].m1_samples.append(m1_score)
                results[patient_idx].m2_samples.append(m2_score)
                
                if len(results[patient_idx].m0_samples) == self.n_samp:
                    if patient_idx not in completed_patients:
                        completed_patients.add(patient_idx)
                        pbar.update(1)
        
        with tqdm(total=num_patients, desc="Generating & Scoring") as pbar:
            tasks = []
            for patient_idx, tokens in enumerate(patient_tokens):
                for sample_idx in range(self.n_samp):
                    tasks.append(process_sample(patient_idx, tokens, sample_idx, pbar))
            
            await asyncio.gather(*tasks)
        
        return [results[i] for i in range(num_patients)]


async def async_main():
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
    )

    df_test = (
        pl.read_parquet(
            data_dir
            / f"{args.data_version}-tokenized"
            / "test"
            / "tokens_timelines.parquet"
        )
        .sample(n=args.test_size)
        .lazy()
    )
    
    test_token_list = df_test.select("tokens").collect().to_series().to_list()

    vocab = Vocabulary().load(
        data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
    )

    DSCG_EXPIRED_ID = vocab("DSCG_Expired")
    TL_END_ID = vocab("TL_END")
    PAD_ID = vocab("PAD")
    TRUNC_ID = vocab("TRUNC")

    logger.info(f"Loaded {len(test_token_list)} patients, {args.n_samp} samples each")

    engine = sgl.Engine(
        model_path=str(model_loc),
        skip_tokenizer_init=True,
        context_length=args.max_len,
    )

    scheduler = InterleavedScheduler(
        engine=engine,
        n_samp=args.n_samp,
        max_len=args.max_len,
        dscg_expired_id=DSCG_EXPIRED_ID,
        tl_end_id=TL_END_ID,
        pad_id=PAD_ID,
        trunc_id=TRUNC_ID,
        vocab_size=len(vocab),
    )

    # Run interleaved generation and scoring
    total_start = time.time()
    results = await scheduler.run_all(test_token_list)
    total_time = time.time() - total_start
    
    engine.shutdown()

    # Compute estimators
    M0 = np.array([np.mean(r.m0_samples) for r in results])
    M1 = np.array([np.mean(r.m1_samples) for r in results])
    M2 = np.array([np.mean(r.m2_samples) for r in results])

    # Load outcomes
    outcome = (
        df_test.join(
            pl.scan_parquet(
                data_dir
                / f"{args.tto_version}-tokenized"
                / "test"
                / "tokens_timelines_outcomes.parquet"
            ),
            how="left",
            on="hospitalization_id",
            validate="1:1",
        )
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )

    # Report results
    logger.info("=" * 50)
    logger.info(f"RESULTS (n={len(test_token_list)}, samples={args.n_samp})")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info("=" * 50)
    
    for name, estm in {"M0": M0, "M1": M1, "M2": M2}.items():
        logger.info(f"{name}: mean={np.mean(estm):.4f}, max={np.max(estm):.4f}")
        log_classification_metrics(y_true=outcome, y_score=estm, logger=logger)
        logger.info(bootstrap_ci(y_true=outcome, y_score=estm))

    logger.info("---fin")


def main():
    asyncio.run(async_main())


if __name__ == '__main__':
    main()