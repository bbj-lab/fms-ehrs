#!/usr/bin/env python3

"""
Model wrapper for Experiment 2 representation mechanics.

This module wraps a pretrained causal LM (e.g., LLaMA) to support different
value representation methods (discrete, soft, xval) and temporal
encoding strategies (time_tokens, time_rope).

The wrapper intercepts the embedding layer and modifies embeddings based on:
- Soft discretization: Replace quantile-token embeddings with convex combinations, and train quantile-token positions with a soft target
- xVal: Handled by a separate wrapper (XValModelWrapper) that operates on [NUM] tokenization and adds a numeric head loss.
  We support both standard multiplicative xVal ("xval") and an affine-shifted
  variant ("xval_affine") that adds a learned bias after scaling.
- Time-Aware RoPE: Use relative timestamps as position IDs for rotary embeddings

Architecture:
    input_ids ──> Token Embedding ──> [Value Encoder] ──> Transformer
                       │                    │                  │
                       v                    v                  v
                  (batch, seq, d)     (modify numeric    (position_ids from
                                       positions)         relative_times)

References
----------
- Soft discretization: ConSE (Norouzi et al., 2014)
- Continuous encoding: xVal (Golkar et al., 2023) with EHR-specific adaptation
- Time-Aware RoPE: position_ids derived from relative timestamps
"""

import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from fms_ehrs.framework.soft_discretization import SoftDiscretizationEncoder
from fms_ehrs.framework.temporal import admission_relative_position_ids
from fms_ehrs.framework.vocabulary import Vocabulary
from fms_ehrs.framework.xval import XValModelWrapper


class RepresentationModelWrapper(nn.Module):
    """Wraps a pretrained causal LM with Exp2 representation mechanics.

    This wrapper modifies the forward pass to:
    1. Apply soft discretization or continuous encoding to numeric token positions
    2. Apply Time-Aware RoPE temporal encoding when enabled

    Parameters
    ----------
    base_model : PreTrainedModel
        The base transformer model (e.g., LLaMA)
    vocab : Vocabulary
        Tokenizer vocabulary with quantile auxiliary data
    representation : {"discrete", "soft", "xval", "xval_affine"}
        Value representation method:
        - discrete: Standard token embeddings (baseline)
        - soft: Convex combinations of adjacent bin embeddings
        - xval: standard xVal wrapper ([NUM] tokenization + multiplicative scaling + numeric head loss)
        - xval_affine: xVal with affine numeric injection (z*e + b)
    temporal : {"time_tokens", "time_rope", "event_order"}
        Temporal encoding method:
        - time_tokens: Use existing time spacing tokens (baseline)
        - time_rope: Use relative timestamps as continuous position IDs for RoPE
        - event_order: No spacing tokens and no admission-relative position IDs;
          the model falls back to its default sequential positions, so ordinary
          RoPE rotates over token index rather than elapsed clinical time.
    num_bins : int
        Number of quantile bins (for soft discretization)
    seconds_per_position : float
        Admission-relative seconds represented by one RoPE position
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        vocab: Vocabulary,
        representation: typing.Literal["discrete", "soft"] = "discrete",
        temporal: typing.Literal[
            "time_tokens", "time_rope", "event_order"
        ] = "time_tokens",
        num_bins: int = 20,
        seconds_per_position: float = 60.0,
        **kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        self.vocab = vocab
        self.representation = representation
        self.temporal = temporal
        self.seconds_per_position = float(seconds_per_position)

        # Get model hidden size
        self.hidden_size = base_model.config.hidden_size

        # Build fast lookup for quantile tokens (Q0, Q1, ..., Qn-1)
        self.q_token_ids: set[int] = set()
        self._build_quantile_token_lookup()

        # Fast boolean lookup for "is this token a Q token?"
        is_q = torch.zeros(len(self.vocab), dtype=torch.bool)
        for tid in self.q_token_ids:
            if 0 <= tid < is_q.numel():
                is_q[tid] = True
        self.register_buffer("is_q_token", is_q, persistent=False)

        # For soft-target training on quantile tokens, we need a stable mapping
        # bin index k -> token id of "Q{k}".
        self.q_token_id_by_bin: torch.Tensor | None = None
        if representation == "soft":
            qids: list[int] = []
            for k in range(int(num_bins)):
                tok = f"Q{k}"
                if tok not in self.vocab.lookup:
                    raise ValueError(
                        f"Soft discretization requires quantile token {tok} in vocab."
                    )
                qids.append(int(self.vocab.lookup[tok]))
            self.q_token_id_by_bin = torch.tensor(qids, dtype=torch.long)
            self.register_buffer(
                "q_token_id_by_bin_buf",
                self.q_token_id_by_bin,
                persistent=False,
            )

        # Initialize value encoder if needed
        self.value_encoder: nn.Module | None = None
        if representation == "soft":
            self.value_encoder = SoftDiscretizationEncoder(
                num_bins=num_bins, embed_dim=self.hidden_size
            )
            self.value_encoder.set_boundaries_from_vocab_aux(
                vocab.aux,
                token_id_lookup=vocab.lookup,
                vocab_size=len(vocab),
            )
            self._seed_bin_embeddings_from_quantile_tokens()

    def _seed_bin_embeddings_from_quantile_tokens(self) -> None:
        """Copy the model's own Q0..Qn-1 rows into the soft bin embeddings.

        `nn.Embedding` initializes from N(0, 1), but the backbones initialize
        token embeddings at `initializer_range` (0.02). Left untouched, the soft
        table starts roughly 50x wider than every other token embedding in the
        same sequence. Seeding from the quantile rows fixes that scale and also
        starts each bin's input representation in agreement with the output
        representation the LM head scores against.
        """
        if self.q_token_id_by_bin is None or self.value_encoder is None:
            return
        embeddings = self.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is None:
            return
        target = self.value_encoder.bin_embeddings.weight
        bins = self.q_token_id_by_bin.to(device=weight.device)
        if int(bins.max()) >= weight.shape[0]:
            return
        with torch.no_grad():
            target.copy_(
                weight[bins].to(dtype=target.dtype, device=target.device)
            )


    def _build_quantile_token_lookup(self) -> None:
        """Build mapping from token IDs to code identifiers for numeric tokens.

        In unfused tokenization, numeric events emit (code_token, quantile_token).
        We need to identify which tokens are quantile tokens (Q0-Q19 for ventiles)
        and map code tokens to their string identifiers.
        """
        # Find all quantile tokens (Q0, Q1, ..., Qn-1)
        for word, token_id in self.vocab.lookup.items():
            if word is not None and isinstance(word, str):
                if word.startswith("Q") and word[1:].isdigit():
                    self.q_token_ids.add(token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        numeric_values: torch.Tensor | None = None,
        relative_times_seconds: torch.Tensor | None = None,
        relative_times: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass with representation mechanics.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len)
        attention_mask : torch.Tensor, optional
            Attention mask of shape (batch_size, seq_len)
        numeric_values : torch.Tensor, optional
            Raw numeric values aligned to tokens, shape (batch_size, seq_len).
            NaN indicates non-numeric positions.
        relative_times_seconds : torch.Tensor, optional
            Relative seconds since admission, shape (batch_size, seq_len).
            Required when temporal="time_rope".
        relative_times : torch.Tensor, optional
            Legacy hour-based relative times. New callers must use
            ``relative_times_seconds``.
        labels : torch.Tensor, optional
            Labels for language modeling loss

        Returns
        -------
        Output from the base model with modified embeddings
        """
        # Get base token embeddings
        # Access the embedding layer (works for LLaMA, GPT-2, etc.)
        if hasattr(self.base_model, "model") and hasattr(
            self.base_model.model, "embed_tokens"
        ):
            # LLaMA-style
            embeddings = self.base_model.model.embed_tokens(input_ids)
        elif hasattr(self.base_model, "transformer") and hasattr(
            self.base_model.transformer, "wte"
        ):
            # GPT-2 style
            embeddings = self.base_model.transformer.wte(input_ids)
        else:
            raise ValueError(
                f"Cannot find embedding layer for model type {type(self.base_model)}"
            )

        # Apply value encoding modifications if using soft (xval is handled by XValModelWrapper).
        if self.representation == "soft" and numeric_values is not None:
            embeddings = self._apply_value_encoding(
                embeddings, input_ids, numeric_values
            )


        # Handle Time-Aware RoPE (pass position_ids to base model).
        if self.temporal == "time_rope":
            if relative_times_seconds is None and relative_times is not None:
                relative_times_seconds = relative_times * 3600.0
            if relative_times_seconds is None:
                raise ValueError(
                    "time_rope requires admission-relative times in seconds."
                )
            kwargs["position_ids"] = admission_relative_position_ids(
                relative_times_seconds,
                seconds_per_position=self.seconds_per_position,
            )

        # Ensure dtype matches the base model parameters.
        #
        # Rationale: Some value encoders operate in float32 by default.
        # When token embeddings are bf16/fp16, adding float32 temporal embeddings will
        # upcast `embeddings` to float32. Passing float32 `inputs_embeds` into a bf16
        # transformer triggers a hard dtype mismatch in torch.nn.Linear:
        #   RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != BFloat16
        #
        # We standardize by casting to the base model's parameter dtype at the boundary.
        base_dtype = getattr(self.base_model, "dtype", None)
        if base_dtype is None:
            base_dtype = next(self.base_model.parameters()).dtype
        if embeddings.dtype != base_dtype:
            embeddings = embeddings.to(dtype=base_dtype)

        # Soft discretization uses a soft target at quantile-token positions.
        # We therefore compute the LM loss explicitly when labels are provided.
        if self.representation == "soft" and labels is not None and numeric_values is not None:
            model_kwargs = dict(kwargs)
            model_kwargs.pop("labels", None)
            # HF Trainer passes num_items_in_batch when the model accepts loss
            # kwargs (our forward has **kwargs). The contract is: return
            # sum(loss) / num_items_in_batch so gradient accumulation and
            # cross-device token averaging stay exact. The base model is called
            # with labels=None here, so we must honor it in our custom loss.
            num_items_in_batch = model_kwargs.pop("num_items_in_batch", None)
            model_kwargs.setdefault("return_dict", True)
            outputs = self.base_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=None,
                **model_kwargs,
            )
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = self._soft_discretization_lm_loss(
                logits=logits,
                input_ids=input_ids,
                labels=labels,
                numeric_values=numeric_values,
                num_items_in_batch=num_items_in_batch,
            )
            out = {"loss": loss, "logits": logits}
            if isinstance(outputs, dict):
                # Preserve optional fields for callers that request them.
                for k in ("hidden_states", "past_key_values", "attentions"):
                    if k in outputs:
                        out[k] = outputs[k]
            else:
                if getattr(outputs, "hidden_states", None) is not None:
                    out["hidden_states"] = outputs.hidden_states
                if getattr(outputs, "past_key_values", None) is not None:
                    out["past_key_values"] = outputs.past_key_values
                if getattr(outputs, "attentions", None) is not None:
                    out["attentions"] = outputs.attentions
            return out

        # Default path: let the base model compute standard causal-LM loss.
        return self.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def _soft_discretization_lm_loss(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        numeric_values: torch.Tensor,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Causal-LM loss with soft targets at quantile-token positions.

        For positions where the next-token label is a quantile token Qk and a numeric
        value v is available, we replace hard cross-entropy with a two-point target
        distribution over adjacent bins determined by the within-boundary interpolation
        weight alpha (from the same construction used for the soft embedding).

        When `num_items_in_batch` is provided (HF Trainer loss-kwargs contract),
        the loss is normalized as sum / num_items_in_batch instead of a per-batch
        mean, so the Trainer's gradient-accumulation handling stays exact.
        Ignoring it would inflate the logged train loss (and gradients) by the
        gradient accumulation factor.
        """
        if self.value_encoder is None or not isinstance(self.value_encoder, SoftDiscretizationEncoder):
            raise ValueError("Soft discretization loss requires SoftDiscretizationEncoder.")
        if self.q_token_id_by_bin is None:
            # Buffer is registered as q_token_id_by_bin_buf.
            q = getattr(self, "q_token_id_by_bin_buf", None)
            if q is None:
                raise ValueError("Missing q_token_id_by_bin buffer for soft discretization.")
            self.q_token_id_by_bin = q

        # Shift for causal LM: logits[t] predicts label[t+1]
        logits_next = logits[:, :-1, :]  # (B, S-1, V)
        labels_next = labels[:, 1:]  # (B, S-1)
        values_next = numeric_values[:, 1:]  # (B, S-1)
        code_ids = input_ids[:, :-1]  # (B, S-1) code token before the quantile token

        ignore_index = -100
        valid = labels_next != ignore_index

        # Start with standard hard CE everywhere.
        hard = F.cross_entropy(
            logits_next.reshape(-1, logits_next.size(-1)),
            labels_next.reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).reshape_as(labels_next)

        # Soft-target positions: quantile label + present numeric value + boundaries available.
        is_q_label = self.is_q_token.to(device=labels_next.device)[labels_next.clamp(min=0)]
        has_value = torch.isfinite(values_next)

        # Determine which codes have boundaries.
        n_boundaries_by_id = self.value_encoder.n_boundaries_by_id
        if n_boundaries_by_id.device != code_ids.device:
            n_boundaries_by_id = n_boundaries_by_id.to(code_ids.device)
        has_boundaries = n_boundaries_by_id[code_ids] > 0

        def _normalize(loss_sum: torch.Tensor) -> torch.Tensor:
            if num_items_in_batch is not None:
                denom = num_items_in_batch
                if torch.is_tensor(denom):
                    denom = denom.to(loss_sum.device)
                return loss_sum / denom
            return loss_sum / valid.sum().clamp(min=1)

        soft_mask = valid & is_q_label & has_value & has_boundaries
        if not torch.any(soft_mask):
            return _normalize(hard[valid].sum())

        # Compute (lower_bin, upper_bin, alpha) for each soft position.
        flat_pos = soft_mask.nonzero(as_tuple=False)  # (N, 2): (batch, pos_in_Sminus1)
        v = values_next[soft_mask].to(dtype=torch.float32)  # (N,)
        cids = code_ids[soft_mask].to(dtype=torch.long)  # (N,)

        boundaries_by_id = self.value_encoder.boundaries_by_id
        n_boundaries_by_id = self.value_encoder.n_boundaries_by_id
        if boundaries_by_id.device != cids.device:
            boundaries_by_id = boundaries_by_id.to(cids.device)
        if n_boundaries_by_id.device != cids.device:
            n_boundaries_by_id = n_boundaries_by_id.to(cids.device)

        b = boundaries_by_id[cids]  # (N, M)
        n_b = n_boundaries_by_id[cids].to(torch.int64)  # (N,)
        m = b.shape[1]
        idx = torch.arange(m, device=cids.device).unsqueeze(0)  # (1, M)
        valid_b = idx < n_b.unsqueeze(1)
        b_eff = b.masked_fill(~valid_b, float("inf"))

        # Bin index = number of boundaries strictly less than v. A value
        # exactly on b_k stays in bin k, matching digitize(..., right=True).
        bin_idx = torch.sum(v.unsqueeze(1) > b_eff, dim=1)  # (N,)
        last_bin = n_b.clamp(max=self.value_encoder.num_bins - 1)
        lo = bin_idx == 0
        hi = bin_idx >= n_b
        # Interior bins interpolate (bin_idx-1, bin_idx). Tails are hard on
        # the same endpoint embedding the encoder uses (Q0 or Q_{K-1}).
        lower = (bin_idx - 1).clamp(0, self.value_encoder.num_bins - 1)
        upper = bin_idx.clamp(0, self.value_encoder.num_bins - 1)
        lower = torch.where(lo, torch.zeros_like(lower), lower)
        upper = torch.where(lo, torch.zeros_like(upper), upper)
        lower = torch.where(hi, last_bin, lower)
        upper = torch.where(hi, last_bin, upper)

        alpha = torch.zeros_like(v)
        mid = (~lo) & (~hi)
        if torch.any(mid):
            bi = bin_idx[mid].to(torch.long)
            b_mid = b_eff[mid]
            v_mid = v[mid]
            lower_b = b_mid.gather(1, (bi - 1).unsqueeze(1)).squeeze(1)
            upper_b = b_mid.gather(1, bi.unsqueeze(1)).squeeze(1)
            denom = upper_b - lower_b
            a = torch.where(
                denom.abs() < 1e-8,
                torch.zeros_like(denom),
                (v_mid - lower_b) / denom,
            ).clamp(0.0, 1.0)
            alpha[mid] = a

        qmap = self.q_token_id_by_bin.to(device=logits.device)
        q_low = qmap[lower.to(torch.long)]
        q_up = qmap[upper.to(torch.long)]

        logp = F.log_softmax(logits_next, dim=-1)
        lp = logp[flat_pos[:, 0], flat_pos[:, 1], :]  # (N, V)
        lp_low = lp.gather(1, q_low.unsqueeze(1)).squeeze(1)
        lp_up = lp.gather(1, q_up.unsqueeze(1)).squeeze(1)
        soft_loss = -((1.0 - alpha) * lp_low + alpha * lp_up)  # (N,)

        # Replace losses at soft positions.
        out = hard.clone()
        out[soft_mask] = soft_loss.to(dtype=out.dtype)
        return _normalize(out[valid].sum())

    def _apply_value_encoding(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        numeric_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply soft discretization to numeric token positions.

        Parameters
        ----------
        embeddings : torch.Tensor
            Token embeddings of shape (batch_size, seq_len, hidden_size)
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len)
        numeric_values : torch.Tensor
            Raw numeric values of shape (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
            Modified embeddings with value encoding applied
        """
        modified = embeddings.clone()

        # Create mask for numeric positions with finite measurements.
        numeric_mask = torch.isfinite(numeric_values)

        # Only modify quantile token positions (Q0..Qn-1) and skip position 0
        q_mask = self.is_q_token[input_ids]
        mask = numeric_mask & q_mask
        if mask.size(1) > 0:
            mask[:, 0] = False

        if not torch.any(mask):
            return modified

        # For a quantile token at position s, the code token is at position s-1.
        prev_ids = torch.roll(input_ids, shifts=1, dims=1)

        # We perform value encoding only when we have per-code bin boundaries.
        # If boundaries are missing for a code (e.g., an unseen code at inference),
        # we skip numeric injection and leave the original embedding unchanged.
        flat_pos = mask.nonzero(as_tuple=False)  # (n, 2) with columns (batch, seq)
        code_ids_all = prev_ids[mask]  # (n,)
        values_all = numeric_values[mask]  # (n,)

        has_meta = self.value_encoder.n_boundaries_by_id.to(device=code_ids_all.device)[code_ids_all] > 0

        if torch.any(has_meta):
            pos = flat_pos[has_meta]
            code_ids = code_ids_all[has_meta]
            values = values_all[has_meta]

            # Compute new embeddings in one vectorized call (fast code-id mode).
            new_embeds = self.value_encoder(values, code_ids=code_ids)

            # Write back only for codes with metadata.
            modified[pos[:, 0], pos[:, 1]] = new_embeds.to(dtype=modified.dtype)
        return modified

    def get_input_embeddings(self):
        """Return the input embeddings layer."""
        if hasattr(self.base_model, "model") and hasattr(
            self.base_model.model, "embed_tokens"
        ):
            return self.base_model.model.embed_tokens
        elif hasattr(self.base_model, "transformer") and hasattr(
            self.base_model.transformer, "wte"
        ):
            return self.base_model.transformer.wte
        return self.base_model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize the token embedding layer."""
        return self.base_model.resize_token_embeddings(new_num_tokens)

    @property
    def config(self):
        """Return the model config."""
        return self.base_model.config

    def parameters(self, recurse: bool = True):
        """Return all parameters including encoder parameters."""
        yield from self.base_model.parameters(recurse=recurse)
        if self.value_encoder is not None:
            yield from self.value_encoder.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return all named parameters."""
        yield from self.base_model.named_parameters(prefix=prefix, recurse=recurse)
        if self.value_encoder is not None:
            encoder_prefix = f"{prefix}value_encoder." if prefix else "value_encoder."
            yield from self.value_encoder.named_parameters(
                prefix=encoder_prefix, recurse=recurse
            )

    def state_dict(self, *args, **kwargs):
        """Return combined state dict."""
        state = {}
        state.update(self.base_model.state_dict(*args, **kwargs))
        if self.value_encoder is not None:
            for k, v in self.value_encoder.state_dict(*args, **kwargs).items():
                state[f"value_encoder.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load combined state dict."""
        base_state = {}
        encoder_state = {}

        for k, v in state_dict.items():
            if k.startswith("value_encoder."):
                encoder_state[k[14:]] = v
            else:
                base_state[k] = v

        result = self.base_model.load_state_dict(base_state, strict=strict)
        if self.value_encoder is not None and encoder_state:
            enc_result = self.value_encoder.load_state_dict(encoder_state, strict=strict)
            if result is not None and enc_result is not None:
                # Merge missing/unexpected keys from both sub-models
                result = type(result)(
                    missing_keys=result.missing_keys + [f"value_encoder.{k}" for k in enc_result.missing_keys],
                    unexpected_keys=result.unexpected_keys + [f"value_encoder.{k}" for k in enc_result.unexpected_keys],
                )
        return result


def create_representation_model(
    base_model: PreTrainedModel,
    vocab: Vocabulary,
    representation: str = "discrete",
    temporal: str = "time_tokens",
    **kwargs,
) -> RepresentationModelWrapper | PreTrainedModel:
    """Factory function to create a representation model.

    Returns the base model unchanged when no representation mechanics are
    needed: the discrete encoder combined with either "time_tokens" or
    "event_order". Both leave embeddings and position IDs untouched, differing
    only in whether the tokenizer inserted spacing tokens. Otherwise, wraps it
    with RepresentationModelWrapper.

    Parameters
    ----------
    base_model : PreTrainedModel
        The pretrained model to wrap
    vocab : Vocabulary
        Tokenizer vocabulary
    representation : str
        Value representation method
    temporal : str
        Temporal encoding method
    **kwargs
        Additional arguments passed to RepresentationModelWrapper

    Returns
    -------
    Model with representation mechanics applied
    """
    if representation == "discrete" and temporal in ("time_tokens", "event_order"):
        # No modifications needed - return base model
        return base_model

    if representation in ("xval", "xval_affine"):
        # xVal wrapper (requires [NUM] tokenization + numeric_values).
        numeric_injection = "mul" if representation == "xval" else "affine"
        seconds_per_position = kwargs.get("seconds_per_position")
        if seconds_per_position is None and kwargs.get("time_rope_scaling") is not None:
            seconds_per_position = 3600.0 / float(kwargs["time_rope_scaling"])
        return XValModelWrapper(
            base_model=base_model,
            vocab=vocab,
            temporal=temporal,
            seconds_per_position=float(seconds_per_position or 60.0),
            clip_sigma=float(kwargs.get("clip_sigma", 5.0)),
            numeric_stats=kwargs.get("numeric_stats", None),
            numeric_loss_weight=float(kwargs.get("numeric_loss_weight", 1.0)),
            numeric_injection=numeric_injection,
        )

    if (
        kwargs.get("seconds_per_position") is None
        and kwargs.get("time_rope_scaling") is not None
    ):
        kwargs = dict(kwargs)
        kwargs["seconds_per_position"] = 3600.0 / float(kwargs["time_rope_scaling"])
    return RepresentationModelWrapper(
        base_model=base_model,
        vocab=vocab,
        representation=representation,
        temporal=temporal,
        **kwargs,
    )


if __name__ == "__main__":
    # Quick test of the wrapper
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # Create a tiny model for testing
    config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    base_model = AutoModelForCausalLM.from_config(config)

    # Create mock vocabulary
    vocab = Vocabulary()
    for i in range(20):
        vocab(f"Q{i}")
    vocab("LAB_glucose")
    vocab.set_aux("LAB_glucose", list(range(50, 200, 8)))  # 19 breakpoints
    vocab.is_training = False

    # Test wrapper
    wrapper = RepresentationModelWrapper(
        base_model=base_model,
        vocab=vocab,
        representation="soft",
        temporal="time_rope",
        num_bins=20,
    )

    # Create mock inputs
    input_ids = torch.randint(0, 100, (2, 32))
    numeric_values = torch.full((2, 32), float("nan"))
    numeric_values[:, 10] = 100.0  # One numeric value
    relative_times = torch.linspace(0, 24, 32).unsqueeze(0).expand(2, -1)

    # Forward pass
    outputs = wrapper(
        input_ids=input_ids,
        numeric_values=numeric_values,
        relative_times=relative_times,
    )
    print(f"Output shape: {outputs.logits.shape}")
    print("Wrapper test passed!")
