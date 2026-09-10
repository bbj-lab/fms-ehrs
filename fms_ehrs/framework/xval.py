#!/usr/bin/env python3

"""
xVal implementation (Golkar et al., 2023) adapted to EHR sequences.

Key engineering choices:
- Standard xVal replaces each numeric value with a dedicated placeholder token
  (we use the literal token string "[NUM]") and encodes the magnitude by
  *multiplying* that placeholder embedding by a (preprocessed) scalar.
- We additionally support an affine variant: at [NUM] positions,
  e_NUM(v) = z · e_NUM + b, so a code-median value (z = 0) is not forced to
  the zero vector.
- Standard xVal adds a separate *number head* trained with a regression loss
  on the numeric values at the "[NUM]" positions, in addition to the standard
  token cross-entropy loss.

EHR-specific compatibility:
- Unlike the original paper (which parses numbers from free text), EHR numeric
  values are code-typed. We therefore normalize values per code token with a
  clipped median / IQR-scale z-score from tokenization:
    z = clip((v - median_c) / scale_c, ±clip_sigma),  scale_c = IQR_c / 1.35
    <data_dir>/<data_version>-tokenized/train/numeric_stats.json

Reference:
  Golkar, S., Pettee, M., Eickenberg, M., et al. (2023).
  xVal: A Continuous Number Encoding for Large Language Models. arXiv:2310.02989.
"""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from fms_ehrs.framework.temporal import admission_relative_position_ids
from fms_ehrs.framework.vocabulary import Vocabulary


def _median_and_iqr_scale(st: dict[str, float]) -> tuple[float, float] | None:
    """Read location and IQR-scale from a per-code stats dict.

    New files store ``scale`` = IQR/1.35. Version-1 numeric_stats.json files
    stored the same quantity under ``std``; accept that key as a fallback.
    """
    scale_raw = st["scale"] if "scale" in st else st.get("std")
    try:
        median = float(st.get("median"))
        scale = float(scale_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(median) or not math.isfinite(scale) or scale <= 0:
        return None
    return median, scale


class XValModelWrapper(nn.Module):
    """Wrap a causal LM with standard xVal numeric encoding.

    Expected tokenization (unfused):
      ... CODE_TOKEN, [NUM], ...
    and `numeric_values` aligned so that the [NUM] position contains the raw scalar,
    and other positions are NaN.

    Parameters
    ----------
    base_model:
        HuggingFace causal LM (e.g. LLaMA).
    vocab:
        Vocabulary used for tokenization (must include "[NUM]" token).
    temporal:
        "time_tokens" (no extra temporal embedding), "time_rope" (Time-Aware
        RoPE), or "event_order" (no spacing tokens and no admission-relative
        position IDs; ordinary RoPE over sequential token index).
    clip_sigma:
        Clip the IQR-scale z-score to [-clip_sigma, clip_sigma].
    numeric_stats:
        Mapping from code-string -> stats dict, as stored under numeric_stats.json["stats"].
        Used to compute per-code median / IQR-scale z. If missing for a code, we skip
        scaling and exclude that position from the numeric loss.
    numeric_loss_weight:
        Weight for numeric regression loss added to token loss.
    numeric_loss_type:
        "mse" (default). (NMSE support can be added when enabling multi-scale xVal.)
    numeric_injection:
        "mul" (standard): e_NUM(v) = z · e_NUM.
        "affine": e_NUM(v) = z · e_NUM + b.
    """

    def __init__(
        self,
        *,
        base_model: PreTrainedModel,
        vocab: Vocabulary,
        temporal: typing.Literal[
            "time_tokens", "time_rope", "event_order"
        ] = "time_tokens",
        seconds_per_position: float = 60.0,
        clip_sigma: float = 5.0,
        numeric_stats: dict[str, dict[str, float]] | None = None,
        numeric_loss_weight: float = 1.0,
        numeric_loss_type: typing.Literal["mse"] = "mse",
        numeric_injection: typing.Literal["mul", "affine"] = "mul",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.vocab = vocab
        self.temporal = temporal
        self.seconds_per_position = float(seconds_per_position)
        self.clip_sigma = float(clip_sigma)
        self.numeric_loss_weight = float(numeric_loss_weight)
        self.numeric_loss_type = numeric_loss_type
        self.numeric_injection = numeric_injection

        if self.numeric_injection not in ("mul", "affine"):
            raise ValueError(
                f"Unsupported numeric_injection={self.numeric_injection!r}; expected 'mul' or 'affine'."
            )

        if "[NUM]" not in self.vocab.lookup:
            raise ValueError('Vocabulary does not contain required token "[NUM]".')
        self.num_token_id = int(self.vocab("[NUM]"))

        hidden_size = base_model.config.hidden_size
        self.number_head = nn.Linear(hidden_size, 1)

        # Optional affine bias (only when enabled to avoid unused parameters in standard runs).
        if self.numeric_injection == "affine":
            self.num_bias = nn.Parameter(torch.empty(hidden_size))
            self._init_num_bias_from_num_embedding()
        else:
            self.num_bias = None

        # Build fast stats lookup by *token id of the code token*.
        vocab_size = len(vocab)
        medians_by_id = torch.zeros((vocab_size,), dtype=torch.float32)
        scales_by_id = torch.ones((vocab_size,), dtype=torch.float32)
        has_by_id = torch.zeros((vocab_size,), dtype=torch.bool)

        if numeric_stats is not None:
            for code, st in numeric_stats.items():
                tok_id = vocab.lookup.get(code)
                if tok_id is None:
                    continue
                parsed = _median_and_iqr_scale(st)
                if parsed is None:
                    continue
                median, scale = parsed
                medians_by_id[tok_id] = median
                scales_by_id[tok_id] = scale
                has_by_id[tok_id] = True

        self.register_buffer("medians_by_id", medians_by_id, persistent=False)
        self.register_buffer("scales_by_id", scales_by_id, persistent=False)
        self.register_buffer("has_stats_by_id", has_by_id, persistent=False)

    def _init_num_bias_from_num_embedding(self) -> None:
        """Seed the affine bias with the model's own [NUM] embedding row.

        The injected embedding is z*e_NUM + b, so b is what the model sees when a
        measurement sits at its code's median (z = 0) - the most common case.
        Seeding b = e_NUM makes that decode to "a number of typical magnitude was
        here" rather than an arbitrary direction, so at initialization the
        injection reads as (1 + z)*e_NUM. It also removes initialization as a
        confound between the 'mul' and 'affine' arms, leaving the affine shift
        itself as the only difference. b remains free to move during training.
        """
        weight = getattr(self._embedding_layer(), "weight", None)
        with torch.no_grad():
            if weight is None or self.num_token_id >= weight.shape[0]:
                nn.init.normal_(self.num_bias, mean=0.0, std=0.02)
                return
            self.num_bias.copy_(
                weight[self.num_token_id].to(
                    dtype=self.num_bias.dtype, device=self.num_bias.device
                )
            )

    def _embedding_layer(self) -> nn.Module | None:
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            return self.base_model.model.embed_tokens
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            return self.base_model.transformer.wte
        return None

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            return self.base_model.model.embed_tokens(input_ids)
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            return self.base_model.transformer.wte(input_ids)
        raise ValueError(f"Cannot find embedding layer for model type {type(self.base_model)}")

    def _normalize_by_code_id(
        self,
        values: torch.Tensor,
        code_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-code median / IQR-scale normalization with clipping.

        Note: callers are responsible for skipping numeric injection when stats are missing.
        """
        device = values.device
        code_ids = code_ids.to(device=device, dtype=torch.long)

        medians = self.medians_by_id.to(device)[code_ids]
        scales = self.scales_by_id.to(device)[code_ids]

        z = (values - medians) / scales
        z = torch.clamp(z, -self.clip_sigma, self.clip_sigma)
        return z

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
        if numeric_values is None:
            raise ValueError("XValModelWrapper requires `numeric_values` aligned to tokens.")

        # Identify numeric positions ([NUM]) with a present numeric value.
        # NOTE: numeric_values is expected to use NaN for non-numeric positions.
        num_mask = (input_ids == self.num_token_id) & torch.isfinite(numeric_values)
        if num_mask.size(1) > 0:
            num_mask[:, 0] = False  # cannot infer code_id for position 0

        # Infer code token ids for [NUM] positions as the immediately previous token.
        code_ids = torch.zeros_like(input_ids)
        code_ids[:, 1:] = input_ids[:, :-1]

        # Determine which [NUM] positions have per-code stats (skip injection if missing).
        has_stats = self.has_stats_by_id.to(device=input_ids.device)[code_ids]

        # Normalize values (only meaningful at [NUM] positions with stats).
        norm_values = self._normalize_by_code_id(
            values=numeric_values.nan_to_num(0.0),
            code_ids=code_ids,
        )

        # Build multiplicative scaling factors: 1 for non-[NUM] positions.
        scale = torch.ones_like(norm_values)
        scale = torch.where(num_mask & has_stats, norm_values, scale)

        # Embed tokens and apply xVal scaling at [NUM] positions.
        embeddings = self._embed_tokens(input_ids)
        embeddings = embeddings * scale.unsqueeze(-1).to(dtype=embeddings.dtype)

        if self.numeric_injection == "affine":
            if self.num_bias is None:
                raise RuntimeError("numeric_injection='affine' requires num_bias to be initialized.")
            # Apply affine shift only at [NUM] positions with per-code stats.
            mask = (num_mask & has_stats).unsqueeze(-1)  # (B, S, 1)
            bias = self.num_bias.to(device=embeddings.device, dtype=embeddings.dtype).view(1, 1, -1)
            embeddings = embeddings + mask.to(dtype=embeddings.dtype) * bias
        elif self.numeric_injection != "mul":
            raise ValueError(f"Unsupported numeric_injection: {self.numeric_injection}")

        # Optional Time-Aware RoPE (pass position_ids to base model)
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

        # Ensure dtype matches base model dtype.
        base_dtype = getattr(self.base_model, "dtype", None) or next(self.base_model.parameters()).dtype
        if embeddings.dtype != base_dtype:
            embeddings = embeddings.to(dtype=base_dtype)

        # Pop keys we hardcode below so callers can pass them without conflict.
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        outputs = self.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # HF ModelOutput uses attribute access; our unit tests use a dict.
        # When labels=None, the model doesn't compute loss: attribute access
        # returns None, but __getitem__ raises KeyError. Use safe access.
        if isinstance(outputs, dict):
            hidden_states = outputs["hidden_states"]
            logits = outputs["logits"]
            token_loss = outputs.get("loss", None)
        else:
            hidden_states = outputs.hidden_states
            logits = outputs.logits
            token_loss = getattr(outputs, "loss", None)

        # Inference mode (labels=None): no loss computation, return hidden_states for extraction.
        if token_loss is None:
            return {
                "logits": logits,
                "hidden_states": hidden_states,
            }

        numeric_loss = torch.tensor(0.0, device=token_loss.device, dtype=token_loss.dtype)

        if self.numeric_loss_type != "mse":
            raise ValueError(f"Unsupported numeric_loss_type: {self.numeric_loss_type}")

        # Align numeric regression with the causal LM objective: logits at position t
        # predict token t+1, so the numeric head predicts the value of a [NUM] token at
        # position t+1 from the hidden state at position t. Therefore compare
        # pred[:, :-1] to target[:, 1:] where input_ids[:, 1:] is [NUM].
        #
        # Guard on the shifted, label-filtered mask rather than the unshifted one:
        # it is a strict subset, so testing the wider mask can pass while the
        # selection is empty, and torch.mean over an empty tensor returns NaN.
        # The numeric loss is supervised, so it requires labels; without them
        # there is nothing to score.
        num_mask_used = num_mask & has_stats
        mask_next = (
            num_mask_used[:, 1:] & (labels[:, 1:] != -100)
            if labels is not None
            else None
        )
        if mask_next is not None and torch.any(mask_next):
            hidden = hidden_states[-1]  # (batch, seq, hidden)
            pred = self.number_head(hidden).squeeze(-1)  # (batch, seq)
            target = norm_values.to(device=pred.device, dtype=pred.dtype)
            numeric_loss = torch.mean(
                (pred[:, :-1][mask_next] - target[:, 1:][mask_next]) ** 2
            )

        total_loss = token_loss + (self.numeric_loss_weight * numeric_loss)

        # Trainer accepts dict-like outputs. We include numeric_loss for logging/debugging.
        return {
            "loss": total_loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "token_loss": token_loss.detach(),
            "numeric_loss": numeric_loss.detach(),
        }


def persist_xval_mechanics(model: XValModelWrapper) -> dict[str, typing.Any]:
    """Serialize trained xVal heads for representation_mechanics.pt."""
    state: dict[str, typing.Any] = {
        "number_head_state": model.number_head.state_dict(),
    }
    if model.numeric_injection == "affine":
        if model.num_bias is None:
            raise RuntimeError("numeric_injection='affine' requires num_bias.")
        state["num_bias"] = model.num_bias.detach().cpu().contiguous().clone()
    return state


def apply_xval_mechanics(
    model: XValModelWrapper, mechanics: dict[str, typing.Any]
) -> None:
    """Restore trained xVal heads from representation_mechanics.pt.

    Affine extraction requires the trained bias. Missing ``num_bias`` is an
    error: the wrapper would otherwise keep the initialization seed
    ``b = e_NUM``, which is not the trained map ``z * e_NUM + b_trained``.
    """
    number_head_state = mechanics.get("number_head_state")
    if number_head_state is not None:
        model.number_head.load_state_dict(number_head_state)
    if model.numeric_injection != "affine":
        return
    bias = mechanics.get("num_bias")
    if bias is None:
        raise RuntimeError(
            "xval_affine requires trained num_bias in representation_mechanics.pt; "
            "refusing to extract with the initialization seed."
        )
    if model.num_bias is None:
        raise RuntimeError("numeric_injection='affine' requires num_bias to be initialized.")
    tensor = bias if isinstance(bias, torch.Tensor) else torch.as_tensor(bias)
    tensor = tensor.to(device=model.num_bias.device, dtype=model.num_bias.dtype)
    if tuple(tensor.shape) != tuple(model.num_bias.shape):
        raise RuntimeError(
            f"num_bias shape {tuple(tensor.shape)} does not match wrapper "
            f"{tuple(model.num_bias.shape)}"
        )
    model.num_bias.data.copy_(tensor)
