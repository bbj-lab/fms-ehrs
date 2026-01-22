#!/usr/bin/env python3

"""
Model wrapper for Experiment 2 representation mechanics.

This module wraps a pretrained causal LM (e.g., LLaMA) to support different
value representation methods (discrete, soft, continuous) and temporal
encoding strategies (time_tokens, time2vec).

The wrapper intercepts the embedding layer and modifies embeddings based on:
- Soft discretization: Replace quantile-token embeddings with convex combinations
- Continuous encoding: Replace quantile-token embeddings with xVal-style scaled embeddings
- Time2Vec: Add learned temporal embeddings based on relative time since admission

Architecture:
    input_ids ──> Token Embedding ──> [Value Encoder] ──> [Time2Vec] ──> Transformer
                       │                    │                  │
                       v                    v                  v
                  (batch, seq, d)     (modify numeric    (add temporal
                                       positions)         embeddings)

References
----------
- Soft discretization: ConSE (Norouzi et al., 2014)
- Continuous encoding: xVal (Golkar et al., 2023) with EHR-specific adaptation
- Time2Vec: Kazemi & Poupart (2019)
"""

import typing

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from fms_ehrs.framework.continuous_encoder import ContinuousValueEncoder
from fms_ehrs.framework.soft_discretization import SoftDiscretizationEncoder
from fms_ehrs.framework.time2vec import Time2VecEmbedding
from fms_ehrs.framework.vocabulary import Vocabulary
from fms_ehrs.framework.xval import XValModelWrapper


class RepresentationModelWrapper(nn.Module):
    """Wraps a pretrained causal LM with Exp2 representation mechanics.

    This wrapper modifies the forward pass to:
    1. Apply soft discretization or continuous encoding to numeric token positions
    2. Add Time2Vec temporal embeddings when enabled

    Parameters
    ----------
    base_model : PreTrainedModel
        The base transformer model (e.g., LLaMA)
    vocab : Vocabulary
        Tokenizer vocabulary with quantile auxiliary data
    representation : {"discrete", "soft", "continuous"}
        Value representation method:
        - discrete: Standard token embeddings (baseline)
        - soft: Convex combinations of adjacent bin embeddings
        - continuous: xVal-style scaled embedding of z-scored values
    temporal : {"time_tokens", "time2vec"}
        Temporal encoding method:
        - time_tokens: Use existing time spacing tokens (baseline)
        - time2vec: Add learned Time2Vec embeddings
    num_bins : int
        Number of quantile bins (for soft discretization)
    time2vec_dim : int
        Internal dimension for Time2Vec before projection
    continuous_num_scales : int
        Number of xVal multiscale embeddings for continuous encoding
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        vocab: Vocabulary,
        representation: typing.Literal["discrete", "soft", "continuous"] = "discrete",
        temporal: typing.Literal["time_tokens", "time2vec"] = "time_tokens",
        num_bins: int = 20,
        time2vec_dim: int = 64,
        continuous_num_scales: int = 1,
        continuous_numeric_stats: dict[str, dict[str, float]] | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.vocab = vocab
        self.representation = representation
        self.temporal = temporal

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
        elif representation == "continuous":
            self.value_encoder = ContinuousValueEncoder(
                embed_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                num_scales=continuous_num_scales,
            )
            self.value_encoder.set_statistics_from_vocab_aux(
                vocab.aux,
                numeric_stats=continuous_numeric_stats,
                token_id_lookup=vocab.lookup,
                vocab_size=len(vocab),
            )

        # Initialize Time2Vec if needed
        self.time2vec_layer: nn.Module | None = None
        if temporal == "time2vec":
            self.time2vec_layer = Time2VecEmbedding(
                hidden_size=self.hidden_size,
                time2vec_dim=time2vec_dim,
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
        relative_times : torch.Tensor, optional
            Relative time in hours since admission, shape (batch_size, seq_len).
            Required when temporal="time2vec".
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

        # Apply value encoding modifications if using soft/continuous
        if self.representation in ("soft", "continuous") and numeric_values is not None:
            embeddings = self._apply_value_encoding(
                embeddings, input_ids, numeric_values
            )

        # Apply Time2Vec if enabled
        if self.temporal == "time2vec" and relative_times is not None:
            embeddings = self.time2vec_layer(embeddings, relative_times)

        # Ensure dtype matches the base model parameters.
        #
        # Rationale: Time2Vec (and some value encoders) operate in float32 by default.
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

        # Pass modified embeddings through the model
        # We bypass the embedding layer by passing inputs_embeds
        return self.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def _apply_value_encoding(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        numeric_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply soft/continuous encoding to numeric token positions.

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

        # Create mask for numeric positions (non-NaN values)
        numeric_mask = ~torch.isnan(numeric_values)

        # Only modify quantile token positions (Q0..Qn-1) and skip position 0
        q_mask = self.is_q_token[input_ids]
        mask = numeric_mask & q_mask
        if mask.size(1) > 0:
            mask[:, 0] = False

        if not torch.any(mask):
            return modified

        # For a quantile token at position s, the code token is at position s-1.
        prev_ids = torch.roll(input_ids, shifts=1, dims=1)
        code_ids = prev_ids[mask]
        values = numeric_values[mask]

        # Compute new embeddings in one vectorized call.
        new_embeds = self.value_encoder(values, code_ids=code_ids)

        # Write back
        modified[mask] = new_embeds.to(dtype=modified.dtype)
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
        if self.time2vec_layer is not None:
            yield from self.time2vec_layer.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return all named parameters."""
        yield from self.base_model.named_parameters(prefix=prefix, recurse=recurse)
        if self.value_encoder is not None:
            encoder_prefix = f"{prefix}value_encoder." if prefix else "value_encoder."
            yield from self.value_encoder.named_parameters(
                prefix=encoder_prefix, recurse=recurse
            )
        if self.time2vec_layer is not None:
            time_prefix = f"{prefix}time2vec_layer." if prefix else "time2vec_layer."
            yield from self.time2vec_layer.named_parameters(
                prefix=time_prefix, recurse=recurse
            )

    def state_dict(self, *args, **kwargs):
        """Return combined state dict."""
        state = {}
        state.update(self.base_model.state_dict(*args, **kwargs))
        if self.value_encoder is not None:
            for k, v in self.value_encoder.state_dict(*args, **kwargs).items():
                state[f"value_encoder.{k}"] = v
        if self.time2vec_layer is not None:
            for k, v in self.time2vec_layer.state_dict(*args, **kwargs).items():
                state[f"time2vec_layer.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load combined state dict."""
        base_state = {}
        encoder_state = {}
        time_state = {}

        for k, v in state_dict.items():
            if k.startswith("value_encoder."):
                encoder_state[k[14:]] = v
            elif k.startswith("time2vec_layer."):
                time_state[k[15:]] = v
            else:
                base_state[k] = v

        self.base_model.load_state_dict(base_state, strict=strict)
        if self.value_encoder is not None and encoder_state:
            self.value_encoder.load_state_dict(encoder_state, strict=strict)
        if self.time2vec_layer is not None and time_state:
            self.time2vec_layer.load_state_dict(time_state, strict=strict)


def create_representation_model(
    base_model: PreTrainedModel,
    vocab: Vocabulary,
    representation: str = "discrete",
    temporal: str = "time_tokens",
    **kwargs,
) -> RepresentationModelWrapper | PreTrainedModel:
    """Factory function to create a representation model.

    If representation is "discrete" and temporal is "time_tokens", returns
    the base model unchanged. Otherwise, wraps it with RepresentationModelWrapper.

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
    if representation == "discrete" and temporal == "time_tokens":
        # No modifications needed - return base model
        return base_model

    if representation == "xval":
        # Canonical xVal wrapper (requires [NUM] tokenization + numeric_values).
        return XValModelWrapper(
            base_model=base_model,
            vocab=vocab,
            temporal=temporal,
            time2vec_dim=int(kwargs.get("time2vec_dim", 64)),
            clip_sigma=float(kwargs.get("clip_sigma", 5.0)),
            numeric_stats=kwargs.get("continuous_numeric_stats", None),
            numeric_loss_weight=float(kwargs.get("numeric_loss_weight", 1.0)),
        )

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
        temporal="time2vec",
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
