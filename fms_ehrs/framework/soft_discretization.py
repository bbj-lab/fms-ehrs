#!/usr/bin/env python3

"""
Soft discretization via convex combinations of adjacent bin embeddings.

This module implements the ConSE-inspired approach for representing continuous
values as weighted interpolations of discrete bin embeddings.

References
----------
[1] Norouzi, M., Mikolov, T., Bengio, S., Singer, Y., Shlens, J., Frome, A.,
    Corrado, G. S., & Dean, J. (2014). Zero-Shot Learning by Convex Combination
    of Semantic Embeddings. ICLR. arXiv:1312.5650.

For a value v falling between bin boundaries b_i and b_{i+1}:
    alpha = (v - b_i) / (b_{i+1} - b_i)
    embedding = (1 - alpha) * E[b_i] + alpha * E[b_{i+1}]

This local support constraint ensures:
- Monotonicity: values closer to b_i receive embeddings closer to E[b_i]
- Interpretability: alpha directly corresponds to relative position
- Computational efficiency: only two embeddings accessed per value
"""

import typing

import numpy as np
import torch
import torch.nn as nn


class SoftDiscretizationEncoder(nn.Module):
    """Encode continuous values via convex combinations of bin embeddings.

    This encoder represents numeric values as weighted interpolations of
    adjacent quantile bin embeddings, preserving the geometric structure
    of the embedding space while enabling smooth representation of
    continuous values.

    Parameters
    ----------
    num_bins : int
        Number of quantile bins (e.g., 10 for deciles, 20 for ventiles)
    embed_dim : int
        Dimension of the embedding space
    bin_boundaries : torch.Tensor, optional
        Pre-computed bin boundaries of shape (num_codes, num_bins - 1).
        If None, boundaries must be set via set_boundaries() before use.

    Example
    -------
    >>> encoder = SoftDiscretizationEncoder(num_bins=20, embed_dim=768)
    >>> # Set boundaries for a specific code (e.g., glucose lab)
    >>> boundaries = torch.linspace(50, 200, 19)  # 19 boundaries for 20 bins
    >>> encoder.set_boundaries("LAB_glucose", boundaries)
    >>> # Encode a value
    >>> value = torch.tensor([85.0])
    >>> embedding = encoder(value, code="LAB_glucose")
    """

    def __init__(
        self,
        num_bins: int,
        embed_dim: int,
        bin_boundaries: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim

        # Learnable bin embeddings: shared across all codes
        # Shape: (num_bins, embed_dim)
        self.bin_embeddings = nn.Embedding(num_bins, embed_dim)

        # Store bin boundaries per code (not learnable parameters)
        self._boundaries: dict[str, torch.Tensor] = bin_boundaries or {}

        # Optional fast path: boundaries indexed by code token-id.
        # These are derived from a (code string -> token id) lookup and registered
        # as buffers for device placement. They are not persisted in checkpoints
        # (they can be reconstructed from vocab.aux at load time).
        self.register_buffer(
            "boundaries_by_id",
            torch.empty(0, 0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "n_boundaries_by_id",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )

    def set_boundaries(self, code: str, boundaries: torch.Tensor) -> None:
        """Set bin boundaries for a specific code.

        Parameters
        ----------
        code : str
            Code identifier (e.g., "LAB_glucose")
        boundaries : torch.Tensor
            Sorted bin boundaries of shape (num_bins - 1,)
        """
        if boundaries.shape[0] != self.num_bins - 1:
            raise ValueError(
                f"Expected {self.num_bins - 1} boundaries, got {boundaries.shape[0]}"
            )
        self._boundaries[code] = boundaries

    def set_boundaries_from_vocab_aux(
        self,
        vocab_aux: dict[str, list[float]],
        *,
        device: torch.device | None = None,
        token_id_lookup: dict[str, int] | None = None,
        vocab_size: int | None = None,
    ) -> None:
        """Set boundaries from vocabulary auxiliary data.

        This supports two modes:
        1) String-keyed mode (legacy): store boundaries in a dict keyed by code string.
        2) Token-id keyed mode (fast): build padded boundary tensors indexed by code token id.

        Parameters
        ----------
        vocab_aux : dict
            Dictionary mapping code names to list of quantile boundaries
        device : torch.device, optional
            Device to place tensors on
        token_id_lookup : dict[str, int], optional
            Mapping from code string to its vocabulary token id. If provided,
            a fast vectorized path will be enabled via `code_ids` in `forward()`.
        vocab_size : int, optional
            Vocabulary size used to size the boundary tensors. Required if
            `token_id_lookup` is provided.
        """
        if token_id_lookup is None:
            for code, breaks in vocab_aux.items():
                boundaries = torch.tensor(breaks, dtype=torch.float32)
                if device is not None:
                    boundaries = boundaries.to(device)
                self._boundaries[code] = boundaries
            return

        if vocab_size is None:
            raise ValueError("vocab_size must be provided when token_id_lookup is used")

        # Build padded boundary tensors:
        # - boundaries_by_id: (vocab_size, num_bins - 1), padded with 0s
        # - n_boundaries_by_id: (vocab_size,), number of valid boundaries per code id
        max_b = max(self.num_bins - 1, 0)
        boundaries_by_id = torch.zeros((vocab_size, max_b), dtype=torch.float32)
        # NOTE: Use int32 (not int16/Short) because DDP initial state sync uses NCCL,
        # and NCCL does not support torch.int16 (Short) for broadcast.
        n_boundaries_by_id = torch.zeros((vocab_size,), dtype=torch.int32)

        for code, breaks in vocab_aux.items():
            tok_id = token_id_lookup.get(code)
            if tok_id is None:
                continue
            if not breaks:
                continue
            # Keep order (assumed sorted) and allow duplicates; duplicates are handled
            # by the denom==0 guard during interpolation.
            b = torch.tensor(breaks, dtype=torch.float32)
            if device is not None:
                b = b.to(device)
            n = min(b.numel(), max_b)
            if n == 0:
                continue
            boundaries_by_id[tok_id, :n] = b[:n]
            n_boundaries_by_id[tok_id] = n

        # Register as buffers (device follows module)
        self.boundaries_by_id = boundaries_by_id
        self.n_boundaries_by_id = n_boundaries_by_id

    def forward(
        self,
        values: torch.Tensor,
        codes: list[str] | None = None,
        *,
        code_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute soft discretization embeddings for values.

        Parameters
        ----------
        values : torch.Tensor
            Numeric values of shape (batch_size,) or (batch_size, 1)
        codes : list[str], optional
            Code identifiers for each value, length batch_size (string-keyed mode)
        code_ids : torch.Tensor, optional
            Code token ids for each value, shape (batch_size,) (fast mode).

        Returns
        -------
        torch.Tensor
            Soft discretized embeddings of shape (batch_size, embed_dim)
        """
        if values.dim() == 2:
            values = values.squeeze(-1)

        if code_ids is not None and self.boundaries_by_id.numel() > 0:
            return self._forward_code_ids(values, code_ids)

        if codes is None:
            raise ValueError("Must provide either `codes` or `code_ids`.")

        batch_size = values.shape[0]
        device = values.device
        embeddings = torch.zeros(batch_size, self.embed_dim, device=device)

        for i, (value, code) in enumerate(zip(values, codes)):
            if code not in self._boundaries:
                embeddings[i] = self.bin_embeddings(
                    torch.tensor(self.num_bins // 2, device=device)
                )
                continue

            boundaries = self._boundaries[code].to(device)
            embedding = self._compute_soft_embedding(value, boundaries)
            embeddings[i] = embedding

        return embeddings

    def _forward_code_ids(self, values: torch.Tensor, code_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized soft discretization for code token ids."""
        if code_ids.dim() != 1:
            raise ValueError("code_ids must be a 1D tensor of token ids")
        if values.shape[0] != code_ids.shape[0]:
            raise ValueError("values and code_ids must have matching length")

        device = values.device
        dtype = self.bin_embeddings.weight.dtype
        n = values.shape[0]

        # Default fallback: middle bin embedding
        mid = self.num_bins // 2
        out = self.bin_embeddings(torch.full((n,), mid, device=device, dtype=torch.long)).to(dtype)

        # Gather boundaries for these codes
        code_ids = code_ids.to(device=device, dtype=torch.long)
        if self.boundaries_by_id.device != device:
            # If the module buffers haven't been moved (e.g., constructed on CPU and
            # code runs before .to(device)), move them lazily.
            self.boundaries_by_id = self.boundaries_by_id.to(device)
            self.n_boundaries_by_id = self.n_boundaries_by_id.to(device)

        boundaries = self.boundaries_by_id[code_ids]  # (n, num_bins-1)
        n_b = self.n_boundaries_by_id[code_ids].to(torch.int64)  # (n,)

        has = n_b > 0
        if not torch.any(has):
            return out

        # Mask out unused boundary slots by setting them to +inf, so comparisons ignore them.
        m = boundaries.shape[1]
        idx = torch.arange(m, device=device).unsqueeze(0)  # (1, m)
        valid = idx < n_b.unsqueeze(1)  # (n, m)
        boundaries_eff = boundaries.masked_fill(~valid, float("inf"))

        # Bin index is the number of boundaries strictly less than the value.
        # This matches searchsorted(..., right=False) semantics.
        bin_idx = torch.sum(values.unsqueeze(1) > boundaries_eff, dim=1)  # (n,)

        # Edge bins: below first boundary (bin 0) and above last boundary (bin n_b)
        # Note: for codes with fewer than (num_bins-1) boundaries, the last valid bin
        # index is n_b, not necessarily (num_bins-1).
        lo = has & (bin_idx == 0)
        hi = has & (bin_idx >= n_b)
        mid_mask = has & (~lo) & (~hi)

        if torch.any(lo):
            out[lo] = self.bin_embeddings(torch.zeros((int(lo.sum()),), device=device, dtype=torch.long))

        if torch.any(hi):
            hi_idx = bin_idx[hi].clamp(0, self.num_bins - 1).to(torch.long)
            out[hi] = self.bin_embeddings(hi_idx)

        if torch.any(mid_mask):
            # Interpolate between (bin_idx-1) and (bin_idx)
            bi = bin_idx[mid_mask].to(torch.long)
            lower_i = bi - 1
            upper_i = bi

            b_eff = boundaries_eff[mid_mask]  # (k, m)
            v = values[mid_mask]

            lower_b = b_eff.gather(1, lower_i.unsqueeze(1)).squeeze(1)
            upper_b = b_eff.gather(1, upper_i.unsqueeze(1)).squeeze(1)
            denom = upper_b - lower_b

            alpha = torch.where(
                denom.abs() < 1e-8,
                torch.full_like(denom, 0.5),
                (v - lower_b) / denom,
            ).clamp(0.0, 1.0)

            lower_e = self.bin_embeddings(lower_i)
            upper_e = self.bin_embeddings(upper_i)
            out[mid_mask] = (1.0 - alpha).unsqueeze(1) * lower_e + alpha.unsqueeze(1) * upper_e

        return out.to(dtype)

    def _compute_soft_embedding(
        self, value: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft embedding for a single value.

        Parameters
        ----------
        value : torch.Tensor
            Scalar value
        boundaries : torch.Tensor
            Bin boundaries of shape (num_bins - 1,)

        Returns
        -------
        torch.Tensor
            Soft embedding of shape (embed_dim,)
        """
        device = value.device

        # Some codes may have fewer valid boundaries than (num_bins - 1) due to
        # degenerate quantiles / duplicates. Treat the effective number of bins
        # as (n_boundaries + 1), bounded by self.num_bins.
        n_boundaries = int(boundaries.numel())
        eff_bins = min(self.num_bins, n_boundaries + 1) if n_boundaries > 0 else 1

        # Find which bin the value falls into
        bin_idx = torch.searchsorted(boundaries, value).clamp(0, eff_bins - 1)

        # Handle edge cases
        if bin_idx == 0:
            # Below first boundary: use first bin embedding
            return self.bin_embeddings(torch.tensor(0, device=device))
        elif bin_idx >= eff_bins - 1:
            # Above last boundary: use last effective bin embedding
            return self.bin_embeddings(torch.tensor(eff_bins - 1, device=device))

        # Compute interpolation weight
        lower_boundary = boundaries[bin_idx - 1]
        upper_boundary = boundaries[bin_idx]

        # Avoid division by zero
        denom = upper_boundary - lower_boundary
        if denom.abs() < 1e-8:
            alpha = torch.tensor(0.5, device=device)
        else:
            alpha = (value - lower_boundary) / denom
            alpha = alpha.clamp(0.0, 1.0)

        # Get adjacent bin embeddings
        lower_embed = self.bin_embeddings(torch.tensor(bin_idx - 1, device=device))
        upper_embed = self.bin_embeddings(torch.tensor(bin_idx, device=device))

        # Convex combination
        return (1 - alpha) * lower_embed + alpha * upper_embed


class SoftDiscretizationLayer(nn.Module):
    """Layer that applies soft discretization to numeric token values.

    This layer is designed to be used within a transformer model, replacing
    the standard discrete quantile token lookup with soft interpolation.

    Parameters
    ----------
    num_bins : int
        Number of quantile bins
    embed_dim : int
        Embedding dimension
    """

    def __init__(self, num_bins: int, embed_dim: int):
        super().__init__()
        self.encoder = SoftDiscretizationEncoder(num_bins, embed_dim)
        self.num_bins = num_bins
        self.embed_dim = embed_dim

    def forward(
        self,
        token_embeddings: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        codes: list[list[str]],
    ) -> torch.Tensor:
        """Apply soft discretization to numeric tokens.

        Parameters
        ----------
        token_embeddings : torch.Tensor
            Standard token embeddings of shape (batch, seq_len, embed_dim)
        numeric_values : torch.Tensor
            Numeric values of shape (batch, seq_len)
        numeric_mask : torch.Tensor
            Boolean mask indicating numeric tokens, shape (batch, seq_len)
        codes : list[list[str]]
            Code identifiers for each position, shape (batch, seq_len)

        Returns
        -------
        torch.Tensor
            Modified embeddings with soft discretization applied to numeric tokens
        """
        batch_size, seq_len, embed_dim = token_embeddings.shape
        output = token_embeddings.clone()

        for b in range(batch_size):
            for s in range(seq_len):
                if numeric_mask[b, s]:
                    value = numeric_values[b, s]
                    code = codes[b][s]
                    soft_embed = self.encoder(
                        value.unsqueeze(0), [code]
                    ).squeeze(0)
                    output[b, s] = soft_embed

        return output


def create_soft_discretization_from_tokenizer(
    tokenizer, embed_dim: int
) -> SoftDiscretizationEncoder:
    """Create a SoftDiscretizationEncoder from a trained tokenizer.

    Parameters
    ----------
    tokenizer : BaseTokenizer
        Trained tokenizer with quantile auxiliary data
    embed_dim : int
        Embedding dimension

    Returns
    -------
    SoftDiscretizationEncoder
        Encoder initialized with tokenizer's bin boundaries
    """
    # Determine number of bins from tokenizer
    num_bins = len(tokenizer.q_tokens)

    encoder = SoftDiscretizationEncoder(num_bins=num_bins, embed_dim=embed_dim)

    # Set boundaries from vocab auxiliary data
    encoder.set_boundaries_from_vocab_aux(tokenizer.vocab.aux)

    return encoder


if __name__ == "__main__":
    # Test soft discretization
    torch.manual_seed(42)

    # Create encoder with 20 bins (ventile)
    encoder = SoftDiscretizationEncoder(num_bins=20, embed_dim=64)

    # Set up boundaries for a test code (glucose-like distribution)
    boundaries = torch.linspace(50, 200, 19)  # 19 breaks for 20 bins
    encoder.set_boundaries("LAB_glucose", boundaries)

    # Test single value
    value = torch.tensor([125.0])
    embedding = encoder(value, codes=["LAB_glucose"])
    print(f"Single value embedding shape: {embedding.shape}")

    # Test batch
    values = torch.tensor([60.0, 100.0, 150.0, 180.0])
    codes = ["LAB_glucose"] * 4
    embeddings = encoder(values, codes=codes)
    print(f"Batch embedding shape: {embeddings.shape}")

    # Verify interpolation property: values close together should have similar embeddings
    v1 = encoder(torch.tensor([100.0]), codes=["LAB_glucose"])
    v2 = encoder(torch.tensor([101.0]), codes=["LAB_glucose"])
    v3 = encoder(torch.tensor([150.0]), codes=["LAB_glucose"])

    dist_12 = torch.norm(v1 - v2).item()
    dist_13 = torch.norm(v1 - v3).item()
    print(f"Distance between 100 and 101: {dist_12:.4f}")
    print(f"Distance between 100 and 150: {dist_13:.4f}")
    print(f"Close values have smaller distance: {dist_12 < dist_13}")

