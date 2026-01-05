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
        self, vocab_aux: dict[str, list[float]], device: torch.device = None
    ) -> None:
        """Set boundaries from vocabulary auxiliary data.

        Parameters
        ----------
        vocab_aux : dict
            Dictionary mapping code names to list of quantile boundaries
        device : torch.device, optional
            Device to place tensors on
        """
        for code, breaks in vocab_aux.items():
            boundaries = torch.tensor(breaks, dtype=torch.float32)
            if device is not None:
                boundaries = boundaries.to(device)
            self._boundaries[code] = boundaries

    def forward(
        self,
        values: torch.Tensor,
        codes: list[str],
    ) -> torch.Tensor:
        """Compute soft discretization embeddings for values.

        Parameters
        ----------
        values : torch.Tensor
            Numeric values of shape (batch_size,) or (batch_size, 1)
        codes : list[str]
            Code identifiers for each value, length batch_size

        Returns
        -------
        torch.Tensor
            Soft discretized embeddings of shape (batch_size, embed_dim)
        """
        if values.dim() == 2:
            values = values.squeeze(-1)

        batch_size = values.shape[0]
        device = values.device

        embeddings = torch.zeros(batch_size, self.embed_dim, device=device)

        for i, (value, code) in enumerate(zip(values, codes)):
            if code not in self._boundaries:
                # Fallback: use middle bin embedding if boundaries not set
                embeddings[i] = self.bin_embeddings(
                    torch.tensor(self.num_bins // 2, device=device)
                )
                continue

            boundaries = self._boundaries[code].to(device)
            embedding = self._compute_soft_embedding(value, boundaries)
            embeddings[i] = embedding

        return embeddings

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

        # Find which bin the value falls into
        bin_idx = torch.searchsorted(boundaries, value).clamp(0, self.num_bins - 1)

        # Handle edge cases
        if bin_idx == 0:
            # Below first boundary: use first bin embedding
            return self.bin_embeddings(torch.tensor(0, device=device))
        elif bin_idx >= self.num_bins - 1:
            # Above last boundary: use last bin embedding
            return self.bin_embeddings(torch.tensor(self.num_bins - 1, device=device))

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

