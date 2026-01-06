#!/usr/bin/env python3

"""
Learned continuous encoder for numeric values.

This module implements MLP-based continuous value encoding inspired by the xVal
approach for representing numeric values in transformers.

References
----------
[1] Golkar, S., Pettee, M., Eickenberg, M., et al. (2023). xVal: A Continuous
    Number Encoding for Large Language Models. arXiv:2310.02989.

The encoder projects z-score normalized values through a small MLP to produce
embeddings that preserve numeric relationships in the embedding space.
"""

import typing

import numpy as np
import torch
import torch.nn as nn


class ContinuousValueEncoder(nn.Module):
    """Encode continuous values via learned MLP projection.

    This encoder normalizes numeric values using per-code z-score normalization
    and projects them through a 2-layer MLP to produce embeddings.

    Parameters
    ----------
    embed_dim : int
        Dimension of output embeddings
    hidden_dim : int, optional
        Hidden dimension of MLP. Defaults to embed_dim.
    activation : str
        Activation function: "gelu" or "relu"
    clip_sigma : float
        Clip normalized values to +/- this many standard deviations

    Example
    -------
    >>> encoder = ContinuousValueEncoder(embed_dim=768)
    >>> encoder.set_statistics("LAB_glucose", mean=100.0, std=25.0)
    >>> values = torch.tensor([85.0, 100.0, 125.0])
    >>> embeddings = encoder(values, codes=["LAB_glucose"] * 3)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        activation: typing.Literal["gelu", "relu"] = "gelu",
        clip_sigma: float = 5.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.clip_sigma = clip_sigma

        # 2-layer MLP: value -> hidden -> embed
        self.mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(self.hidden_dim, embed_dim),
        )

        # Per-code normalization statistics (not learnable)
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

        # Optional fast path: stats indexed by code token-id.
        # These are derived from a (code string -> token id) lookup and registered
        # as buffers for device placement. They are not persisted in checkpoints
        # (they can be reconstructed from vocab.aux at load time).
        self.register_buffer(
            "means_by_id",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "stds_by_id",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "has_stats_by_id",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )

    def set_statistics(self, code: str, mean: float, std: float) -> None:
        """Set normalization statistics for a specific code.

        Parameters
        ----------
        code : str
            Code identifier (e.g., "LAB_glucose")
        mean : float
            Mean value for z-score normalization
        std : float
            Standard deviation for z-score normalization
        """
        self._means[code] = mean
        self._stds[code] = max(std, 1e-8)  # Avoid division by zero

    def set_statistics_from_vocab_aux(
        self,
        vocab_aux: dict[str, list[float]],
        *,
        token_id_lookup: dict[str, int] | None = None,
        vocab_size: int | None = None,
    ) -> None:
        """Estimate statistics from vocabulary auxiliary data (quantile breaks).

        Uses the median and IQR of the breaks as robust estimates of mean and std.

        Parameters
        ----------
        vocab_aux : dict
            Dictionary mapping code names to list of quantile boundaries
        token_id_lookup : dict[str, int], optional
            Mapping from code string to its vocabulary token id. If provided,
            a fast vectorized path will be enabled via `code_ids` in `forward()`.
        vocab_size : int, optional
            Vocabulary size used to size the stats tensors. Required if
            `token_id_lookup` is provided.
        """
        if token_id_lookup is not None and vocab_size is None:
            raise ValueError("vocab_size must be provided when token_id_lookup is used")

        if token_id_lookup is not None:
            means_by_id = torch.zeros((vocab_size,), dtype=torch.float32)
            stds_by_id = torch.ones((vocab_size,), dtype=torch.float32)
            has_by_id = torch.zeros((vocab_size,), dtype=torch.bool)

        for code, breaks in vocab_aux.items():
            if len(breaks) == 0:
                continue
            breaks_arr = np.array(breaks)
            # Use median of breaks as estimate of mean
            mean = float(np.median(breaks_arr))
            # Use IQR / 1.35 as robust estimate of std (for normal distribution)
            q25_idx = len(breaks_arr) // 4
            q75_idx = 3 * len(breaks_arr) // 4
            if q75_idx > q25_idx:
                iqr = breaks_arr[q75_idx] - breaks_arr[q25_idx]
                std = float(iqr / 1.35)
            else:
                std = float(np.std(breaks_arr))
            self.set_statistics(code, mean, std)

            if token_id_lookup is not None:
                tok_id = token_id_lookup.get(code)
                if tok_id is not None:
                    means_by_id[tok_id] = mean
                    stds_by_id[tok_id] = max(std, 1e-8)
                    has_by_id[tok_id] = True

        if token_id_lookup is not None:
            self.means_by_id = means_by_id
            self.stds_by_id = stds_by_id
            self.has_stats_by_id = has_by_id

    def normalize(self, values: torch.Tensor, codes: list[str]) -> torch.Tensor:
        """Z-score normalize values using per-code statistics.

        Parameters
        ----------
        values : torch.Tensor
            Raw numeric values of shape (batch_size,)
        codes : list[str]
            Code identifiers for each value

        Returns
        -------
        torch.Tensor
            Normalized values, clipped to +/- clip_sigma
        """
        normalized = torch.zeros_like(values)

        for i, (value, code) in enumerate(zip(values, codes)):
            if code in self._means:
                mean = self._means[code]
                std = self._stds[code]
                z = (value.item() - mean) / std
                z = max(-self.clip_sigma, min(self.clip_sigma, z))
                normalized[i] = z
            else:
                # If statistics not available, use value as-is
                normalized[i] = value

        return normalized

    def forward(
        self,
        values: torch.Tensor,
        codes: list[str] | None = None,
        *,
        code_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode continuous values to embeddings.

        Parameters
        ----------
        values : torch.Tensor
            Numeric values of shape (batch_size,) or (batch_size, 1)
        codes : list[str], optional
            Code identifiers for each value, length batch_size (string-keyed mode)
        code_ids : torch.Tensor, optional
            Code token ids for each value, shape (batch_size,) (fast mode)

        Returns
        -------
        torch.Tensor
            Continuous embeddings of shape (batch_size, embed_dim)
        """
        if values.dim() == 2:
            values = values.squeeze(-1)

        if code_ids is not None and self.means_by_id.numel() > 0:
            normalized = self._normalize_code_ids(values, code_ids)
        else:
            if codes is None:
                raise ValueError("Must provide either `codes` or `code_ids`.")
            normalized = self.normalize(values, codes)

        # Project through MLP
        # Shape: (batch_size, 1) -> (batch_size, embed_dim)
        embeddings = self.mlp(normalized.unsqueeze(-1))

        return embeddings

    def _normalize_code_ids(self, values: torch.Tensor, code_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized normalization for code token ids."""
        if code_ids.dim() != 1:
            raise ValueError("code_ids must be a 1D tensor of token ids")
        if values.shape[0] != code_ids.shape[0]:
            raise ValueError("values and code_ids must have matching length")

        device = values.device
        code_ids = code_ids.to(device=device, dtype=torch.long)

        if self.means_by_id.device != device:
            # Lazily move buffers if needed.
            self.means_by_id = self.means_by_id.to(device)
            self.stds_by_id = self.stds_by_id.to(device)
            self.has_stats_by_id = self.has_stats_by_id.to(device)

        means = self.means_by_id[code_ids]
        stds = self.stds_by_id[code_ids]
        has = self.has_stats_by_id[code_ids]

        # Match legacy behavior: if stats missing, use value as-is (no clipping).
        z = (values - means) / stds
        z = torch.clamp(z, -self.clip_sigma, self.clip_sigma)
        return torch.where(has, z, values)


class ContinuousValueLayer(nn.Module):
    """Layer that applies continuous encoding to numeric token values.

    This layer is designed to be used within a transformer model, replacing
    the discrete quantile token lookup with learned continuous encoding.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    hidden_dim : int, optional
        Hidden dimension of MLP
    """

    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        self.encoder = ContinuousValueEncoder(embed_dim, hidden_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        token_embeddings: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        codes: list[list[str]],
    ) -> torch.Tensor:
        """Apply continuous encoding to numeric tokens.

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
            Modified embeddings with continuous encoding for numeric tokens
        """
        batch_size, seq_len, embed_dim = token_embeddings.shape
        output = token_embeddings.clone()

        for b in range(batch_size):
            for s in range(seq_len):
                if numeric_mask[b, s]:
                    value = numeric_values[b, s]
                    code = codes[b][s]
                    cont_embed = self.encoder(
                        value.unsqueeze(0), [code]
                    ).squeeze(0)
                    output[b, s] = cont_embed

        return output


class CodeAwareContinuousEncoder(nn.Module):
    """Continuous encoder with code-specific projection heads.

    This encoder uses a shared backbone MLP but adds code-specific
    linear layers to allow different codes to have different learned
    transformations while sharing most parameters.

    Parameters
    ----------
    embed_dim : int
        Dimension of output embeddings
    hidden_dim : int, optional
        Hidden dimension of shared backbone
    num_code_heads : int
        Number of distinct code projection heads
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        num_code_heads: int = 100,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.num_code_heads = num_code_heads

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
        )

        # Code-specific projection heads
        self.code_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, embed_dim)
            for _ in range(num_code_heads)
        ])

        # Default head for unknown codes
        self.default_head = nn.Linear(self.hidden_dim, embed_dim)

        # Code to head index mapping
        self._code_to_head: dict[str, int] = {}

        # Normalization statistics
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def register_code(self, code: str, head_idx: int = None) -> int:
        """Register a code and assign it a projection head.

        Parameters
        ----------
        code : str
            Code identifier
        head_idx : int, optional
            Specific head index to use. If None, assigns next available.

        Returns
        -------
        int
            Assigned head index
        """
        if code in self._code_to_head:
            return self._code_to_head[code]

        if head_idx is None:
            head_idx = len(self._code_to_head) % self.num_code_heads

        self._code_to_head[code] = head_idx
        return head_idx

    def set_statistics(self, code: str, mean: float, std: float) -> None:
        """Set normalization statistics for a code."""
        self._means[code] = mean
        self._stds[code] = max(std, 1e-8)
        self.register_code(code)

    def forward(
        self,
        values: torch.Tensor,
        codes: list[str],
    ) -> torch.Tensor:
        """Encode values with code-specific projections.

        Parameters
        ----------
        values : torch.Tensor
            Numeric values of shape (batch_size,)
        codes : list[str]
            Code identifiers

        Returns
        -------
        torch.Tensor
            Embeddings of shape (batch_size, embed_dim)
        """
        if values.dim() == 2:
            values = values.squeeze(-1)

        batch_size = values.shape[0]
        device = values.device
        embeddings = torch.zeros(batch_size, self.embed_dim, device=device)

        for i, (value, code) in enumerate(zip(values, codes)):
            # Normalize
            if code in self._means:
                z = (value.item() - self._means[code]) / self._stds[code]
                z = max(-5.0, min(5.0, z))
            else:
                z = value.item()

            # Backbone
            hidden = self.backbone(torch.tensor([[z]], device=device))

            # Code-specific head
            if code in self._code_to_head:
                head_idx = self._code_to_head[code]
                embedding = self.code_heads[head_idx](hidden)
            else:
                embedding = self.default_head(hidden)

            embeddings[i] = embedding.squeeze(0)

        return embeddings


def create_continuous_encoder_from_tokenizer(
    tokenizer, embed_dim: int, hidden_dim: int = None
) -> ContinuousValueEncoder:
    """Create a ContinuousValueEncoder from a trained tokenizer.

    Parameters
    ----------
    tokenizer : BaseTokenizer
        Trained tokenizer with quantile auxiliary data
    embed_dim : int
        Embedding dimension
    hidden_dim : int, optional
        Hidden dimension of MLP

    Returns
    -------
    ContinuousValueEncoder
        Encoder initialized with statistics from tokenizer
    """
    encoder = ContinuousValueEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
    encoder.set_statistics_from_vocab_aux(tokenizer.vocab.aux)
    return encoder


if __name__ == "__main__":
    # Test continuous encoder
    torch.manual_seed(42)

    # Create encoder
    encoder = ContinuousValueEncoder(embed_dim=64)

    # Set up statistics for test code (glucose-like)
    encoder.set_statistics("LAB_glucose", mean=100.0, std=25.0)

    # Test single value
    value = torch.tensor([125.0])
    embedding = encoder(value, codes=["LAB_glucose"])
    print(f"Single value embedding shape: {embedding.shape}")

    # Test batch
    values = torch.tensor([60.0, 100.0, 150.0, 180.0])
    codes = ["LAB_glucose"] * 4
    embeddings = encoder(values, codes=codes)
    print(f"Batch embedding shape: {embeddings.shape}")

    # Verify that the MLP is learning something meaningful
    # Close values should have similar embeddings (after training)
    v1 = encoder(torch.tensor([100.0]), codes=["LAB_glucose"])
    v2 = encoder(torch.tensor([101.0]), codes=["LAB_glucose"])
    v3 = encoder(torch.tensor([150.0]), codes=["LAB_glucose"])

    dist_12 = torch.norm(v1 - v2).item()
    dist_13 = torch.norm(v1 - v3).item()
    print(f"Distance between 100 and 101: {dist_12:.4f}")
    print(f"Distance between 100 and 150: {dist_13:.4f}")

    # Test code-aware encoder
    print("\nTesting CodeAwareContinuousEncoder...")
    aware_encoder = CodeAwareContinuousEncoder(embed_dim=64, num_code_heads=10)
    aware_encoder.set_statistics("LAB_glucose", mean=100.0, std=25.0)
    aware_encoder.set_statistics("LAB_creatinine", mean=1.0, std=0.3)

    values = torch.tensor([100.0, 1.2])
    codes = ["LAB_glucose", "LAB_creatinine"]
    embeddings = aware_encoder(values, codes=codes)
    print(f"Code-aware embeddings shape: {embeddings.shape}")

