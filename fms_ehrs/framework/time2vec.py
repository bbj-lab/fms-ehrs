"""
Time2Vec: Learned temporal embeddings for EHR sequences.

Reference:
    Kazemi, S. M., & Poupart, P. (2019). Time2Vec: Learning a Vector 
    Representation of Time. NeurIPS 2019.

This implementation encodes **relative time** (hours since admission) rather
than absolute timestamps. This design choice is grounded in MIMIC-IV's 
deidentification policy:

    "A single date shift was assigned to each subject_id. As a result, the 
    data for a single patient are internally consistent... Conversely, 
    distinct patients are not temporally comparable."
    
Encoding relative time preserves clinically meaningful temporal patterns
(e.g., deterioration over hours, medication timing) while avoiding spurious 
cross-patient correlations from arbitrary date shifts.

Time2Vec operates orthogonally to RoPE:
- RoPE: Encodes sequence position (token index) via rotation matrices
- Time2Vec: Encodes temporal semantics (hours since admission) via learned 
  periodic and linear components

Time2Vec is composed with token embeddings before transformer processing.
"""

import math
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Learnable time representation with periodic and linear components.
    
    Architecture:
        t2v(τ) = [ω₀·τ + φ₀, sin(ω₁·τ + φ₁), ..., sin(ωₖ·τ + φₖ)]
        
    Where:
        - τ: relative time (e.g., hours since admission)
        - ω: learnable frequency parameters
        - φ: learnable phase parameters
        - First component is linear, remaining are periodic (sinusoidal)
    
    The periodic components capture cyclical patterns (circadian rhythms,
    medication schedules), while the linear component captures trends.
    
    Args:
        embed_dim: Output embedding dimension
        num_periodic: Number of periodic (sinusoidal) components
                     Total dimension = 1 (linear) + num_periodic
        learnable_frequencies: If True, learn ω; else use fixed log-spaced
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_periodic: int | None = None,
        learnable_frequencies: bool = True,
        init_period_max_hours: float = 24.0,
        init_period_min_hours: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Default: all but one dimension are periodic
        if num_periodic is None:
            num_periodic = embed_dim - 1
        self.num_periodic = num_periodic
        
        # Total components: 1 linear + num_periodic sinusoidal
        total_components = 1 + num_periodic
        
        # Linear component parameters
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        
        # Periodic component parameters
        if learnable_frequencies:
            # Initialize angular frequencies ω with log-spacing across periods.
            #
            # In Eq. (Time2Vec), periodic components are sin(ω_i τ + φ_i), so the
            # period in the units of τ is (2π / ω_i). If τ is measured in hours,
            # a 24-hour period corresponds to ω = 2π/24 (not 1/24).
            period_max = float(init_period_max_hours)
            period_min = float(init_period_min_hours)
            if not (0.0 < period_min <= period_max):
                raise ValueError(
                    f"Expected 0 < init_period_min_hours <= init_period_max_hours, got {period_min} and {period_max}."
                )
            w_min = (2.0 * math.pi) / period_max
            w_max = (2.0 * math.pi) / period_min
            init_freqs = torch.exp(torch.linspace(math.log(w_min), math.log(w_max), num_periodic))
            self.periodic_weights = nn.Parameter(init_freqs)
        else:
            # Fixed log-spaced frequencies
            period_max = float(init_period_max_hours)
            period_min = float(init_period_min_hours)
            if not (0.0 < period_min <= period_max):
                raise ValueError(
                    f"Expected 0 < init_period_min_hours <= init_period_max_hours, got {period_min} and {period_max}."
                )
            w_min = (2.0 * math.pi) / period_max
            w_max = (2.0 * math.pi) / period_min
            self.register_buffer(
                'periodic_weights',
                torch.exp(torch.linspace(math.log(w_min), math.log(w_max), num_periodic))
            )
        
        self.periodic_biases = nn.Parameter(torch.zeros(num_periodic))
        
        # Projection to model hidden dimension if needed
        if total_components != embed_dim:
            self.projection = nn.Linear(total_components, embed_dim)
        else:
            self.projection = None
    
    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute time embeddings for relative timestamps.
        
        Args:
            tau: Relative time values in hours since admission.
                 Shape: (batch_size,) or (batch_size, seq_len)
                 
        Returns:
            Time embeddings of shape (batch_size, embed_dim) or 
            (batch_size, seq_len, embed_dim)
        """
        # Save original shape for output
        original_ndim = tau.dim()
        
        # Add final dimension for broadcasting with weights
        # (batch,) -> (batch, 1) or (batch, seq) -> (batch, seq, 1)
        tau = tau.unsqueeze(-1)
        
        # Linear component: ω₀·τ + φ₀
        # tau: (..., 1), weights: scalar -> (..., 1)
        linear_out = self.linear_weight * tau + self.linear_bias
        
        # Periodic components: sin(ωᵢ·τ + φᵢ)
        # tau: (..., 1), weights: (num_periodic,) -> (..., num_periodic)
        periodic_input = tau * self.periodic_weights + self.periodic_biases
        periodic_out = torch.sin(periodic_input)
        
        # Concatenate linear and periodic components
        # (..., 1) + (..., num_periodic) -> (..., 1 + num_periodic)
        time_embedding = torch.cat([linear_out, periodic_out], dim=-1)
        
        # Project to target dimension if needed
        if self.projection is not None:
            time_embedding = self.projection(time_embedding)
        
        # If original input was 1D (batch,), squeeze the middle dimension
        # (batch, 1, embed) -> (batch, embed)
        if original_ndim == 1:
            time_embedding = time_embedding.squeeze(-2)
        
        return time_embedding
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_periodic={self.num_periodic}, "
            f"learnable_frequencies={isinstance(self.periodic_weights, nn.Parameter)}"
        )


class Time2VecEmbedding(nn.Module):
    """
    Wrapper that combines Time2Vec with token embeddings.
    
    This module:
    1. Computes Time2Vec embeddings from relative timestamps
    2. Projects to model hidden dimension
    3. Composes with token embeddings (additive or concatenative)
    
    Args:
        hidden_size: Model hidden dimension (for projection)
        time2vec_dim: Internal Time2Vec dimension before projection
        num_periodic: Number of periodic components
    """
    
    def __init__(
        self,
        hidden_size: int,
        time2vec_dim: int = 64,
        num_periodic: int | None = None,
        compose: str = "add",
    ):
        super().__init__()
        
        self.time2vec = Time2Vec(
            embed_dim=time2vec_dim,
            num_periodic=num_periodic,
            learnable_frequencies=True,
        )
        
        # Project Time2Vec output to model hidden dimension
        self.projection = nn.Linear(time2vec_dim, hidden_size)

        if compose not in ("add", "concat"):
            raise ValueError(f"Unknown Time2VecEmbedding compose={compose!r}. Expected 'add' or 'concat'.")
        self.compose = compose

        # If concatenating, fuse (token || time) back to hidden_size.
        self.fuse = None
        if self.compose == "concat":
            self.fuse = nn.Linear(2 * hidden_size, hidden_size)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add temporal embeddings to token embeddings.
        
        Args:
            token_embeddings: Shape (batch_size, seq_len, hidden_size)
            timestamps: Relative time in hours. Shape (batch_size, seq_len)
            
        Returns:
            Temporally-enhanced embeddings (batch_size, seq_len, hidden_size)
        """
        # Compute Time2Vec embeddings
        time_emb = self.time2vec(timestamps)  # (batch, seq, time2vec_dim)
        
        # Project to hidden dimension
        time_emb = self.projection(time_emb)  # (batch, seq, hidden_size)
        
        if self.compose == "add":
            combined = token_embeddings + time_emb
        else:
            assert self.fuse is not None
            combined = self.fuse(torch.cat([token_embeddings, time_emb], dim=-1))
        
        # Normalize
        return self.layer_norm(combined)

