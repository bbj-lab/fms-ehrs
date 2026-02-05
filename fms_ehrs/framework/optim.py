from __future__ import annotations

import dataclasses
import typing

import torch


class CombinedOptimizer(torch.optim.Optimizer):
    """A thin wrapper that steps multiple optimizers as one.

    HuggingFace Trainer expects a single optimizer object. Muon is typically used
    only for 2D "hidden layer" weight matrices, while embeddings, heads, and 1D
    parameters are better handled by AdamW. This wrapper exposes a single
    `Optimizer` interface while delegating to two underlying optimizers.
    """

    def __init__(self, *, muon: torch.optim.Optimizer, aux: torch.optim.Optimizer):
        # NOTE: torch.optim.Optimizer requires `param_groups` and `defaults`.
        # We expose a synthetic view by concatenating param groups. The real state
        # lives inside the child optimizers.
        self.muon = muon
        self.aux = aux
        param_groups = list(muon.param_groups) + list(aux.param_groups)
        super().__init__(param_groups, defaults={})

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        return f"CombinedOptimizer(muon={self.muon!r}, aux={self.aux!r})"

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.aux.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: typing.Callable | None = None):
        # Respect closure semantics if provided (rare in our code path).
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.aux.step()
        return loss

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "aux": self.aux.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.aux.load_state_dict(state_dict["aux"])


@dataclasses.dataclass(frozen=True)
class MuonConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    eps: float = 1e-7


@dataclasses.dataclass(frozen=True)
class AdamWConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # DDP wraps the underlying module in `.module`.
    return model.module if hasattr(model, "module") else model


def _is_embedding_or_head_param(name: str) -> bool:
    # Keep this intentionally conservative: avoid Muon on embeddings and output heads.
    # This covers HF causal LMs (LLaMA/GPT2-like), and our xVal numeric head.
    n = name.lower()
    return any(
        k in n
        for k in (
            "embed_tokens",
            ".wte",  # GPT-2 style token embedding
            "lm_head",
            "output_layer",
            "classifier",
            "number_head",
        )
    )


def split_named_parameters_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[tuple[str, torch.Tensor]]]:
    """Split model params into (muon_params, aux_named_params).

    - Muon: 2D parameters excluding embeddings/heads.
    - Aux (AdamW): everything else (including embeddings, heads, 1D params).

    We return aux as (name, tensor) to enable standard AdamW no-decay filtering.
    """
    model = _unwrap_model(model)
    muon_params: list[torch.Tensor] = []
    aux_named: list[tuple[str, torch.Tensor]] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Only use Muon on 2D parameters that are not embeddings/heads.
        if p.ndim == 2 and not _is_embedding_or_head_param(name):
            muon_params.append(p)
        else:
            aux_named.append((name, p))

    return muon_params, aux_named


def build_muon_with_aux_adamw(
    *,
    model: torch.nn.Module,
    muon: MuonConfig,
    adamw: AdamWConfig,
) -> CombinedOptimizer:
    """Create a Muon(+AdamW) optimizer for transformer-style models.

    Muon is applied to 2D hidden-layer weight matrices. AdamW is applied to:
    - embeddings and heads (even though they are 2D)
    - 1D parameters (biases, LayerNorm weights, etc.)
    - any other non-2D parameters
    """
    muon_params, aux_named = split_named_parameters_for_muon(model)

    if not muon_params:
        raise ValueError("Muon optimizer requested, but no eligible 2D hidden-layer parameters were found.")

    # AdamW param grouping: no decay for biases and layer norms (standard convention).
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, p in aux_named:
        if p.ndim < 2 or name.endswith(".bias") or "layernorm" in name.lower() or ".ln_" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    muon_optim = torch.optim.Muon(
        muon_params,
        lr=float(muon.lr),
        weight_decay=float(muon.weight_decay),
        momentum=float(muon.momentum),
        nesterov=bool(muon.nesterov),
        ns_steps=int(muon.ns_steps),
        eps=float(muon.eps),
    )
    aux_groups: list[dict] = []
    if decay:
        aux_groups.append({"params": decay, "weight_decay": float(adamw.weight_decay)})
    if no_decay:
        aux_groups.append({"params": no_decay, "weight_decay": 0.0})
    aux_optim = torch.optim.AdamW(
        aux_groups,
        lr=float(adamw.lr),
        betas=tuple(map(float, adamw.betas)),
        eps=float(adamw.eps),
    )

    return CombinedOptimizer(muon=muon_optim, aux=aux_optim)

