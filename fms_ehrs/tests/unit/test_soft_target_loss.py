import unittest

import torch
import torch.nn as nn

from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
from fms_ehrs.framework.vocabulary import Vocabulary


class _DummyInner(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class _FixedLogitsCausalLM(nn.Module):
    """Minimal CausalLM-like module that returns fixed logits.

    Supports:
      - .config.hidden_size
      - .model.embed_tokens(input_ids)
      - forward(inputs_embeds=..., labels=None|tensor, return_dict=True)
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self.model = _DummyInner(vocab_size, hidden_size)
        self.dtype = torch.float32
        self.fixed_logits: torch.Tensor | None = None

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask=None,
        labels=None,
        return_dict: bool = True,
        **kwargs,
    ):
        if self.fixed_logits is None:
            raise ValueError("fixed_logits must be set by the test before forward().")
        return {"logits": self.fixed_logits, "loss": torch.tensor(0.0, device=inputs_embeds.device)}


class TestSoftTargetLoss(unittest.TestCase):
    def test_soft_target_reduces_loss_when_adjacent_bin_has_high_prob(self):
        # Vocab: code token + quantile tokens.
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        q0 = vocab("Q0")
        q1 = vocab("Q1")
        q2 = vocab("Q2")
        q3 = vocab("Q3")
        pad = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")

        # Soft discretization with 4 bins (3 boundaries).
        # Aux must be set before freezing the vocab.
        vocab.set_aux("CODE_A", [0.0, 1.0, 2.0])
        vocab.is_training = False

        base = _FixedLogitsCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            base.model.embed_tokens.weight.fill_(0.0)

        wrapper = RepresentationModelWrapper(
            base_model=base,
            vocab=vocab,
            representation="soft",
            temporal="time_tokens",
            num_bins=4,
        )

        # Sequence: CODE_A, Q1, PAD
        input_ids = torch.tensor([[code_id, q1, pad]], dtype=torch.long)
        labels = input_ids.clone()
        numeric_values = torch.tensor([[float("nan"), 0.5, float("nan")]], dtype=torch.float32)

        # Fixed logits: at position 0, strongly prefer Q0 over Q1.
        logits = torch.full((1, 3, len(vocab)), -2.0, dtype=torch.float32)
        logits[0, 0, q0] = 2.0
        logits[0, 0, q1] = 0.0
        base.fixed_logits = logits

        out = wrapper(input_ids=input_ids, numeric_values=numeric_values, labels=labels, return_dict=True)

        # Hard CE at step 0 would target label Q1 (at position 1).
        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        hard = nn.CrossEntropyLoss(reduction="none")(logits_next.reshape(-1, logits_next.size(-1)), labels_next.reshape(-1))
        hard_loss = hard.mean()

        # Soft-target should be strictly smaller because it assigns weight to Q0.
        self.assertLess(out["loss"].item(), hard_loss.item())


if __name__ == "__main__":
    unittest.main()

