import unittest

import torch
import torch.nn as nn

from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper
from fms_ehrs.framework.soft_discretization import SoftDiscretizationEncoder
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

    def test_top_bin_soft_target_is_the_last_quantile_token(self):
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        q_ids = [vocab(f"Q{k}") for k in range(4)]
        pad = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.set_aux("CODE_A", [0.0, 1.0, 2.0])
        vocab.is_training = False

        base = _FixedLogitsCausalLM(vocab_size=len(vocab), hidden_size=4)
        wrapper = RepresentationModelWrapper(
            base_model=base,
            vocab=vocab,
            representation="soft",
            temporal="time_tokens",
            num_bins=4,
        )

        input_ids = torch.tensor([[code_id, q_ids[3], pad]], dtype=torch.long)
        labels = input_ids.clone()
        # Above the last boundary: encoder uses E_{K-1}; loss must target Q3, not Q2.
        numeric_values = torch.tensor([[float("nan"), 3.0, float("nan")]], dtype=torch.float32)
        logits = torch.full((1, 3, len(vocab)), -4.0, dtype=torch.float32)
        logits[0, 0, q_ids[3]] = 3.0
        base.fixed_logits = logits

        out = wrapper(
            input_ids=input_ids,
            numeric_values=numeric_values,
            labels=labels,
            return_dict=True,
        )
        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        hard = nn.CrossEntropyLoss(reduction="none")(
            logits_next.reshape(-1, logits_next.size(-1)),
            labels_next.reshape(-1),
        )
        self.assertAlmostEqual(out["loss"].item(), hard.mean().item(), places=5)

    def test_soft_discretization_rejects_invalid_boundaries(self):
        encoder = SoftDiscretizationEncoder(num_bins=3, embed_dim=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            encoder.set_boundaries("CODE_A", torch.tensor([0.0, float("nan")]))
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            encoder.set_boundaries("CODE_A", torch.tensor([2.0, 1.0]))

    def test_bin_embeddings_are_seeded_from_the_models_quantile_rows(self):
        """Soft bin embeddings must start at the backbone's embedding scale.

        A bare nn.Embedding initializes from N(0, 1) while the backbones use
        initializer_range=0.02, so an unseeded table would start ~50x wider than
        every other token embedding in the sequence.
        """
        vocab = Vocabulary()
        vocab("CODE_A")
        quantile_ids = [vocab(f"Q{k}") for k in range(4)]
        vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.set_aux("CODE_A", [0.0, 1.0, 2.0])
        vocab.is_training = False

        base = _FixedLogitsCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            # Give each quantile row a distinct, small-scale signature.
            for bin_index, token_id in enumerate(quantile_ids):
                base.model.embed_tokens.weight[token_id] = 0.02 * (bin_index + 1)

        wrapper = RepresentationModelWrapper(
            base_model=base,
            vocab=vocab,
            representation="soft",
            temporal="time_tokens",
            num_bins=4,
        )

        seeded = wrapper.value_encoder.bin_embeddings.weight
        expected = base.model.embed_tokens.weight[torch.tensor(quantile_ids)]
        self.assertTrue(torch.allclose(seeded, expected))


if __name__ == "__main__":
    unittest.main()

