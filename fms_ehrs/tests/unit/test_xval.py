import unittest

import torch
import torch.nn as nn

from fms_ehrs.framework.vocabulary import Vocabulary
from fms_ehrs.framework.xval import XValModelWrapper


class _DummyInner(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class _DummyCausalLM(nn.Module):
    """Minimal CausalLM-like module for testing wrappers.

    Supports:
      - .config.hidden_size
      - .model.embed_tokens(input_ids)
      - forward(inputs_embeds=..., labels=..., output_hidden_states=True, return_dict=True)
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self.model = _DummyInner(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dtype = torch.float32
        self.last_inputs_embeds = None

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask=None,
        labels=None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        self.last_inputs_embeds = inputs_embeds.detach().clone()
        hidden = inputs_embeds
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            # Standard causal LM loss shape: (batch, seq, vocab) vs (batch, seq)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        out = {
            "loss": loss if loss is not None else torch.tensor(0.0, device=logits.device),
            "logits": logits,
        }
        if output_hidden_states:
            out["hidden_states"] = (hidden,)
        return out


class TestXValModelWrapper(unittest.TestCase):
    def test_xval_scales_num_embedding_and_computes_numeric_loss(self):
        # Build minimal vocab with required tokens.
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        num_id = vocab("[NUM]")
        pad_id = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)

        # Make embeddings deterministic: all ones.
        with torch.no_grad():
            base.model.embed_tokens.weight.fill_(1.0)
            base.lm_head.weight.fill_(0.0)  # neutralize token loss dependence

        # Make number head deterministic: predict 0 everywhere.
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "std": 1.0}},
            numeric_loss_weight=1.0,
        )
        with torch.no_grad():
            wrapper.number_head.weight.zero_()
            wrapper.number_head.bias.zero_()

        input_ids = torch.tensor([[code_id, num_id, pad_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 2.0, float("nan"), float("nan")]], dtype=torch.float32)
        labels = input_ids.clone()

        out = wrapper(input_ids=input_ids, numeric_values=numeric_values, labels=labels)

        # [NUM] embedding should be scaled by normalized value (=2.0).
        # Base embed is ones -> scaled becomes 2s at position 1.
        self.assertTrue(torch.allclose(base.last_inputs_embeds[0, 1], torch.full((4,), 2.0)))

        # Numeric loss: pred=0, target=2 => MSE = 4
        self.assertTrue(torch.allclose(out["numeric_loss"], torch.tensor(4.0)))

    def test_xval_missing_stats_skips_injection_and_numeric_loss(self):
        # Build minimal vocab with required tokens.
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        num_id = vocab("[NUM]")
        pad_id = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            base.model.embed_tokens.weight.fill_(1.0)
            base.lm_head.weight.fill_(0.0)

        # No stats for CODE_A -> wrapper should skip injection.
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={},  # missing per-code stats
            numeric_loss_weight=1.0,
        )
        with torch.no_grad():
            wrapper.number_head.weight.zero_()
            wrapper.number_head.bias.zero_()

        input_ids = torch.tensor([[code_id, num_id, pad_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 2.0, float("nan"), float("nan")]], dtype=torch.float32)
        labels = input_ids.clone()

        out = wrapper(input_ids=input_ids, numeric_values=numeric_values, labels=labels)

        # [NUM] embedding should be unchanged (base embed is ones).
        self.assertTrue(torch.allclose(base.last_inputs_embeds[0, 1], torch.full((4,), 1.0)))

        # Numeric loss should be 0 because we exclude missing-stat positions.
        self.assertTrue(torch.allclose(out["numeric_loss"], torch.tensor(0.0)))


if __name__ == "__main__":
    unittest.main()

