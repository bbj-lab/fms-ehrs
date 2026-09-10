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
            numeric_stats={"CODE_A": {"median": 0.0, "scale": 1.0}},
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

    def test_xval_excludes_nonfinite_or_masked_numeric_targets(self):
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
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "scale": 1.0}},
        )

        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        labels = input_ids.clone()
        labels[0, 1] = -100
        out = wrapper(
            input_ids=input_ids,
            numeric_values=torch.tensor([[float("nan"), float("inf"), float("nan")]]),
            labels=labels,
        )

        self.assertTrue(torch.allclose(base.last_inputs_embeds[0, 1], torch.ones(4)))
        self.assertTrue(torch.allclose(out["numeric_loss"], torch.tensor(0.0)))

    def test_multiplicative_injection_at_median_collapses_to_zero(self):
        """z = 0 scales the [NUM] row to the zero vector."""
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        num_id = vocab("[NUM]")
        pad_id = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 5.0, float("nan")]])
        stats = {"CODE_A": {"median": 5.0, "scale": 1.0}}

        base_mul = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            base_mul.model.embed_tokens.weight.normal_(0.0, 0.02)
        mul = XValModelWrapper(
            base_model=base_mul,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats=stats,
        )
        mul(input_ids=input_ids, numeric_values=numeric_values)
        self.assertTrue(
            torch.allclose(base_mul.last_inputs_embeds[0, 1], torch.zeros(4), atol=1e-6)
        )

    def test_affine_bias_is_seeded_from_the_num_embedding_row(self):
        """At z=0 the affine variant should decode to the model's [NUM] row.

        A random bias would make a median-valued measurement land on an
        arbitrary direction, and would make initialization a second difference
        between the 'mul' and 'affine' arms.
        """
        vocab = Vocabulary()
        vocab("CODE_A")
        num_id = vocab("[NUM]")
        vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            base.model.embed_tokens.weight.normal_(0.0, 0.02)

        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "scale": 1.0}},
            numeric_injection="affine",
        )

        self.assertTrue(
            torch.allclose(
                wrapper.num_bias, base.model.embed_tokens.weight[num_id]
            )
        )

    def test_affine_injection_at_median_value_recovers_the_num_embedding(self):
        """z = 0 must inject e_NUM, not the zero vector that 'mul' collapses to."""
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        num_id = vocab("[NUM]")
        pad_id = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        # Value exactly at the code's median => z = 0.
        numeric_values = torch.tensor([[float("nan"), 5.0, float("nan")]])
        stats = {"CODE_A": {"median": 5.0, "scale": 1.0}}

        base_affine = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            base_affine.model.embed_tokens.weight.normal_(0.0, 0.02)
        affine = XValModelWrapper(
            base_model=base_affine,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats=stats,
            numeric_injection="affine",
        )
        affine(input_ids=input_ids, numeric_values=numeric_values)
        num_row = base_affine.model.embed_tokens.weight[num_id]
        self.assertTrue(
            torch.allclose(base_affine.last_inputs_embeds[0, 1], num_row, atol=1e-6)
        )

    def test_trained_affine_bias_is_used_instead_of_the_embedding_seed(self):
        """Extraction must apply z * e_NUM + b_trained, not b = e_NUM."""
        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        num_id = vocab("[NUM]")
        pad_id = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 7.0, float("nan")]])
        stats = {"CODE_A": {"median": 5.0, "scale": 1.0}}  # z = 2.0

        base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            base.model.embed_tokens.weight.fill_(1.0)
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats=stats,
            numeric_injection="affine",
        )
        trained_bias = torch.full((4,), 3.0)
        with torch.no_grad():
            wrapper.num_bias.copy_(trained_bias)

        wrapper(input_ids=input_ids, numeric_values=numeric_values)
        # z * e_NUM + b_trained = 2 * 1 + 3 = 5
        self.assertTrue(
            torch.allclose(base.last_inputs_embeds[0, 1], torch.full((4,), 5.0))
        )

    def test_persist_and_reload_restores_trained_affine_bias(self):
        from fms_ehrs.framework.xval import apply_xval_mechanics, persist_xval_mechanics

        vocab = Vocabulary()
        code_id = vocab("CODE_A")
        num_id = vocab("[NUM]")
        pad_id = vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        stats = {"CODE_A": {"median": 5.0, "scale": 1.0}}
        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 5.0, float("nan")]])

        trained = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        restored_base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        with torch.no_grad():
            trained.model.embed_tokens.weight.fill_(1.0)
            restored_base.model.embed_tokens.weight.fill_(1.0)

        source = XValModelWrapper(
            base_model=trained,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats=stats,
            numeric_injection="affine",
        )
        trained_bias = torch.arange(4, dtype=torch.float32)
        with torch.no_grad():
            source.num_bias.copy_(trained_bias)
            source.number_head.weight.fill_(0.5)
            source.number_head.bias.fill_(-0.25)

        mechanics = persist_xval_mechanics(source)
        self.assertIn("num_bias", mechanics)
        self.assertTrue(torch.allclose(mechanics["num_bias"], trained_bias))

        restored = XValModelWrapper(
            base_model=restored_base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats=stats,
            numeric_injection="affine",
        )
        apply_xval_mechanics(restored, mechanics)
        self.assertTrue(torch.allclose(restored.num_bias, trained_bias))
        restored(input_ids=input_ids, numeric_values=numeric_values)
        # z = 0, so the injected vector is b_trained, not e_NUM.
        self.assertTrue(
            torch.allclose(restored_base.last_inputs_embeds[0, 1], trained_bias)
        )

    def test_affine_reload_rejects_missing_num_bias(self):
        from fms_ehrs.framework.xval import apply_xval_mechanics

        vocab = Vocabulary()
        vocab("CODE_A")
        vocab("[NUM]")
        vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "scale": 1.0}},
            numeric_injection="affine",
        )
        with self.assertRaisesRegex(RuntimeError, "num_bias"):
            apply_xval_mechanics(
                wrapper, {"number_head_state": wrapper.number_head.state_dict()}
            )

    def test_mul_persist_omits_num_bias(self):
        from fms_ehrs.framework.xval import persist_xval_mechanics

        vocab = Vocabulary()
        vocab("CODE_A")
        vocab("[NUM]")
        vocab("PAD")
        vocab("TL_START")
        vocab("TL_END")
        vocab.is_training = False

        base = _DummyCausalLM(vocab_size=len(vocab), hidden_size=4)
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "scale": 1.0}},
        )
        mechanics = persist_xval_mechanics(wrapper)
        self.assertNotIn("num_bias", mechanics)
        self.assertIn("number_head_state", mechanics)

    def test_xval_numeric_loss_is_finite_when_every_target_is_label_masked(self):
        """An eligible [NUM] whose label is ignored must not yield a NaN loss.

        The eligibility mask is computed before the causal shift and the
        `labels != -100` filter, so it can be non-empty while the scored
        selection is empty. Averaging over an empty selection returns NaN and
        would poison the total loss.
        """
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
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "scale": 1.0}},
            numeric_loss_weight=1.0,
        )

        # A finite value at a [NUM] position with per-code stats, so the
        # pre-shift eligibility mask is non-empty, but its label is ignored.
        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 2.0, float("nan")]])
        labels = input_ids.clone()
        labels[0, 1] = -100

        out = wrapper(
            input_ids=input_ids, numeric_values=numeric_values, labels=labels
        )

        self.assertTrue(torch.isfinite(out["numeric_loss"]))
        self.assertTrue(torch.allclose(out["numeric_loss"], torch.tensor(0.0)))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_legacy_std_key_is_read_as_iqr_scale(self):
        """Version-1 numeric_stats.json stored IQR/1.35 under the key 'std'."""
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
        wrapper = XValModelWrapper(
            base_model=base,
            vocab=vocab,
            temporal="time_tokens",
            numeric_stats={"CODE_A": {"median": 0.0, "std": 1.0}},
        )

        input_ids = torch.tensor([[code_id, num_id, pad_id]], dtype=torch.long)
        numeric_values = torch.tensor([[float("nan"), 2.0, float("nan")]])
        wrapper(input_ids=input_ids, numeric_values=numeric_values)

        self.assertTrue(
            torch.allclose(base.last_inputs_embeds[0, 1], torch.full((4,), 2.0))
        )


if __name__ == "__main__":
    unittest.main()

