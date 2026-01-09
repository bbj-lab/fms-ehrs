import unittest

import torch
import torch.nn as nn

from fms_ehrs.framework.model_wrapper import RepresentationModelWrapper


class _DummyVocab:
    """Minimal vocab stub for RepresentationModelWrapper tests."""

    def __init__(self, size: int):
        self.lookup = {"PAD": 0, "TL_START": 1, "TL_END": 2, "Q0": 3}
        self.aux = {}
        self._size = size

    def __len__(self) -> int:
        return self._size


class _DummyEmbedHost(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        emb = nn.Embedding(vocab_size, hidden_size)
        self.embed_tokens = emb.to(dtype=dtype)


class _DummyBaseModel(nn.Module):
    """Base-model stub that records inputs_embeds dtype."""

    def __init__(self, vocab_size: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self.model = _DummyEmbedHost(vocab_size=vocab_size, hidden_size=hidden_size, dtype=dtype)
        self.last_inputs_dtype = None

    def forward(self, *, inputs_embeds, **kwargs):
        self.last_inputs_dtype = inputs_embeds.dtype
        # Return a minimal HF-like output dict
        return {"loss": inputs_embeds.sum() * 0.0}


class TestRepresentationModelWrapperDType(unittest.TestCase):
    def test_time2vec_does_not_upcast_inputs_embeds_against_base_dtype(self):
        # Base model initialized in bf16 (common on A100)
        base_dtype = torch.bfloat16
        vocab_size = 64
        hidden_size = 32

        base_model = _DummyBaseModel(vocab_size=vocab_size, hidden_size=hidden_size, dtype=base_dtype)
        wrapper = RepresentationModelWrapper(
            base_model=base_model,
            vocab=_DummyVocab(size=vocab_size),
            representation="discrete",
            temporal="time2vec",
            num_bins=20,
            time2vec_dim=8,
        )

        # Input ids on CPU; embeddings start as bf16, but Time2Vec path produces float32 intermediates.
        input_ids = torch.tensor([[1, 3, 2, 0]], dtype=torch.long)  # TL_START, Q0, TL_END, PAD
        attention_mask = (input_ids != 0).long()
        relative_times = torch.tensor([[0.0, 1.0, 2.0, 0.0]], dtype=torch.float32)

        _ = wrapper(input_ids=input_ids, attention_mask=attention_mask, relative_times=relative_times, labels=input_ids)
        self.assertEqual(base_model.last_inputs_dtype, base_dtype)


if __name__ == "__main__":
    unittest.main()

