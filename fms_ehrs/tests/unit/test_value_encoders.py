import unittest

import torch

from fms_ehrs.framework.soft_discretization import SoftDiscretizationEncoder


class TestSoftDiscretizationEncoder(unittest.TestCase):
    def test_continuity_at_boundary(self):
        # 5 bins => 4 boundaries
        num_bins = 5
        embed_dim = 3
        enc = SoftDiscretizationEncoder(num_bins=num_bins, embed_dim=embed_dim)

        # Deterministic bin embeddings: embedding[i] = [i, i, i]
        with torch.no_grad():
            w = torch.arange(num_bins, dtype=torch.float32).unsqueeze(1).repeat(1, embed_dim)
            enc.bin_embeddings.weight.copy_(w)

        token_id_lookup = {"CODE_A": 10}
        vocab_aux = {"CODE_A": [0.0, 1.0, 2.0, 3.0]}
        enc.set_boundaries_from_vocab_aux(
            vocab_aux,
            token_id_lookup=token_id_lookup,
            vocab_size=32,
        )

        code_ids = torch.tensor([10, 10, 10], dtype=torch.long)
        v_left = torch.tensor([1.0], dtype=torch.float32)  # exactly at boundary b1
        v_right = torch.tensor([1.0 + 1e-6], dtype=torch.float32)

        # At boundary, embedding should equal bin 1 embedding exactly.
        e_at = enc(v_left, code_ids=code_ids[:1])[0]
        self.assertTrue(torch.allclose(e_at, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6))

        # Just above boundary, embedding should be very close to bin 1 as well.
        e_above = enc(v_right, code_ids=code_ids[:1])[0]
        self.assertTrue(torch.allclose(e_above, torch.tensor([1.0, 1.0, 1.0]), atol=1e-3))

    def test_vectorized_matches_reference(self):
        num_bins = 6
        embed_dim = 4
        enc = SoftDiscretizationEncoder(num_bins=num_bins, embed_dim=embed_dim)

        # Random but deterministic embeddings
        torch.manual_seed(0)
        with torch.no_grad():
            enc.bin_embeddings.weight.copy_(torch.randn_like(enc.bin_embeddings.weight))

        token_id_lookup = {"A": 3, "B": 7}
        vocab_aux = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],  # 5 boundaries for 6 bins
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        enc.set_boundaries_from_vocab_aux(
            vocab_aux,
            token_id_lookup=token_id_lookup,
            vocab_size=16,
        )

        values = torch.tensor([0.5, 4.5, 15.0, 49.0], dtype=torch.float32)
        code_ids = torch.tensor([3, 3, 7, 7], dtype=torch.long)

        out_fast = enc(values, code_ids=code_ids)

        # Build a reference using the string-keyed slow path
        enc_slow = SoftDiscretizationEncoder(num_bins=num_bins, embed_dim=embed_dim)
        with torch.no_grad():
            enc_slow.bin_embeddings.weight.copy_(enc.bin_embeddings.weight)
        enc_slow.set_boundaries_from_vocab_aux(vocab_aux)

        codes = ["A", "A", "B", "B"]
        out_slow = enc_slow(values, codes=codes)

        self.assertTrue(torch.allclose(out_fast, out_slow, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

