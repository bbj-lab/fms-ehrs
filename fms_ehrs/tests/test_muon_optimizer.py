import pytest
import torch

from fms_ehrs.framework.optim import AdamWConfig, MuonConfig, build_muon_with_aux_adamw, split_named_parameters_for_muon


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon not available in this torch build")
def test_split_named_parameters_for_muon_excludes_embeddings_and_heads():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Match HF naming conventions used by our splitter.
            self.embed_tokens = torch.nn.Embedding(16, 8)
            self.proj = torch.nn.Linear(8, 8, bias=True)  # 2D weight should be Muon
            self.lm_head = torch.nn.Linear(8, 16, bias=False)  # head should be aux
            self.number_head = torch.nn.Linear(8, 1, bias=True)  # xVal head should be aux

        def forward(self, x):
            h = self.embed_tokens(x)
            h = self.proj(h)
            return self.lm_head(h).sum()

    m = Dummy()
    muon_params, aux_named = split_named_parameters_for_muon(m)
    aux_names = {n for (n, _) in aux_named}

    # Muon should include proj.weight only.
    assert any(p is m.proj.weight for p in muon_params)
    assert all(p is not m.embed_tokens.weight for p in muon_params)
    assert all(p is not m.lm_head.weight for p in muon_params)
    assert all(p is not m.number_head.weight for p in muon_params)

    # Embeddings/heads must remain in aux.
    assert any("embed_tokens" in n for n in aux_names)
    assert any("lm_head" in n for n in aux_names)
    assert any("number_head" in n for n in aux_names)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon not available in this torch build")
def test_build_muon_with_aux_adamw_steps():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 8)
            self.proj = torch.nn.Linear(8, 8, bias=True)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, x):
            h = self.embed_tokens(x)
            h = self.proj(h)
            return self.lm_head(h).mean()

    m = Dummy()
    opt = build_muon_with_aux_adamw(
        model=m,
        muon=MuonConfig(lr=1e-3, weight_decay=0.0),
        adamw=AdamWConfig(lr=1e-3, weight_decay=0.0),
    )

    x = torch.randint(0, 16, (4, 5))
    loss = m(x)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    sd = opt.state_dict()
    opt.load_state_dict(sd)

