"""FeatureCVAE: shapes, param budget, loss decomposition, deterministic CF."""

import torch

from paper2_causalechonet.cvae import FeatureCVAE, cvae_loss
from paper1_echofusenet.models import count_parameters


def _cvae(D=48):
    return FeatureCVAE(representation_dim=D, n_classes=5, latent_dim=16, hidden_dim=64)


def test_forward_shapes():
    model = _cvae().train()
    x = torch.randn(8, 48)
    cond = torch.randint(0, 5, (8,))
    x_hat, mu, logvar = model(x, cond)
    assert x_hat.shape == x.shape
    assert mu.shape == (8, 16) == logvar.shape


def test_accepts_onehot_condition():
    model = _cvae().eval()
    x = torch.randn(4, 48)
    idx = torch.tensor([0, 1, 2, 3])
    onehot = torch.nn.functional.one_hot(idx, 5).float()
    a, _, _ = model(x, idx)
    b, _, _ = model(x, onehot)
    assert torch.allclose(a, b, atol=1e-6)


def test_param_budget_small():
    # The CVAE is meant to be lightweight (~0.2M at full size); at these test
    # dims it is far smaller, but must stay well under the whole-system 1M budget.
    assert count_parameters(_cvae(D=768)) < 350_000


def test_cvae_loss_decomposition():
    x = torch.randn(6, 48)
    x_hat = x.clone()  # perfect reconstruction -> recon term is ~0
    mu = torch.zeros(6, 16)
    logvar = torch.zeros(6, 16)  # matches prior -> KL is ~0
    out = cvae_loss(x, x_hat, mu, logvar, beta=1.0)
    assert out["recon"].item() < 1e-6
    assert out["kl"].item() < 1e-6
    assert torch.allclose(out["total"], out["recon"] + out["kl"])


def test_kl_positive_when_posterior_off_prior():
    x = torch.randn(6, 48)
    out = cvae_loss(x, x, torch.ones(6, 16), torch.zeros(6, 16), beta=2.0)
    assert out["kl"].item() > 0
    assert torch.allclose(out["total"], out["recon"] + 2.0 * out["kl"])


def test_counterfactual_is_deterministic_in_eval():
    model = _cvae().eval()
    x = torch.randn(5, 48)
    src = torch.zeros(5, dtype=torch.long)
    tgt = torch.full((5,), 2, dtype=torch.long)
    cf1 = model.counterfactual(x, src, tgt)
    cf2 = model.counterfactual(x, src, tgt)
    assert cf1.shape == x.shape
    assert torch.allclose(cf1, cf2)  # posterior-mean path -> reproducible


def test_target_condition_changes_counterfactual():
    model = _cvae().eval()
    x = torch.randn(5, 48)
    src = torch.zeros(5, dtype=torch.long)
    cf_to_1 = model.counterfactual(x, src, torch.full((5,), 1, dtype=torch.long))
    cf_to_4 = model.counterfactual(x, src, torch.full((5,), 4, dtype=torch.long))
    assert not torch.allclose(cf_to_1, cf_to_4)  # conditioning actually steers the edit
