import pytest
import torch
from torch import nn

from modul8 import ThreeFactor


def reference(layer, x, mod, post_fn=None):
    """-d/dw of sum(M * stopgrad(post) * layer(x)) / n, via autograd."""
    out = layer(x)
    post = out.detach() if post_fn is None else post_fn(out.detach())
    ch = 1 if isinstance(layer, nn.modules.conv._ConvNd) else out.ndim - 1
    n = out.numel() // out.shape[ch]
    params = [layer.weight] + ([layer.bias] if layer.bias is not None else [])
    grads = torch.autograd.grad((mod * post * out).sum() / n, params)
    return [-g for g in grads]


LAYERS = [
    (lambda: nn.Linear(5, 3), (4, 5)),
    (lambda: nn.Linear(5, 3), (4, 7, 5)),
    (lambda: nn.Conv2d(4, 6, 3), (2, 4, 8, 8)),
    (lambda: nn.Conv2d(4, 6, 3, stride=2, padding=1, dilation=2, groups=2), (2, 4, 9, 9)),
    (lambda: nn.Conv2d(4, 6, 3, padding="same"), (2, 4, 8, 8)),
    (lambda: nn.Conv2d(4, 6, 3, padding=1, padding_mode="circular"), (2, 4, 8, 8)),
    (lambda: nn.Conv1d(4, 6, 3), (2, 4, 10)),
    (lambda: nn.Conv3d(2, 3, 2), (2, 2, 4, 4, 4)),
]


@pytest.mark.parametrize("make,shape", LAYERS)
def test_matches_autograd_reference(make, shape):
    torch.manual_seed(0)
    layer, x = make(), torch.randn(shape)
    hebb = ThreeFactor(layer, bias=True)
    layer(x)
    hebb.modulate(0.7)
    ref_w, ref_b = reference(layer, x, 0.7)
    assert torch.allclose(layer.weight.grad, ref_w, atol=1e-6)
    assert torch.allclose(layer.bias.grad, ref_b, atol=1e-6)


@pytest.mark.parametrize("make,shape", [LAYERS[0], LAYERS[2]])
def test_per_sample_modulator(make, shape):
    torch.manual_seed(0)
    layer, x = make(), torch.randn(shape)
    m = torch.randn(shape[0])
    hebb = ThreeFactor(layer)
    layer(x)
    hebb.modulate(m)
    ref_w, _ = reference(layer, x, m.reshape(-1, *([1] * (len(shape) - 1))))
    assert torch.allclose(layer.weight.grad, ref_w, atol=1e-6)


def test_mask_keeps_backprop_elsewhere():
    torch.manual_seed(0)
    layer, x = nn.Linear(5, 3), torch.randn(4, 5)
    mask = torch.tensor([True, False, False]).unsqueeze(1)  # only output unit 0
    hebb = ThreeFactor(layer, mask=mask)
    layer(x).pow(2).sum().backward()
    bp = layer.weight.grad.clone()
    bias_bp = layer.bias.grad.clone()
    hebb.modulate(1.0)
    ref_w, _ = reference(layer, x, 1.0)
    assert torch.allclose(layer.weight.grad[0], ref_w[0], atol=1e-6)
    assert torch.equal(layer.weight.grad[1:], bp[1:])
    assert torch.equal(layer.bias.grad, bias_bp)  # bias untouched by default


def test_post_module_and_upstream_backprop():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(8, 6)
    hebb = ThreeFactor(net[2], post=net[3])
    net(x).sum().backward()
    upstream = net[0].weight.grad.clone()
    hebb.modulate(1.0)
    ref_w, _ = reference(net[2], net[1](net[0](x)).detach(), 1.0, post_fn=torch.relu)
    assert torch.allclose(net[2].weight.grad, ref_w, atol=1e-6)
    assert upstream.abs().sum() > 0 and torch.equal(net[0].weight.grad, upstream)


def test_capture_lifecycle():
    layer, x = nn.Linear(5, 3), torch.randn(4, 5)
    hebb = ThreeFactor(layer)
    with pytest.raises(RuntimeError):
        hebb.modulate(1.0)
    layer.eval()
    layer(x)
    with pytest.raises(RuntimeError):  # eval-mode forwards are not captured
        hebb.modulate(1.0)
    layer.train()
    layer(x)
    layer(x)  # two passes accumulate
    hebb.modulate(1.0)
    ref_w, _ = reference(layer, x, 1.0)
    assert torch.allclose(layer.weight.grad, 2 * ref_w, atol=1e-6)
    hebb.remove()
    assert not layer._forward_hooks


def test_rejects_unsupported():
    with pytest.raises(TypeError):
        ThreeFactor().add(nn.ConvTranspose2d(2, 2, 3))
    with pytest.raises(ValueError):
        ThreeFactor(nn.Linear(2, 2, bias=False), bias=True)


def test_learns_with_optimizer():
    """Reward-modulated Hebbian readout learns a linear bandit-style task."""
    torch.manual_seed(0)
    layer = nn.Linear(10, 1, bias=False)
    target = torch.randn(1, 10)
    hebb = ThreeFactor(layer)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)

    def err():
        x = torch.randn(512, 10)
        return (layer(x) - x @ target.t()).pow(2).mean().item()

    before = err()
    hebb.reset()  # err() ran a training-mode forward; drop that capture
    for _ in range(300):
        x = torch.randn(64, 10)
        y = layer(x)
        # Per-sample signed error divided by post gives the delta rule as a
        # three-factor update; here it simply checks the plumbing end to end.
        y = y.detach()
        y_safe = torch.where(y.abs() < 1e-3, torch.full_like(y, 1e-3), y)
        m = (x @ target.t() - y) / y_safe
        opt.zero_grad()
        hebb.modulate(m)
        opt.step()
    assert err() < 0.05 * before


@pytest.mark.parametrize("make,shape", [LAYERS[0], LAYERS[3]])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_oja_decay_term(make, shape, sign):
    torch.manual_seed(0)
    layer, x = make(), torch.randn(shape)
    m = sign * torch.rand(shape[0])
    mb = m.reshape(-1, *([1] * (len(shape) - 1)))
    hebb = ThreeFactor(layer, bias=True, rule="oja")
    out = layer(x).detach()
    hebb.modulate(m)
    ch = 1 if out.ndim > 2 else out.ndim - 1
    n = out.numel() // out.shape[ch]
    decay = (mb.abs() * out**2).sum(dim=[d for d in range(out.ndim) if d != ch]) / n
    ref_w, ref_b = reference(layer, x, mb)
    w = layer.weight.detach()
    exp_w = ref_w + decay.reshape(-1, *([1] * (w.ndim - 1))) * w
    assert torch.allclose(layer.weight.grad, exp_w, atol=1e-6)
    assert torch.allclose(layer.bias.grad, ref_b + decay * layer.bias.detach(), atol=1e-6)


def _train_norm(rule, steps=400):
    torch.manual_seed(0)
    layer = nn.Linear(8, 4, bias=False)
    hebb = ThreeFactor(layer, rule=rule)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    scale = torch.linspace(0.5, 2.0, 8)
    for _ in range(steps):
        layer(torch.randn(64, 8) * scale)
        hebb.modulate(1.0)
        opt.step()
    return layer.weight.detach().norm(dim=1)


def test_oja_bounds_weights_where_hebb_explodes():
    assert _train_norm("hebb").min() > 1e3
    assert torch.allclose(_train_norm("oja"), torch.ones(4), atol=0.05)


def test_rejects_unknown_rule():
    with pytest.raises(ValueError):
        ThreeFactor(nn.Linear(2, 2), rule="bcm")
