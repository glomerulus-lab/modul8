import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modul8 import LocalHead, LocalLoss, similarity_matrix


def _reference(layer, relu, head, x, y):
    """Autograd gradient of the local loss w.r.t. the layer and head params,
    with the layer's input detached."""
    params = [layer.weight, layer.bias, *head.parameters()]
    loss = head(relu(layer(x.detach())), y)
    return torch.autograd.grad(loss, params)


@pytest.mark.parametrize("loss", ["pred", "sim", "predsim"])
@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_update_is_local_loss_gradient(loss, kind):
    torch.manual_seed(0)
    if kind == "linear":
        layer, x, shape = nn.Linear(7, 6), torch.randn(9, 7), (6,)
    else:
        layer, x, shape = nn.Conv2d(3, 5, 3, padding=1), torch.randn(9, 3, 8, 8), (5, 8, 8)
    relu = nn.ReLU()
    y = torch.randint(0, 4, (9,))
    local = LocalLoss(layer, 4, posts=[relu], shapes=[shape], loss=loss,
                      dim_in_decoder=64)
    head = local.heads[0]
    # The pred decoder is zero-initialised, which makes dL/dh zero for pure
    # pred; perturb it so the test is not trivial.
    for p in head.parameters():
        nn.init.normal_(p, std=0.3)
    with local.paused():
        want = _reference(layer, relu, head, x, y)

    with torch.no_grad():
        relu(layer(x))
    local.modulate(y)
    got = [layer.weight.grad, layer.bias.grad, *(p.grad for p in head.parameters())]
    assert any(g.abs().sum() > 0 for g in got[:2])
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=1e-5, rtol=1e-4)


def test_custom_head_and_linear_units():
    torch.manual_seed(0)
    layer = nn.Linear(4, 3)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(3, 2))

        def forward(self, h, y):
            return F.cross_entropy(h @ self.w, y)

    head = Head()
    local = LocalLoss(layer, 2, heads=[head], derivative=None)
    x, y = torch.randn(5, 4), torch.randint(0, 2, (5,))
    with local.paused():
        want = torch.autograd.grad(head(layer(x), y), [layer.weight, layer.bias, head.w])
    layer(x)
    local.modulate(y)
    for g, w in zip((layer.weight.grad, layer.bias.grad, head.w.grad), want):
        assert torch.allclose(g, w, atol=1e-6)


def test_head_grads_accumulate():
    torch.manual_seed(0)
    layer, relu = nn.Linear(4, 3), nn.ReLU()
    local = LocalLoss(layer, 2, posts=[relu], shapes=[(3,)])
    x, y = torch.randn(5, 4), torch.randint(0, 2, (5,))
    relu(layer(x))
    local.modulate(y)
    first = [p.grad.clone() for p in local.parameters()]
    relu(layer(x))
    local.modulate(y)
    for p, g in zip(local.parameters(), first):
        assert torch.allclose(p.grad, 2 * g)


def test_shared_post_shares_head_and_error():
    torch.manual_seed(0)
    a, b, relu = nn.Linear(4, 3), nn.Linear(4, 3), nn.ReLU()
    local = LocalLoss([a, b], 2, posts=[relu, relu], shapes=[(3,), (3,)])
    assert len(local.heads) == 1
    head = local.heads[0]
    for p in head.parameters():
        nn.init.normal_(p, std=0.3)
    x, y = torch.randn(6, 4), torch.randint(0, 2, (6,))
    params = [a.weight, a.bias, b.weight, b.bias]
    with local.paused():
        want = torch.autograd.grad(head(relu(a(x) + b(x)), y), params)
    with torch.no_grad():
        relu(a(x) + b(x))
    local.modulate(y)
    for p, w in zip(params, want):
        assert torch.allclose(p.grad, w, atol=1e-5)


def test_no_error_crosses_layers():
    torch.manual_seed(0)
    l1, r1, l2, r2 = nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU()
    local = LocalLoss([l1, l2], 2, posts=[r1, r2], shapes=[(5,), (3,)])
    x, y = torch.randn(6, 4), torch.randint(0, 2, (6,))
    r2(l2(r1(l1(x))))
    local.modulate(y)
    g1 = l1.weight.grad.clone()
    with torch.no_grad():
        l2.weight.mul_(-3.0)
        for p in local.heads[1].parameters():
            p.add_(1.0)
    r2(l2(r1(l1(x))))
    local.modulate(y)
    assert torch.allclose(l1.weight.grad, g1)


def test_lazy_build_and_parameters():
    layer, relu = nn.Conv2d(3, 4, 3), nn.ReLU()
    local = LocalLoss(layer, 10, posts=[relu])
    with pytest.raises(RuntimeError, match="not built"):
        list(local.parameters())
    relu(layer(torch.randn(2, 3, 6, 6)))
    local.modulate(torch.tensor([1, 2]))
    assert local.shapes == [(4, 4, 4)]
    assert len(list(local.parameters())) > 0


def test_validation_leaves_grads_untouched():
    a, b = nn.Linear(3, 3), nn.Linear(3, 3)
    local = LocalLoss([a, b], 2, shapes=[(3,), (3,)])
    a(torch.randn(4, 3))  # b never runs
    with pytest.raises(RuntimeError, match="no activity"):
        local.modulate(torch.zeros(4, dtype=torch.long))
    assert a.weight.grad is None
    local.reset()
    b(a(torch.randn(4, 3)))
    with pytest.raises(ValueError, match="batch"):
        local.modulate(torch.zeros(5, dtype=torch.long))
    assert a.weight.grad is None
    local.reset()
    b(a(torch.randn(4, 3)))
    b(a(torch.randn(4, 3)))
    with pytest.raises(RuntimeError, match="exactly one"):
        local.modulate(torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="heads"):
        LocalLoss([a, b], 2, heads=[nn.Identity()])
    with pytest.raises(ValueError, match="loss"):
        LocalLoss(a, 2, loss="recon")


def test_state_dict_round_trip_and_paused():
    torch.manual_seed(0)
    layer, relu = nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()
    local = LocalLoss(layer, 3, posts=[relu], shapes=[(4, 8, 8)])
    for p in local.parameters():
        nn.init.normal_(p)
    other = LocalLoss(nn.Conv2d(3, 4, 3, padding=1), 3, posts=[nn.ReLU()])
    other.load_state_dict(local.state_dict())
    for p, q in zip(local.parameters(), other.parameters()):
        assert torch.equal(p, q)
    with local.paused():
        relu(layer(torch.randn(2, 3, 8, 8)))
    with pytest.raises(RuntimeError, match="no activity"):
        local.modulate(torch.tensor([0, 1]))


def test_decoder_pool_limits_input_size():
    head = LocalHead((128, 32, 32), 10, loss="pred", dim_in_decoder=4096)
    assert head.decoder.in_features <= 4096
    head = LocalHead((64, 7, 7), 10, loss="pred", dim_in_decoder=1000)
    assert head.decoder.in_features <= 1000
    assert head.pool(torch.zeros(1, 64, 7, 7)).numel() == head.decoder.in_features


def test_similarity_matrix():
    x = torch.randn(5, 8, 4, 4)
    r = similarity_matrix(x)
    assert r.shape == (5, 5)
    assert torch.allclose(r.diagonal(), torch.ones(5), atol=1e-5)
    assert torch.allclose(r, r.t())
    y = F.one_hot(torch.tensor([0, 1, 0]), 3).float()
    ry = similarity_matrix(y)
    assert ry[0, 2] > 0.99 and ry[0, 1] < 0


def test_dead_units_give_finite_grads():
    head = LocalHead((6, 4, 4), 3, loss="sim")
    h = torch.rand(4, 6, 4, 4)
    h[0] = 0  # a sample whose every unit is dead
    h = h.requires_grad_()
    head(h, torch.tensor([0, 1, 2, 0])).backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
