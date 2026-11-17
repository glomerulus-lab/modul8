import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modul8 import LinearFA, LocalHead, LocalLoss, similarity_matrix


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


# -- bio variant ----------------------------------------------------------------

def test_linear_fa_gradients():
    torch.manual_seed(0)
    fa = LinearFA(5, 3)
    x = torch.randn(4, 5, requires_grad=True)
    out = fa(x)
    assert torch.allclose(out, F.linear(x, fa.weight, fa.bias))
    g = torch.randn(4, 3)
    out.backward(g)
    assert torch.allclose(x.grad, g @ fa.feedback)          # B, not W
    assert torch.allclose(fa.weight.grad, g.t() @ x.detach())
    assert torch.allclose(fa.bias.grad, g.sum(0))
    assert [n for n, _ in fa.named_parameters()] == ["weight", "bias"]


def test_linear_fa_with_feedback_equal_to_weight_is_backprop():
    torch.manual_seed(0)
    fa = LinearFA(5, 3)
    fa.feedback.copy_(fa.weight.detach())
    x = torch.randn(4, 5, requires_grad=True)
    (fa(x) ** 2).sum().backward()
    x2 = x.detach().clone().requires_grad_()
    (F.linear(x2, fa.weight, fa.bias) ** 2).sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-6)


@pytest.mark.parametrize("loss", ["pred", "sim", "predsim"])
@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_bio_update_is_local_loss_gradient(loss, kind):
    torch.manual_seed(0)
    if kind == "linear":
        layer, x, shape = nn.Linear(7, 6), torch.randn(9, 7), (6,)
    else:
        layer, x, shape = nn.Conv2d(3, 8, 3, padding=1), torch.randn(9, 3, 8, 8), (8, 8, 8)
    relu = nn.ReLU()
    y = torch.randint(0, 4, (9,))
    local = LocalLoss(layer, 4, posts=[relu], shapes=[shape], loss=loss, bio=True,
                      target_proj_size=16, dim_in_decoder=128, beta=0.5)
    head = local.heads[0]
    for p in head.parameters():
        nn.init.normal_(p, std=0.3)
    with local.paused():
        want = _reference(layer, relu, head, x, y)
    with torch.no_grad():
        relu(layer(x))
    local.modulate(y)
    got = [layer.weight.grad, layer.bias.grad, *(p.grad for p in head.parameters())]
    assert layer.weight.grad.abs().sum() > 0
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=1e-5, rtol=1e-4)


def test_bio_heads_structure():
    conv = LocalHead((128, 32, 32), 10, loss="predsim", bio=True)
    assert conv.pool is not None and conv.decoder.in_features <= 4096
    assert isinstance(conv.decoder, LinearFA) and conv.decoder.out_features == 128
    assert not hasattr(conv, "sim_proj")
    assert torch.count_nonzero(conv.decoder.weight) == 0      # zero init, as in the reference
    assert torch.count_nonzero(conv.decoder.feedback) > 0
    sim = LocalHead((16, 8, 8), 10, loss="sim", bio=True, dim_in_decoder=256)
    assert sim.pool is not None                              # pooled for sim too
    assert list(sim.parameters()) == []                      # nothing to learn
    assert {n for n, _ in conv.named_buffers()} == {"proj_y", "decoder.feedback"}


def test_bio_pred_error_nonzero_at_zero_init():
    torch.manual_seed(0)
    layer, relu = nn.Linear(5, 6), nn.ReLU()
    local = LocalLoss(layer, 3, posts=[relu], shapes=[(6,)], loss="pred", bio=True)
    relu(layer(torch.randn(8, 5)))
    local.modulate(torch.randint(0, 3, (8,)))
    assert layer.weight.grad.abs().sum() > 0  # flows through B, not the zero W


def test_bio_fixed_matrices_do_not_train_and_are_reproducible():
    def make(seed):
        return LocalLoss(nn.Linear(5, 6), 3, posts=[nn.ReLU()], shapes=[(6,)], bio=True,
                         generator=torch.Generator().manual_seed(seed))

    a, b = make(1), make(1)
    for name in ("proj_y", "decoder.feedback"):
        ta = a.heads[0].get_buffer(name)
        assert torch.equal(ta, b.heads[0].get_buffer(name))
        assert not torch.equal(ta, make(2).heads[0].get_buffer(name))

    head = a.heads[0]
    before = {n: t.clone() for n, t in head.named_buffers()}
    opt = torch.optim.SGD(list(a.parameters()), lr=1.0)
    lin, relu = a.layers[0], nn.ReLU()
    a.remove()
    a = LocalLoss(lin, 3, posts=[relu], heads=[head])
    for _ in range(3):
        opt.zero_grad()
        relu(lin(torch.randn(8, 5)))
        a.modulate(torch.randint(0, 3, (8,)))
        opt.step()
    for n, t in head.named_buffers():
        assert torch.equal(t, before[n])

    c = make(3)
    c.load_state_dict(a.state_dict())
    for n, t in head.named_buffers():
        assert torch.equal(c.heads[0].get_buffer(n), t)


# -- layers other than Linear / Conv (autograd fallback) -------------------------


class _Wrapped(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


@pytest.mark.parametrize("bio", [False, True])
def test_wrapped_conv_matches_plain_conv(bio):
    """The autograd fallback gives LocalLoss the same layer and head updates
    as the closed form on a plain nn.Conv2d."""
    grads = []
    for wrap in (False, True):
        torch.manual_seed(0)
        conv, relu = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()
        layer = _Wrapped(conv) if wrap else conv
        x, y = torch.randn(6, 3, 8, 8), torch.randint(0, 4, (6,))
        local = LocalLoss(layer, 4, posts=[relu], shapes=[(8, 8, 8)], bio=bio,
                          generator=torch.Generator().manual_seed(1))
        relu(layer(x))
        local.modulate(y)
        grads.append([conv.weight.grad, conv.bias.grad,
                      *(p.grad for p in local.parameters())])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


class _OrthoProj(nn.Module):
    """A conv-like layer with an orthogonally parametrized 1x1 channel map
    behind a fixed expansion, as in examples/orthogonal_filter.py."""

    def __init__(self, c_in, d, c_out):
        super().__init__()
        self.proj = nn.utils.parametrizations.orthogonal(nn.Linear(c_in, d, bias=False))
        self.register_buffer("expand", torch.randn(c_out, d))
        self.bias = nn.Parameter(torch.randn(c_out))

    def forward(self, x):
        z = F.conv2d(x, self.proj.weight[:, :, None, None])
        return F.conv2d(z, self.expand[:, :, None, None], self.bias)


def test_parametrized_layer_update_is_local_loss_gradient():
    torch.manual_seed(0)
    layer, relu = _OrthoProj(6, 3, 5), nn.ReLU()
    head = LocalHead((5, 4, 4), 3)
    x, y = torch.randn(8, 6, 4, 4), torch.randint(0, 3, (8,))
    local = LocalLoss(layer, 3, posts=[relu], heads=[head])
    relu(layer(x))
    local.modulate(y)
    params = [layer.proj.parametrizations.weight.original, layer.bias]
    ref = torch.autograd.grad(head(relu(layer(x)), y), params)
    for p, r in zip(params, ref):
        assert p.grad is not None and torch.allclose(p.grad, r, atol=1e-6)


# -- integration: LocalLoss inside a small network -------------------------------


class _SmallNet(nn.Module):
    """Two conv blocks (conv -> ReLU -> BN -> max-pool -> dropout), a hidden
    linear layer and a readout, like the CIFAR VGG examples. With `custom`,
    the second conv is an orthogonally parametrized layer, so LocalLoss uses
    the autograd fallback for it."""

    def __init__(self, custom: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = _OrthoProj(6, 4, 8) if custom else nn.Conv2d(6, 8, 3, padding=1)
        self.fc1 = nn.Linear(8 * 2 * 2, 10)
        self.relus = nn.ModuleList(nn.ReLU() for _ in range(3))
        self.bn1 = nn.BatchNorm2d(6, affine=False)
        self.bn2 = nn.BatchNorm2d(8, affine=False)
        self.drop = nn.Dropout(0.3)
        self.readout = nn.Linear(10, 4)

    def forward(self, x):
        r1, r2, r3 = self.relus
        x = self.drop(F.max_pool2d(self.bn1(r1(self.conv1(x))), 2))
        x = self.drop(F.max_pool2d(self.bn2(r2(self.conv2(x))), 2))
        x = self.drop(r3(self.fc1(x.flatten(1))))
        return self.readout(x)

    def hidden(self):
        return [self.conv1, self.conv2, self.fc1], list(self.relus)


def _detached_reference(model, layers, posts, heads, x, y):
    """Gradients of the summed local losses, by autograd, with every hidden
    layer's input detached so no error crosses between layers."""
    acts = []
    handles = [layer.register_forward_pre_hook(
        lambda m, args: tuple(a.detach() for a in args)) for layer in layers]
    handles += [post.register_forward_hook(lambda m, i, o: acts.append(o)) for post in posts]
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    loss = sum(head(h, y) for head, h in zip(heads, acts))
    params = ([p for layer in layers for p in layer.parameters()]
              + [p for head in heads for p in head.parameters()])
    return torch.autograd.grad(loss, params)


@pytest.mark.parametrize("custom", [False, True], ids=["plain", "fallback"])
@pytest.mark.parametrize("bio", [False, True], ids=["std", "bio"])
def test_network_updates_match_detached_autograd(custom, bio):
    """In a full network (BN, pooling, dropout and a flatten between the
    layers, hidden layers run without autograd), every layer's and head's
    LocalLoss update equals the gradient of its own local loss."""
    import copy

    torch.manual_seed(0)
    model = _SmallNet(custom).train()
    layers, posts = model.hidden()
    local = LocalLoss(layers, 4, posts=posts, shapes=[(6, 8, 8), (8, 4, 4), (10,)],
                      loss="predsim", beta=0.5, bio=bio, dim_in_decoder=64,
                      generator=torch.Generator().manual_seed(1))
    # The pred decoders start at zero; perturb every head so no term is trivial.
    with torch.no_grad():
        for p in local.parameters():
            p.normal_(std=0.3)
    ref_model, ref_heads = copy.deepcopy(model), copy.deepcopy(local.heads)
    x, y = torch.randn(10, 3, 8, 8), torch.randint(0, 4, (10,))

    torch.manual_seed(2)  # same dropout masks in both passes
    with torch.no_grad():
        model(x)
    local.modulate(y)
    got = ([p.grad for layer in layers for p in layer.parameters()]
           + [p.grad for p in local.parameters()])

    torch.manual_seed(2)
    want = _detached_reference(ref_model, *ref_model.hidden(), ref_heads, x, y)

    assert len(got) == len(want)
    for layer in layers:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=1e-5, rtol=1e-4)
