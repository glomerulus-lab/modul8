import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modul8 import DirectFeedback, ThreeFactor


# -- ThreeFactor: per-layer modulators -----------------------------------------

def test_mapping_modulator_matches_separate_instances():
    torch.manual_seed(0)
    a, b = nn.Linear(5, 3), nn.Linear(3, 4)
    x = torch.randn(6, 5)
    ma, mb = torch.randn(6, 3), torch.randn(6, 4)

    tf = ThreeFactor([a, b], bias=True)
    b(a(x))
    tf.modulate({a: ma, b: mb})
    got = [p.grad.clone() for p in (a.weight, a.bias, b.weight, b.bias)]
    tf.remove()

    ta, tb = ThreeFactor(a, bias=True), ThreeFactor(b, bias=True)
    b(a(x))
    ta.modulate(ma)
    tb.modulate(mb)
    want = [a.weight.grad, a.bias.grad, b.weight.grad, b.bias.grad]
    for g, w in zip(got, want):
        assert torch.allclose(g, w)


def test_mapping_modulator_must_cover_managed_layers():
    a, b, c = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
    tf = ThreeFactor([a, b])
    b(a(torch.randn(3, 2)))
    with pytest.raises(KeyError, match="no modulator"):
        tf.modulate({a: 1.0})
    with pytest.raises(KeyError, match="does not manage"):
        tf.modulate({a: 1.0, b: 1.0, c: 1.0})


def test_reduce_validation():
    with pytest.raises(ValueError, match="reduce"):
        ThreeFactor(nn.Linear(2, 2), reduce="mean")


def test_reduce_sample_matches_mean_loss_gradient_for_conv():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 4, 3)
    x = torch.randn(5, 3, 8, 8)
    out = conv(x)
    G = torch.randn_like(out)  # dLoss/dout for loss = sum(G * out) / batch
    ref = torch.autograd.grad((G * out).sum() / x.shape[0], [conv.weight, conv.bias])

    tf = ThreeFactor(conv, bias=True, reduce="sample", post_fn=torch.ones_like)
    conv(x)
    tf.modulate(-G)
    assert torch.allclose(conv.weight.grad, ref[0], atol=1e-5)
    assert torch.allclose(conv.bias.grad, ref[1], atol=1e-5)


# -- DirectFeedback -------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.relu = nn.Linear(6, 8), nn.ReLU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_dfa_with_true_feedback_equals_backprop():
    """If B is the actual backward path (W2), DFA computes the exact gradient."""
    torch.manual_seed(0)
    net = MLP()
    x, y = torch.randn(10, 6), torch.randint(0, 4, (10,))

    logits = net(x)
    loss = F.cross_entropy(logits, y)
    ref = torch.autograd.grad(loss, [net.fc1.weight, net.fc1.bias])

    dfa = DirectFeedback(net.fc1, n_out=4, posts=[net.relu])
    dfa.load_state_dict({"feedback": [net.fc2.weight.detach().clone()]})
    logits = net(x)
    dfa.modulate(logits.detach().softmax(1) - F.one_hot(y, 4))
    assert torch.allclose(net.fc1.weight.grad, ref[0], atol=1e-6)
    assert torch.allclose(net.fc1.bias.grad, ref[1], atol=1e-6)


def test_dfa_creates_feedback_lazily_with_layer_shapes():
    conv, fc = nn.Conv2d(1, 3, 3), nn.Linear(5, 7)
    dfa = DirectFeedback([conv, fc], n_out=10)
    assert dfa.feedback == [None, None]
    conv(torch.randn(2, 1, 6, 6))
    fc(torch.randn(2, 5))
    dfa.modulate(torch.randn(2, 10))
    assert dfa.feedback[0].shape == (10, 3 * 4 * 4)
    assert dfa.feedback[1].shape == (10, 7)
    assert conv.weight.grad is not None and fc.bias.grad is not None


def test_dfa_generator_is_reproducible_and_state_round_trips():
    def run(seed):
        fc = nn.Linear(4, 5)
        dfa = DirectFeedback(fc, n_out=3, generator=torch.Generator().manual_seed(seed))
        fc(torch.randn(2, 4))
        dfa.modulate(torch.randn(2, 3))
        return dfa

    a, b, c = run(1), run(1), run(2)
    assert torch.equal(a.feedback[0], b.feedback[0])
    assert not torch.equal(a.feedback[0], c.feedback[0])

    fresh = DirectFeedback(nn.Linear(4, 5), n_out=3)
    fresh.load_state_dict(a.state_dict())
    assert torch.equal(fresh.feedback[0], a.feedback[0])


def test_dfa_input_validation():
    fc = nn.Linear(4, 5)
    with pytest.raises(ValueError, match="posts"):
        DirectFeedback(fc, n_out=3, posts=[None, None])
    dfa = DirectFeedback(fc, n_out=3)
    with pytest.raises(RuntimeError, match="no activity"):
        dfa.modulate(torch.randn(2, 3))
    fc(torch.randn(2, 4))
    with pytest.raises(ValueError, match="shape"):
        dfa.modulate(torch.randn(2, 4))
    with pytest.raises(ValueError, match="batch"):
        dfa.modulate(torch.randn(3, 3))
    dfa.load_state_dict({"feedback": [torch.randn(3, 9)]})
    with pytest.raises(ValueError, match="expected"):
        dfa.modulate(torch.randn(2, 3))


def test_dfa_paused_and_remove():
    fc = nn.Linear(4, 5)
    dfa = DirectFeedback(fc, n_out=3)
    with dfa.paused():
        fc(torch.randn(2, 4))
    with pytest.raises(RuntimeError):
        dfa.modulate(torch.randn(2, 3))
    dfa.remove()
    fc(torch.randn(2, 4))
    assert dfa.tf._sites == []
