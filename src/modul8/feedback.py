"""Direct feedback alignment (DFA) as a three-factor rule.

Each managed layer gets a fixed random feedback matrix ``B`` that projects the
network's output error ``e`` (e.g. ``softmax - one_hot``) straight to its units.
The update is

    dW  =  -(B^T e) * f'(a) * pre

i.e. a three-factor rule whose per-neuron modulator is ``-(B^T e)`` and whose
postsynaptic factor is the derivative of the layer's nonlinearity. No error is
propagated backwards through the network.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from .three_factor import ThreeFactor


def relu_derivative(post: Tensor) -> Tensor:
    """relu'(a), from either the ReLU's output or its input."""
    return (post > 0).to(post.dtype)


def tanh_derivative(post: Tensor) -> Tensor:
    """tanh'(a) = 1 - tanh(a)^2, from the tanh's output."""
    return 1 - post * post


def sigmoid_derivative(post: Tensor) -> Tensor:
    """sigmoid'(a) = s (1 - s), from the sigmoid's output."""
    return post * (1 - post)


class DirectFeedback:
    """Train layers by direct feedback alignment.

    Args:
        layers: a layer, or an iterable of ``nn.Linear`` / ``nn.ConvNd`` layers.
        n_out: size of the output error passed to :meth:`modulate`.
        posts: for each layer, the module whose output feeds ``derivative``
            (typically the nonlinearity after the layer), or ``None`` to use the
            layer's own output (pre-activation). Defaults to all ``None``.
        derivative: maps that activity to ``f'(a)``. Default
            :func:`relu_derivative`. ``None`` uses a factor of 1 (linear units).
        scale: entries of each ``B`` are ``N(0, scale^2 / n_out)``.
        bias: also train the biases.
        reduce: passed to :class:`ThreeFactor`; the default ``"sample"`` gives
            updates on the scale of a mean-loss gradient, so SGD works too.
        generator: ``torch.Generator`` used to draw ``B`` (for reproducibility).

    Each ``B`` is created on the first :meth:`modulate`, when the layer's
    output shape is known, on the layer's device and dtype.

    Usage::

        dfa = DirectFeedback([model.fc1, model.fc2], n_out=10,
                             posts=[model.relu1, model.relu2])
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()                       # backprop for the readout
        dfa.modulate(logits.detach().softmax(1) - F.one_hot(y, 10))
        opt.step()
    """

    def __init__(
        self,
        layers: Union[nn.Module, Iterable[nn.Module]],
        n_out: int,
        *,
        posts: Optional[Sequence[Optional[nn.Module]]] = None,
        derivative: Optional[Callable[[Tensor], Tensor]] = relu_derivative,
        scale: float = 1.0,
        bias: bool = True,
        reduce: str = "sample",
        generator: Optional[torch.Generator] = None,
    ):
        layers = [layers] if isinstance(layers, nn.Module) else list(layers)
        if not layers:
            raise ValueError("no layers given")
        if posts is None:
            posts = [None] * len(layers)
        posts = list(posts)
        if len(posts) != len(layers):
            raise ValueError(f"got {len(posts)} posts for {len(layers)} layers")
        if n_out < 1:
            raise ValueError("n_out must be >= 1")

        post_fn = derivative if derivative is not None else torch.ones_like
        self.tf = ThreeFactor()
        for layer, post in zip(layers, posts):
            self.tf.add(layer, post=post, post_fn=post_fn, bias=bias, reduce=reduce)
        self.layers = layers
        self.n_out = n_out
        self.scale = float(scale)
        self.generator = generator
        self.feedback: list[Optional[Tensor]] = [None] * len(layers)

    def _matrix(self, i: int, unit_shape: torch.Size) -> Tensor:
        layer = self.layers[i]
        size = math.prod(unit_shape)
        B = self.feedback[i]
        if B is None:
            B = torch.randn(self.n_out, size, generator=self.generator)
            B = (B * (self.scale / math.sqrt(self.n_out))).to(
                device=layer.weight.device, dtype=layer.weight.dtype)
            self.feedback[i] = B
        elif B.shape != (self.n_out, size):
            raise ValueError(
                f"feedback matrix {i} has shape {tuple(B.shape)}, "
                f"expected {(self.n_out, size)} for this layer's output"
            )
        return B

    @torch.no_grad()
    def modulate(self, error: Tensor) -> None:
        """Deliver the output error and write the DFA update into ``.grad``.

        Args:
            error: ``(batch, n_out)``, the loss gradient with respect to the
                network output (``softmax - one_hot`` for mean cross-entropy,
                up to the 1/batch factor that ``reduce="sample"`` applies).
        """
        error = torch.as_tensor(error).detach()
        if error.ndim != 2 or error.shape[1] != self.n_out:
            raise ValueError(f"error must have shape (batch, {self.n_out}), "
                             f"got {tuple(error.shape)}")
        mods = {}
        for i, (layer, site) in enumerate(zip(self.layers, self.tf._sites)):
            if not site.captures:
                raise RuntimeError(
                    "no activity captured for a layer since the last modulate()/reset(); "
                    "run a forward pass in training mode first"
                )
            post = site.captures[-1][1]
            if post.shape[0] != error.shape[0]:
                raise ValueError(f"error has batch {error.shape[0]} but the layer saw "
                                 f"batch {post.shape[0]}")
            B = self._matrix(i, post.shape[1:])
            mods[layer] = -(error.to(B.dtype) @ B).reshape(post.shape)
        self.tf.modulate(mods)

    # -- state ---------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"feedback": [None if B is None else B.clone() for B in self.feedback]}

    def load_state_dict(self, state: dict) -> None:
        feedback = state["feedback"]
        if len(feedback) != len(self.layers):
            raise ValueError(f"state has {len(feedback)} matrices for {len(self.layers)} layers")
        self.feedback = [
            None if B is None else torch.as_tensor(B).to(
                device=layer.weight.device, dtype=layer.weight.dtype)
            for B, layer in zip(feedback, self.layers)
        ]

    # -- delegation ----------------------------------------------------------

    def reset(self) -> None:
        self.tf.reset()

    def enable(self, mode: bool = True) -> None:
        self.tf.enable(mode)

    @contextmanager
    def paused(self):
        with self.tf.paused():
            yield self

    def remove(self) -> None:
        self.tf.remove()

    def __enter__(self) -> "DirectFeedback":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
