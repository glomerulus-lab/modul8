"""Layer-local error signals (Nøkland & Eidnes, 2019) as a three-factor rule.

Each managed layer gets a small auxiliary head that reads the layer's
(post-nonlinearity) activity ``h`` and computes a local loss ``L(h, y)`` from
the labels. The layer's per-neuron error is the gradient of that loss with
respect to its own activity, and the update is

    dW  =  -(dL/dh) * f'(a) * pre

the same three-factor form as :class:`~modul8.DirectFeedback`, with
``dL/dh`` in place of DFA's random projection of the output error. No error
crosses from one layer to another: each layer is trained by backprop confined
to the layer and its head.

Heads follow the reference implementation of "Training Neural Networks with
Local Error Signals" (github.com/anokland/local-loss):

    pred     average-pool conv activity to at most ``dim_in_decoder``
             features, then a linear classifier (weights zero-initialised);
             cross-entropy.
    sim      a bias-free 3x3 conv (conv layers) or linear map (linear layers),
             then the similarity matrix of the batch, matched (MSE) to the
             similarity matrix of the one-hot labels.
    predsim  (1 - beta) * pred + beta * sim, beta = 0.99 by default.

With ``bio=True`` the heads are the reference code's "more biologically
plausible" (``--bio``) versions, in which no error depends on a learned
weight the layer does not own:

    * the labels are replaced by a fixed random projection ``P y_onehot``
      (``target_proj_size`` = 128 dimensions);
    * pred: average-pooled activity -> :class:`LinearFA` -> binary
      cross-entropy against ``(P y_onehot > 0)``. ``LinearFA`` learns its
      weights by their true gradient but sends the error back to ``h``
      through a fixed random matrix (feedback alignment);
    * sim: the similarity matrix of the average-pooled activity itself (no
      learned conv), matched to the similarity matrix of ``P y_onehot``.

The reference code's bio recipes use ``beta = 0.01`` for predsim (mostly
pred), not the default 0.99.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .feedback import relu_derivative
from .three_factor import ThreeFactor, first_param

_LOSSES = ("pred", "sim", "predsim")


def similarity_matrix(x: Tensor, std: bool = True) -> Tensor:
    """Batch x batch adjusted cosine similarity of the rows of ``x``.

    For feature maps ``(batch, C, H, W)`` with ``C > 3`` and ``H > 1`` (and
    ``std``), each channel is first summarised by its standard deviation over
    positions, as in the reference implementation; otherwise ``x`` is
    flattened. Rows are centred and normalised before the Gram matrix.
    """
    if x.ndim == 4 and std and x.shape[1] > 3 and x.shape[2] > 1:
        # Clamped so a constant channel (e.g. all-zero ReLU output) gets a
        # zero gradient instead of NaN.
        x = x.flatten(2).var(dim=2).clamp_min(1e-12).sqrt()
    else:
        x = x.flatten(1)
    xc = x - x.mean(dim=1, keepdim=True)
    # vector_norm has a zero (not NaN) gradient at an all-zero row.
    xn = xc / (1e-8 + torch.linalg.vector_norm(xc, dim=1, keepdim=True))
    return (xn @ xn.t()).clamp(-1, 1)


def _decoder_pool(c: int, h: int, w: int, max_dim: int) -> Optional[nn.AvgPool2d]:
    """Average pool that brings ``c x h x w`` down to at most ``max_dim``
    features, doubling the kernel height then width in turn (as the
    reference implementation does)."""
    ks_h = ks_w = 1
    out_h, out_w = h, w
    while c * out_h * out_w > max_dim and ks_h < h:
        ks_h *= 2
        out_h = math.ceil(h / ks_h)
        if c * out_h * out_w > max_dim:
            ks_w *= 2
            out_w = math.ceil(w / ks_w)
    if ks_h == 1 and ks_w == 1:
        return None
    pad_h = (ks_h * (out_h - h // ks_h)) // 2
    pad_w = (ks_w * (out_w - w // ks_w)) // 2
    return nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))


class _LinearFAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, feedback):
        ctx.save_for_backward(x, weight, feedback)
        ctx.has_bias = bias is not None
        out = x @ weight.t()
        return out if bias is None else out + bias

    @staticmethod
    def backward(ctx, grad):
        x, weight, feedback = ctx.saved_tensors
        # The error to the input goes through the fixed feedback matrix, not
        # weight^T; weight and bias get their true gradients.
        gx = grad @ feedback if ctx.needs_input_grad[0] else None
        gw = grad.t() @ x if ctx.needs_input_grad[1] else None
        gb = grad.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return gx, gw, gb, None


class LinearFA(nn.Module):
    """Linear layer trained by feedback alignment: the forward pass is
    ``x W^T + b`` and W, b get their true gradients, but the gradient passed
    back to ``x`` is ``grad @ B`` for a fixed random ``B`` (a buffer, same
    shape as W) instead of ``grad @ W``. Input must be 2-D.

    W and B are drawn from U(-1/sqrt(in), 1/sqrt(in)); the bias starts at 0.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        bound = 1 / math.sqrt(in_features)

        def uniform():
            u = torch.rand(out_features, in_features, generator=generator)
            return (2 * u - 1) * bound

        self.weight = nn.Parameter(uniform())
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("feedback", uniform())

    def forward(self, x: Tensor) -> Tensor:
        return _LinearFAFunction.apply(x, self.weight, self.bias, self.feedback)


class LocalHead(nn.Module):
    """The default local loss for one layer: ``forward(h, y)`` returns the
    scalar loss for activity ``h`` and integer labels ``y``.

    Args:
        shape: per-sample activity shape, ``(features,)`` for a linear layer
            or ``(C, H, W)`` for a 2-D conv layer.
        n_classes: number of classes.
        loss: ``"pred"``, ``"sim"`` or ``"predsim"``.
        beta: weight of the sim term in ``predsim``.
        dim_in_decoder: maximum input size of the pred classifier; conv
            activity is average-pooled down to it.
        similarity_std: summarise each feature-map channel by its spatial
            standard deviation in :func:`similarity_matrix`.
        bio: use the reference code's biologically plausible heads (see the
            module docstring): random label projection, pooled activity for
            sim, and a feedback-alignment pred classifier with a BCE loss.
        target_proj_size: size of the random label projection when ``bio``.
        generator: ``torch.Generator`` for the fixed random matrices of the
            bio heads (the label projection and the feedback matrix).
    """

    def __init__(self, shape: Sequence[int], n_classes: int, loss: str = "predsim",
                 beta: float = 0.99, dim_in_decoder: int = 4096,
                 similarity_std: bool = True, bio: bool = False,
                 target_proj_size: int = 128,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        shape = tuple(int(s) for s in shape)
        if len(shape) not in (1, 3):
            raise ValueError(
                f"default heads support linear (features,) and 2-D conv (C, H, W) activity, "
                f"got shape {shape}; pass your own heads= for other layers"
            )
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
        self.loss = loss
        self.beta = float(beta)
        self.n_classes = n_classes
        self.similarity_std = similarity_std
        self.bio = bio
        conv = len(shape) == 3
        pred = loss in ("pred", "predsim")
        sim = loss in ("sim", "predsim")

        # bio: sim reads the pooled activity too, so the pool is always built.
        self.pool = (_decoder_pool(*shape, dim_in_decoder) if conv and (pred or bio)
                     else None)
        if bio:
            # A fixed random projection of the one-hot labels, with
            # nn.Linear's default init, U(-1/sqrt(n_classes), 1/sqrt(n_classes)).
            u = torch.rand(target_proj_size, n_classes, generator=generator)
            self.register_buffer("proj_y", (2 * u - 1) / math.sqrt(n_classes))
        elif sim:
            c = shape[0]
            self.sim_proj = (nn.Conv2d(c, c, 3, 1, 1, bias=False) if conv
                             else nn.Linear(c, c, bias=False))
        if pred:
            if self.pool is None:
                n_in = math.prod(shape)
            else:
                n_in = self.pool(torch.zeros(1, *shape)).numel()
            self.decoder = (LinearFA(n_in, target_proj_size, generator=generator) if bio
                            else nn.Linear(n_in, n_classes))
            nn.init.zeros_(self.decoder.weight)

    def forward(self, h: Tensor, y: Tensor) -> Tensor:
        pred_w, sim_w = {"pred": (1.0, 0.0), "sim": (0.0, 1.0),
                         "predsim": (1 - self.beta, self.beta)}[self.loss]
        total = h.new_zeros(())
        onehot = F.one_hot(y, self.n_classes).to(h.dtype)
        z = h if self.pool is None else self.pool(h)
        if self.bio:
            target = onehot @ self.proj_y.t()
        if pred_w:
            logits = self.decoder(z.flatten(1))
            if self.bio:
                pred = F.binary_cross_entropy_with_logits(logits, (target > 0).to(h.dtype))
            else:
                pred = F.cross_entropy(logits, y)
            total = total + pred_w * pred
        if sim_w:
            if self.bio:
                rh = similarity_matrix(z, self.similarity_std)
                ry = similarity_matrix(target, self.similarity_std)
            else:
                rh = similarity_matrix(self.sim_proj(h), self.similarity_std)
                ry = similarity_matrix(onehot, self.similarity_std)
            total = total + sim_w * F.mse_loss(rh, ry)
        return total


class LocalLoss:
    """Train layers with layer-local error signals.

    Args:
        layers: a layer, or an iterable of layers: ``nn.Linear`` / ``nn.ConvNd``,
            or any module with parameters (see :class:`~modul8.ThreeFactor`).
        n_classes: number of classes (for the default heads).
        posts: for each layer, the module whose output the local loss reads
            (typically the nonlinearity after the layer), or ``None`` for the
            layer's own output. Layers that share a post module (e.g. a
            residual block's conv and its projection shortcut) share one head
            and one error. Defaults to all ``None``.
        heads: optional custom heads, one per distinct post (in order of first
            appearance), each a module with ``forward(h, y) -> scalar loss``.
            Default: a :class:`LocalHead` per post, built from ``shapes`` or
            on the first :meth:`modulate`.
        shapes: per-layer activity shapes (without the batch dimension), to
            build the default heads now so :meth:`parameters` can go to an
            optimizer before training.
        loss, beta, dim_in_decoder, similarity_std, bio, target_proj_size,
        generator: passed to :class:`LocalHead` (``generator`` draws the bio
            heads' fixed random matrices, one head after another).
        derivative: maps the post activity to ``f'(a)``. Default
            :func:`relu_derivative`; ``None`` means linear units.
        bias: also train the biases. Default ``True``.

    The value written to each layer's ``.grad`` equals the backprop gradient
    of its local loss with respect to the layer's parameters (given that
    ``derivative`` is the derivative of the post module). The heads' own
    gradients are accumulated into their ``.grad``; add
    :meth:`parameters` to an optimizer.

    Usage::

        local = LocalLoss([model.conv1, model.fc1], n_classes=10,
                          posts=[model.relu1, model.relu2],
                          shapes=[(16, 26, 26), (128,)])
        opt = torch.optim.Adam([*model.parameters(), *local.parameters()])
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()                       # backprop for the readout
        local.modulate(y)
        opt.step()
    """

    def __init__(
        self,
        layers: Union[nn.Module, Iterable[nn.Module]],
        n_classes: int,
        *,
        posts: Optional[Sequence[Optional[nn.Module]]] = None,
        heads: Optional[Sequence[nn.Module]] = None,
        shapes: Optional[Sequence[Sequence[int]]] = None,
        loss: str = "predsim",
        beta: float = 0.99,
        dim_in_decoder: int = 4096,
        similarity_std: bool = True,
        bio: bool = False,
        target_proj_size: int = 128,
        generator: Optional[torch.Generator] = None,
        derivative: Optional[Callable[[Tensor], Tensor]] = relu_derivative,
        bias: bool = True,
    ):
        layers = [layers] if isinstance(layers, nn.Module) else list(layers)
        if not layers:
            raise ValueError("no layers given")
        posts = [None] * len(layers) if posts is None else list(posts)
        if len(posts) != len(layers):
            raise ValueError(f"got {len(posts)} posts for {len(layers)} layers")
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")

        post_fn = derivative if derivative is not None else torch.ones_like
        self.tf = ThreeFactor()
        for layer, post in zip(layers, posts):
            # reduce="sample" with a modulator of batch * dL/dh makes the
            # update exactly the gradient of the (batch-mean) local loss.
            self.tf.add(layer, post=post, post_fn=post_fn, bias=bias, reduce="sample")
        self.layers = layers
        self.n_classes = n_classes
        self.head_kwargs = dict(loss=loss, beta=beta, dim_in_decoder=dim_in_decoder,
                                similarity_std=similarity_std, bio=bio,
                                target_proj_size=target_proj_size, generator=generator)

        # Group layers by post module: one head and one error per group.
        index: dict[object, int] = {}
        self.groups: list[list[int]] = []
        for i, post in enumerate(posts):
            key = id(post) if post is not None else ("layer", i)
            if key not in index:
                index[key] = len(self.groups)
                self.groups.append([])
            self.groups[index[key]].append(i)

        self.custom_heads = heads is not None
        self.shapes: Optional[list[tuple[int, ...]]] = None
        if heads is not None:
            heads = list(heads)
            if len(heads) != len(self.groups):
                raise ValueError(f"got {len(heads)} heads for {len(self.groups)} distinct posts")
            self.heads: list[Optional[nn.Module]] = heads
        else:
            self.heads = [None] * len(self.groups)
            if shapes is not None:
                self.build(shapes)
        self.last_losses: list[Optional[Tensor]] = [None] * len(self.groups)

    # -- heads ---------------------------------------------------------------

    def build(self, shapes: Sequence[Sequence[int]]) -> None:
        """Create the default heads from per-layer activity shapes (no-op for
        heads that already exist)."""
        shapes = [tuple(int(s) for s in shape) for shape in shapes]
        if len(shapes) != len(self.layers):
            raise ValueError(f"got {len(shapes)} shapes for {len(self.layers)} layers")
        for group in self.groups:
            if any(shapes[i] != shapes[group[0]] for i in group):
                raise ValueError("layers sharing a post module must have the same output shape")
        self.shapes = shapes
        for g, group in enumerate(self.groups):
            if self.heads[g] is None:
                w = first_param(self.layers[group[0]])
                self.heads[g] = LocalHead(shapes[group[0]], self.n_classes,
                                          **self.head_kwargs).to(device=w.device, dtype=w.dtype)

    def parameters(self) -> Iterator[nn.Parameter]:
        """The heads' parameters (pass them to an optimizer)."""
        if any(h is None for h in self.heads):
            raise RuntimeError("heads are not built yet; pass shapes= or call build()")
        for head in self.heads:
            yield from head.parameters()

    # -- update --------------------------------------------------------------

    @torch.no_grad()
    def modulate(self, y: Tensor) -> None:
        """Compute each layer's local error from labels ``y`` and write the
        update into ``.grad`` (layers) and accumulate head gradients.

        Args:
            y: ``(batch,)`` integer class labels for the captured forward pass.
        """
        y = torch.as_tensor(y).detach()
        if y.ndim != 1:
            raise ValueError(f"y must have shape (batch,), got {tuple(y.shape)}")
        sites = self.tf._sites
        # Validate every layer before writing any .grad.
        for site in sites:
            if not site.captures:
                raise RuntimeError(
                    "no activity captured for a layer since the last modulate()/reset(); "
                    "run a forward pass in training mode first"
                )
            if len(site.captures) > 1:
                raise RuntimeError(
                    "LocalLoss.modulate() expects exactly one captured forward pass per call; "
                    "use reset() to discard extra passes"
                )
            post = site.captures[0][1]
            if post.ndim < 2 or post.shape[0] != y.shape[0]:
                raise ValueError(f"labels have batch {y.shape[0]} but the layer saw "
                                 f"activity of shape {tuple(post.shape)}")
        if any(h is None for h in self.heads):
            self.build([site.captures[0][1].shape[1:] for site in sites])

        # One group at a time, so only one error tensor is alive at once.
        for g, group in enumerate(self.groups):
            head = self.heads[g]
            h = sites[group[0]].captures[0][1]
            params = [p for p in head.parameters() if p.requires_grad]
            with torch.enable_grad():
                h_ = h.detach().requires_grad_()
                loss = head(h_, y.to(h.device))
                grads = torch.autograd.grad(loss, [h_, *params])
            for p, grad in zip(params, grads[1:]):
                p.grad = grad if p.grad is None else p.grad + grad
            # dL/dh carries the 1/batch of the mean loss; reduce="sample"
            # divides by batch again, so scale it back.
            modulator = grads[0].mul_(-h.shape[0])
            for i in group:
                sites[i].modulate(modulator)
            self.last_losses[g] = loss.detach()

    # -- state ---------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "shapes": self.shapes,
            "heads": [None if h is None else h.state_dict() for h in self.heads],
        }

    def load_state_dict(self, state: dict) -> None:
        heads = state["heads"]
        if len(heads) != len(self.heads):
            raise ValueError(f"state has {len(heads)} heads for {len(self.heads)} distinct posts")
        if any(h is None for h in self.heads) and state.get("shapes") is not None:
            self.build(state["shapes"])
        for head, sd in zip(self.heads, heads):
            if sd is None:
                continue
            if head is None:
                raise RuntimeError("cannot load a head that has not been built")
            head.load_state_dict(sd)

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

    def __enter__(self) -> "LocalLoss":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
