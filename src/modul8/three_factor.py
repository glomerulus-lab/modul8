"""Three-factor Hebbian learning for unmodified ``nn.Linear`` / ``nn.ConvNd`` layers.

The rule is

    dw_ij  ∝  M * post_i * pre_j

where ``pre`` is the layer's input, ``post`` its output (or the output of a
later module such as the nonlinearity), and ``M`` is a global modulatory signal
(reward, negative error, ...) supplied by the training loop.

With ``rule="oja"`` each unit also gets Oja's activity-dependent decay, which
keeps its weight norm bounded:

    dw_ij  ∝  M * post_i * pre_j  -  |M| * post_i**2 * w_ij

With ``noise=sigma`` the layer's output is perturbed, ``y + sigma * xi``, and
the perturbation replaces the activity as the postsynaptic factor:

    dw_ij  ∝  M * (xi_i / sigma) * pre_j

which is node perturbation or REINFORCE depending on the modulator supplied.

For ``nn.Linear`` / ``nn.ConvNd`` layers the update is computed in closed
form. Any other module with parameters (a custom layer, a parametrized
layer, a transposed conv, ...) is handled by an autograd fallback: at
:meth:`ThreeFactor.modulate` the module is re-run on its captured inputs and
every trainable parameter gets the vector-Jacobian product of the output with
``M * post`` -- for a Linear / Conv layer, exactly the closed-form update.
The fallback needs a module whose forward is deterministic (no dropout or
batch-norm statistics inside it) and returns one tensor; it does not support
``mask``, ``pre_fn`` or ``rule="oja"``.

Activities are captured with forward hooks, so the layers themselves are
untouched. :meth:`ThreeFactor.modulate` writes the update into ``param.grad``
(as ``-dw``), replacing whatever backprop put there, so any ``torch.optim``
optimizer applies it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd

_CONV_WEIGHT_GRAD = {
    1: torch.nn.grad.conv1d_weight,
    2: torch.nn.grad.conv2d_weight,
    3: torch.nn.grad.conv3d_weight,
}

Activity = Callable[[Tensor], Tensor]
_RULES = ("hebb", "oja")
_REDUCTIONS = ("all", "sample")


def _is_conv(module: nn.Module) -> bool:
    return isinstance(module, _ConvNd) and not isinstance(module, _ConvTransposeNd)


def first_param(module: nn.Module) -> Tensor:
    """The module's first parameter (for its device and dtype)."""
    try:
        return next(module.parameters())
    except StopIteration:
        raise ValueError(f"{type(module).__name__} has no parameters") from None


class _Site:
    """One layer under Hebbian control: its hooks, options and captured activity."""

    def __init__(
        self,
        module: nn.Module,
        mask: Optional[Tensor],
        bias: bool,
        post: Optional[nn.Module],
        pre_fn: Optional[Activity],
        post_fn: Optional[Activity],
        rule: str,
        noise: float,
        reduce: Optional[str] = None,
        channel_dim: int = 1,
    ):
        # Linear / Conv layers get the closed-form update; any other module
        # gets the same update by autograd (see _vjp).
        self.generic = not (isinstance(module, nn.Linear) or _is_conv(module))
        if rule not in _RULES:
            raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
        if noise < 0:
            raise ValueError("noise must be >= 0")
        if reduce is None:
            reduce = "sample" if noise > 0 else "all"
        if reduce not in _REDUCTIONS:
            raise ValueError(f"reduce must be one of {_REDUCTIONS}, got {reduce!r}")
        if noise > 0 and (post is not None or post_fn is not None or rule != "hebb"):
            raise ValueError(
                "noise replaces the postsynaptic activity with the injected perturbation, "
                'so it cannot be combined with `post`, `post_fn` or rule="oja"'
            )
        self.params: list[nn.Parameter] = []
        if self.generic:
            if mask is not None or pre_fn is not None or rule != "hebb":
                raise ValueError(
                    f"`mask`, `pre_fn` and rule=\"oja\" need an nn.Linear / nn.ConvNd weight; "
                    f"{type(module).__name__} is trained by the autograd fallback"
                )
            self.params = [p for name, p in module.named_parameters() if p.requires_grad
                           and (bias or not (name == "bias" or name.endswith(".bias")))]
            if not self.params:
                raise ValueError(f"{type(module).__name__} has no trainable parameters")
        elif bias and module.bias is None:
            raise ValueError("bias=True but the module has no bias")
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool)
            # Raises if the mask cannot broadcast to the weight.
            torch.broadcast_shapes(mask.shape, module.weight.shape)
            if mask.ndim > module.weight.ndim:
                raise ValueError("mask has more dimensions than the weight")

        self.module = module
        self.mask = mask
        self.bias = bias
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.rule = rule
        self.noise = float(noise)
        self.reduce = reduce
        self.channel_dim = channel_dim  # used by the autograd fallback only
        # [pre, post] per forward call; with noise, `post` holds xi / sigma.
        # (For the autograd fallback, `pre` is the tuple of all inputs.)
        self.captures: list[list] = []
        self.enabled = True
        self._replaying = False  # the fallback's own re-run of the module

        self.handles = [module.register_forward_hook(self._capture)]
        if post is not None:
            self.handles.append(post.register_forward_hook(self._capture_post))

    # -- hooks ---------------------------------------------------------------

    def _capture(self, module, inputs, output):
        if not (self.enabled and module.training) or self._replaying:
            return None
        if self.generic:
            pre = tuple(i.detach() if torch.is_tensor(i) else i for i in inputs)
        else:
            pre = inputs[0].detach()
        if self.noise == 0:
            self.captures.append([pre, output.detach()])
            return None
        xi = torch.randn_like(output)
        self.captures.append([pre, xi / self.noise])
        return output + self.noise * xi  # replaces the layer's output

    def _capture_post(self, module, inputs, output):
        if self.enabled and self.module.training and self.captures and not self._replaying:
            if output.shape != self.captures[-1][1].shape:
                raise ValueError(
                    f"`post` module output shape {tuple(output.shape)} does not match "
                    f"the layer output shape {tuple(self.captures[-1][1].shape)}"
                )
            self.captures[-1][1] = output.detach()

    # -- update --------------------------------------------------------------

    def _deltas(self, pre: Tensor, post: Tensor, modulator: Tensor):
        """Return (dw, db) for one captured forward pass."""
        m = self.module
        w = m.weight
        conv = _is_conv(m)
        pre = pre.to(w.dtype)
        post = post.to(w.dtype)

        if conv and post.ndim == w.ndim - 1:  # unbatched input
            pre, post = pre.unsqueeze(0), post.unsqueeze(0)
        ch = 1 if conv else post.ndim - 1  # channel / feature dim of `post`

        if self.pre_fn is not None:
            pre = self.pre_fn(pre)
        g, mod, post, n = self._signal(post, modulator, ch)

        if conv:
            padding = m.padding
            if isinstance(padding, str) or m.padding_mode != "zeros":
                mode = "constant" if m.padding_mode == "zeros" else m.padding_mode
                pre = F.pad(pre, m._reversed_padding_repeated_twice, mode=mode)
                padding = 0
            dw = _CONV_WEIGHT_GRAD[w.ndim - 2](
                pre, w.shape, g, m.stride, padding, m.dilation, m.groups
            )
            db = g.sum(dim=[d for d in range(g.ndim) if d != 1])
        else:
            g2 = g.reshape(-1, g.shape[-1])
            dw = g2.t() @ pre.reshape(-1, pre.shape[-1])
            db = g2.sum(0)

        if self.rule == "oja":
            # Per-unit decay |M| * post^2, averaged like `g`. |M| rather than M
            # so that a negative modulator can never turn the decay into growth.
            decay = (mod.abs() * post * post).expand_as(post) / n
            decay = decay.sum(dim=[d for d in range(post.ndim) if d != ch])
            dw = dw - decay.reshape(-1, *([1] * (w.ndim - 1))) * w
            if self.bias:
                db = db - decay * m.bias
        return dw, db

    def _vjp(self, inputs: tuple, post: Tensor, modulator: Tensor) -> list[Tensor]:
        """Autograd fallback for modules other than Linear / Conv: the update
        of every trained parameter is the vector-Jacobian product of the
        module's output with ``g = M * post / n``, from re-running the module
        on its captured (detached) inputs. For a Linear / Conv layer this is
        exactly the closed-form ``g x pre``."""
        ch = 0
        if post.ndim > 1:
            if not -post.ndim <= self.channel_dim < post.ndim:
                raise ValueError(
                    f"channel_dim={self.channel_dim} is out of range for "
                    f"{type(self.module).__name__} output of shape {tuple(post.shape)}"
                )
            ch = self.channel_dim % post.ndim
        g, *_ = self._signal(post, modulator, ch)
        self._replaying = True
        try:
            with torch.enable_grad():
                out = self.module(*inputs)
        finally:
            self._replaying = False
        if not torch.is_tensor(out) or out.shape != g.shape:
            raise ValueError(
                f"re-running {type(self.module).__name__} gave output of shape "
                f"{tuple(getattr(out, 'shape', ()))}, expected {tuple(g.shape)}"
            )
        grads = torch.autograd.grad(out, self.params, g.to(out.dtype), allow_unused=True)
        return [torch.zeros_like(p) if d is None else d for p, d in zip(self.params, grads)]

    def _signal(self, post: Tensor, modulator: Tensor, ch: int):
        """``(g, mod, post, n)``: the factor ``g = M * f(post) / n`` that
        multiplies the presynaptic activity, the broadcast modulator, the
        transformed postsynaptic activity and the averaging count. ``ch`` is
        the channel / feature dim of ``post``."""
        if self.post_fn is not None:
            post = self.post_fn(post)

        mod = modulator.to(device=post.device, dtype=post.dtype)
        if mod.ndim == 1 and post.ndim > 1:  # per-sample signal
            if mod.shape[0] != post.shape[0]:
                raise ValueError(
                    f"per-sample modulator has {mod.shape[0]} entries "
                    f"but the batch has {post.shape[0]}"
                )
            mod = mod.reshape(-1, *([1] * (post.ndim - 1)))

        # Third factor times postsynaptic activity. reduce="all" averages over
        # every sample and every output position; reduce="sample" (the default
        # with noise) averages over samples only and sums positions, the same
        # scaling as the gradient of a per-sample mean loss.
        if self.reduce == "sample":
            n = post.shape[0] if post.ndim > 1 else 1
        else:
            n = post.numel() // post.shape[ch]
        g = (mod * post).expand_as(post) / n
        return g, mod, post, n

    def modulate(self, modulator: Tensor):
        if not self.captures:
            raise RuntimeError(
                "no activity captured for this layer since the last modulate()/reset(); "
                "run a forward pass in training mode first"
            )
        if self.generic:
            total = None
            for inputs, post in self.captures:
                d = self._vjp(inputs, post, modulator)
                total = d if total is None else [a + b for a, b in zip(total, d)]
            self.captures.clear()
            for p, d in zip(self.params, total):
                p.grad = -d
            return

        m = self.module
        dw = torch.zeros_like(m.weight)
        db = torch.zeros_like(m.bias) if self.bias else None
        for pre, post in self.captures:
            w_, b_ = self._deltas(pre, post, modulator)
            dw += w_
            if db is not None:
                db += b_
        self.captures.clear()

        # Optimizers descend, so the gradient is the negated Hebbian update.
        if self.mask is None:
            m.weight.grad = -dw
        else:
            mask = self.mask.to(dw.device)
            backprop = m.weight.grad if m.weight.grad is not None else torch.zeros_like(dw)
            m.weight.grad = torch.where(mask, -dw, backprop)
        if db is not None:
            m.bias.grad = -db

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.captures.clear()


class ThreeFactor:
    """Train selected layers with a three-factor Hebbian rule.

    Args:
        modules: a layer, or an iterable of layers, to put under Hebbian
            control. More can be added (with per-layer options) via :meth:`add`.
            ``nn.Linear`` / ``nn.ConvNd`` layers use the closed-form update;
            any other module with parameters uses the autograd fallback (see
            the module docstring), which trains all of its trainable
            parameters (those named ``bias`` only with ``bias=True``).
        mask: bool tensor broadcastable to the layer's weight. ``True`` entries
            are learned with the Hebbian rule; ``False`` entries keep their
            backprop gradient. ``None`` (default) means the whole weight.
        bias: also learn the bias with the rule (presynaptic activity = 1).
        post: module whose output is used as postsynaptic activity instead of
            the layer's own output, e.g. the ReLU that follows it. Must produce
            the same shape as the layer.
        rule: ``"hebb"`` (default) for the plain rule ``M * post * pre``, or
            ``"oja"`` to add Oja's decay ``-|M| * post**2 * w``, which bounds
            each unit's weight norm.
        noise: standard deviation ``sigma`` of Gaussian noise added to the
            layer's output on every captured forward pass. The update becomes
            ``M * (xi / sigma) * pre``, averaged over samples: an estimate of
            the gradient when ``M`` is a per-sample ``clean_loss - noisy_loss``
            (node perturbation, see :meth:`paused`) or ``baseline - noisy_loss``
            (REINFORCE). Not combinable with ``post``, ``post_fn`` or
            ``rule="oja"``. No noise is added in eval mode or while paused.
        pre_fn, post_fn: optional transforms applied to the pre/postsynaptic
            activity before the update is computed (e.g. centering).
        reduce: how the update is averaged. ``"all"`` (the default without
            noise) averages over samples and output positions; ``"sample"``
            (the default with noise) averages over samples and sums over
            positions, matching the scale of a mean-loss gradient.
        channel_dim: for the autograd fallback with ``reduce="all"``, the
            output dim that holds the channels / features; every other dim
            is averaged over. Default 1, the layout of a conv; pass ``-1``
            for a channel-last module, e.g. one whose output is
            ``(batch, seq, features)``. Ignored for ``nn.Linear`` /
            ``nn.ConvNd`` layers (features last and channels at dim 1,
            respectively) and with ``reduce="sample"``.

    Usage::

        hebb = ThreeFactor(model.fc1)

        loss = criterion(model(x), y)
        opt.zero_grad()
        loss.backward()                      # backprop for everything else
        hebb.modulate(baseline - loss.item())  # overwrite .grad of Hebbian weights
        opt.step()

    The modulator's sign convention is reward-like: positive values reinforce
    the current pre/post correlation (``dw = +lr * M * post * pre``).
    """

    def __init__(
        self,
        modules: Union[nn.Module, Iterable[nn.Module], None] = None,
        *,
        mask: Optional[Tensor] = None,
        bias: bool = False,
        post: Optional[nn.Module] = None,
        pre_fn: Optional[Activity] = None,
        post_fn: Optional[Activity] = None,
        rule: str = "hebb",
        noise: float = 0.0,
        reduce: Optional[str] = None,
        channel_dim: int = 1,
    ):
        self._sites: list[_Site] = []
        if modules is None:
            return
        # A single layer; containers (nn.Sequential, nn.ModuleList) give their
        # members.
        if isinstance(modules, nn.Module) and not hasattr(modules, "__iter__"):
            modules = [modules]
        modules = list(modules)
        if len(modules) > 1 and (mask is not None or post is not None):
            raise ValueError("`mask` and `post` are per-layer; use add() for each layer")
        for m in modules:
            self.add(
                m, mask=mask, bias=bias, post=post, pre_fn=pre_fn, post_fn=post_fn, rule=rule,
                noise=noise, reduce=reduce, channel_dim=channel_dim,
            )

    def add(
        self,
        module: nn.Module,
        *,
        mask: Optional[Tensor] = None,
        bias: bool = False,
        post: Optional[nn.Module] = None,
        pre_fn: Optional[Activity] = None,
        post_fn: Optional[Activity] = None,
        rule: str = "hebb",
        noise: float = 0.0,
        reduce: Optional[str] = None,
        channel_dim: int = 1,
    ) -> "ThreeFactor":
        """Put ``module`` under Hebbian control. Returns ``self`` for chaining."""
        if any(s.module is module for s in self._sites):
            raise ValueError("module is already managed by this ThreeFactor")
        self._sites.append(_Site(module, mask, bias, post, pre_fn, post_fn, rule, noise, reduce,
                                 channel_dim))
        return self

    @torch.no_grad()
    def modulate(
        self, modulator: Union[float, Tensor, Mapping[nn.Module, Union[float, Tensor]]]
    ) -> None:
        """Deliver the third factor and write the Hebbian update into ``.grad``.

        Call after ``loss.backward()`` (if any) and before ``optimizer.step()``.

        Args:
            modulator: a scalar, a 1-D per-sample tensor of shape ``(batch,)``,
                or any tensor broadcastable to the postsynaptic activity
                (e.g. ``(batch, out_features)`` for a per-neuron signal).
                Or a mapping ``{layer: modulator}`` giving each managed layer
                its own signal; it must cover exactly the managed layers.

        The update is averaged over samples and output positions, summed over
        all forward passes since the last call, and the captured activity is
        then cleared.
        """
        if isinstance(modulator, Mapping):
            managed = {id(s.module) for s in self._sites}
            given = {id(m) for m in modulator}
            if given - managed:
                raise KeyError("modulator given for a layer this ThreeFactor does not manage")
            if managed - given:
                raise KeyError("no modulator given for some managed layer(s)")
            by_id = {id(m): v for m, v in modulator.items()}
            for site in self._sites:
                site.modulate(torch.as_tensor(by_id[id(site.module)]).detach())
            return
        modulator = torch.as_tensor(modulator).detach()
        for site in self._sites:
            site.modulate(modulator)

    def reset(self) -> None:
        """Discard captured activity without updating."""
        for site in self._sites:
            site.captures.clear()

    def enable(self, mode: bool = True) -> None:
        """Turn activity capture on/off and noise injection on/off (both are also off in eval mode)."""
        for site in self._sites:
            site.enabled = mode

    @contextmanager
    def paused(self):
        """Context manager: forward passes inside neither capture activity nor add noise.

        Use it for the clean pass of node perturbation::

            with hebb.paused():
                clean = loss_fn(model(x), y)
            noisy = loss_fn(model(x), y)
            hebb.modulate(clean - noisy)
        """
        previous = [site.enabled for site in self._sites]
        self.enable(False)
        try:
            yield self
        finally:
            for site, was in zip(self._sites, previous):
                site.enabled = was

    def remove(self) -> None:
        """Remove all hooks, returning the layers to plain backprop."""
        for site in self._sites:
            site.remove()
        self._sites.clear()

    def __enter__(self) -> "ThreeFactor":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
