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

Activities are captured with forward hooks, so the layers themselves are
untouched. :meth:`ThreeFactor.modulate` writes the update into ``param.grad``
(as ``-dw``), replacing whatever backprop put there, so any ``torch.optim``
optimizer applies it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Union

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


def _is_conv(module: nn.Module) -> bool:
    return isinstance(module, _ConvNd) and not isinstance(module, _ConvTransposeNd)


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
    ):
        if not (isinstance(module, nn.Linear) or _is_conv(module)):
            raise TypeError(
                f"ThreeFactor supports nn.Linear and nn.Conv1d/2d/3d, got {type(module).__name__}"
            )
        if rule not in _RULES:
            raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
        if noise < 0:
            raise ValueError("noise must be >= 0")
        if noise > 0 and (post is not None or post_fn is not None or rule != "hebb"):
            raise ValueError(
                "noise replaces the postsynaptic activity with the injected perturbation, "
                'so it cannot be combined with `post`, `post_fn` or rule="oja"'
            )
        if bias and module.bias is None:
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
        # [pre, post] per forward call; with noise, `post` holds xi / sigma.
        self.captures: list[list[Tensor]] = []
        self.enabled = True

        self.handles = [module.register_forward_hook(self._capture)]
        if post is not None:
            self.handles.append(post.register_forward_hook(self._capture_post))

    # -- hooks ---------------------------------------------------------------

    def _capture(self, module, inputs, output):
        if not (self.enabled and module.training):
            return None
        if self.noise == 0:
            self.captures.append([inputs[0].detach(), output.detach()])
            return None
        xi = torch.randn_like(output)
        self.captures.append([inputs[0].detach(), xi / self.noise])
        return output + self.noise * xi  # replaces the layer's output

    def _capture_post(self, module, inputs, output):
        if self.enabled and self.module.training and self.captures:
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

        # Third factor times postsynaptic activity, averaged over every
        # sample and every output position. With noise, average over samples
        # only (summing positions), so that a per-sample modulator makes the
        # result an estimate of the true gradient of the mean loss.
        if self.noise > 0:
            n = post.shape[0] if post.ndim > 1 else 1
        else:
            n = post.numel() // post.shape[ch]
        g = (mod * post).expand_as(post) / n

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

    def modulate(self, modulator: Tensor):
        if not self.captures:
            raise RuntimeError(
                "no activity captured for this layer since the last modulate()/reset(); "
                "run a forward pass in training mode first"
            )
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
    """Train selected Linear/Conv weights with a three-factor Hebbian rule.

    Args:
        modules: a layer, or an iterable of layers, to put under Hebbian
            control. More can be added (with per-layer options) via :meth:`add`.
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
    ):
        self._sites: list[_Site] = []
        if modules is None:
            return
        if isinstance(modules, nn.Module) and (
            isinstance(modules, nn.Linear) or _is_conv(modules)
        ):
            modules = [modules]
        modules = list(modules)
        if len(modules) > 1 and (mask is not None or post is not None):
            raise ValueError("`mask` and `post` are per-layer; use add() for each layer")
        for m in modules:
            self.add(
                m, mask=mask, bias=bias, post=post, pre_fn=pre_fn, post_fn=post_fn, rule=rule,
                noise=noise,
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
    ) -> "ThreeFactor":
        """Put ``module`` under Hebbian control. Returns ``self`` for chaining."""
        if any(s.module is module for s in self._sites):
            raise ValueError("module is already managed by this ThreeFactor")
        self._sites.append(_Site(module, mask, bias, post, pre_fn, post_fn, rule, noise))
        return self

    @torch.no_grad()
    def modulate(self, modulator: Union[float, Tensor]) -> None:
        """Deliver the third factor and write the Hebbian update into ``.grad``.

        Call after ``loss.backward()`` (if any) and before ``optimizer.step()``.

        Args:
            modulator: a scalar, a 1-D per-sample tensor of shape ``(batch,)``,
                or any tensor broadcastable to the postsynaptic activity
                (e.g. ``(batch, out_features)`` for a per-neuron signal).

        The update is averaged over samples and output positions, summed over
        all forward passes since the last call, and the captured activity is
        then cleared.
        """
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
