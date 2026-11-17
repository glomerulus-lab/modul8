"""CIFAR-10 VGG8 whose convs learn only a low-rank orthogonal channel filter.

A model in the spirit of Neural LoFi (github.com/IdePHICS/Neural-LoFi-Theory):
each layer learns a low-dimensional projection of its input features and
then expands it back out with a fixed random map. Every conv is split, as in
MobileNet, into a spatial part and a channel part:

    depthwise 3x3 (fixed, random)  ->  pointwise 1x1 = G @ P

where, for a layer with C_in input and C_out output channels,

  * P (d x C_in) is learned and kept orthonormal-rowed by
    torch.nn.utils.parametrizations.orthogonal, so it is an orthogonal
    projection of the C_in channels onto a d-dimensional subspace;
  * G (C_out x d) is a fixed Gaussian matrix, entries N(0, 1/d), that expands
    the d projected channels to C_out.

Only P (and the pointwise bias) is learned in the convs. When d >= C_in (the
first layer, C_in = 3) there is nothing to project away, so that layer uses
d = C_in and P is a learned rotation.

The rest of the network -- conv -> ReLU -> BN blocks, max-pools, the hidden
fully connected layer and the readout -- is the VGG of examples/cifar10_vgg.py.

Methods (as in examples/cifar10_vgg.py; the readout is always trained by
backprop):

    gd         every learned parameter trained by backprop
    local      layer-local error signals (Nøkland & Eidnes, 2019): each hidden
               layer -- every conv and the hidden fc layer -- is trained by
               the gradient of a local loss computed by a modul8.LocalHead on
               its ReLU output (--local-loss pred, sim or predsim)
    local-bio  the same with the reference code's "bio" heads
    frozen     hidden layers left at their random initialisation

modul8.LocalLoss writes its updates straight into `layer.weight.grad`, so it
needs plain Linear / Conv layers and cannot drive the parametrized P. Here
the same update is computed by autograd instead: each hidden layer's input
(and the readout's) is detached, so the backward pass of a layer's local
loss stops at that layer. That is backprop confined to the layer and its
head -- what LocalLoss computes for an ordinary layer -- and it reaches P
through the orthogonal parametrization.

Data pipeline, optimizers, lr schedule, alignment probe (--align-every),
results CSV and plot are those of examples/cifar10_resnet.py. Weight decay
applies to the fully connected weights only: not to the projections'
unconstrained `original` tensors (decaying them means nothing for P, which
stays orthonormal), nor to biases. Each epoch also logs max |P P^T - I|
(column ortho_err in the CSV) as a check that the projections stay
orthonormal.

Run from the repository root:

    uv run --group examples python examples/orthogonal_filter.py --d 32
    uv run --group examples python examples/orthogonal_filter.py --d 32 --methods gd local
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from cifar10_resnet import (accuracy, add_args, augment, make_optimizer, make_schedule, norm,
                            output_shapes, progress, run)
from cifar10_vgg import ARCHS, Block
from modul8 import LocalHead

METHODS = ("gd", "local", "local-bio", "frozen")


class OrthogonalFilterConv(nn.Module):
    """Fixed random depthwise kxk conv, then a 1x1 pointwise conv G @ P with
    P (d x C_in) learned orthonormal-rowed and G (C_out x d) fixed Gaussian."""

    def __init__(self, c_in: int, c_out: int, d: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = True):
        super().__init__()
        self.d = d = min(d, c_in)
        self.padding = padding

        # Spatial part: one random kxk filter per input channel, not learned.
        # He-scaled for the k*k fan-in.
        depthwise = torch.randn(c_in, 1, kernel_size, kernel_size)
        self.register_buffer("depthwise", depthwise * math.sqrt(2 / kernel_size**2))

        # Learned channel projection. The parametrization acts on this 2-D
        # weight; forward views it as the (d, C_in, 1, 1) conv kernel (applying
        # `orthogonal` to a 4-D conv weight would treat it as a batch of 1x1
        # matrices).
        self.proj = orthogonal(nn.Linear(c_in, d, bias=False))

        # Fixed random expansion; N(0, 1/d) keeps the expected squared norm.
        self.register_buffer("expand", torch.randn(c_out, d) / math.sqrt(d))

        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None

    def pointwise_weight(self) -> torch.Tensor:
        """The full (C_out, C_in, 1, 1) pointwise kernel G @ P."""
        return (self.expand @ self.proj.weight)[:, :, None, None]

    def forward(self, x):
        x = F.conv2d(x, self.depthwise, padding=self.padding, groups=x.shape[1])
        x = F.conv2d(x, self.proj.weight[:, :, None, None])  # project: C_in -> d
        return F.conv2d(x, self.expand[:, :, None, None], self.bias)  # expand: d -> C_out

    def extra_repr(self) -> str:
        c_out, d = self.expand.shape
        return f"{self.depthwise.shape[0]} -> {d} -> {c_out}"


class OrthogonalFilterVGG(nn.Module):
    """vgg8 (Nøkland & Eidnes) with every conv an OrthogonalFilterConv."""

    def __init__(self, d: int = 32, arch: str = "vgg8", feat_mult: float = 1.0,
                 num_hidden: int = 1024, dropout: float = 0.0, num_classes: int = 10,
                 batchnorm: bool = True, input_size: int = 32):
        super().__init__()
        layers, c, size = [], 3, input_size
        for v in ARCHS[arch]:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                size //= 2
            else:
                w = int(v * feat_mult)
                layers.append(Block(OrthogonalFilterConv(c, w, d), norm(w, batchnorm),
                                    dropout, spatial=True))
                c = w
        self.convs = nn.Sequential(*layers)
        self.hidden_fc = Block(nn.Linear(c * size * size, num_hidden),
                               nn.BatchNorm1d(num_hidden, affine=False) if batchnorm
                               else nn.Identity(), dropout, spatial=False)
        self.fc = nn.Linear(num_hidden, num_classes)
        self.hidden_grad = True

        nn.init.kaiming_normal_(self.hidden_fc.layer.weight, nonlinearity="relu")
        nn.init.zeros_(self.hidden_fc.layer.bias)

    def features(self, x):
        return self.hidden_fc(self.convs(x).flatten(1))

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.hidden_grad):
            f = self.features(x)
        return self.fc(f)

    def hidden(self):
        """(layer, post module) for every conv and the hidden fc layer."""
        blocks = [m for m in self.convs if isinstance(m, Block)] + [self.hidden_fc]
        return [(b.layer, b.relu) for b in blocks]



# -- training ------------------------------------------------------------------


def orthogonality_error(model) -> float:
    """max |P P^T - I| over every conv's projection."""
    errs = []
    with torch.no_grad():
        for conv in (m for m in model.modules() if isinstance(m, OrthogonalFilterConv)):
            P = conv.proj.weight
            errs.append((P @ P.T - torch.eye(conv.d, device=P.device)).abs().max().item())
    return max(errs)


def param_groups(named_params, wd: float):
    """Weight decay on weights only: the projections' unconstrained
    `original` tensors and all biases are excluded."""
    decay, no_decay = [], []
    for name, p in named_params:
        (no_decay if "parametrizations" in name or name.endswith("bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class LocalSignals:
    """Layer-local training by autograd. Forward pre-hooks detach the input
    of every hidden layer and of the readout; forward hooks keep each hidden
    layer's post (ReLU) output. `loss(y)` sums the heads' local losses on
    those outputs, so backward sends each layer only its own head's error."""

    def __init__(self, model, hidden, heads):
        self.heads = heads
        self.acts: list[torch.Tensor] = []
        self.enabled = True
        self.handles = [model.fc.register_forward_pre_hook(self._detach)]
        for layer, post in hidden:
            self.handles.append(layer.register_forward_pre_hook(self._detach))
            self.handles.append(post.register_forward_hook(self._capture))

    def _detach(self, module, inputs):
        return tuple(i.detach() for i in inputs) if self.enabled else None

    def _capture(self, module, inputs, output):
        if self.enabled and module.training:
            self.acts.append(output)

    def loss(self, y):
        """Sum of the local losses of the last training forward pass."""
        acts, self.acts = self.acts, []
        if len(acts) != len(self.heads):
            raise RuntimeError(f"captured {len(acts)} activities for {len(self.heads)} heads; "
                               "run exactly one training-mode forward pass first")
        return sum(head(h, y) for head, h in zip(self.heads, acts))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.acts.clear()


def alignment(model, local, layers, x, y) -> list[float]:
    """Cosine between each layer's local-loss gradient and its backprop
    gradient on (x, y). Leaves parameters, .grad and batch-norm statistics
    unchanged; both passes see the same dropout masks."""
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    per_layer = [list(layer.parameters()) for layer in layers]
    params = [p for ps in per_layer for p in ps]
    devices = [x.device] if x.is_cuda else []

    def grads(enabled):
        local.enabled = enabled
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            logits = model(x)
            loss = local.loss(y) if enabled else F.cross_entropy(logits, y)
        return torch.autograd.grad(loss, params)

    try:
        true, est = grads(False), grads(True)
    finally:
        local.enabled = True
        local.acts.clear()
        for k, v in model.named_buffers():
            v.copy_(buffers[k])

    cos, i = [], 0
    for ps in per_layer:
        a = torch.cat([g.flatten() for g in true[i:i + len(ps)]])
        b = torch.cat([g.flatten() for g in est[i:i + len(ps)]])
        cos.append(F.cosine_similarity(a, b, dim=0).item())
        i += len(ps)
    return cos


def train(method: str, data, args, seed: int, build_model, label: str = "") -> dict:
    """One run; same signature and history as cifar10_resnet.train, plus the
    per-epoch orthogonality error under history["extra"]."""
    (x, y), (x_test, y_test) = data
    device = x.device
    torch.manual_seed(seed)
    model = build_model(args).to(device)
    hidden = model.hidden()
    layers = [layer for layer, _ in hidden]
    hidden_named = [(n, p) for layer in layers for n, p in layer.named_parameters()]

    # The readout is always trained by backprop.
    opts = [make_optimizer(param_groups(model.fc.named_parameters(), args.wd), args.opt,
                           args.lr, args.wd, args.momentum)]
    local = None
    readout_scale = 1.0  # multiplies the readout's loss before backward
    if method == "gd":
        opts.append(make_optimizer(param_groups(hidden_named, args.wd), args.opt, args.lr,
                                   args.wd, args.momentum))
    elif method in ("local", "local-bio"):
        bio = method == "local-bio"
        beta = args.local_bio_beta if bio else args.local_beta
        gen = torch.Generator().manual_seed(seed)  # bio heads' fixed random matrices
        heads = nn.ModuleList(
            LocalHead(shape, 10, loss=args.local_loss, beta=beta,
                      dim_in_decoder=args.local_dim_in_decoder, bio=bio,
                      target_proj_size=args.local_target_proj_size, generator=gen)
            for shape in output_shapes(model, layers, device)).to(device)
        local = LocalSignals(model, hidden, heads)
        # As in the reference code: the readout starts at zero and, for
        # predsim, its loss is scaled by (1 - beta).
        if args.local_readout_zero_init:
            nn.init.zeros_(model.fc.weight)
        if args.local_readout_scale is not None:
            readout_scale = args.local_readout_scale
        elif args.local_loss == "predsim":
            readout_scale = 1 - beta
        # The heads are trained alongside the layers they serve.
        opts.append(make_optimizer(param_groups(hidden_named + list(heads.named_parameters()),
                                                args.local_wd),
                                   args.local_opt, args.local_lr, args.local_wd, args.momentum))
    else:  # frozen: the hidden parameters are in no optimizer
        model.hidden_grad = False

    steps_per_epoch = len(x) // args.batch_size
    total = args.epochs * steps_per_epoch
    schedulers = [make_schedule(opt, total, args.warmup_epochs * steps_per_epoch,
                                args.schedule, args.min_lr_frac) for opt in opts]

    # A fixed held-out-from-training batch for the alignment probe.
    gen = torch.Generator().manual_seed(seed)
    probe = torch.randperm(len(x_test), generator=gen)[:args.batch_size].to(device)
    x_probe, y_probe = x_test[probe], y_test[probe]

    history = {"train": [], "test": [], "loss": [], "align": [], "extra": {"ortho_err": []}}
    t_run = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = correct = seen = 0
        perm = torch.randperm(len(x), device=device)
        for k in range(steps_per_epoch):
            idx = perm[k * args.batch_size:(k + 1) * args.batch_size]
            xb, yb = x[idx], y[idx]
            if args.augment:
                xb = augment(xb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            objective = readout_scale * loss
            if local is not None:
                objective = objective + local.loss(yb)
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            objective.backward()
            for opt in opts:
                opt.step()
            for sched in schedulers:
                sched.step()

            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
            if args.progress and (k + 1) % 20 == 0:
                progress(f"{label} ep {epoch:>3}", k + 1, steps_per_epoch,
                         loss_sum / seen, 100.0 * correct / seen, t0)

        train_acc = 100.0 * correct / seen  # running, on augmented batches
        test_acc = accuracy(model, x_test, y_test)
        ortho = orthogonality_error(model)
        history["train"].append(train_acc)
        history["test"].append(test_acc)
        history["loss"].append(loss_sum / seen)
        history["extra"]["ortho_err"].append(ortho)
        align = ""
        if local is not None and args.align_every and (epoch % args.align_every == 0
                                                       or epoch == args.epochs):
            cos = alignment(model, local, layers, x_probe, y_probe)
            history["align"].append((epoch, cos))
            align = f"  align {statistics.fmean(cos):+.3f}"
        if args.progress:
            sys.stderr.write("\r\033[K")
        print(f"{label} ep {epoch:>3}  loss {loss_sum / seen:.3f}  train {train_acc:6.2f}%  "
              f"test {test_acc:6.2f}%{align}  |PP^T-I| {ortho:.1e}  ({time.time() - t0:.0f}s)",
              flush=True)

    history["time"] = time.time() - t_run
    if local is not None:
        local.remove()
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_args(p, ARCHS, "vgg8")
    p.set_defaults(methods=list(METHODS))
    p.add_argument("--d", type=int, default=32,
                   help="dimension of each conv's learned projection (capped at C_in)")
    p.add_argument("--feat-mult", type=float, default=1.0, help="width multiplier for the convs")
    p.add_argument("--num-hidden", type=int, default=1024,
                   help="units in the hidden fully connected layer")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout after each hidden layer's batch norm")
    args = p.parse_args()
    if unsupported := [m for m in args.methods if m not in METHODS]:
        p.error(f"methods {unsupported} are not supported here (choose from {list(METHODS)})")

    mult = "" if args.feat_mult == 1 else f" ({args.feat_mult:g}x)"
    args.arch_label = f"{args.arch}{mult}, orthogonal d={args.d}"
    if args.out is None:
        suffix = "" if args.feat_mult == 1 else f"_{args.feat_mult:g}x"
        args.out = f"cifar10_{args.arch}{suffix}_orthogonal_d{args.d}.png"

    run(args, lambda a: OrthogonalFilterVGG(a.d, a.arch, a.feat_mult, a.num_hidden, a.dropout,
                                            batchnorm=a.batchnorm),
        train_fn=train)


if __name__ == "__main__":
    main()
