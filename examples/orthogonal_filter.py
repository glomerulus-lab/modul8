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
fully connected layer and the readout -- is the VGG of examples/cifar10_vgg.py,
and training (methods gd, dfa, local, local-bio and frozen; options; alignment
probe; results CSV and plot) is that of examples/cifar10_resnet.py. The
three-factor rules (DirectFeedback, LocalLoss) handle OrthogonalFilterConv
through modul8's autograd fallback: the layer is re-run on its captured input
and the modulated error is backpropagated through it alone, so P's update
goes through the orthogonal parametrization into its unconstrained
`original` tensor.

Weight decay applies to the fully connected weights only: not to the
projections' `original` tensors (decaying them means nothing for P, which
stays orthonormal), nor to biases. Each epoch also logs max |P P^T - I|
(ortho_err, also in the CSV) as a check that the projections stay
orthonormal.

Run from the repository root:

    uv run --group examples python examples/orthogonal_filter.py --d 32
    uv run --group examples python examples/orthogonal_filter.py --d 32 --methods gd local
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from cifar10_resnet import add_args, norm, run
from cifar10_vgg import ARCHS, Block


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

    def param_groups(self, named_params, wd: float):
        """Optimizer groups (used by cifar10_resnet.train): weight decay on
        weights only, not on the projections' unconstrained `original`
        tensors or on biases."""
        decay, no_decay = [], []
        for name, p in named_params:
            (no_decay if "parametrizations" in name or name.endswith("bias")
             else decay).append(p)
        return [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]

    @torch.no_grad()
    def epoch_metrics(self) -> dict[str, float]:
        """max |P P^T - I| over every conv's projection (logged per epoch)."""
        errs = []
        for conv in (m for m in self.modules() if isinstance(m, OrthogonalFilterConv)):
            P = conv.proj.weight
            errs.append((P @ P.T - torch.eye(conv.d, device=P.device)).abs().max().item())
        return {"ortho_err": max(errs)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_args(p, ARCHS, "vgg8")
    p.add_argument("--d", type=int, default=32,
                   help="dimension of each conv's learned projection (capped at C_in)")
    p.add_argument("--feat-mult", type=float, default=1.0, help="width multiplier for the convs")
    p.add_argument("--num-hidden", type=int, default=1024,
                   help="units in the hidden fully connected layer")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout after each hidden layer's batch norm")
    args = p.parse_args()

    mult = "" if args.feat_mult == 1 else f" ({args.feat_mult:g}x)"
    args.arch_label = f"{args.arch}{mult}, orthogonal d={args.d}"
    if args.out is None:
        suffix = "" if args.feat_mult == 1 else f"_{args.feat_mult:g}x"
        args.out = f"cifar10_{args.arch}{suffix}_orthogonal_d{args.d}.png"

    run(args, lambda a: OrthogonalFilterVGG(a.d, a.arch, a.feat_mult, a.num_hidden, a.dropout,
                                            batchnorm=a.batchnorm))


if __name__ == "__main__":
    main()
