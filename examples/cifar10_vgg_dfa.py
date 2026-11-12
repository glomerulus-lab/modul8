"""CIFAR-10 VGG nets trained by direct feedback alignment vs. backprop.

The VGG variants are those of Nøkland & Eidnes, "Training Neural Networks
with Local Error Signals" (ICML 2019; github.com/anokland/local-loss),
including their widened versions: --feat-mult scales every conv layer's width,
so the paper's VGG11B(3x) is `--arch vgg11b --feat-mult 3`. After the convs
comes one hidden fully connected layer (--num-hidden, 1024 by default, not
scaled), then the linear readout.

Methods, training loop, alignment probe and plot are those of
examples/cifar10_resnet_dfa.py (see its docstring): every hidden layer -- each
conv and the hidden fc layer -- is trained by the method, and the readout is
always trained by backprop.

Differences from the paper's networks, to fit this repo's DFA convention:

  * Batch norm (affine=False) sits *after* the ReLU, conv -> ReLU -> BN,
    so the ReLU's derivative is each layer's exact local derivative; the paper
    uses conv -> BN -> nonlinearity.
  * ReLU throughout; the paper's recipes use leaky ReLU.
  * Max-pooling follows the BN (and dropout). DFA ignores the pool, so units
    the pool discards still receive DFA updates.

The paper trains 400 epochs with Adam; the default here is a 30-epoch run.
Its recipes use dropout 0.2 for vgg8b and 0.3 for vgg11b (3x) (--dropout).

Run from the repository root:

    uv run --group examples python examples/cifar10_vgg_dfa.py --arch vgg8b
    uv run --group examples python examples/cifar10_vgg_dfa.py --arch vgg11b --feat-mult 3
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from cifar10_resnet_dfa import add_args, norm, run

# Nøkland & Eidnes's configurations: conv widths (before --feat-mult), "M" is
# a 2x2 max-pool.
ARCHS = {
    "vgg6a": [128, "M", 256, "M", 512, "M", 512],
    "vgg6b": [128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg8a": [128, 256, "M", 256, 512, "M", 512, "M", 512],
    "vgg8b": [128, 256, "M", 256, 512, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg11a": [128, 256, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "vgg11b": [128, 128, 128, 256, "M", 256, 512, "M", 512, 512, "M", 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
              512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class Block(nn.Module):
    """layer -> ReLU -> BN -> dropout; the ReLU is the layer's own module
    (DFA hooks it)."""

    def __init__(self, layer: nn.Module, bn: nn.Module, dropout: float, spatial: bool):
        super().__init__()
        self.layer = layer
        self.relu = nn.ReLU()
        self.bn = bn
        drop = nn.Dropout2d if spatial else nn.Dropout
        self.drop = drop(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.bn(self.relu(self.layer(x))))


class VGG(nn.Module):
    def __init__(self, arch: str, feat_mult: float = 1.0, num_hidden: int = 1024,
                 dropout: float = 0.0, num_classes: int = 10, batchnorm: bool = True,
                 input_size: int = 32):
        super().__init__()
        layers, c, size = [], 3, input_size
        for v in ARCHS[arch]:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                size //= 2
            else:
                w = int(v * feat_mult)
                layers.append(Block(nn.Conv2d(c, w, 3, 1, 1), norm(w, batchnorm),
                                    dropout, spatial=True))
                c = w
        self.convs = nn.Sequential(*layers)
        self.hidden_fc = Block(nn.Linear(c * size * size, num_hidden),
                               nn.BatchNorm1d(num_hidden, affine=False) if batchnorm
                               else nn.Identity(), dropout, spatial=False)
        self.fc = nn.Linear(num_hidden, num_classes)
        # When False, the hidden layers run without autograd: only the readout
        # gets a backprop gradient (used for dfa and frozen).
        self.hidden_grad = True

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m is not self.fc:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_args(p, ARCHS, "vgg8b")
    p.add_argument("--feat-mult", type=float, default=1.0,
                   help="width multiplier for every conv layer (the paper's '(2x)', '(3x)')")
    p.add_argument("--num-hidden", type=int, default=1024,
                   help="units in the hidden fully connected layer")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout after each hidden layer's batch norm")
    args = p.parse_args()

    mult = f"{args.feat_mult:g}x"
    args.arch_label = args.arch if args.feat_mult == 1 else f"{args.arch} ({mult})"
    if args.out is None:
        suffix = "" if args.feat_mult == 1 else f"_{mult}"
        args.out = f"cifar10_{args.arch}{suffix}_dfa.png"

    run(args, lambda a: VGG(a.arch, a.feat_mult, a.num_hidden, a.dropout,
                            batchnorm=a.batchnorm))


if __name__ == "__main__":
    main()
