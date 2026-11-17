"""CIFAR-10 ResNets trained by DFA or local error signals vs. backprop.

A CIFAR-style ResNet (ResNet-20/32/44/56/110, or the wider ResNet-18/34) is
trained on CIFAR-10 once per method, and test accuracy is tracked per epoch.
Every conv -- including the stem and the 1x1 projection shortcuts -- is a
hidden layer trained by the method; the readout is always trained by
backprop. Methods, training loop, alignment probe and outputs are those of
examples/cifar10_common.py (see its docstring).

Block layout. Following examples/mnist_learning_curve.py, batch norm (with
affine=False, so it has nothing to train) sits *after* each ReLU:

    h   = BN(ReLU(conv1(x)))
    out = BN(ReLU(conv2(h) + shortcut(x)))

so the ReLU's derivative is each conv's exact local derivative, which is what
DFA uses. conv2 and a projection shortcut feed the same units (their sum), so
by default they share one feedback matrix (--no-share-shortcut-feedback gives
each its own) and, for local, one head. With --dfa-feedback tied, each conv's
feedback is drawn per channel and shared across spatial positions instead of
per unit.

Run from the repository root:

    uv run --group examples python examples/cifar10_resnet.py --arch resnet20
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from cifar10_common import add_args, norm, run

# name: (stage widths, blocks per stage, stem width)
ARCHS = {
    "resnet20": ((16, 32, 64), (3, 3, 3), 16),
    "resnet32": ((16, 32, 64), (5, 5, 5), 16),
    "resnet44": ((16, 32, 64), (7, 7, 7), 16),
    "resnet56": ((16, 32, 64), (9, 9, 9), 16),
    "resnet110": ((16, 32, 64), (18, 18, 18), 16),
    "resnet18": ((64, 128, 256, 512), (2, 2, 2, 2), 64),
    "resnet34": ((64, 128, 256, 512), (3, 4, 6, 3), 64),
}


class BasicBlock(nn.Module):
    """Two 3x3 convs with a residual connection; every ReLU is its own module
    (DFA hooks a layer's nonlinearity, so it must not be shared)."""

    def __init__(self, c_in: int, c_out: int, stride: int, batchnorm: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1)
        self.relu1 = nn.ReLU()
        self.bn1 = norm(c_out, batchnorm)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.bn2 = norm(c_out, batchnorm)
        # 1x1 projection when the shape changes, identity otherwise.
        self.shortcut = (nn.Conv2d(c_in, c_out, 1, stride)
                         if stride != 1 or c_in != c_out else None)

    def forward(self, x):
        h = self.bn1(self.relu1(self.conv1(x)))
        s = x if self.shortcut is None else self.shortcut(x)
        return self.bn2(self.relu2(self.conv2(h) + s))

    def hidden(self):
        """(layer, post module) pairs; the shortcut shares conv2's ReLU."""
        pairs = [(self.conv1, self.relu1), (self.conv2, self.relu2)]
        if self.shortcut is not None:
            pairs.append((self.shortcut, self.relu2))
        return pairs


class ResNet(nn.Module):
    def __init__(self, arch: str, num_classes: int = 10, batchnorm: bool = True):
        super().__init__()
        widths, blocks, stem = ARCHS[arch]
        self.stem = nn.Conv2d(3, stem, 3, 1, 1)
        self.stem_relu = nn.ReLU()
        self.stem_bn = norm(stem, batchnorm)
        layers, c = [], stem
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock(c, w, 2 if (i > 0 and j == 0) else 1, batchnorm))
                c = w
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(c, num_classes)
        # When False, the hidden layers run without autograd: only the readout
        # gets a backprop gradient (used for dfa, local and frozen).
        self.hidden_grad = True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def features(self, x):
        x = self.stem_bn(self.stem_relu(self.stem(x)))
        x = self.blocks(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.hidden_grad):
            f = self.features(x)
        return self.fc(f)

    def hidden(self):
        """(layer, post module) for every conv; the fc readout is excluded."""
        pairs = [(self.stem, self.stem_relu)]
        for block in self.blocks:
            pairs += block.hidden()
        return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_args(p, ARCHS, "resnet20")
    run(p.parse_args(), lambda a: ResNet(a.arch, batchnorm=a.batchnorm))


if __name__ == "__main__":
    main()
