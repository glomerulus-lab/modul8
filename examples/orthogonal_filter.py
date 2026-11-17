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
fully connected layer and the readout -- is the vgg8 of
examples/cifar10_vgg.py, and the model exposes the same `hidden()` /
`hidden_grad` interface, so it can later be trained with the local rules via
that example's training loop.

Training is plain backprop on every learned parameter (the projections P,
the conv biases, the hidden fully connected layer and the readout), with the
data pipeline, optimizer and lr schedule of examples/cifar10_resnet.py. Each
epoch also logs max |P P^T - I| as a check that the projections stay
orthonormal.

Run from the repository root:

    uv run --group examples python examples/orthogonal_filter.py --d 32
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from csv import DictWriter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from cifar10_resnet import accuracy, augment, load_cifar10, make_optimizer, make_schedule, norm, progress
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


# -- training (plain backprop) -------------------------------------------------


def orthogonality_error(model) -> float:
    """max |P P^T - I| over every conv's projection."""
    errs = []
    for conv in (m for m in model.modules() if isinstance(m, OrthogonalFilterConv)):
        P = conv.proj.weight
        errs.append((P @ P.T - torch.eye(conv.d, device=P.device)).abs().max().item())
    return max(errs)


def param_groups(model, wd: float):
    """Weight decay on the fully connected weights only. The projections'
    unconstrained `original` tensors are excluded (decaying them has no
    meaning for P, which stays orthonormal), as are the biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if "parametrizations" in name or name.endswith("bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train(args, data, seed: int) -> dict:
    (x, y), (x_test, y_test) = data
    device = x.device
    torch.manual_seed(seed)
    model = OrthogonalFilterVGG(args.d, args.arch, args.feat_mult, args.num_hidden,
                                args.dropout, batchnorm=args.batchnorm).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"seed {seed}  d={args.d}  trainable parameters {n_params:,}", flush=True)

    opt = make_optimizer(param_groups(model, args.wd), args.opt, args.lr, args.wd,
                         args.momentum)
    steps_per_epoch = len(x) // args.batch_size
    sched = make_schedule(opt, args.epochs * steps_per_epoch,
                          args.warmup_epochs * steps_per_epoch, args.schedule,
                          args.min_lr_frac)

    history = {"epoch": [], "loss": [], "train": [], "test": [], "ortho_err": []}
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
            if args.progress and (k + 1) % 20 == 0:
                progress(f"seed {seed} ep {epoch:>3}", k + 1, steps_per_epoch,
                         loss_sum / seen, 100.0 * correct / seen, t0)

        train_acc = 100.0 * correct / seen  # running, on augmented batches
        test_acc = accuracy(model, x_test, y_test)
        ortho = orthogonality_error(model)
        for key, val in zip(history, (epoch, loss_sum / seen, train_acc, test_acc, ortho)):
            history[key].append(val)
        if args.progress:
            sys.stderr.write("\r\033[K")
        print(f"seed {seed} ep {epoch:>3}  loss {loss_sum / seen:.3f}  train {train_acc:6.2f}%  "
              f"test {test_acc:6.2f}%  |PP^T-I| {ortho:.1e}  ({time.time() - t0:.0f}s)",
              flush=True)

    history["time"] = time.time() - t_run
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--d", type=int, default=32,
                   help="dimension of each conv's learned projection (capped at C_in)")
    p.add_argument("--arch", choices=list(ARCHS), default="vgg8")
    p.add_argument("--feat-mult", type=float, default=1.0, help="width multiplier for the convs")
    p.add_argument("--num-hidden", type=int, default=1024,
                   help="units in the hidden fully connected layer")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batchnorm", action=argparse.BooleanOptionalAction, default=True,
                   help="batch norm (affine=False) after each ReLU")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-size", type=int, default=0,
                   help="use only the first n training images (0: all 50,000)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="random crop + horizontal flip")
    p.add_argument("--opt", choices=("sgd", "adam"), default="sgd",
                   help="optimizer (adam means AdamW)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4,
                   help="weight decay (fully connected weights only)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (Nesterov)")
    p.add_argument("--schedule", choices=("constant", "cosine", "step"), default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=1, help="linear lr warmup")
    p.add_argument("--min-lr-frac", type=float, default=0.0,
                   help="final lr as a fraction of the initial lr")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="disable the progress bar")
    p.add_argument("--data", default="data", help="dataset download directory")
    p.add_argument("--csv", help="per-epoch results table; a timestamp is added to the name "
                                 "(default: cifar10_<arch>_orthogonal_d<d>.csv)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    (x, y), test = load_cifar10(args.data, args.device)
    if args.train_size:
        x, y = x[:args.train_size], y[:args.train_size]
    if len(x) < args.batch_size:
        sys.exit("--train-size must be at least --batch-size")
    data = ((x, y), test)

    base = Path(args.csv or f"cifar10_{args.arch}_orthogonal_d{args.d}.csv")
    csv = base.with_name(f"{base.stem}_{time.strftime('%Y%m%d-%H%M%S')}{base.suffix or '.csv'}")

    rows = []
    for seed in range(args.seeds):
        hist = train(args, data, seed)
        print(f"seed {seed} final train {hist['train'][-1]:6.2f}%  "
              f"test {hist['test'][-1]:6.2f}%  best test {max(hist['test']):6.2f}%  "
              f"({hist['time']:.0f}s)\n", flush=True)
        rows += [{"seed": seed, "d": args.d, "arch": args.arch,
                  **{k: hist[k][i] for k in ("epoch", "loss", "train", "test", "ortho_err")}}
                 for i in range(len(hist["epoch"]))]
        # Rewritten after every seed, so a crash keeps the finished runs.
        with open(csv, "w", newline="") as f:
            writer = DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
