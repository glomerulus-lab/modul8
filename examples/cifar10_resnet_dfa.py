"""CIFAR-10 ResNets trained by direct feedback alignment vs. backprop.

A CIFAR-style ResNet (ResNet-20/32/44/56/110, or the wider ResNet-18/34) is
trained on CIFAR-10 once per method, and test accuracy is tracked per epoch.

Methods (they differ only in how the hidden layers -- every conv, including
the stem and the 1x1 projection shortcuts -- learn; the final linear readout
is always trained by backprop):

    gd      hidden layers trained by backprop (the usual network)
    dfa     direct feedback alignment: each hidden conv's third factor is the
            output error (softmax - one-hot) sent through a fixed random
            matrix B; the postsynaptic factor is relu'(a)
    frozen  hidden layers left at their random initialisation (control: a
            method must beat this to be learning useful features)

Block layout. Following examples/mnist_learning_curve.py, batch norm (with
affine=False, so it has nothing to train) sits *after* each ReLU:

    h   = BN(ReLU(conv1(x)))
    out = BN(ReLU(conv2(h) + shortcut(x)))

so the ReLU's derivative is each conv's exact local derivative, which is what
DFA uses. conv2 and a projection shortcut feed the same units (their sum), so
by default they share one feedback matrix (--no-share-shortcut-feedback gives
each its own).

For dfa and frozen no error is backpropagated into the hidden layers at all:
the features are computed without autograd and only the readout builds a
graph, so a training step costs roughly a forward pass plus the DFA update.

With --align-every, the cosine between each layer's DFA update and its true
backprop gradient is measured on a held-out batch: alignment near 0 means DFA
is not following the gradient for that layer.

Run from the repository root:

    uv run --group examples python examples/cifar10_resnet_dfa.py --arch resnet20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

from modul8 import DirectFeedback

METHODS = ("gd", "dfa", "frozen")
LABELS = {
    "gd": "Backprop",
    "dfa": "Direct feedback alignment",
    "frozen": "Frozen random hidden",
}

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


# -- model -------------------------------------------------------------------


def norm(c: int, batchnorm: bool) -> nn.Module:
    return nn.BatchNorm2d(c, affine=False) if batchnorm else nn.Identity()


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
        # gets a backprop gradient (used for dfa and frozen).
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


# -- data --------------------------------------------------------------------


def load_cifar10(root: str, device):
    train, test = (datasets.CIFAR10(root, train=t, download=True) for t in (True, False))

    def raw(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255)
        return x, torch.tensor(ds.targets)

    (x, y), (xt, yt) = raw(train), raw(test)
    # Per-channel standardisation with the training set's statistics.
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True)
    return (((x - mean) / std).to(device), y.to(device)), (((xt - mean) / std).to(device),
                                                          yt.to(device))


def augment(x, pad: int = 4):
    """Standard CIFAR augmentation on a GPU batch: random crop after zero
    padding by `pad`, and random horizontal flip."""
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (i[:, None] + torch.arange(h, device=x.device))[:, :, None]  # (b, h, 1)
    cols = (j[:, None] + torch.arange(w, device=x.device))[:, None, :]  # (b, 1, w)
    batch = torch.arange(b, device=x.device)[:, None, None]
    x = xp[batch, :, rows, cols].permute(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)
    flip = torch.rand(b, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


# -- training helpers ----------------------------------------------------------


@torch.no_grad()
def accuracy(model, x, y, batch: int = 1000) -> float:
    was = model.training
    model.eval()
    correct = sum((model(xb).argmax(1) == yb).sum().item()
                  for xb, yb in zip(x.split(batch), y.split(batch)))
    model.train(was)
    return 100.0 * correct / len(y)


def progress(label: str, step: int, total: int, loss: float, acc: float, t0: float):
    """Redraw a one-line progress bar in place on stderr."""
    width = 24
    filled = width * step // total
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r{label} [{bar}] {step:>{len(str(total))}}/{total}  "
                     f"loss {loss:.3f}  train {acc:6.2f}%  ({time.time() - t0:.0f}s)")
    sys.stderr.flush()


def make_optimizer(params, kind: str, lr: float, wd: float, momentum: float):
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=momentum > 0,
                               weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def make_schedule(opt, total: int, warmup: int, schedule: str, floor: float):
    """Linear warmup over `warmup` steps, then cosine or step decay to
    `floor` x the initial lr at `total`."""

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        if schedule == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
        if schedule == "step":  # /10 at 50% and 75% of the post-warmup steps
            return max(floor, 0.1 ** ((t >= 0.5) + (t >= 0.75)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


@torch.no_grad()
def output_shapes(model, layers, device):
    """Per-sample output shape of each layer, from one eval-mode forward."""
    shapes, handles = {}, []
    for layer in layers:
        handles.append(layer.register_forward_hook(
            lambda m, i, o: shapes.__setitem__(id(m), o.shape[1:])))
    was = model.training
    model.eval()
    model(torch.zeros(1, 3, 32, 32, device=device))
    model.train(was)
    for h in handles:
        h.remove()
    return [shapes[id(layer)] for layer in layers]


def feedback_matrices(model, hidden, n_out: int, scale: float, share_shortcut: bool,
                      generator: torch.Generator, device):
    """Draw DFA's fixed random matrices B, N(0, scale^2 / n_out), as
    DirectFeedback would, optionally sharing one B between a block's conv2 and
    its projection shortcut (they feed the same units)."""
    shapes = output_shapes(model, [layer for layer, _ in hidden], device)
    by_post: dict[int, torch.Tensor] = {}
    mats = []
    for (layer, post), shape in zip(hidden, shapes):
        if share_shortcut and id(post) in by_post:
            mats.append(by_post[id(post)])
            continue
        B = torch.randn(n_out, math.prod(shape), generator=generator)
        B = (B * (scale / math.sqrt(n_out))).to(device=layer.weight.device,
                                               dtype=layer.weight.dtype)
        by_post[id(post)] = B
        mats.append(B)
    return mats


def dfa_alignment(model, dfa, layers, x, y) -> list[float]:
    """Cosine between each layer's DFA update and its backprop gradient on
    (x, y). Leaves parameters, grads and batch-norm statistics unchanged."""
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    params = [p for layer in layers for p in (layer.weight, layer.bias)]
    saved = [None if p.grad is None else p.grad.clone() for p in params]
    hidden_grad = model.hidden_grad

    dfa.reset()
    model.hidden_grad = True
    for p in params:
        p.grad = None
    logits = model(x)
    F.cross_entropy(logits, y).backward()
    true = [torch.cat([layer.weight.grad.flatten(), layer.bias.grad.flatten()])
            for layer in layers]
    dfa.modulate(logits.detach().softmax(1) - F.one_hot(y, logits.shape[1]))
    est = [torch.cat([layer.weight.grad.flatten(), layer.bias.grad.flatten()])
           for layer in layers]
    cos = [F.cosine_similarity(a, b, dim=0).item() for a, b in zip(true, est)]

    model.hidden_grad = hidden_grad
    for p, g in zip(params, saved):
        p.grad = g
    model.fc.zero_grad(set_to_none=True)
    for k, v in model.named_buffers():
        v.copy_(buffers[k])
    return cos


# -- one run -----------------------------------------------------------------


def train(method: str, data, args, seed: int, label: str = "") -> dict:
    """Train one model; return its per-epoch history."""
    (x, y), (x_test, y_test) = data
    device = x.device
    torch.manual_seed(seed)
    model = ResNet(args.arch, batchnorm=args.batchnorm).to(device)
    hidden = model.hidden()
    layers = [layer for layer, _ in hidden]
    hidden_params = [p for layer in layers for p in layer.parameters()]

    # The readout is always trained by backprop.
    opts = [make_optimizer(model.fc.parameters(), args.opt, args.lr, args.wd, args.momentum)]
    dfa = None
    if method == "gd":
        opts.append(make_optimizer(hidden_params, args.opt, args.lr, args.wd, args.momentum))
    elif method == "dfa":
        model.hidden_grad = False
        dfa = DirectFeedback(layers, n_out=10, posts=[post for _, post in hidden])
        dfa.load_state_dict({"feedback": feedback_matrices(
            model, hidden, 10, args.dfa_scale, args.share_shortcut_feedback,
            torch.Generator().manual_seed(seed), device)})
        opts.append(make_optimizer(hidden_params, args.dfa_opt, args.dfa_lr, args.dfa_wd,
                                   args.momentum))
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

    history = {"train": [], "test": [], "loss": [], "align": []}
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
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss.backward()  # the readout (and, for gd, every layer)
            if dfa is not None:
                # Output error e = softmax - one-hot, i.e. dLoss/dlogits per sample.
                dfa.modulate(logits.detach().softmax(1) - F.one_hot(yb, 10))
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
        history["train"].append(train_acc)
        history["test"].append(test_acc)
        history["loss"].append(loss_sum / seen)
        align = ""
        if dfa is not None and args.align_every and (epoch % args.align_every == 0
                                                     or epoch == args.epochs):
            cos = dfa_alignment(model, dfa, layers, x_probe, y_probe)
            history["align"].append((epoch, cos))
            align = f"  align {statistics.fmean(cos):+.3f}"
        if args.progress:
            sys.stderr.write("\r\033[K")
        print(f"{label} ep {epoch:>3}  loss {loss_sum / seen:.3f}  train {train_acc:6.2f}%  "
              f"test {test_acc:6.2f}%{align}  ({time.time() - t0:.0f}s)", flush=True)

    history["time"] = time.time() - t_run
    if dfa is not None:
        dfa.remove()
    return history


# -- reporting ---------------------------------------------------------------


def plot(results, args, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    style = {"gd": ("#2a78d6", "o"), "dfa": ("#7b5ce0", "P"), "frozen": ("#008300", "X")}
    aligns = [run["align"][-1][1] for run in results.get("dfa", []) if run["align"]]

    ncols = 2 if aligns else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.5), dpi=150,
                             facecolor=surface, squeeze=False)
    axes = axes[0]

    def tidy(ax, xlabel, ylabel, title):
        ax.set_facecolor(surface)
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylabel(ylabel, color=ink)
        ax.set_title(title, color=ink, loc="left", fontsize=11)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")

    # Left: test accuracy per epoch, mean (± s.d. over seeds).
    ax = axes[0]
    epochs = range(1, args.epochs + 1)
    for method, runs in results.items():
        color, marker = style[method]
        curves = list(zip(*(run["test"] for run in runs)))
        mean = [statistics.fmean(c) for c in curves]
        ax.plot(epochs, mean, color=color, marker=marker, lw=2, ms=6,
                markevery=max(1, args.epochs // 10), markeredgecolor=surface,
                markeredgewidth=1, label=LABELS[method])
        if len(runs) > 1:
            sd = [statistics.stdev(c) for c in curves]
            ax.fill_between(epochs, [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)], color=color, alpha=0.15, lw=0)
    seeds = len(next(iter(results.values())))
    tidy(ax, "Epoch", "Test accuracy (%)",
         f"CIFAR-10, {args.arch}" + (f" (mean ± s.d., {seeds} seeds)" if seeds > 1 else ""))
    ax.legend(frameon=False, loc="lower right", labelcolor=ink)

    # Right: final DFA/backprop alignment per layer, input to output.
    if aligns:
        ax = axes[1]
        per_layer = list(zip(*aligns))
        mean = [statistics.fmean(c) for c in per_layer]
        idx = range(len(mean))
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.plot(idx, mean, color=style["dfa"][0], marker=style["dfa"][1], lw=2, ms=6,
                markeredgecolor=surface, markeredgewidth=1)
        tidy(ax, "Hidden layer (stem → last block)", "cos(DFA update, gradient)",
             "DFA alignment with backprop, final epoch")

    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", choices=list(ARCHS), default="resnet20")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-size", type=int, default=0,
                   help="use only the first n training images (0: all 50,000)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="random crop + horizontal flip")
    p.add_argument("--batchnorm", action=argparse.BooleanOptionalAction, default=True,
                   help="batch norm (affine=False) after each ReLU")
    # Backprop parameters: the readout for every method, and the hidden layers for gd.
    p.add_argument("--opt", choices=("sgd", "adam"), default="sgd",
                   help="optimizer for backprop parameters (adam means AdamW)")
    p.add_argument("--lr", type=float, default=0.1, help="lr for backprop parameters")
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay for backprop parameters")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (Nesterov)")
    # DFA hidden layers. The DFA signal is a deterministic (if biased) direction,
    # so Adam is fine here.
    p.add_argument("--dfa-opt", choices=("sgd", "adam"), default="adam",
                   help="optimizer for dfa hidden layers")
    p.add_argument("--dfa-lr", type=float, default=1e-3, help="lr for dfa hidden layers")
    p.add_argument("--dfa-wd", type=float, default=0.0, help="weight decay for dfa hidden layers")
    p.add_argument("--dfa-scale", type=float, default=1.0,
                   help="scale of the dfa random feedback matrices")
    p.add_argument("--share-shortcut-feedback", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="one feedback matrix for a block's conv2 and its projection shortcut")
    p.add_argument("--align-every", type=int, default=5,
                   help="epochs between DFA/backprop alignment probes (0: off)")
    p.add_argument("--schedule", choices=("constant", "cosine", "step"), default="cosine",
                   help="lr schedule, applied to every optimizer")
    p.add_argument("--warmup-epochs", type=int, default=1, help="linear lr warmup")
    p.add_argument("--min-lr-frac", type=float, default=0.0,
                   help="final lr as a fraction of the initial lr")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="disable the progress bar")
    p.add_argument("--data", default="data", help="dataset download directory")
    p.add_argument("--out", help="plot path (default: cifar10_<arch>_dfa.png)")
    p.add_argument("--json", help="also write per-epoch histories to this JSON file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    (x, y), test = load_cifar10(args.data, args.device)
    if args.train_size:
        x, y = x[:args.train_size], y[:args.train_size]
    if len(x) < args.batch_size:
        p.error("--train-size must be at least --batch-size")
    data = ((x, y), test)

    results = {m: [] for m in args.methods}
    for seed in range(args.seeds):
        for method in args.methods:
            hist = train(method, data, args, seed, label=f"seed {seed}  {method:<6}")
            results[method].append(hist)
            print(f"seed {seed}  {method:<6} final train {hist['train'][-1]:6.2f}%  "
                  f"test {hist['test'][-1]:6.2f}%  ({hist['time']:.0f}s)\n", flush=True)

    print(f"Final accuracy (%), {args.arch}, mean over {args.seeds} seed(s)")
    print(f"{'method':<8} {'train':>8} {'test':>8} {'best test':>10}")
    for method, runs in results.items():
        print(f"{method:<8} {statistics.fmean(r['train'][-1] for r in runs):8.2f} "
              f"{statistics.fmean(r['test'][-1] for r in runs):8.2f} "
              f"{statistics.fmean(max(r['test']) for r in runs):10.2f}")

    aligns = [r["align"][-1][1] for r in results.get("dfa", []) if r["align"]]
    if aligns:
        print("\nDFA alignment with the backprop gradient, final epoch (cosine per layer)")
        print("  " + " ".join(f"{statistics.fmean(c):+.2f}" for c in zip(*aligns)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
        print(f"\nSaved {args.json}")
    out = args.out or f"cifar10_{args.arch}_dfa.png"
    plot(results, args, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
