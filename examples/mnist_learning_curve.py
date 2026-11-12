"""MNIST learning curves: three-factor Hebbian conv layers vs. gradient descent.

A LeNet-style convnet is trained on subsets of MNIST (or Fashion-MNIST, with
``--dataset fmnist``) of increasing size, once
per method, and test accuracy is plotted against training-set size.

Methods (they differ only in how the two conv layers learn; the fully
connected layers are always trained by backprop):

    gd         conv layers trained by backprop (the usual network)
    hebb       conv layers trained by the plain three-factor rule
    oja        conv layers trained by the three-factor rule with Oja decay
    np         node perturbation: noise on the conv outputs, modulated by
               (clean loss - noisy loss) per sample; two forward passes
    reinforce  the same noise, modulated by (running-average loss - noisy
               loss) per sample; one forward pass
    frozen     conv layers left at their random initialisation (control: a
               method must beat this to be learning useful conv features)

For hebb, oja and reinforce the third factor is a per-sample reward prediction
error: a running average of the loss minus that sample's loss.

Run from the repository root:

    uv run --group examples python examples/mnist_learning_curve.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

from modul8 import ThreeFactor

METHODS = ("gd", "hebb", "oja", "np", "reinforce", "frozen")
LABELS = {
    "gd": "Backprop",
    "hebb": "3-factor Hebbian",
    "oja": "3-factor Oja",
    "np": "Node perturbation",
    "reinforce": "REINFORCE",
    "frozen": "Frozen random conv",
}


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(self.relu1(self.conv1(x)), 2)
        x = F.max_pool2d(self.relu2(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


DATASETS = {
    "mnist": (datasets.MNIST, "MNIST"),
    "fmnist": (datasets.FashionMNIST, "Fashion-MNIST"),
}


def load_dataset(name: str, root: str, device):
    cls = DATASETS[name][0]
    train, test = (cls(root, train=t, download=True) for t in (True, False))
    # Standardise both splits with the training set's statistics.
    x_train = train.data.float().div(255)
    mean, std = x_train.mean(), x_train.std()

    def tensors(ds):
        x = (ds.data.float().div(255) - mean) / std
        return x.unsqueeze(1).to(device), ds.targets.to(device)

    return tensors(train), tensors(test)


@torch.no_grad()
def accuracy(model, x, y) -> float:
    model.eval()
    correct = sum(
        (model(xb).argmax(1) == yb).sum().item()
        for xb, yb in zip(x.split(2000), y.split(2000))
    )
    model.train()
    return 100.0 * correct / len(y)


def train(method: str, x, y, x_test, y_test, args, seed: int) -> float:
    torch.manual_seed(seed)
    model = LeNet().to(x.device)
    convs = [model.conv1, model.conv2]
    conv_params = [p for c in convs for p in c.parameters()]
    conv_ids = {id(p) for p in conv_params}
    rest = [p for p in model.parameters() if id(p) not in conv_ids]

    opts = [torch.optim.Adam(rest, lr=args.lr)]
    hebb = None
    if method == "gd":
        opts.append(torch.optim.Adam(conv_params, lr=args.lr))
    elif method in ("hebb", "oja"):
        hebb = ThreeFactor()
        hebb.add(model.conv1, post=model.relu1, rule=method)
        hebb.add(model.conv2, post=model.relu2, rule=method)
        # Plain SGD on the conv weights: the step is exactly lr * (Hebbian update).
        opts.append(torch.optim.SGD([c.weight for c in convs], lr=args.hebb_lr))
    elif method in ("np", "reinforce"):
        # The update is a noisy estimate of the gradient. Plain SGD averages
        # that noise out over steps; Adam would rescale it to full-size steps.
        noise, lr = ((args.np_noise, args.np_lr) if method == "np"
                     else (args.reinforce_noise, args.reinforce_lr))
        hebb = ThreeFactor(convs, noise=noise, bias=True)
        opts.append(torch.optim.SGD(conv_params, lr=lr))
    # "frozen": the conv parameters are in no optimizer.

    n = len(x)
    baseline = None
    step = 0
    while step < args.steps:
        for idx in torch.randperm(n, device=x.device).split(args.batch_size):
            if method == "np":  # clean pass: no noise, nothing captured
                with torch.no_grad(), hebb.paused():
                    clean = F.cross_entropy(model(x[idx]), y[idx], reduction="none")
            losses = F.cross_entropy(model(x[idx]), y[idx], reduction="none")
            for opt in opts:
                opt.zero_grad()
            losses.mean().backward()
            if method == "np":
                hebb.modulate(clean - losses.detach())
            elif hebb is not None:
                mean = losses.mean().item()
                baseline = mean if baseline is None else 0.9 * baseline + 0.1 * mean
                hebb.modulate(baseline - losses.detach())  # per-sample reward signal
            for opt in opts:
                opt.step()
            step += 1
            if step >= args.steps:
                break

    acc = accuracy(model, x_test, y_test)
    if hebb is not None:
        hebb.remove()
    return acc


def plot(results, sizes, path: str, dataset: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    style = {  # fixed colour + marker per method, so identity is not colour-alone
        "gd": ("#2a78d6", "o"),
        "hebb": ("#eb6834", "s"),
        "oja": ("#1baf7a", "^"),
        "np": ("#eda100", "D"),
        "reinforce": ("#e87ba4", "v"),
        "frozen": ("#008300", "X"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for method, accs in results.items():
        color, marker = style[method]
        mean = [statistics.fmean(a) for a in accs]
        ax.plot(sizes, mean, color=color, marker=marker, lw=2, ms=6,
                markeredgecolor=surface, markeredgewidth=1, label=LABELS[method])
        if len(accs[0]) > 1:
            sd = [statistics.stdev(a) for a in accs]
            ax.fill_between(sizes, [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)], color=color, alpha=0.15, lw=0)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Training examples", color=ink)
    ax.set_ylabel("Test accuracy (%)", color=ink)
    seeds = len(next(iter(results.values()))[0])
    ax.set_title(f"{DATASETS[dataset][1]} learning curves, LeNet" +
                 (f" (mean ± s.d., {seeds} seeds)" if seeds > 1 else ""),
                 color=ink, loc="left", fontsize=11)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="lower right", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[100, 300, 1000, 3000, 10000, 60000])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=1500, help="optimizer steps per run")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam lr for backprop weights")
    p.add_argument("--hebb-lr", type=float, default=1e-3, help="SGD lr for Hebbian weights")
    # Node perturbation wants small noise (its modulator shrinks with the noise);
    # REINFORCE's variance grows as 1/noise^2, so it wants larger noise and a
    # smaller learning rate.
    p.add_argument("--np-noise", type=float, default=0.01, help="output noise std for np")
    p.add_argument("--np-lr", type=float, default=1e-5, help="SGD lr for np weights")
    p.add_argument("--reinforce-noise", type=float, default=0.1,
                   help="output noise std for reinforce")
    p.add_argument("--reinforce-lr", type=float, default=1e-5,
                   help="SGD lr for reinforce weights")
    p.add_argument("--dataset", choices=list(DATASETS), default="mnist",
                   help="mnist or fmnist (Fashion-MNIST)")
    p.add_argument("--data", default="data", help="dataset download directory")
    p.add_argument("--out", help="plot path (default: <dataset>_learning_curve.png)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    (x, y), (x_test, y_test) = load_dataset(args.dataset, args.data, args.device)
    sizes = sorted(min(s, len(x)) for s in args.sizes)
    results = {m: [[] for _ in sizes] for m in args.methods}

    for seed in range(args.seeds):
        # Same nested subsets for every method: the first n of one permutation.
        perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).to(x.device)
        for i, n in enumerate(sizes):
            idx = perm[:n]
            for method in args.methods:
                t0 = time.time()
                acc = train(method, x[idx], y[idx], x_test, y_test, args, seed)
                results[method][i].append(acc)
                print(f"seed {seed}  n={n:>6}  {method:<9} {acc:6.2f}%  ({time.time() - t0:.0f}s)",
                      flush=True)

    print("\nTest accuracy (%), mean over seeds")
    print(f"{'n':>7}  " + "  ".join(f"{m:>9}" for m in args.methods))
    for i, n in enumerate(sizes):
        print(f"{n:>7}  " + "  ".join(f"{statistics.fmean(results[m][i]):9.2f}" for m in args.methods))

    out = args.out or f"{args.dataset}_learning_curve.png"
    plot(results, sizes, out, args.dataset)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
