# modul8

Three-factor Hebbian learning for standard PyTorch layers.

Pick weights in an ordinary `nn.Linear` or `nn.Conv1d/2d/3d` and learn them with

```
Δw_ij = lr · M · post_i · pre_j
```

(local pre/post activity × a global modulatory signal `M`) instead of backprop.
The layers are not subclassed or replaced: activity is captured with forward
hooks, and the update is written into `param.grad`, so your existing optimizer
applies it.

*Developed with the help of Claude (obviously).*

## Install

modul8 needs Python ≥ 3.11 and PyTorch ≥ 2.6. Install the PyTorch build you
want (CPU or a specific CUDA version) first, following
[pytorch.org](https://pytorch.org/get-started/locally/); otherwise the
installer pulls the default `torch` wheel from PyPI.

With uv, in your project:

```sh
uv add git+https://github.com/glomerulus-lab/modul8
uv add /path/to/modul8              # or from a local clone
```

With pip:

```sh
pip install git+https://github.com/glomerulus-lab/modul8
pip install /path/to/modul8         # or from a local clone
pip install -e /path/to/modul8      # editable, to work on modul8 itself
```

Then `from modul8 import ThreeFactor, DirectFeedback`.

The `cpu`, `cu124` and `cu134` extras only choose a PyTorch index when working
inside this repository with uv (see [Develop](#develop)). Elsewhere they just
add `torchvision`, so install PyTorch yourself as above.

## Use

```python
import torch
from modul8 import ThreeFactor

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 3), torch.nn.ReLU(),
    torch.nn.Flatten(), torch.nn.Linear(16 * 26 * 26, 10),
)
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

hebb = ThreeFactor(model[0], post=model[1])   # conv weights; post = ReLU output

baseline = 0.0
for x, y in loader:
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()                         # backprop for all other parameters
    hebb.modulate(baseline - loss.item())   # replace .grad on the Hebbian weights
    opt.step()
    baseline = 0.9 * baseline + 0.1 * loss.item()
```

### The modulator

`modulate(M)` takes the third factor. It is **reward-like**: positive `M`
strengthens the current pre/post correlation, so pass a negated error
(e.g. `baseline - loss`). It can be

- a scalar,
- a 1-D per-sample tensor of shape `(batch,)`,
- any tensor broadcastable to the postsynaptic activity (per-neuron signals),
- a mapping `{layer: M}` giving each managed layer its own signal (it must
  cover exactly the layers this `ThreeFactor` manages).

By default the update is averaged over samples and output positions
(`reduce="all"`). With `reduce="sample"` it is averaged over samples and summed
over positions, the same scale as the gradient of a per-sample mean loss; this
is the default when `noise` is set.

### Options

```python
hebb = ThreeFactor()                         # several layers, per-layer options
hebb.add(model.conv1, post=model.relu1)
hebb.add(model.fc, mask=mask, bias=True)
```

| option | meaning |
| --- | --- |
| `mask` | bool tensor broadcastable to the weight. `True` entries are Hebbian, `False` entries keep their backprop gradient. Default: whole weight. |
| `bias` | also learn the bias with the rule (pre = 1). Default `False`: bias stays on backprop. |
| `post` | module whose output is the postsynaptic activity (e.g. the following nonlinearity) instead of the layer's raw output. |
| `rule` | `"hebb"` (default) or `"oja"`; see below. |
| `noise` | standard deviation of Gaussian noise added to the layer's output; the noise becomes the postsynaptic factor. See below. |
| `pre_fn`, `post_fn` | transforms applied to the activities before the update (centering, thresholds, ...). |
| `reduce` | `"all"` (average over samples and positions) or `"sample"` (average over samples only). Default: `"sample"` with `noise`, else `"all"`. |

### Keeping weights bounded: `rule="oja"`

Plain Hebbian growth is unbounded. `ThreeFactor(layer, rule="oja")` adds Oja's
activity-dependent decay to each unit:

```
Δw_ij = lr · ( M · post_i · pre_j  −  |M| · post_i² · w_ij )
```

For a linear unit with a positive modulator this is the modulated Oja rule,
which drives each unit's weight vector towards unit norm. The decay uses `|M|`
so that a negative modulator still shrinks the weights instead of growing them.
The decay acts on the bias too when `bias=True`, and uses the same postsynaptic
activity as the Hebbian term (so it follows `post=` and `post_fn=`).

### Node perturbation and REINFORCE: `noise=σ`

The rules above correlate the modulator with the layer's activity, which is not
a gradient estimate. With `noise=σ` the layer's output is perturbed on every
captured forward pass and the perturbation is used as the postsynaptic factor:

```
forward:  y = layer(x) + σ·ξ,   ξ ~ N(0, 1)
update:   Δw_ij = lr · M · (ξ_i / σ) · pre_j      (averaged over samples)
```

Which algorithm this is depends only on the modulator your loop passes. Use a
**per-sample** modulator of shape `(batch,)`; then the value written to `.grad`
is an unbiased estimate (for small σ) of the backprop gradient of the mean
loss, at the same scale.

```python
hebb = ThreeFactor(model.conv1, noise=0.01, bias=True)
loss_fn = lambda out, y: F.cross_entropy(out, y, reduction="none")
```

**Node perturbation**: compare against a noise-free pass. Two forward passes.

```python
with torch.no_grad(), hebb.paused():        # clean: no noise, nothing captured
    clean = loss_fn(model(x), y)
noisy = loss_fn(model(x), y)
opt.zero_grad()
noisy.mean().backward()                     # only if other layers use backprop
hebb.modulate(clean - noisy.detach())
opt.step()
```

**REINFORCE**: compare against any baseline that does not depend on the
noise (a running average, or zero). One forward pass, and the loss need not be
differentiable; `M` can be any per-sample reward.

```python
noisy = loss_fn(model(x), y)
opt.zero_grad()
noisy.mean().backward()
hebb.modulate(baseline - noisy.detach())
opt.step()
baseline = 0.9 * baseline + 0.1 * noisy.mean().item()
```

Practical notes:

- **Variance.** Both estimates are unbiased but noisy, and the noise grows with
  the number of perturbed units. For the two conv layers of LeNet on MNIST
  (about 6,300 units, batch 64) a single node-perturbation estimate is roughly
  40 to 160 times larger in norm than the gradient it estimates. Expect to need
  small learning rates and many more steps than backprop.
- **Choosing σ.** Node perturbation's estimate does not depend on σ once it is
  small, so use a small value to keep the forward pass nearly clean. REINFORCE's
  variance grows as 1/σ², so it needs a larger σ and a smaller learning rate.
- **Optimizer.** Prefer plain SGD for these weights. Adam rescales a
  noise-dominated estimate into full-size steps.
- `noise` cannot be combined with `post`, `post_fn` or `rule="oja"`. No noise
  is added in eval mode, inside `paused()`, or after `enable(False)`.
- A scalar modulator works but gives an estimate scaled by 1/batch with higher
  variance.

### Direct feedback alignment: `DirectFeedback`

A single scalar reward has to assign credit to every unit at once, and its
variance grows with the number of units. Direct feedback alignment (DFA;
Nøkland, 2016) gives each unit its own third factor instead: the network's
output error `e`, projected to the layer through a fixed random matrix `B`.

```
Δw_ij = −lr · (Bᵀe)_i · f'(a_i) · pre_j
```

This is still a three-factor rule: the per-neuron modulator is `−(Bᵀe)`, and
the postsynaptic factor is the derivative of the layer's nonlinearity. No error
travels backwards through the network, and the matrices `B` are never trained;
the forward weights come to align with them.

```python
import torch.nn.functional as F
from modul8 import DirectFeedback

dfa = DirectFeedback(
    [model.conv1, model.fc1],          # layers to train with DFA
    n_out=10,                          # size of the output error
    posts=[model.relu1, model.relu2],  # each layer's nonlinearity
    generator=torch.Generator().manual_seed(0),
)

logits = model(x)
loss = F.cross_entropy(logits, y)
opt.zero_grad()
loss.backward()                        # backprop for the readout
dfa.modulate(logits.detach().softmax(1) - F.one_hot(y, 10))
opt.step()
```

For mean cross-entropy, `softmax − one_hot` is the gradient of the loss with
respect to the logits (up to the 1/batch that `reduce="sample"` applies). For
another loss, pass its gradient with respect to the network output.

| option | meaning |
| --- | --- |
| `posts` | per layer, the module whose output feeds `derivative` (usually its nonlinearity); `None` uses the layer's own output. |
| `derivative` | maps that activity to `f'(a)`. Default `relu_derivative`; `tanh_derivative` and `sigmoid_derivative` take the nonlinearity's output. `None` means linear units. |
| `scale` | entries of `B` are drawn from `N(0, scale² / n_out)`. Default 1. |
| `bias` | also train the biases. Default `True`. |
| `reduce` | default `"sample"`: updates are on the scale of a mean-loss gradient, so SGD and Adam both work. |
| `generator` | `torch.Generator` used to draw `B`, for reproducibility. |

Practical notes:

- Each `B` is created on the first `modulate()`, sized from the layer's actual
  output (channels × positions for conv layers), on the layer's device and dtype.
- Results depend on `B`, so seed the `generator` and save the matrices with
  `dfa.state_dict()` / `dfa.load_state_dict()` alongside your checkpoints.
- If `B` equals the true backward path (for one hidden layer, the readout's
  weight), the update is exactly the backprop gradient.
- DFA works well for fully connected layers and less well for conv layers.
  Batch norm after each layer usually helps; DFA's update ignores the norm's
  own derivative.
- The signal is a biased but deterministic direction, not a noisy estimate, so
  Adam is fine here (unlike for `noise=`).
- `paused()`, `reset()`, `enable()` and `remove()` work as for `ThreeFactor`;
  `DirectFeedback` is also a context manager.

### Behaviour to know

- Activity is captured on every forward pass while the layer is in **training
  mode**; all passes since the last `modulate()` are summed into the update.
  Use `model.eval()`, `hebb.enable(False)` or `hebb.reset()` for forwards that
  should not count.
- Backprop still flows *through* a Hebbian layer to the layers before it; only
  that layer's own gradient is replaced. `loss.backward()` is optional if
  nothing else needs it.
- Optimizer features (momentum, weight decay, Adam scaling) apply to the
  Hebbian update as they would to a gradient. With `rule="oja"`, optimizer weight decay is usually
  unnecessary.
- `hebb.remove()` detaches the hooks; `ThreeFactor` also works as a context
  manager.

## Example: MNIST learning curves

`examples/mnist_learning_curve.py` trains a LeNet-style convnet on MNIST
subsets of increasing size and plots test accuracy for backprop, the Hebbian
and Oja rules, node perturbation, REINFORCE, direct feedback alignment (`dfa`),
and a frozen-random control. Every method trains the four hidden layers
(conv1, conv2, fc1, fc2) its own way; the fc3 readout is always trained by
backprop.

```sh
uv run --group examples python examples/mnist_learning_curve.py
```

Add `--dataset fmnist` to use Fashion-MNIST instead (the plot then defaults to
`fmnist_learning_curve.png`). The per-method learning rates and noise levels
were tuned on MNIST only.

By default each hidden layer's ReLU is followed by batch norm with
`affine=False` (layer → ReLU → BN; no learnable parameters; `--no-batchnorm`
removes it). BN sits after the ReLU so that, for DFA, the ReLU's derivative is
still the layer's exact local derivative. The hidden-layer
learning rate follows a cosine schedule with a 500-step warmup (`--schedule`,
`--warmup`, `--min-lr-frac`). Each run reports train and test accuracy and
shows a progress bar (`--eval-every`, `--no-progress`). DFA has its own
`--dfa-lr` and `--dfa-scale`.

Results on the full Fashion-MNIST training set (60,000 images, batch 128,
30,000 steps, other options at their defaults; mean over 3 seeds, test ± s.d.):

| method | train (%) | test (%) | gap (pp) |
| --- | ---: | ---: | ---: |
| Backprop (`gd`) | 100.00 | 90.41 ± 0.30 | 9.59 |
| Direct feedback alignment (`dfa`) | 93.74 | 89.02 ± 0.14 | 4.72 |
| Node perturbation (`np`) | 79.37 | 78.08 ± 0.60 | 1.29 |
| Frozen random hidden (`frozen`) | 74.89 | 73.78 ± 0.91 | 1.11 |

```sh
uv run --group examples python examples/mnist_learning_curve.py --dataset fmnist \
    --sizes 60000 --seeds 3 --methods gd frozen np dfa --batch-size 128 --steps 30000
```

## Example: DFA on CIFAR-10 ResNets

`examples/cifar10_resnet_dfa.py` trains a CIFAR-style ResNet (`--arch`
`resnet20/32/44/56/110` or `resnet18/34`) on CIFAR-10 with backprop (`gd`),
DFA (`dfa`) and a frozen-random control (`frozen`). Every conv, including the
stem and the 1x1 projection shortcuts, is trained the method's way; the linear
readout is always trained by backprop.

```sh
uv run --group examples python examples/cifar10_resnet_dfa.py --arch resnet20
```

Each block is `BN(ReLU(conv1(x)))`, then `BN(ReLU(conv2(h) + shortcut(x)))`,
with `affine=False` batch norm after the ReLU as in the MNIST example. A
block's conv2 and projection shortcut feed the same units, so they share one
feedback matrix (`--no-share-shortcut-feedback` gives each its own). For
`dfa` and `frozen` the hidden layers run without autograd, so no backward pass
goes through the network. Every `--align-every` epochs the script reports the
cosine between each layer's DFA update and its true gradient, and the plot
shows this alignment per layer next to the accuracy curves. Default
hyperparameters: SGD (lr 0.1, Nesterov momentum, wd 5e-4) for backprop
weights, Adam (lr 1e-3) for DFA, cosine schedule with 1 warmup epoch, standard
crop + flip augmentation. The DFA settings have not been tuned.
`--dfa-feedback tied` draws each conv's feedback per channel and shares it
across spatial positions instead of drawing one value per unit.

## Example: DFA on CIFAR-10 VGG nets

`examples/cifar10_vgg_dfa.py` runs the same comparison on the VGG variants of
Nøkland & Eidnes (2019), including their widened versions (`--feat-mult`):

```sh
uv run --group examples python examples/cifar10_vgg_dfa.py --arch vgg8b
uv run --group examples python examples/cifar10_vgg_dfa.py --arch vgg11b --feat-mult 3
```

The convs are followed by one hidden fully connected layer (`--num-hidden`,
default 1024), then the backprop-trained readout. Layers are conv → ReLU → BN
(`affine=False`), as in the other examples, rather than the paper's
conv → BN → leaky ReLU. `--dropout` adds dropout after each hidden layer. The
paper's recipes use 0.2 for vgg8b and 0.3 for vgg11b (3x).

Results for vgg8 (full training set, 80 epochs, tied feedback, SGD lr 0.01 for
backprop weights, Adam lr 0.01 with weight decay 1e-5 for DFA, other options at
their defaults; mean over 3 seeds):

| method | train (%) | test (%) | best test (%) | gap (pp) |
| --- | ---: | ---: | ---: | ---: |
| Backprop (`gd`) | 99.62 | 88.37 | 88.44 | 11.25 |
| Direct feedback alignment (`dfa`) | 82.05 | 73.03 | 73.15 | 9.02 |
| Frozen random hidden (`frozen`) | 45.36 | 47.96 | 48.06 | −2.60 |

At the final epoch, the cosine between each DFA update and the backprop
gradient rises from the first conv to the hidden fully connected layer:
+0.03, +0.11, +0.14, +0.20, +0.23, +0.34.

```sh
uv run --group examples python examples/cifar10_vgg_dfa.py --arch vgg8 \
    --methods frozen gd dfa --seeds 3 --epochs 80 --dfa-feedback tied \
    --lr 0.01 --dfa-lr 0.01 --dfa-wd 1e-5
```

## Develop

```sh
uv sync                     # add --extra cpu, cu124 or cu134 to pick a PyTorch build
uv run pytest
```

Run the examples with `--group examples` (and the same extra), e.g.
`uv run --extra cu124 --group examples python examples/mnist_learning_curve.py`.
