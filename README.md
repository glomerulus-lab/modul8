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

## Install

```sh
uv add /path/to/modul8        # or: uv add git+<repo-url>
```

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
- any tensor broadcastable to the postsynaptic activity (per-neuron signals).

The update is averaged over samples and output positions.

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
and Oja rules, node perturbation, REINFORCE, and a frozen-random-conv control.

```sh
uv run --group examples python examples/mnist_learning_curve.py
```

Add `--dataset fmnist` to use Fashion-MNIST instead (the plot then defaults to
`fmnist_learning_curve.png`). The per-method learning rates and noise levels
were tuned on MNIST only.

## Develop

```sh
uv sync
uv run pytest
```
