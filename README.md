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
| `pre_fn`, `post_fn` | transforms applied to the activities before the update (centering, thresholds, ...). |

### Behaviour to know

- Activity is captured on every forward pass while the layer is in **training
  mode**; all passes since the last `modulate()` are summed into the update.
  Use `model.eval()`, `hebb.enable(False)` or `hebb.reset()` for forwards that
  should not count.
- Backprop still flows *through* a Hebbian layer to the layers before it; only
  that layer's own gradient is replaced. `loss.backward()` is optional if
  nothing else needs it.
- Optimizer features (momentum, weight decay, Adam scaling) apply to the
  Hebbian update as they would to a gradient. Plain Hebbian growth is
  unbounded, so weight decay or a normalising `post_fn` is usually wanted.
- `hebb.remove()` detaches the hooks; `ThreeFactor` also works as a context
  manager.

## Develop

```sh
uv sync
uv run pytest
```
