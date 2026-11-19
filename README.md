# NNM-49

NNM-49 is a neural network with **49 trainable parameters** for experiments in function approximation. The research release includes its trained checkpoint, a local desktop chat app, a CIL arithmetic compiler, and an English event translator.

## What it did

- **Learned a synthetic curve.** On 257 held-out inputs, NNM-49 reached an MSE of `0.001491`, compared with `0.009541` for a fitted linear model. This benchmark covers one function and one seed.
- **Compiled and reran math.** The CIL compiler turned `x * (2 + 3 * 4) - 5` into 32 bytes of bytecode. Studio ran it at `x = 3` and `x = 5`, returning `37` and `65`.
- **Fit and tested an equation.** Studio’s equation fitter recovered `y = x² − 2x + 3` from five synthetic points and matched all three held-out points.

[Setup, exact prompts, and outputs](releases/nnm-49/README.md#worked-examples) · [Model card](releases/nnm-49/MODEL_CARD.md) · [Research license](releases/nnm-49/LICENSE)

## Start Studio

From the repository root with Python 3.12:

```sh
cd releases/nnm-49
python -m pip install -r requirements.txt
python studio.py
```

Run `python benchmark.py` to reproduce the model benchmark.
