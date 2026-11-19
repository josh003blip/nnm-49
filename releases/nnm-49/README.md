# NNM-49 — standalone original neural model

**Research release candidate.** The 49-parameter original NumPy neural regressor, packaged separately from the later adaptive Cognitive OS. It maps one scalar input through 16 tanh neurons to one scalar output. This is not an LLM or a general reasoning system.

## Files

- [nnm49.py](nnm49.py) — original standalone NeuralCore implementation
- [core.npz](core.npz) — original trained checkpoint, included in the standalone app
- [studio.py](studio.py) — local chat window and terminal prompts
- [research_chat.py](research_chat.py) — prompt routing and bounded equation fitting
- [cil.py](cil.py), [compiler_model.py](compiler_model.py), and [event_language.py](event_language.py) — CIL runtime, arithmetic compiler, and English event translator
- [requirements.txt](requirements.txt) — pinned NumPy dependency
- [benchmark.py](benchmark.py) — minimal, deterministic scalar-regression benchmark
- [test_nnm49.py](test_nnm49.py) — parameter count, training, and checkpoint tests
- [MODEL_CARD.md](MODEL_CARD.md) — intended use and limitations
- [LICENSE](LICENSE) — NNM-49 Noncommercial Research License, scoped to this release directory

The standalone benchmark measures scalar-regression performance.

## Install and run

From the repository root:

```bash
cd releases/nnm-49
python -m pip install -r requirements.txt
python -m unittest -v
python benchmark.py
python studio.py
```

The window uses Python's Tkinter. On systems with a separate Tk package, install it or use the terminal interface.

~~~sh
python studio.py --ask "Run x*x + 2 at 3" --ask "Try that at 5"
~~~

### Worked examples

Run these prompts in NNM Studio. Use the same chat for follow-ups.

**Predict with the released weights.** Ask `What would the model predict for 0.5?` The bundled 49-parameter checkpoint returns `0.340689088191`.

**Compile an expression and rerun it at a new input.** The arithmetic compiler folds the constant expression `2 + 3 * 4` to `14` and emits 32 bytes of CIL bytecode.

```text
You: Compile x * (2 + 3 * 4) - 5
CIL: (program (add (mul x 14.0) -5.0))
You: Try that at 3
Result: 37
You: Try that at 5
Result: 65
```

**Fit a quadratic and check it on separate data.** Ask:

```text
Fit an equation to [(-2,11), (-1,6), (0,3), (1,2), (2,3)]; test on [(3,6), (4,11), (-3,18)]
```

Studio’s equation fitter selected `y = x² − 2x + 3` from constant, linear, and quadratic candidates using leave-one-out error on the five fit points. It compiled the selected equation and then evaluated it on the three held-out points.

| Held-out x | Predicted y | Expected y |
|---:|---:|---:|
| 3 | 6 | 6 |
| 4 | 11 | 11 |
| −3 | 18 | 18 |

Holdout mean squared error was `0` for this exact synthetic quadratic.

**Ask a question with a missing time.** Give the English event translator `Maya started painting at 2:00. At 2:12, was she painting?` It asks when she stopped. Reply `2:30`, and it uses CIL interval arithmetic to confirm she was painting at `2:12`.

The arithmetic frontend accepts numbers, `x`, parentheses, addition, subtraction, multiplication, and unary minus. It induces and audits operator lowerings using separate sets of example inputs. The release includes the original arithmetic compiler experiment and English event translator. Studio keeps the current conversation in memory until you start a new chat or quit.

### Optional general chat

If you already run a local language model with an OpenAI-compatible chat endpoint, start Studio with the opt-in adapter.

~~~sh
python studio.py --local-llm --llm-port 8081 --llm-model cognitive-qwen3-1.7b
~~~

The adapter connects only to `127.0.0.1`. The language model handles open-ended prompts and can propose an expression for the arithmetic compiler. Studio validates each proposed action before it runs. General chat uses a separately installed, running language server.

### Use the model from Python

Load the bundled original checkpoint.

```python
from nnm49 import NeuralCore
model = NeuralCore.load("core.npz")
print(model.predict([0.0, 0.5, 1.0]))
```

Train and save a fresh model:

```python
import numpy as np
from nnm49 import NeuralCore
x = np.linspace(-1, 1, 256)
y = np.tanh(1.4*x - 0.3)
model = NeuralCore(seed=7)
model.train(x, y, steps=4000)
model.save("my_core.npz")
```

## Benchmark results

The [standalone benchmark](benchmark.py) trained NNM-49 on 256 synthetic examples from `tanh(1.4x − 0.3)` with `x` in `[-1, 1]`. It ran 4,000 training steps and evaluated 257 separate inputs from the same interval. Lower mean squared error (MSE) is better.

| Predictor | Held-out MSE |
|---|---:|
| NNM-49 before training | 0.297147 |
| Fitted constant | 0.348017 |
| Fitted linear model | 0.009541 |
| Trained NNM-49 | **0.001491** |

The trained network had about **84% lower MSE** than the linear fit on this seeded task. On 128 extrapolation inputs from `[1.2, 2]`, its MSE rose to `0.063430`. These results cover one synthetic function and one seed.

The nine release tests cover the original checkpoint, compiler and follow-ups, separate equation holdouts, event translation, and the optional language adapter with a simulated endpoint. To reproduce the reported results, run `python benchmark.py` and `python -m unittest -v` with Python 3.12 and NumPy 2.2.6.

## Contribute

Replicate the results with recorded Python/NumPy versions and seeds, report negative results, test new function families, compare matched-compute baselines, and open GitHub issues/PRs. Please do not represent the model as having autonomous formula invention or self-improving learning rules.

## Licensing and scope

Files in this release directory are offered under the custom [NNM-49 Noncommercial Research License](LICENSE): noncommercial research and publication are allowed with attribution; commercial users must notify the copyright holder and obtain a separate written agreement before commercial use. [Open a GitHub issue](https://github.com/josh003blip/nnm-49/issues/new) titled `NNM-49 commercial licensing inquiry` to begin discussing terms. Notification alone does not grant commercial rights. This is source-available, not OSI-approved open source. Legal review is recommended. The rest of the repository is not automatically covered by this scoped license. The standalone neural implementation is a copy of the original; no changes were made to the original source, weights, or adaptive models. Confirm contributor rights and third-party provenance before external redistribution.
