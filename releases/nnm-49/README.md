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

### What you can type

| Prompt | What runs |
|---|---|
| `What would the model predict for 0.5?` | CIL calls the original trained NNM checkpoint |
| `Compile x * (2 + 3 * 4) - 5` | The arithmetic compiler emits CIL and bytecode |
| `Run x*x + 2 at 3` then `Try that at 5` | The CIL VM evaluates the expression at both inputs |
| `Fit an equation to [(0,1), (1,3), (2,5), (3,7)]; test on [(4,9), (5,11)]` | The fitter compares small polynomials using the fit set and evaluates the chosen compiled expression on the supplied holdout |
| `Maya was painting from 2:00 to 2:30. At 2:12, was she painting?` | The English event translator calls CIL for interval arithmetic |

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

## Existing recorded evidence

The [standalone benchmark](benchmark.py) measures primitive held-out MSE **0.0014912751** versus a fitted linear baseline **0.0095407298**, and extrapolation MSE **0.0634297298**. These are one-seed synthetic results, not independent replication or broad architecture superiority.

The standalone tests and benchmark ran locally with Python 3.12 and NumPy 2.2.6. The benchmark reproduced held-out MSE **0.0014912751**. Chat tests cover the checkpoint, compiler and follow-ups, disjoint equation holdouts, event translation, and communication with a simulated local language endpoint.

## Contribute

Replicate the results with recorded Python/NumPy versions and seeds, report negative results, test new function families, compare matched-compute baselines, and open GitHub issues/PRs. Please do not represent the model as having autonomous formula invention or self-improving learning rules.

## Licensing and scope

Files in this release directory are offered under the custom [NNM-49 Noncommercial Research License](LICENSE): noncommercial research and publication are allowed with attribution; commercial users must notify the copyright holder and obtain a separate written agreement before commercial use. [Open a GitHub issue](https://github.com/josh003blip/nnm-49/issues/new) titled `NNM-49 commercial licensing inquiry` to begin discussing terms. Notification alone does not grant commercial rights. This is source-available, not OSI-approved open source. Legal review is recommended. The rest of the repository is not automatically covered by this scoped license. The standalone neural implementation is a copy of the original; no changes were made to the original source, weights, or adaptive models. Confirm contributor rights and third-party provenance before external redistribution.
