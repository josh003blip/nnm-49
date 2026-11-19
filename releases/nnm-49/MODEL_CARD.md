# Model card: NNM-49

**Model type:** NumPy 1 → 16 tanh → 1 scalar regressor, 49 trainable float64 parameters (392 raw parameter bytes).

**Intended use:** educational and research experiments on numerical function approximation, small-network training, and reproducible baselines.

**Inputs/outputs:** the checkpoint maps a numeric scalar to a numeric prediction. Studio handles supported text prompts and runs the CIL compiler and VM. A separately installed local language model can provide broader chat and drafting.

**Training:** full-batch gradient descent; benchmark uses 256 synthetic examples sampled uniformly in [-1, 1] with target tanh(1.4x - 0.3), 4,000 steps.

**Evidence:** the [standalone benchmark](benchmark.py) reproduced a held-out MSE of 0.001491 and extrapolation MSE of 0.063430 on the original seeded synthetic task. Studio equation fitting is a deterministic numerical routine. The compiler learns arithmetic lowerings from examples and verifies them on separate inputs.

**Limitations:** one synthetic function family, one recorded seed, weaker extrapolation, no confidence intervals or broad matched-compute comparisons. Small parameter count does not imply superior efficiency on realistic workloads.

**Safety/privacy:** Do not treat numeric outputs as validated predictions for consequential applications. Studio keeps conversations in memory and connects to a loopback language server only when explicitly enabled.

**Provenance:** original implementation [nnm49.py](nnm49.py). This model is independent of later adaptive NNM experiments.

**Reproducibility:** Python 3.12 with NumPy pinned by [requirements](requirements.txt). Follow the [release README](README.md).

**License:** [NNM-49 Noncommercial Research License](LICENSE). Noncommercial research is permitted; commercial use requires advance notification and a separate written agreement. This is not an OSI-approved open-source license.
