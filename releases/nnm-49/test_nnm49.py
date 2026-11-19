import unittest
import numpy as np
from nnm49 import NeuralCore

class NeuralCoreTests(unittest.TestCase):
    def test_parameter_count(self):
        m = NeuralCore()
        self.assertEqual(sum(a.size for a in (m.w, m.b, m.v, m.c)), 49)

    def test_training_reduces_error(self):
        x = np.linspace(-1, 1, 128)
        y = np.tanh(1.4*x - .3)
        m = NeuralCore(seed=7)
        before = np.mean((m.predict(x) - y)**2)
        m.train(x, y, steps=4000)
        after = np.mean((m.predict(x) - y)**2)
        self.assertLess(after, before)
        self.assertLess(after, .003)

    def test_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path
        m = NeuralCore(seed=7)
        x = np.array([-.5, 0, .5])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "core.npz"
            m.save(path)
            loaded = NeuralCore.load(path)
            np.testing.assert_allclose(m.predict(x), loaded.predict(x), atol=0, rtol=0)

if __name__ == "__main__":
    unittest.main()
