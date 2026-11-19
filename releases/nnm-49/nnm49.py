import numpy as np

class NeuralCore:
    """Trainable 1→16 tanh→1 regressor, 49 float64 parameters; no LLM."""
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, .7, 16); self.b = np.zeros(16)
        self.v = rng.normal(0, .2, 16); self.c = np.zeros(1)

    def predict(self, x):
        x = np.asarray(x)
        return np.tanh(x[..., None]*self.w+self.b) @ self.v + self.c[0]

    def train(self, x, y, steps=1000, rate=.02):
        x = np.asarray(x); y = np.asarray(y)
        for _ in range(steps):
            h = np.tanh(x[:, None]*self.w+self.b)
            e = (h @ self.v + self.c[0]-y)*2/len(x)
            dh = e[:, None]*self.v*(1-h*h)
            grads = ((dh*x[:, None]).sum(0), dh.sum(0), h.T@e, np.array([e.sum()]))
            for param, grad in zip((self.w,self.b,self.v,self.c), grads): param -= rate*grad

    def save(self, path): np.savez(path, w=self.w, b=self.b, v=self.v, c=self.c)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            for name in ('w','b','v','c'):
                value = data[name]
                if value.shape != getattr(obj,name).shape or not np.isfinite(value).all():
                    raise ValueError('invalid checkpoint')
                setattr(obj,name,value.copy())
        return obj
