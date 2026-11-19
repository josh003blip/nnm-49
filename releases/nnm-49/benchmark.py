"""Independent minimal benchmark for the original 49-parameter core."""
import json
import numpy as np
from nnm49 import NeuralCore

def mse(a,b):
    return float(np.mean((np.asarray(a)-np.asarray(b))**2))

def target(x):
    return np.tanh(1.4*np.asarray(x)-.3)

def main():
    rng=np.random.default_rng(2026)
    x=rng.uniform(-1,1,256)
    y=target(x)
    test=rng.uniform(-1,1,257)
    extrap=rng.uniform(1.2,2,128)
    model=NeuralCore(seed=7)
    before=mse(model.predict(test),target(test))
    model.train(x,y,steps=4000)
    slope,intercept=np.polyfit(x,y,1)
    result={
        "seed":2026,"parameters":49,"training_steps":4000,
        "train_count":len(x),"test_count":len(test),
        "primitive_mse":{"untrained":before,"trained":mse(model.predict(test),target(test)),
                         "linear":mse(slope*test+intercept,target(test)),
                         "constant":mse(np.full_like(test,y.mean()),target(test))},
        "extrapolation_mse":mse(model.predict(extrap),target(extrap))
    }
    assert result["primitive_mse"]["trained"] < .003
    assert result["primitive_mse"]["trained"] < result["primitive_mse"]["linear"]
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
