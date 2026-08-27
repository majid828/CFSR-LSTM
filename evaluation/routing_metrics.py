import numpy as np


def expected_delay_from_probs(probs, dt=1.0):
    probs=np.asarray(probs); lags=np.arange(probs.shape[-1]); return np.sum(probs*lags,axis=-1)*dt

def routing_entropy(probs, eps=1e-12):
    p=np.asarray(probs); return -np.sum(p*np.log(p+eps),axis=-1)

def delay_summary(probs,dt=1.0):
    d=expected_delay_from_probs(probs,dt); return {"mean_delay":float(np.mean(d)),"median_delay":float(np.median(d)),"std_delay":float(np.std(d))}
