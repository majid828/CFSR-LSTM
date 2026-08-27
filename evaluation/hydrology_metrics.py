import numpy as np


def _arr(x): return np.asarray(x, dtype=float)

def rmse(obs, pred):
    o,p=_arr(obs),_arr(pred); return float(np.sqrt(np.mean((o-p)**2)))

def mae(obs,pred):
    o,p=_arr(obs),_arr(pred); return float(np.mean(np.abs(o-p)))

def nse(obs,pred):
    o,p=_arr(obs),_arr(pred); d=np.sum((o-o.mean())**2); return float(1-np.sum((o-p)**2)/(d+1e-12))

def kge(obs,pred):
    o,p=_arr(obs),_arr(pred); r=np.corrcoef(o,p)[0,1] if np.std(o)>0 and np.std(p)>0 else 0.0
    alpha=np.std(p)/(np.std(o)+1e-12); beta=np.mean(p)/(np.mean(o)+1e-12)
    return float(1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2))

def log_nse(obs,pred,eps=1e-6): return nse(np.log(_arr(obs)+eps),np.log(_arr(pred)+eps))

def low_flow_bias(obs,pred,quantile=0.2):
    o,p=_arr(obs),_arr(pred); m=o<=np.quantile(o,quantile); return float((p[m].sum()-o[m].sum())/(o[m].sum()+1e-12))
