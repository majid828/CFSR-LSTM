import numpy as np


def event_peak_errors(obs,pred,event_id,eps=1e-6):
    obs=np.asarray(obs); pred=np.asarray(pred); event_id=np.asarray(event_id)
    out=[]
    for eid in np.unique(event_id):
        if eid<0: continue
        m=event_id==eid; op=float(obs[m].max()); pp=float(pred[m].max())
        out.append({"event":int(eid),"obs_peak":op,"pred_peak":pp,"relative_error_pct":100*abs(pp-op)/(op+eps)})
    return out

def peak_rmse(obs,pred,event_id):
    e=event_peak_errors(obs,pred,event_id); return float(np.sqrt(np.mean([(x['pred_peak']-x['obs_peak'])**2 for x in e]))) if e else np.nan

def quantile_rmse(obs,pred,q=0.95):
    obs=np.asarray(obs); pred=np.asarray(pred); m=obs>=np.quantile(obs,q); return float(np.sqrt(np.mean((pred[m]-obs[m])**2))) if np.any(m) else np.nan

def extreme_underprediction_frequency(obs,pred,event_id):
    e=event_peak_errors(obs,pred,event_id); return float(np.mean([x['pred_peak']<x['obs_peak'] for x in e])) if e else np.nan
