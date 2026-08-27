import numpy as np


def event_timing_errors(obs,pred,event_id,dt=1.0):
    obs=np.asarray(obs); pred=np.asarray(pred); event_id=np.asarray(event_id); out=[]
    for eid in np.unique(event_id):
        if eid<0: continue
        idx=np.where(event_id==eid)[0]
        if len(idx)==0: continue
        io=idx[np.argmax(obs[idx])]; ip=idx[np.argmax(pred[idx])]
        out.append({"event":int(eid),"obs_peak_index":int(io),"pred_peak_index":int(ip),"abs_timing_error":abs(ip-io)*dt,"signed_timing_error":(ip-io)*dt})
    return out

def mean_absolute_timing_error(obs,pred,event_id,dt=1.0):
    e=event_timing_errors(obs,pred,event_id,dt); return float(np.mean([x['abs_timing_error'] for x in e])) if e else np.nan
