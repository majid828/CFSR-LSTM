import numpy as np
from scipy.stats import wilcoxon, ttest_rel


def paired_tests(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<2: return {"n":len(a),"wilcoxon_p":np.nan,"paired_t_p":np.nan}
    try: wp=float(wilcoxon(a,b).pvalue)
    except ValueError: wp=1.0
    return {"n":len(a),"wilcoxon_p":wp,"paired_t_p":float(ttest_rel(a,b).pvalue)}

def bootstrap_mean_difference(a,b,n_boot=2000,seed=42):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b; rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot): vals.append(np.mean(rng.choice(d,size=len(d),replace=True)))
    lo,hi=np.quantile(vals,[0.025,0.975]); return {"mean_difference":float(np.mean(d)),"ci95":[float(lo),float(hi)]}
