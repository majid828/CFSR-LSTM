import argparse, numpy as np, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--time",type=int,default=0); p.add_argument("--out",default="results/figures/routing_kernel.png"); a=p.parse_args(); d=np.load(a.file); probs=d["routing_probs"]; probs=probs[0] if probs.ndim==3 else probs
plt.figure(figsize=(7,4)); plt.bar(np.arange(probs.shape[-1]),probs[a.time]); plt.xlabel("Lag"); plt.ylabel("Probability"); plt.tight_layout(); plt.savefig(a.out,dpi=180)
