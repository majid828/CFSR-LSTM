import argparse, numpy as np, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--out",default="results/figures/memory_timescale.png"); a=p.parse_args(); d=np.load(a.file)
plt.figure(figsize=(8,4)); plt.hist(d["fast_forget"].ravel(),bins=30,alpha=.6,label="Fast"); plt.hist(d["slow_forget"].ravel(),bins=30,alpha=.6,label="Slow"); plt.xlabel("Forget gate"); plt.legend(); plt.tight_layout(); plt.savefig(a.out,dpi=180)
