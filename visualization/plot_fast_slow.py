import argparse, numpy as np, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--out",default="results/figures/fast_slow.png"); a=p.parse_args(); d=np.load(a.file)
plt.figure(figsize=(12,4)); plt.plot(d["q_fast"],label="Fast"); plt.plot(d["q_slow"],label="Slow"); plt.plot(d["q_total"],label="Total",alpha=.8); plt.legend(); plt.tight_layout(); plt.savefig(a.out,dpi=180)
