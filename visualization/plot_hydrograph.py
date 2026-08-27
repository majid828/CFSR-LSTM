import argparse, numpy as np, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--out",default="results/figures/hydrograph.png"); a=p.parse_args(); d=np.load(a.file)
plt.figure(figsize=(12,4)); plt.plot(d["q_obs"],label="Observed"); plt.plot(d["q_pred"],label="Predicted"); plt.xlabel("Time step"); plt.ylabel("Discharge"); plt.legend(); plt.tight_layout(); plt.savefig(a.out,dpi=180)
