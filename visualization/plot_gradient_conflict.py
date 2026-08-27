import argparse, pandas as pd, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument("csv",nargs="?",default="results/logs/full_cfsr.csv"); p.add_argument("--out",default="results/figures/gradient_conflict.png"); a=p.parse_args(); d=pd.read_csv(a.csv)
plt.figure(figsize=(8,4)); plt.plot(d["epoch"],d["grad_cosine"],marker="o"); plt.axhline(0,linewidth=1); plt.xlabel("Epoch"); plt.ylabel("Mean cosine(g_A,g_S)"); plt.tight_layout(); plt.savefig(a.out,dpi=180)
