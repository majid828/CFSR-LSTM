import argparse, pandas as pd, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("--out",default="results/figures/pareto_frontier.png"); a=p.parse_args(); d=pd.read_csv(a.csv)
plt.figure(figsize=(6,5)); plt.scatter(d["slow_error"],d["peak_error"]); [plt.annotate(r["model"],(r["slow_error"],r["peak_error"])) for _,r in d.iterrows()]; plt.xlabel("Slow-flow error"); plt.ylabel("Peak error"); plt.tight_layout(); plt.savefig(a.out,dpi=180)
