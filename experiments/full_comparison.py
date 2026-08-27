"""Run the component-wise ablation sequence recommended in Table 1 of the methodology draft."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import json
from utils.config_loader import load_config, deep_update
from data.dataloader import build_dataloaders
from training.train_cfsr import build_model
from training.trainer import CFSRTrainer
from utils.seed import set_seed

VARIANTS={
    "dual_memory":{"model":{"use_partition_gate":False,"use_routing":False},"loss":{"lambda_peak":0.0,"lambda_timing":0.0},"training":{"use_gradient_surgery":False}},
    "dual_partition":{"model":{"use_partition_gate":True,"use_routing":False},"loss":{"lambda_peak":0.0,"lambda_timing":0.0},"training":{"use_gradient_surgery":False}},
    "dual_routing":{"model":{"use_partition_gate":True,"use_routing":True},"training":{"use_gradient_surgery":False}},
    "cfsr_lstm":{"model":{"use_partition_gate":True,"use_routing":True},"training":{"use_gradient_surgery":True}},
}

if __name__ == "__main__":
    base=load_config("config.yaml"); summaries={}
    for name,over in VARIANTS.items():
        set_seed(base.get("seed",42)); cfg=deep_update(base,over)
        cfg=deep_update(cfg,{"training":{"log_path":f"results/logs/{name}.csv","checkpoint_path":f"results/checkpoints/{name}.pt"}})
        tr,va,_,_=build_dataloaders(cfg); hist=CFSRTrainer(build_model(cfg),cfg).fit(tr,va)
        summaries[name]=hist[-1] if hist else {}
    Path("results/tables").mkdir(parents=True,exist_ok=True)
    Path("results/tables/full_comparison.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
