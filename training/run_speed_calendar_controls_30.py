"""Speed-task climatology-only and residual-only injection runs (30 folds each), the speed
counterpart of run_climatology_ablation_30.py / run_od_residual_ablation_30.py. Every argument is
copied from that fold's Speed-Hybrid run; only the injected signal file and output dir differ.
Writes speed_calendar_controls_results.json (same schema as the volume *_ablation_results.json).
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

GTS = "/home/ncrc/work/gts"
PY = sys.executable
hyb = {e["fold_start"]: e for e in json.load(open(f"{GTS}/multi_fold_speed_results_ext.json"))
       if e["model"] == "speed_hybrid_routed"}

jobs = []
for kind in ("climatology", "residual"):
    for fold in range(30):
        a = dict(hyb[fold]["args"])
        a["routed_signal_suffix"] = f"_{kind}_fold{fold}_ext"
        a["out"] = f"sfold{fold}_hybrid_{kind}_ext"
        if os.path.exists(f"{GTS}/{a['out']}/summary.json"):
            continue
        cmd = [PY, "-u", "train_speed_model.py"] + [x for k, v in a.items() for x in (f"--{k}", str(v))]
        jobs.append((kind, fold, a["out"], cmd))
print(len(jobs), "jobs", flush=True)


def run(job):
    kind, fold, out, cmd = job
    with open(f"{GTS}/logs_speedctl_{out}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=GTS, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{out}: rc={rc}", flush=True)


with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, jobs))

res = []
for kind in ("climatology", "residual"):
    for fold in range(30):
        f = f"{GTS}/sfold{fold}_hybrid_{kind}_ext/summary.json"
        if os.path.exists(f):
            res.append({"kind": kind, "fold_start": fold, **json.load(open(f))})
json.dump(res, open(f"{GTS}/speed_calendar_controls_results.json", "w"), indent=1)
print("SPEED_CONTROLS_ALL_DONE", len(res), flush=True)
