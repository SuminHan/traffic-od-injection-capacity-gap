"""
Seed-control (free-lunch) checkpoints for our own lightweight model ("Ours"): a second plain
(no-injection) checkpoint per fold, identical to the Section 6 tfold*/sfold*_only_ext run except
for --seed 1. Needed so Ours' selective-injection result (Table absperf) can be netted against its
own selection-noise baseline, exactly as Table freelunch does for the ten other architectures.

Resumable: skips any fold whose output dir already has test_predictions.csv. Runs N_PAR jobs at
once (the model is small; one job uses a fraction of the GPU).
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

GTS = "/home/ncrc/work/gts"
PY = sys.executable
N_PAR = 3

TASKS = [("multi_fold_traffic_results_ext.json", "traffic_only", "train_traffic_model.py"),
         ("multi_fold_speed_results_ext.json", "speed_only", "train_speed_model.py")]

jobs = []
for results_file, plain_key, script in TASKS:
    for e in json.load(open(f"{GTS}/{results_file}")):
        if e["model"] != plain_key:
            continue
        a = dict(e["args"])
        a["seed"] = 1
        a["out"] = a["out"] + "_s1"
        if os.path.exists(f"{GTS}/{a['out']}/test_predictions.csv"):
            continue
        cmd = [PY, "-u", script] + [x for k, v in a.items() for x in (f"--{k}", str(v))]
        jobs.append((a["out"], cmd))

print(f"{len(jobs)} jobs to run", flush=True)


def run(job):
    out, cmd = job
    with open(f"{GTS}/logs_seedctl_ours_{out}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=GTS, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{out}: rc={rc}", flush=True)
    return rc


with ThreadPoolExecutor(N_PAR) as ex:
    rcs = list(ex.map(run, jobs))
print("SEEDCTL_OURS_ALL_DONE", "failures:", sum(r != 0 for r in rcs), flush=True)
