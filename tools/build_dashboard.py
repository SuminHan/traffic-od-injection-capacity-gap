import json, os, glob
import numpy as np

GTS = "/home/ncrc/work/gts"
RUNS = ["sweep_topk5", "sweep_topk10", "sweep_topk15", "sweep_topk30", "sweep_topk60",
        "sweep_topk120", "sweep_topk250", "sweep_dense", "sweep_notime_topk15"]

def load_run(name):
    d = f"{GTS}/{name}"
    log_path = f"{d}/train_log.jsonl"
    recs = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    summary = None
    sp = f"{d}/summary.json"
    if os.path.exists(sp):
        try:
            summary = json.load(open(sp))
        except Exception:
            pass
    return recs, summary

def status_for(recs, summary, target_epochs=80):
    if summary is not None:
        return "done"
    if not recs:
        return "pending"
    if len(recs) < 5:
        return "improving"
    recent = [r["val_loss"] for r in recs[-8:]]
    if len(recent) >= 6:
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        if second_half < first_half * 0.995:
            return "improving"
        else:
            return "plateaued"
    return "improving"

rows = []
for name in RUNS:
    recs, summary = load_run(name)
    st = status_for(recs, summary)
    last = recs[-1] if recs else None
    rows.append({
        "name": name, "status": st, "n_epochs": len(recs),
        "train_loss": last["train_loss"] if last else None,
        "val_loss": last["val_loss"] if last else None,
        "val_od_mae": last["val_od_mae"] if last else None,
        "elapsed_s": last["elapsed_s"] if last else None,
        "summary": summary,
        "val_loss_series": [r["val_loss"] for r in recs],
        "train_loss_series": [r["train_loss"] for r in recs],
    })

with open(f"{GTS}/dashboard_data.json", "w") as f:
    json.dump({"rows": rows}, f)
print("dashboard data written,", len(rows), "runs")
for r in rows:
    print(f"  {r['name']:24s} status={r['status']:10s} epoch={r['n_epochs']:3d} val_loss={r['val_loss']}")
