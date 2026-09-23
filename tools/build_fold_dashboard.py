import json, os

GTS = "/home/ncrc/work/gts"
RUNS = ["foldtopk5", "foldtopk10", "foldtopk15", "foldtopk30", "foldtopk60", "foldtopk120",
        "foldtopk250", "folddense", "foldnotime_topk15"]

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

def status_for(recs, summary):
    if summary is not None:
        return "done"
    if not recs:
        return "pending"
    if len(recs) < 5:
        return "improving"
    recent = [r["val_loss"] for r in recs[-8:]]
    if len(recent) >= 6:
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        return "improving" if second_half < first_half * 0.995 else "plateaued"
    return "improving"

rows = []
for name in RUNS:
    recs, summary = load_run(name)
    st = status_for(recs, summary)
    if name == "folddense" and not recs and summary is None:
        st = "skipped"
    last = recs[-1] if recs else None
    rows.append({
        "name": name, "status": st, "n_epochs": len(recs),
        "train_loss": last["train_loss"] if last else None,
        "val_loss": last["val_loss"] if last else None,
        "elapsed_s": last["elapsed_s"] if last else None,
        "summary": summary,
        "val_loss_series": [r["val_loss"] for r in recs],
    })

winner = None
wp = f"{GTS}/winner.json"
if os.path.exists(wp):
    winner = json.load(open(wp))

hybrid_summary = None
snapshot_winner_summary = None
if winner:
    tk = winner["args"]["topk"]
    hp = f"{GTS}/hybrid_winner_topk{tk}/summary.json"
    sp = f"{GTS}/snapshot_winner_topk{tk}/summary.json"
    if os.path.exists(hp):
        hybrid_summary = json.load(open(hp))
    if os.path.exists(sp):
        snapshot_winner_summary = json.load(open(sp))

with open(f"{GTS}/fold_dashboard_data.json", "w") as f:
    json.dump({"rows": rows, "winner": winner, "hybrid_summary": hybrid_summary,
               "snapshot_winner_summary": snapshot_winner_summary}, f)
print("fold dashboard data written,", len(rows), "runs")
for r in rows:
    print(f"  {r['name']:24s} status={r['status']:10s} epoch={r['n_epochs']:3d} val_loss={r['val_loss']}")
