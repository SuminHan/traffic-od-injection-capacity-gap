"""
Fills status_report_template.html's placeholders with live per-fold data from
multi_fold_traffic_results.json (volume, 23-fold, static/complete) and
multi_fold_speed_results.json (speed, 23-fold, grows as run_multi_fold_speed.py progresses)
and writes status_report.html. Safe to re-run at any point while the speed sweep is still going --
it just shows however many folds are done so far.
"""
import json

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
N_FOLDS = 23


def per_fold_rows(path):
    d = json.load(open(f"{GTS}/{path}"))
    by_fold = {}
    for r in d:
        by_fold.setdefault(r["fold_start"], {})[r["model"]] = r
    rows = []
    for fs in sorted(by_fold):
        m = by_fold[fs]
        only_key = [k for k in m if k.endswith("only")]
        hyb_key = [k for k in m if "hybrid" in k]
        if not only_key or not hyb_key:
            continue
        o, h = m[only_key[0]], m[hyb_key[0]]
        label = o.get("fold_label", "")
        test_period = label.split("test=")[-1] if "test=" in label else ""
        row = {"fold": fs, "test": test_period}
        for met in ["test_mse", "test_rmse", "test_mae", "test_mape"]:
            if met in o and met in h:
                row[met + "_pct"] = (h[met] - o[met]) / o[met] * 100
        rows.append(row)
    return rows


def cell(v):
    cls = "pos" if v < 0 else "neg"
    return f'<td class="{cls}">{v:+.1f}%</td>'


def build_table(rows):
    out = ["<div class=\"tablewrap\"><table>"]
    out.append("<tr><th>fold</th><th>test 기간</th><th>MSE Δ</th><th>RMSE Δ</th><th>MAE Δ</th><th>MAPE Δ</th></tr>")
    for r in rows:
        out.append(
            "<tr>" + f'<td>#{r["fold"]}</td><td>{r["test"]}</td>'
            + cell(r.get("test_mse_pct", 0)) + cell(r.get("test_rmse_pct", 0))
            + cell(r.get("test_mae_pct", 0)) + cell(r.get("test_mape_pct", 0)) + "</tr>"
        )
    out.append("</table></div>")
    return "\n".join(out)


vrows = per_fold_rows("multi_fold_traffic_results.json")
srows = per_fold_rows("multi_fold_speed_results.json")

tmpl = open(f"{GTS}/status_report_template.html").read()
tmpl = tmpl.replace("__VOLUME_FOLD_TABLE__", build_table(vrows))
tmpl = tmpl.replace("__SPEED_FOLD_TABLE__", build_table(srows))
tmpl = tmpl.replace("__SPEED_DONE__", str(len(srows)))

with open(f"{GTS}/status_report.html", "w") as f:
    f.write(tmpl)
print(f"wrote status_report.html: volume {len(vrows)}/{N_FOLDS} folds, speed {len(srows)}/{N_FOLDS} folds")
