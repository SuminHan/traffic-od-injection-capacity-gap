# When Does Population-Mobility OD Injection Help Traffic Forecasting?

### A Capacity-Gap Account and a Validation-Gated Remedy

Code, analysis scripts, and per-fold results for a submission to *IEEE Transactions on Knowledge
and Data Engineering* (TKDE). Raw data and trained checkpoints are not included; everything else
behind every number and figure in the paper is here.

- **Paper:** [`paper/main.pdf`](paper/main.pdf) (12 pages)
- **Supplementary material (Appendices A–F):** [`paper/supplement.pdf`](paper/supplement.pdf)
- **Datasets** (contents, schemas, download links, samples): [`data/README.md`](data/README.md)

---

## In one paragraph

Forecast origin–destination (OD) population movement in Seoul, routed onto the road network, is
injected into traffic forecasters (121 volume sensors, 396 speed sensors, 30-fold rolling-origin
evaluation, 2023–2026). **Injected uniformly**, it improves a lightweight graph-recurrent model
(MSE −7.2% volume, −8.2% speed) but fails to generalize: across ten published architectures it
helps only the two smallest and is significantly harmful for every architecture above 10⁵
parameters — a *capacity gap*. **Injected selectively** — a per-sensor choice between the plain and
injected checkpoint, made on each fold's validation month, with no retraining — it improves all
eleven models in absolute RMSE and is significant in 18 of 20 architecture/task combinations,
including 8 of the 10 where uniform injection was significantly harmful.

## Key results

**Figure 6 — uniform injection fails with capacity; selective injection recovers it.** Each row:
plain (○) → uniform injection (●, green = better, red = worse) → selective injection (◆).

![capacity law](figures/rendered/fig_capacity_law.png)

**Table 8 — selective vs. uniform injection, 30 folds** (MSE change vs. plain; bold = significant):

| Model | Params | Volume: selective | Volume: uniform | Speed: selective | Speed: uniform |
|---|---:|---|---|---|---|
| STID | 31k | **−10.00%** | **−9.02%** | **−3.39%** | +0.24% |
| GMAN | 55k | **−14.14%** | **−7.53%** | **−5.54%** | +0.19% |
| STGCN | 62k | **−7.84%** | −2.09% | **−3.73%** | +0.09% |
| PDFormer | 95k | **−6.73%** | −2.70% | **−1.65%** | **+3.25%** |
| STAEformer | 160k | **−4.49%** | +0.74% | **−1.23%** | **+4.71%** |
| MTGNN | 335k | −1.37% (p=0.074) | **+11.07%** | **−1.71%** | **+4.24%** |
| DCRNN | 382k | **−4.06%** | +0.95% | **−1.51%** | **+1.76%** |
| GTS | 395k | **−5.10%** | −0.19% | **−1.44%** | **+3.86%** |
| Graph WaveNet | 673k | −0.46% (p=0.36) | **+25.45%** | **−1.87%** | **+3.77%** |
| AGCRN | 774k | **−1.86%** | **+11.92%** | **−1.45%** | **+0.91%** |
| *Ours (lightweight)*¹ | 62k | **−9.41%** | **−7.17%** | **−9.23%** | **−8.17%** |

¹ Ours is not part of Table 8; shown for reference. Net of its selection-noise baseline, its selective
gain (−5.82% / −7.81%) does not exceed uniform injection — selection helps where uniform injection fails.

**Checks that bound these claims** (all in the paper):
- *Calendar confounding* (Table 5): a calendar climatology of the routed signal recovers 87% of the
  lightweight model's uniform gain; the calendar-free residual alone recovers 34%.
- *Selection noise* (Table 10): net of a seed-only "free lunch" baseline, selective injection stays
  favorable in 17 of 20 combinations; the 3 exceptions are exactly the 3 that uniform injection
  harmed most (Graph WaveNet, MTGNN, AGCRN on volume).
- *Within-architecture capacity sweep* (Table 11): scaling STID 31k → 419k parameters moves the
  injection effect from −9.02% to +3.37% (significantly harmful).

## Where each paper result comes from

| Paper item | Result files (`results/`) | Script |
|---|---|---|
| Table 4, Fig. 3 (lightweight model) | `summaries/multi_fold_{traffic,speed}_results_ext.json` | `training/train_traffic_model.py`, `train_speed_model.py` |
| Table 5 (signal controls) | `summaries/{naive_signal,climatology,od_residual,dual_signal}_ablation_results.json` | `training/run_*_ablation_30.py` |
| Table 6, Fig. 6 (uniform, ten architectures) | `summaries/multi_fold_baseline_*_results_ext30.json` | `training/train_baseline_model*.py`, `figures/scripts/make_fig_capacity_law.py` |
| Table 8 (selective) | `per_fold/selective_injection_*_fold_results.csv` | `analysis/selective_injection_*.py` |
| Table 9 (absolute RMSE) | same as Tables 6, 8 | `analysis/selective_injection_ours.py` (Ours row) |
| Table 10 (free lunch) | `per_fold/seed_control_*_fold_results.csv`, `summaries/seed_control_*_summary.csv` | `analysis/seed_control_selective_*.py`, `training/run_seedctl_*.py` |
| Table 11 (STID sweep) | `summaries/stid_fixed_volume_capacity_sweep_summary.csv` | `training/train_baseline_model_extra2.py` |
| Suppl. Table 13 (adaptive threshold) | `per_fold/noise_calibrated_adaptive_*`, `summaries/noise_calibrated_adaptive_full_summary.csv` | `analysis/noise_calibrated_adaptive_full.py` |
| Suppl. Table 14 (cross-dataset) | `summaries/{pemsbayh,metrlah,pemsd7h}_capacity_summary.csv` | `training/train_benchmark_model.py` |
| Suppl. random-selection control | `summaries/selective_injection_random_control_*_summary.csv` | `analysis/selective_injection_random_control*.py` |

STID always refers to the corrected architecture (`stid_fixed` in file names); the first
implementation had a calendar-embedding bug (Appendix F) and its results are not used.

## Repository layout

```
models/           model definitions (reimplemented baselines; our model is in training/train_*_model.py)
preprocessing/    raw data -> tensors: OD tensor, traffic tensors, routing weights W, routed signals
training/         training entry points and multi-fold / seed-control runners
analysis/         selective injection, free-lunch (seed-control), random-selection control,
                  adaptive threshold -- the scripts behind Sections 7.2-7.3 and Appendix B
figures/scripts/  regenerate the paper figures from results/
figures/rendered/ rendered figures
results/          per-fold and summary result files behind every table
data/             dataset documentation and small samples (no raw data)
paper/            main.pdf + supplement.pdf (separate files, per TKDE submission rules)
tools/            internal progress dashboards used during the study (not needed to reproduce)
```

## Reproducing

This is research code, not a packaged library.

- Path constants at the top of `preprocessing/` and `training/` scripts point at our server layout
  (`/home/ncrc/work/gts`); raw-data paths use the placeholder `/path/to/raw_data`. Adjust both.
- Several conda environments were used (PyTorch for training; pandas/scipy for analysis;
  matplotlib/geopandas for figures). `requirements.txt` is a best-effort union.
- One 30-fold sweep for one (architecture, task) takes 1–14 GPU-hours on a single H200;
  reproducing every table is a multi-week compute job.

## Citation

```bibtex
@article{han2026odinjection,
  title   = {When Does Population-Mobility {OD} Injection Help Traffic Forecasting?
             A Capacity-Gap Account and a Validation-Gated Remedy},
  author  = {Han, Sumin},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  note    = {under review},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The paper text and figures
follow standard IEEE copyright policy upon publication.
