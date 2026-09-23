# When Does Population-Mobility OD Injection Help Traffic Forecasting?

### A Capacity-Gap Account Across Eleven Models

Code, analysis scripts, and per-fold results for a submission to *IEEE Transactions on Knowledge
and Data Engineering* (TKDE). Raw data and trained checkpoints are not included; everything else
behind every number and figure in the paper is here.

- **Paper:** [`paper/main.pdf`](paper/main.pdf) (12 pages)
- **Supplementary material (Appendices A–E):** [`paper/supplement.pdf`](paper/supplement.pdf)
- **Datasets** (contents, schemas, download links, samples): [`data/README.md`](data/README.md)

---

## In one paragraph

Forecast origin–destination (OD) population movement in Seoul, routed onto the road network, is
injected into eleven traffic forecasters (121 volume sensors, 396 speed sensors, 30-fold
rolling-origin evaluation, 2023–2026). Injected through a learned gate, it improves a lightweight
graph-recurrent model (MSE −7.2% volume, −8.2% speed), but the same mechanism helps only the two
smallest of ten published architectures and is significantly *harmful*, on at least one task, for
every architecture above 10⁵ parameters — a **capacity gap**. A per-sensor selective-injection
remedy turns out to be dominated by simply *averaging* the plain and injected checkpoints, so we
instead test the signal without any fusion or selection mechanism at all: as an **ensemble
partner** against a second plain seed. That test confirms the capacity gap independent of fusion
design, and is sharper than the original uniform-injection test.

## Key results

**Figure 6 — the capacity gap, two ways.** *(a, b)* Absolute RMSE per architecture, plain (○) →
uniformly injected (●, green improves / red worsens). *(c)* The ensemble-controlled test: how much
better the OD-injected model is as an ensemble partner than a second plain seed, for all eleven
models — below zero, the signal adds value beyond ensembling.

![capacity law](figures/rendered/fig_capacity_law.png)

**Table 7 — does the signal add value beyond ensembling?** (MSE change vs. plain, 30 folds; bold =
significant after Benjamini–Hochberg correction across all 22 rows):

| Model | Params | Task | Uniform | Avg(plain, injected) | Avg(plain, 2nd seed) | OD partner gain |
|---|---:|---|---|---|---|---|
| STID | 31k | V / S | **−9.02%** / +0.24% | −13.19% / −4.24% | −6.33% / −2.22% | **−6.87%** / **−2.02%** |
| *Ours (lightweight)* | 37k | V / S | **−7.17%** / **−8.17%** | −11.62% / −8.47% | −7.72% / −6.73% | **−3.90%** / **−1.73%** |
| GMAN | 55k | V / S | **−7.53%** / +0.19% | −18.10% / −6.82% | −11.17% / −4.34% | **−6.94%** / **−2.48%** |
| STGCN | 62k | V / S | −2.09% / +0.09% | −13.90% / −5.45% | −10.05% / −3.66% | **−3.85%** / **−1.79%** |
| PDFormer | 95k | V / S | −2.70% / **+3.25%** | −12.11% / −3.14% | −11.32% / −3.25% | −0.79% / +0.11% |
| STAEformer | 160k | V / S | +0.74% / **+4.71%** | −10.69% / −2.93% | −11.62% / −4.12% | +0.94% / **+1.19%** |
| MTGNN | 335k | V / S | **+11.07%** / **+4.24%** | −6.98% / −3.13% | −8.31% / −3.40% | +1.34% / +0.27% |
| DCRNN | 382k | V / S | +0.95% / **+1.76%** | −9.92% / −3.50% | −8.88% / −3.50% | −1.04% / +0.00% |
| GTS | 395k | V / S | −0.19% / **+3.86%** | −10.86% / −3.17% | −10.23% / −3.72% | −0.63% / **+0.55%** |
| Graph WaveNet | 673k | V / S | **+25.45%** / **+3.77%** | −2.83% / −4.29% | −10.08% / −5.61% | **+7.24%** / **+1.32%** |
| AGCRN | 774k | V / S | **+11.92%** / **+0.91%** | −6.97% / −3.88% | −11.06% / −3.86% | **+4.08%** / −0.02% |

For every model up to 62k parameters, the injected model is a significantly better ensemble
partner than a second seed, on **both** tasks — including speed, where uniform injection showed no
gain at all. For no model of 95k parameters or more is it better, and for five combinations it is
significantly worse. Correlation with log-parameters: ρ=0.77 (p=2.5×10⁻⁵), sharper than under
uniform injection (ρ=0.67).

**Checks that bound these claims** (all in the paper):
- *Calendar confounding* (Table 5): a calendar climatology of the routed signal recovers 87% of the
  lightweight model's volume gain and 96% of its speed gain; the calendar-free residual alone
  recovers 34% (volume) / 71% (speed) — speed carries roughly twice the genuine day-specific share.
- *Selective injection is dominated by averaging* (supplementary Appendix B): the per-sensor
  selector beats simple averaging in only 1 of 22 combinations. Net of a seed-only selection-noise
  baseline, it is significant in 11 of 20 combinations and reverses in exactly the 3 that uniform
  injection harmed most.
- *Within-architecture capacity sweep* (Table 8): scaling STID 31k → 419k parameters moves the
  injection effect from −9.02% to +3.37% (significantly harmful).
- *Cross-dataset replication* (Table 9): the capacity law replicates in direction on three further
  public benchmarks (PEMS-BAY, METR-LA, PeMSD7) with a calendar signal, reaching significance on
  PEMS-BAY.

## Where each paper result comes from

| Paper item | Result files (`results/`) | Script |
|---|---|---|
| Table 4, Fig. 3 (lightweight model) | `summaries/multi_fold_{traffic,speed}_results_ext.json` | `training/train_traffic_model.py`, `train_speed_model.py` |
| Table 5 (signal controls, volume) | `summaries/{naive_signal,climatology,od_residual,dual_signal}_ablation_results.json` | `training/run_*_ablation_30.py` |
| Table 5 (signal controls, speed) | `summaries/speed_calendar_controls_{results.json,summary.csv}` | `preprocessing/build_speed_calendar_controls_30.py`, `training/run_speed_calendar_controls_30.py` |
| Table 6, Fig. 6a–b (uniform, ten architectures) | `summaries/multi_fold_baseline_*_results_ext30.json` | `training/train_baseline_model*.py` |
| Table 7, Fig. 6c (ensemble-controlled test) | `per_fold/ensemble_check_*_fold_results.csv`, `summaries/{ensemble_check_summary,ensemble_control_table}.csv` | `analysis/ensemble_baseline_check.py`, `figures/scripts/make_fig_capacity_law.py` |
| Table 8 (STID capacity sweep) | `summaries/stid_fixed_volume_capacity_sweep_summary.csv` | `training/train_baseline_model_extra2.py` |
| Table 9 (cross-dataset) | `summaries/{pemsbayh,metrlah,pemsd7h}_capacity_summary.csv` | `training/train_benchmark_model.py` |
| Suppl. Appendix B (selective injection, in full) | `per_fold/selective_injection_*_fold_results.csv`, `per_fold/seed_control_*_fold_results.csv`, `summaries/freelunch_net_significance.csv` | `analysis/selective_injection_*.py`, `analysis/seed_control_selective_*.py`, `analysis/freelunch_net_significance.py` |
| Suppl. Appendix B.1 (adaptive threshold) | `per_fold/noise_calibrated_adaptive_*`, `summaries/noise_calibrated_adaptive_full_summary.csv` | `analysis/noise_calibrated_adaptive_full.py` |
| Suppl. random-selection control | `summaries/selective_injection_random_control_*_summary.csv` | `analysis/selective_injection_random_control*.py` |

STID always refers to the corrected architecture (`stid_fixed` in file names); the first
implementation had a calendar-embedding bug (Appendix E) and its results are not used.

## Repository layout

```
models/           model definitions (reimplemented baselines; our model is in training/train_*_model.py)
preprocessing/    raw data -> tensors: OD tensor, traffic tensors, routing weights W, routed signals
training/         training entry points and multi-fold / seed-control runners
analysis/         ensemble-controlled test, selective injection, free-lunch (seed-control),
                  random-selection control, adaptive threshold -- the scripts behind Section 7 and
                  Appendix B
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
             A Capacity-Gap Account Across Eleven Models},
  author  = {Han, Sumin},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  note    = {under review},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The paper text and figures
follow standard IEEE copyright policy upon publication.
