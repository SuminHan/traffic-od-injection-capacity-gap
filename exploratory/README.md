# Exploratory (not part of the paper)

Pilots run to answer a specific side question, kept for provenance but not part of the paper's
claims or figures.

## `pilot_bottleneck_warmstart.py`

Question: does the capacity gap reflect genuine signal redundancy, or could it be an artifact of
large models being harder to jointly optimize from random init -- i.e., would warm-starting a
large model's OD-injection pathway from an already-trained small model (rather than learning it
jointly, from scratch, at large scale) recover the benefit seen at small scale?

Setup: STID at hidden=256 (the paper's own reversal case, Table VIII: -9.02% at hidden=64 ->
+3.37% at hidden=256), backbone unchanged, but the injection pathway (td_proj/od_proj/gate_net/
step_head) runs through a fixed-width bottleneck (64) instead of scaling with the backbone, either
randomly initialized or warm-started from the hidden=64 model's own (already proven to work)
pathway weights, then fine-tuned end-to-end. 5 folds only (0-4), full 40-epoch/patience-8 training
per fold.

Result: no evidence for the warm-start hypothesis. Warm-starting is still net harmful on average
(+1.09% MSE vs. plain, mean of 5 folds), wins only 2 of 5 folds against plain and only 2 of 5
against the original hidden=256 result, and is not close to significant (paired t-test
p=0.67-0.71 against both comparisons). This is consistent with, not a challenge to, the paper's
account: if the failure were mainly an optimization-difficulty artifact, warm-starting from a
checkpoint that already knows how to use the signal should have recovered a meaningfully larger
share of the small model's -9% benefit, and it did not.

Code changes this required (kept in `models/models_baselines.py` and
`training/train_baseline_model_extra2.py`, both additive -- no existing class or CLI behavior was
changed): `BottleneckODInjectionWrapper` and the `--inject_bottleneck_dim` /
`--inject_warmstart_from` flags.

n=5 folds is underpowered by this paper's own standards (every claim in the paper itself uses
30-fold rolling-origin sweeps); this pilot was intentionally not extended to a full sweep, since
the effect (even directionally) was nowhere near the small model's benefit, making a larger sweep
an unlikely use of compute. Not cited or referenced anywhere in the paper.
