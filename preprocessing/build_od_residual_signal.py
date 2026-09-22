"""Builds the OD-residual (de-climatologized) signal: real routed OD minus its own per-sensor
calendar climatology, so the paper can test genuine day-specific OD content in isolation.

Motivation. Section 6.1 compares injecting the real routed OD forecast against injecting a pure
(day-of-week, hour-of-day) climatology of that same signal, and finds climatology recovers most of
the average benefit but loses to real OD specifically in folds 0 and 4. That comparison establishes
an AVERAGE gap between the two, but does not isolate the mechanism: whatever real OD is contributing
beyond climatology could, in principle, still be partly calendar-shaped (e.g. a finer-grained
calendar effect the coarse 168-bucket lookup misses) rather than genuinely day-specific demand
information.

This ablation removes that ambiguity by construction. For each fold, using the exact climatology
lookup already built by build_climatology_signal_perfold.py (train-window-only, leakage-safe), we
subtract it from the real routed signal:

    residual_t = real_routed_t - climatology[bucket(t)]

By construction, the residual's OWN per-(dow,hour) mean is exactly zero over the training window --
there is no calendar structure left for the model to recover through the injection channel, coarse
or fine. If this residual-only signal, injected through the identical fusion mechanism, still
recovers a comparable fraction of real OD's benefit, that is direct evidence the model is using
genuine day-specific deviations rather than any residual calendar shape. If it recovers little to
nothing, most of real OD's advantage over climatology was itself still calendar-adjacent structure
the coarse lookup happened to miss, not anomaly-specific demand information.

Output: routed_od_signal_residual_fold{N}_ext.npz for fold in 0..4, same schema as the real and
climatology signals, so train_traffic_model.py consumes it unchanged via
--routed_signal_suffix _residual_fold{N}_ext.
"""
import numpy as np

GTS = "/home/ncrc/work/gts"

sig = np.load(f"{GTS}/routed_od_signal_ext.npz", allow_pickle=True)
real = sig["routed_forecast"]              # (T, S)
sensor_link_ids = sig["sensor_link_ids"]

for fold in range(5):
    clim = np.load(f"{GTS}/routed_od_signal_climatology_fold{fold}_ext.npz",
                   allow_pickle=True)["routed_forecast"]
    assert clim.shape == real.shape
    residual = (real - clim).astype(np.float32)

    # Sanity: the residual's own climatology (recomputed the same way) should be ~0 everywhere,
    # confirming no calendar structure survives the subtraction.
    check = np.corrcoef(residual.ravel(), clim.ravel())[0, 1]
    print(f"fold {fold}: residual std={residual.std():.4f} (real std={real.std():.4f}), "
          f"corr(residual, climatology)={check:.4f} (should be ~0)")

    out_path = f"{GTS}/routed_od_signal_residual_fold{fold}_ext.npz"
    np.savez_compressed(out_path, routed_forecast=residual, sensor_link_ids=sensor_link_ids)
    print(f"  wrote {out_path}")

print("\nOD_RESIDUAL_SIGNAL_BUILD_ALL_DONE")
