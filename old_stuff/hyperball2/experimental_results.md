# Baseline LR Search Experimental Results

Date: 2026-05-08

All LR-search runs used a single searched subgroup, `all`. Lower `val_bpb` is better.

## Reference

| Run | val_bpb | Gap vs baseline | Training seconds | MFU % | Notes |
|---|---:|---:|---:|---:|---|
| `hyperball_baseline4.log` | 1.0395 | 0.0000 | 353.34 | 10.588 | Fixed hand schedule: effective LR_MULT starts at 1.5 and warms down to 0.075. |

Baseline effective LR_MULT medians:

| Step range | Median effective LR_MULT |
|---|---:|
| 0-150 | 1.5000 |
| 150-900 | 1.3192 |
| 900-1350 | 0.4138 |

## Sweep Results

| Rank | Experiment | val_bpb | Gap | Final LR | Searches | Train seconds | MFU % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `rollout8_smooth` | 1.0505 | +0.0110 | 0.16037 | 85 | 1052.0 | 3.5563 |
| 2 | `rollout4_wide_grid` | 1.0691 | +0.0296 | 0.093576 | 169 | 983.9 | 3.8023 |
| 3 | `rollout4_smooth` | 1.0723 | +0.0328 | 0.11299 | 169 | 971.66 | 3.8502 |
| 4 | `rollout2_smooth` | 1.0740 | +0.0345 | 0.12928 | 338 | 951.49 | 3.9319 |
| 5 | `rollout4_strict` | 1.0762 | +0.0367 | 0.06958 | 169 | 988.46 | 3.7848 |
| 6 | `rollout4_loose` | 1.0781 | +0.0386 | 0.23383 | 169 | 951.55 | 3.9316 |
| 7 | `rotating_smooth` | 1.0835 | +0.0440 | 0.14017 | 1350 | 1656.0 | 2.2592 |
| 8 | `rotating_safe` | 1.0971 | +0.0576 | 0.11138 | 1350 | 1709.4 | 2.1886 |
| 9 | `next_batch_safe` | 1.0975 | +0.0580 | 0.11138 | 1350 | 1707.1 | 2.1915 |
| 10 | `rollout2_safe` | 1.1031 | +0.0636 | 0.068399 | 338 | 946.54 | 3.9524 |
| 11 | `rollout4_smooth_norm` | 1.1039 | +0.0644 | 0.80523 | 169 | 1045.2 | 3.5795 |
| 12 | `rollout4_safe` | 1.1269 | +0.0874 | 0.042006 | 169 | 906.32 | 4.1278 |
| 13 | `rollout8_safe` | 1.1272 | +0.0877 | 0.035705 | 85 | 933.71 | 4.0067 |
| 14 | `one_step_heldout` | 1.1481 | +0.1086 | 0.025797 | 1350 | 729.98 | 5.1250 |

## LR Trace Summary

| Experiment | Median LR 0-150 | Median LR 150-900 | Median LR 900-1350 | Comment |
|---|---:|---:|---:|---|
| Baseline equivalent | 1.5000 | 1.3192 | 0.4138 | Reference schedule. |
| `rollout8_smooth` | 1.3652 | 0.6335 | 0.2307 | Best result; still decays much earlier and lower than baseline. |
| `rollout4_wide_grid` | 1.2092 | 0.2640 | 0.0988 | Wider grid helped over ordinary rollout4, but LR still collapsed in the middle. |
| `rollout4_smooth` | 1.1458 | 0.2501 | 0.1193 | Smooth/inertial variant is clearly better than greedy safe search. |
| `rollout2_smooth` | 0.9363 | 0.1693 | 0.1364 | Short horizon remains too conservative. |
| `rollout4_smooth_norm` | 1.1771 | 0.7767 | 0.7920 | Norm feedback prevented LR collapse, but performance got worse. |
| `one_step_heldout` | 0.2953 | 0.0494 | 0.0303 | Classic myopic failure. |

## Takeaways

1. The best run was `rollout8_smooth` at `1.0505`, which is close-ish but still materially worse than the baseline at `1.0395`.
2. One-step and greedy "safe" selection are too conservative. They quickly reduce LR because immediate held-out loss prefers small changes over useful noisy training.
3. Longer lookahead helps. Moving from one-step to 2/4/8-step smooth rollout improved the result, with 8-step smooth winning.
4. Smoothing/inertia helps a lot. It consistently beats the corresponding greedy safe variants.
5. The current progress score is probably too crude. It uses absolute rollout train loss, not improvement versus the rollout start, so it does not directly reward learning velocity.
6. Norm feedback as implemented is not enough. It kept LR high, but the result was worse, suggesting the update-norm target is poorly calibrated or the model needs a shaped schedule rather than a constant high LR.
7. Search overhead is large. Baseline MFU was `10.588%`; the better search runs were around `3.5-3.9%`, and per-step searches were around `2.2%`.

## New Ideas To Try

These remain scheduler-agnostic: they do not inspect or encode the baseline's ground-truth LR schedule.

1. **Block-level LR decisions.**
   Commit to one LR for a block of 16-64 real training steps, then score the block on held-out batches. This avoids the next-step loss trap and makes LR selection care about medium-term learning. Start with 32-step blocks and candidates `{0.7x, 1.0x, 1.3x}` around the current LR.

2. **Search a local LR trajectory, not a constant rollout LR.**
   During trial rollout, evaluate two-parameter schedules such as `(start_lr, end_lr)` over the horizon with linear interpolation in log space. This can discover "stay high now, decay later" without knowing the baseline schedule.

3. **Use improvement-based scoring.**
   Replace `score = val_after + alpha * train_loss` with:
   `score = val_after - val_before - alpha * (train_before - train_after)`.
   This directly rewards learning progress instead of penalizing high absolute train loss.

4. **Use confidence intervals for largest-safe selection.**
   Estimate validation noise across the rotating held-out batches. Pick the largest LR whose mean loss is within one standard error of the best. This should be more principled than the fixed `safe_loss_atol`/`rtol`.

5. **Delayed LR adaptation with a low-pass hypergradient.**
   Evaluate `lr / r`, `lr`, and `lr * r`, fit a quadratic in `log(lr)` from validation loss after rollout, and move the LR center a small amount along the fitted optimum. This avoids jumping to noisy argmins.

6. **Learning-speed constrained search.**
   Require each candidate to achieve a minimum training loss decrease over the rollout, then choose the safest validation candidate among those. This directly prevents "do almost nothing" from winning.

7. **Stability-edge LR search.**
   Track activation spikes, update rotations, and loss jumps. Increase LR until a stability statistic approaches a threshold, then back off. This treats validation loss as a guardrail, not the primary immediate objective.

8. **Early calibration phase plus online adaptation.**
   Run an LR range test for the first 50-100 steps using held-out safeguards, choose the largest stable LR, then only adapt every 32-64 steps. This gives the controller a good initial scale without hard-coding a schedule.

9. **Population-based perturbations from checkpoints.**
   Every 100-200 steps, fork 3-5 short shadow rollouts from the current checkpoint with different LR multipliers. Promote the best LR, not the shadow weights. This is expensive, but it tests LR decisions at a horizon long enough to matter.

10. **Validation-set curriculum for LR search.**
    Early in training, score on next-train/fresh-train batches to avoid over-penalizing useful SGD noise. Later, shift weight to held-out validation. The rule can be based on observed train/val gap, not on step fraction.

11. **Adaptive horizon.**
    Use short horizons while loss is changing quickly and longer horizons when candidate rankings become noisy or conservative. Trigger horizon increases when the chosen LR keeps decreasing for several searches.

12. **Search LR and momentum together.**
    The Muon momentum schedule is still hand-shaped. Try a one-dimensional "effective step aggressiveness" controller that adjusts LR while inversely adjusting momentum, or a small 2D search over global LR and Muon momentum.

## Suggested Next Sweep

Prioritize ideas that attack the observed early LR collapse:

| Proposed run | Core change |
|---|---|
| `block32_ci` | 32-step committed LR blocks, largest LR within 1 SE of best held-out loss. |
| `block64_ci` | Same as above with 64-step blocks. |
| `traj8_smooth` | 8-step rollout over `(start_lr, end_lr)` log-linear LR trajectory. |
| `traj16_smooth` | 16-step rollout over `(start_lr, end_lr)`, search every 32 steps. |
| `improvement_score8` | 8-step rollout with improvement-based train/val scoring. |
| `min_train_progress8` | Reject candidates without enough rollout train-loss improvement. |
| `hypergrad8` | Quadratic fit in log LR, low-pass update. |
| `stability_edge` | Increase LR until update/activation stability guardrail is approached. |
| `range_test_then_adapt` | First 100 steps calibrate LR scale, then block-level adaptation. |
| `fresh_then_heldout` | Early score on fresh train batches, later score on held-out gap. |
| `pbt100` | Every 100 steps, run shadow LR rollouts and adopt LR only. |
| `lr_momentum_pair` | Joint search over global LR and Muon momentum. |

