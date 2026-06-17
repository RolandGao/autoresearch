# cifar_baseline2_backprop4 Notes

Goal: train loss below 0.88 in at most 8 legal steps, where each step is one forward pass and one backward pass.

Baseline before changes:

```text
loss_step 08/20 loss=0.989336
```

Best verified result reached during this pass:

```text
loss_step 01/20 loss=1.756765
loss_step 02/20 loss=1.466759
loss_step 03/20 loss=1.331660
loss_step 04/20 loss=1.263923
loss_step 05/20 loss=1.090856
loss_step 06/20 loss=1.005516
loss_step 07/20 loss=0.955370
loss_step 08/20 loss=0.921626
```

This improved step 8, but did not reach the target.

## Things That Worked

- One-step warmup for target updates.
  - Step 1 uses no target-update momentum and gain 1.
  - This avoided the huge first-step spike from the original default path.
  - It reproduced the useful early behavior of older direct-solve variants while preserving later acceleration.

- Revising inactive-negative ReLU backward handling.
  - Instead of forcing `x < 0` and `dy < 0` positions to a hard zero target, pass a bounded negative direction backward.
  - This improved the early trajectory and helped step 8.

- Updating existing BatchNorm scale parameters in the backward pass.
  - `BatchNorm.weight.requires_grad` is false, but the parameter exists and affects the forward pass.
  - A backward-only target update for gamma improved the best step-8 result from about `0.931` to about `0.922`.
  - Larger BN gamma gain was worse; conservative updates were best.

- Slightly higher CE final target scale.
  - Raising `TARGET_DELTA_SCALE_FINAL` from `2.35` to about `3.1` helped after the warmup and BN changes.
  - Higher values became unstable.

- Slightly looser trainable weight RMS projection.
  - Moving the default projection cap from `1.0` to `1.1` gave a small improvement.
  - Larger caps did not help.

## Things That Did Not Work

- Raw target update gain increases.
  - Higher `TARGET_UPDATE_GAIN` destabilized training.
  - Gains like `2.0` and `2.5` produced worse or exploding step-8 losses.

- Lower/no momentum after warmup.
  - Reducing or removing target-update momentum made the early curve smoother in some cases but missed step 8 badly.

- Extending the warmup to 2 or more steps.
  - One warmup step was best.
  - Two warmup steps slowed the descent and worsened step 8.

- Stronger CE schedule compression.
  - Making the CE target scale reach a much stronger value by step 8 caused instability.
  - Values around `4.0+` eventually blew up.

- Probability-residual CE delta.
  - Using `target_probs - softmax(logits)` was too weak in the initial huge-logit regime.
  - It failed to escape quickly enough.

- Softmax-Newton CE delta.
  - Pure Newton and logit-then-Newton variants were worse than the logit target.
  - They were too weak or poorly matched to the target-backprop solves here.

- Head retargeting to predicted feature targets.
  - Trying to update the head again for `x + dx` destabilized even at very small gains.

- Head-only optimization/scaling at step 8.
  - The step-8 head was already locally optimal on frozen features.
  - Scalar logit temperature scaling worsened the loss.
  - Optimizing only the linear head on the step-8 representation did not get below the current loss.

- Prototype feature targets for the head.
  - Sending class-prototype feature targets backward was much worse.
  - The conv stack did not realize those artificial targets well.

- Late-layer update gain.
  - Increasing updates for final conv/head-adjacent ops destabilized if applied early.
  - Applying it only on step 8 was mostly neutral or worse.

- Damping late-layer updates.
  - Reducing or disabling late updates after step 5 stalled training.
  - Those late updates are necessary.

- Updating the frozen whitening convolution.
  - Allowing the backward pass to update frozen conv parameters did not help step 8.

- Conv normal-equation ridge damping.
  - Adding `TARGET_CONV_W_LAMBDA` did not improve the result.
  - Once the damping was large enough to matter, it worsened step 8.

- Rejecting parameter updates when local suffix loss got worse.
  - A guard that restored module params after worse local `train_loss_after_dw` stalled around `1.20`.
  - Local greedy rejection was too conservative.

- Letting early BatchNorm scale updates bypass the freeze.
  - Keeping early BN gamma updates alive after early convs froze made the loss worse.

## Useful Diagnostics

- The head alone cannot solve the first-batch problem from the initial representation.
  - Direct head-only fits on initial features bottomed out around `1.7+`.

- At step 8, local suffix probes showed the late target could imply around `0.90`, but the actual realized forward loss stayed around `0.92`.
  - The remaining gap appears to be representation realization, not head scaling.

- Existing historical logs reached below `0.88`, but only around step 14-16.
  - The current work compressed the step-8 loss substantially but did not compress the full crossing point to 8 steps.
