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

## Follow-up Pass: Best Verified `0.884737`

Best clean no-env result after the later backward-pass tuning:

```text
loss_step 01/20 loss=1.768774
loss_step 02/20 loss=1.444249
loss_step 03/20 loss=1.298850
loss_step 04/20 loss=1.094660
loss_step 05/20 loss=0.971039
loss_step 06/20 loss=0.920249
loss_step 07/20 loss=0.898178
loss_step 08/20 loss=0.884737
```

This was a large improvement over `0.921626`, but still did not reach the `<0.88` target.

### Things That Worked

- Moving `LATE_UPDATE_FREEZE_STEP` from `5` to `4`.
  - This was the first reproducible improvement in the new pass.
  - Alone it reached about `0.917820`.

- Splitting the head weight update target from the representation target.
  - `TARGET_HEAD_DW_DELTA_SCALE=1.0` kept the head update closer to the CE/logit target while allowing the stronger representation target to continue propagating.
  - This was important for stabilizing stronger late-block gains.

- Enabling backward updates for the frozen whitening convolution.
  - This did not help in the older baseline, but it helped after the freeze/head/late-gain changes.
  - In the stronger late-gain branch it improved step 8 to about `0.893811`.
  - A separate frozen-conv scale was tested; the default `1.0` was best among the tried values.

- Targeted late-block gain.
  - Applying late gain to op `16+` from step 6 helped once the head update was stabilized.
  - Uniform gain had a knee: too much improved steps 5-6 but caused step-8 overshoot.
  - A tuned global late gain around `1.6`-`1.65` worked best after op-specific gains were added.

- Op-specific late gains in the final block.
  - Dampening op 16 (`layers.3.conv1`) helped; `TARGET_OP16_GAIN=0.2` was best among tried values.
  - Boosting op 18 (`layers.3.norm1`) helped a lot; `TARGET_OP18_GAIN=3.0` was best among tried values.
  - Mildly boosting op 20 (`layers.3.conv2`) helped; `TARGET_OP20_GAIN=1.3` was best among tried values.
  - Dampening op 21 (`layers.3.norm2`) helped when op 18 was boosted; values around `0.5`-`0.65` were best.

- Slightly stronger final CE target scale in the tuned branch.
  - After late-block tuning, `TARGET_DELTA_SCALE_FINAL=3.6` beat the earlier `3.1`.
  - Larger values started to hurt step 8 again.

- A tiny head weight scale above 1.
  - `TARGET_HEAD_WEIGHT_SCALE=1.05` gave a small improvement in the tuned branch.
  - Larger head scaling became worse.

### Things That Did Not Work

- Standard CE optimizer-style updates.
  - Pure SGD/Adam/Muon-style CE backward updates were nowhere close in 8 steps.
  - Adding a CE-gradient correction hook into the target-backward pass destabilized training, even with very small learning rates.

- Lowering the final CE scale globally.
  - Analytically, the step-8 logits preferred a lower logit-delta scale locally, but lowering the schedule hurt the realized forward loss.
  - The stronger target was still useful as a representation driver.

- Step-8-only CE scale drops.
  - Softer step-8 logit targets improved the local logit CE target but worsened the realized model loss.

- More literal ReLU ignore behavior.
  - Reducing `RELU_REVIVE_CAP`, setting it to zero, or ignoring zero-dy rows in conv solves all worsened step 8.
  - The bounded negative inactive ReLU target remained better.

- Broadly unfreezing earlier late updates.
  - Letting ops before 16 update after the freeze generally worsened step 8.
  - The old freeze boundary was too late, but the op boundary around 16 remained important.

- Larger uniform late gain.
  - Stronger late gain often reduced step 5-6 losses but overshot by step 8.
  - Op-specific gains were much better than one large scalar.

- Head retargeting to predicted `x + dx` features.
  - Gating it to step 8 and trying small or large gains still worsened the final loss.

- Prototype feature targets.
  - Still much worse in the tuned branch, landing around `1.2+` at step 8.

- Conv weight ridge damping.
  - `TARGET_CONV_W_LAMBDA` remained neutral at tiny values and worse when large enough to matter.

- Tuning `TARGET_X_LAMBDA`.
  - Changing the input-target ridge moved only the fourth decimal in the best branch.

- BatchNorm tangent backward mode and BN gamma max changes.
  - Tangent mode was effectively a wash or slightly worse.
  - Changing `TARGET_BN_WEIGHT_MAX` had no effect in the tuned branch, suggesting the max clamp was not active.

- Head op gain changes.
  - Damping or boosting op 25 was worse; the default op 25 scaling was best.
