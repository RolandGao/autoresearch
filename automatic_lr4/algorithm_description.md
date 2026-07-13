The file implements a deterministic, interval-by-interval coordinate search over five optimizer hyperparameters. Every candidate in a search segment starts from exactly the same training snapshot, so candidates are compared after seeing identical batches, augmentations, optimizer state, and RNG state.

Source: [cifar_search2.py](/workspace/neural_networks_optimization/autoresearch/automatic_lr4/cifar_search2.py)

## 1. Search space

The active parameters, in search order, are:

1. `muon_lr`
2. `muon_momentum`
3. `whiten_bias_lr`
4. `bn_bias_lr`
5. `head_lr`

The four learning rates use an integer state \(k\):

\[
lr(k)=\operatorname{round\_to\_2\_significant\_figures}
       \left(lr_{\mathrm{initial}}\cdot 0.6^k\right)
\]

Thus:

- increasing \(k\) makes the LR smaller;
- decreasing \(k\) makes it larger;
- initial state is normally \(k=0\);
- a special `"zero"` state represents exactly zero, although the standard neighbor search never generates it.

The LR rounding is equivalent to:

```python
round(value, 2 - 1 - floor(log10(abs(value))))
```

Momentum is an index into:

```python
[0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.9,
 0.95, 0.99]
```

Its initial value is `0.6`, hence initial index `6`.

With the default batch size of 2000, all four initial learning rates equal `1.0`. The three SGD learning rates are generally initialized as:

```python
initial_lr = batch_size / 2000
```

A search point is a tuple of states in the active-parameter order, for example:

```python
(0, 6, 0, 0, 0)
```

## 2. Candidate evaluation

Before searching a segment, save a snapshot containing:

- every model tensor;
- both optimizer state dictionaries;
- batch-stream state:
  - current prepared epoch images;
  - shuffled indices;
  - current batch index;
  - loader epoch number;
  - CPU RNG state;
  - every CUDA RNG state.

To evaluate a candidate:

1. Restore the segment-start snapshot.
2. Convert its point states to actual hyperparameter values.
3. Train for the segment’s requested number of steps.
4. Stop early if a non-finite loss appears.
5. If every requested step completed with finite loss, calculate TTA accuracy.
6. Otherwise assign accuracy `-inf`.

This makes each candidate a counterfactual replay from the same state, rather than continuing from the preceding candidate.

Candidate results are cached by point. A previously evaluated point normally is not retrained. Nested cooldown searches can require reevaluation if the cached result does not cover the requested cooldown parameters.

### One training step

For each step:

1. Install the candidate Muon LR and momentum.
2. Install the three candidate SGD LRs in their corresponding parameter groups.
3. Read the next batch from the stateful batch stream.
4. Run the model in training mode.
5. Compute mean cross-entropy with label smoothing `0.2`.
6. Backpropagate.
7. Step SGD and then Muon.
8. Clear gradients.

The optimizers are:

- fused Nesterov SGD, momentum `0.85`, for:
  - whitening-layer bias;
  - trainable BatchNorm biases;
  - classifier head;
- Muon for trainable four-dimensional convolution weights.

### Candidate score

The score is CIFAR test-set accuracy with TTA level 2. Despite its name, `tta_val_acc`, this uses the test loader.

For every input batch:

```text
mirror(x) = 0.5 model(x) + 0.5 model(horizontal_flip(x))
```

Then reflect-pad by one pixel and extract:

- the upper-left 32×32 crop;
- the lower-right 32×32 crop.

The final logits are:

```text
0.5 * mirror(original)
+ 0.5 * mean(mirror(upper_left), mirror(lower_right))
```

Inference deliberately calls `model.train()`, so BatchNorm uses and updates training-mode statistics. Snapshot restoration prevents one candidate’s evaluation from contaminating the next candidate.

## 3. Comparing two points

A point is better if its TTA accuracy is higher.

Ties are resolved lexicographically using:

1. smaller total LR-state distance from the initial scale:

   ```python
   sum(abs(state) for each log-LR coordinate)
   ```

   A zero-LR sentinel has infinite distance.

2. lexicographically smaller tuple of actual hyperparameter values;
3. lexicographically smaller tuple of state strings.

Equivalently, minimize:

```python
(
    -tta_accuracy,
    total_log_lr_state_distance,
    actual_hparam_values,
    tuple(map(str, point)),
)
```

Some outer decisions—notably cooldown/main alternation—require a strictly higher accuracy and do not use these tie-breakers.

## 4. Ordinary coordinate search

Starting from a middle point, repeatedly perform complete coordinate sweeps until a sweep accepts no change.

A sweep visits coordinates in this order:

```text
muon_lr
muon_momentum
whiten_bias_lr
bn_bias_lr
head_lr
```

Changes accepted earlier in a sweep become the middle point for later coordinates.

### Searching one coordinate

Generate the two adjacent states where valid:

```python
state - 1
state + 1
```

Sort them by actual hyperparameter value, smallest first. Consequently:

- LR search explores the smaller-LR direction first;
- momentum search explores lower momentum first.

Search the first direction. If it produces an improvement over the middle point, accept it immediately and do not inspect the other direction. Otherwise search the second direction.

### Searching one direction

Given middle point `M` and first neighboring point `P`:

```python
delta = P[index] - M[index]
current = P
parent = M
best = M
went_above_middle = False
small_lr_counter = 0
```

Repeat:

1. Evaluate `current`. When nested cooldown searching is active, `parent` supplies seed states for that candidate’s cooldown search.
2. Replace `best` with `current` if `current` wins under the full comparison rule.
3. If `current` became the best and its accuracy is strictly greater than `M`:
   - set `went_above_middle = True`;
   - reset `small_lr_counter = 0`.
4. Otherwise:
   - remember whether any prior best point was strictly above `M`;
   - if an above-middle point has been found, stop when `current.accuracy < best.accuracy`;
   - if no above-middle point has been found, stop when `current.accuracy < M.accuracy`;
   - non-finite accuracy counts as below;
   - if this direction is toward a smaller LR than `M`, increment `small_lr_counter`, and stop when it reaches 3.
5. Advance one more state in the same direction:

   ```python
   parent = current
   current[index] += delta
   ```

   Momentum stops at the choice-list boundaries.

Return `best` only if at least one evaluated point was strictly more accurate than the middle point. Otherwise return the middle point.

There is no general step limit in the larger-LR direction. It stops only on a score decline, non-finite result, or momentum boundary.

After a direction produces a best point, the normal policy requests one final evaluation of that point without the direction-specific cooldown restriction. This is usually a cache hit. This finalization is disabled when each main candidate performs its own nested cooldown search.

## 5. Special first-interval LR search

The first main search of the first interval treats every log-LR coordinate differently. Momentum still uses ordinary directional search.

For each LR coordinate:

1. Start with the current middle point as a probe.
2. Probe up to 20 consecutive states in the smaller direction.
3. Independently probe up to 20 consecutive states in the larger direction.
4. Maintain the maximum finite accuracy observed across both sides.
5. Stop a direction when:
   - the candidate is invalid/non-finite; or
   - its accuracy is more than `0.02` below the maximum observed so far.
6. Keep every finite probe whose accuracy is within `0.02` of the final maximum.
7. Select the one with the largest actual LR.

The selected point is accepted whenever it differs from the middle point, even if the ordinary tie-breaking comparison would not call it better.

Only one coordinate sweep is performed in this initial-search mode; it is not repeated to convergence.

## 6. Optional full-grid mode

`FULL_GRID_SEARCH` defaults to `False`.

If enabled, ordinary non-initial searches replace directional search for:

- `whiten_bias_lr`
- `bn_bias_lr`
- `head_lr`

For that coordinate, evaluate states:

```python
0, -1, -2, ..., -20
```

with every other coordinate fixed at the current middle point, then select the best under the normal comparison rule.

Because the LR factor is `0.6`, negative states correspond to progressively larger learning rates.

Muon LR and momentum still use directional search.

## 7. Segment search

A generic segment search is:

```python
def search_segment(start_snapshot, initial_point, names, steps):
    cache = previous_candidate_results_if_any
    middle = initial_point
    center_path = [middle]

    evaluate(middle)

    while True:
        accepted = []

        for coordinate in configured_order:
            next_point = search_that_coordinate(middle)

            if better(next_point, middle):
                middle = next_point
                accepted.append(middle)

        center_path.extend(accepted)

        if not accepted:
            break

    best_result = cache[middle]
    restore(start_snapshot)
    return best_result, center_path, all_candidate_results
```

Cooldown searches use this same algorithm, restricted to:

```text
muon_lr
whiten_bias_lr
bn_bias_lr
head_lr
```

Muon momentum is not cooldown-searchable.

## 8. Training intervals and cooldown lookahead

Default configuration:

```text
CIFAR training examples: 50,000
batch size:              2,000
batches per epoch:       25
epochs:                   8
total committed steps:   ceil(8 × 25) = 200
main interval N:          40 steps
cooldown maximum M:       40 steps
```

Therefore there are five 40-step committed intervals. For an interval beginning at global step `s`:

```python
cooldown_steps = min(40, 200 - s - interval_steps)
```

The five cooldown lengths are consequently:

```text
40, 40, 40, 40, 0
```

Cooldown training is lookahead used to score hyperparameters. It is never committed to the real training trajectory.

### First interval

1. Snapshot the interval start.
2. Run the special initial main search for 40 steps per candidate.
3. Use the selected main point as the initial main solution.
4. Enter the main/cooldown alternation described below.

The initial main search scores candidates directly after the main 40 steps, without cooldown.

### Later intervals

When cooldown steps remain, later intervals initially skip main hyperparameter search:

1. Train the inherited main point for 40 steps from the interval snapshot.
2. Score it immediately after those 40 steps.
3. Treat it as the initial selected main solution.
4. Enter cooldown search.

If no cooldown remains, perform an ordinary main search instead.

## 9. Alternating cooldown and main search

Let:

- `selected_main` be the currently chosen main hyperparameters;
- `selected_score` be their current accepted score;
- `cooldown_states` be carried cooldown-search states from the previous interval, if any.

While cooldown steps are available:

### A. Search cooldown hyperparameters

1. Restore the interval-start snapshot.
2. Train `selected_main` for the main interval length.
3. From that main endpoint, search the four cooldown LRs for `cooldown_steps`.
4. Initialize each cooldown LR from:
   - the carried best cooldown state, if present;
   - otherwise the corresponding selected-main LR state.
5. Score at the end of cooldown training.

If cooldown accuracy is not strictly greater than `selected_score`, stop alternating.

Otherwise:

- accept the cooldown score;
- save its best states;
- save its actual LR values as fixed cooldown hyperparameters.

### B. Search main hyperparameters against fixed cooldown

Search all five main parameters from the interval-start snapshot. Each main candidate is scored as follows:

1. Train it for the main interval length.
2. Record its immediate post-main accuracy as `main_tta_val_acc`.
3. Restore/use its exact post-main training state.
4. Train the fixed accepted cooldown LRs for the cooldown length.
5. Use the post-cooldown accuracy as the candidate’s comparison score.

By default, `SEARCH_COOLDOWN_OF_MAIN=False`, so every main candidate receives exactly the same fixed cooldown LR values.

If the new main search’s best score is not strictly greater than the accepted cooldown score, stop. Otherwise accept the new main point and repeat from cooldown search.

Thus the alternation is:

```text
cooldown search
→ if strictly improved: fixed-cooldown main search
→ if strictly improved: cooldown search again
→ ...
```

## 10. Optional per-main-candidate cooldown search

If `SEARCH_COOLDOWN_OF_MAIN=True`, the fixed-cooldown main search instead allows candidate-specific cooldown tuning.

When moving along one main coordinate, only the cooldown-searchable parameters in that coordinate’s group are retuned:

- changing `muon_lr` or `muon_momentum` permits cooldown search of `muon_lr`;
- changing `head_lr` permits cooldown search of `head_lr`;
- changing `whiten_bias_lr` permits cooldown search of `whiten_bias_lr`;
- changing `bn_bias_lr` permits cooldown search of `bn_bias_lr`.

For the initial or unrestricted evaluation of a point, all four cooldown LRs can be searched.

Cooldown initial states are assembled in this priority order:

1. interval-level carried cooldown states;
2. cached cooldown-best states for this same candidate;
3. cooldown-best states from the parent point in the search direction;
4. the candidate’s own main LR state for any still-missing requested parameter.

## 11. Committing an interval

Search evaluations leave no candidate state committed.

After all searching and alternation:

1. Restore the interval-start snapshot.
2. Train exactly `interval_steps` using only the selected main hyperparameters.
3. Record every actual training loss.
4. Keep this resulting model, optimizer, batch-stream, and RNG state as the starting state of the next interval.

No cooldown steps are executed during this committed replay.

The next interval receives:

- the selected main point;
- accepted cooldown states, but only if a cooldown search was successfully accepted.

Training stops early only if the committed replay produces a non-finite loss. Otherwise it continues until all 200 main training steps have been committed.