work under autoresearch directory. python is /venv/main/bin/python.
do some basic hyperparameter tuning. every run should only change one hyperparam from the previous best hyperparam setup. use the hparam search space below. try out the values for one hparam, find the best one, lock in that one, and then search for the next hparam. 
warmdown_ratio: 0.5, 0.7, 0.9
warmup_ratio: 0, 0.05, 0.1
total_batch_size: 2**18, 2**17, 2**16
lr_multiplier (this one will multiply each of the four lr hparams by the same constant): 0.5,1,1.5
weight_decay: 0.1,0.2,0.3
do not actually train yourself, modify train2.py so that i can run the code to do the hparam search. 

keep the existing logging, but add a second logging that logs to a file, give a few word description for each run and the stats for that run. 


work under autoresearch directory. python is /venv/main/bin/python.
fix batch size at 2**17, warmup at 0. 
try the following
linear lr decay x warmdown (0.7) x lr_mult (0.5,1.0,1.5) x wd (0.1,0.2,0.3)
exponential lr decay with end_ratio (0.01) x warmdown (1.0, aka warmdown immediately) x lr_mult (1.0,2.0,3.0) x wd(0.1,0.2,0.3)

exponential lr decay formula is end_ratio**progress. the below is some reference code. 

p = cur_step / max_steps
if p < warmup_threshold:
    return p / warmup_threshold
if p < cooldown_threshold:
    return 1
cooldown_p = (p - cooldown_threshold) / (1 - cooldown_threshold)
if schedule_name == "linear":
    return 1 - cooldown_p * (1 - end_ratio)
if schedule_name == "exponential":
    return end_ratio**cooldown_p

modify train2.py for this, and log to hparam_serach2.log this time. this time, it's no longer "factor" search but "product" search. under each of linear and exponential, try all the combinations. 

for linear lr, search lr (1.5,2.0,2.5) x (0.025,0.05,0.1)
for exp scheduler, search end_ratio (0.01,0.03,0.1) x lr_mult (1.0, 3.0) x wd (0.1, 0.3)


Architecture:


work under autoresearch directory. python is /venv/main/bin/python.
modify only train3.py. train.py is our baseline. someone online has discoverfed the following improvements on H100. We want to see how much of these changes can improve on our A100. Modify train3.py such that when i run train3.py, it will run each of the changes one by one and see which ones worked which ones did not. write the report to hparam_search4.log. be careful of some combo effects like the first two, or the wd and the lr. improved_train.py contains the reference implementation of all the changes below. 

Depth 8 → 9, aspect ratio tuned to keep dim=512
Window pattern SSSL → SSSSL (4:1 short:long ratio)

Short window seq_len/2 → seq_len/8
RoPE base 10K → 200K
Embedding LR 0.6 → 0.9 (effective with WD)
Unembedding LR 0.004 → 0.005
Warmdown ratio 0.5 → 0.75
Final LR fraction 0.0 → 0.05
Muon momentum warmup 300 → 200 steps
Init & regularization:

Transformer init scale ×0.68 (narrow optimum: 0.66 and 0.70 both worse)
x0 skip scalar init 0.1 → 0.05
Weight decay added to lm_head (0.01), embeddings (0.001), value embeddings (0.003)

the following changes worked. put them as default into train.py
final_lr_frac: 0->0.05
lm_head_wd: 0->0.01, embedding_wd: 0->0.001, value_embedding_wd: 0->0.003
short_window_div: 2->8

150 runs. 

std for one run. 

cnn: fov up, speed up, precision down
transformer: fov up, speed down, precision same

rebase train3.py on the recent changes in train.py and have new experiments 
[]

10 runs different seed
mean val_bpb:   1.040451
stddev:         0.001608
min:            1.037853
max:            1.043623
range:          0.005770

same_seed, seed 42 x10:
  mean:   1.039560
  stdev:  0.000165
  min:    1.039284
  max:    1.039785
  range:  0.000501

seed_sweep, seeds 42..51:
  mean:   1.040167
  stdev:  0.001511
  min:    1.037529
  max:    1.043113
  range:  0.005584


work under autoresearch directory. python is /venv/main/bin/python.
modify only train_scheduler.py
implement a hparam beam search scheduler. let's take lr for example.
beam search takes input k.
at the start, we try out many different lrs {0.5,0.75,1.0,1.25,1.5,1.75,2.0}
we train from the randomly initialized checkpoint on each of them for 50 steps, use the avg of the last 3 steps as the avg_loss metric.
find the k lrs that achieve the lowest avg_loss. for each of the k checkpoints, try out 0.8*lr, 1.0*lr, 1.2*lr for 50 steps to get 3*k checkpoints. find the k checkpoints that achieve the lowest avg_loss. repeat this process, until we get to 1350 steps.
we can't put all checkpoints in memory because memory is limited, so we have to save them to files. but we only need to keep k checkpoints at any given time. we can always delete the worst checkpoint to make room for the new checkpoint such that we keep at most k checkpoints at any given time.
the point of this is to find a good lr scheduler.
common values of k that i plan to try are 1 and 3. 
modify only train_scheduler.py
for lr, we are actually talking about LR_MULT in the codebase.
ask questions before you implement. 



> beam_search2.log 2>&1

pip install kernels pyarrow requests rustbpe tiktoken matplotlib

TODO: cooldown period for beam search might be necessary.