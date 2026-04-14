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