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

original train has linear weight decay

algorithm description
i train a neural network. 
it chooses initial lr from (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0). train each of them for 20 steps, and then finds the one with the lowest avg train loss over the last 3 steps. uses that checkpoint to continue. from that lr, it creates three child runs with (0.8, 1.0, 1.2) * lr, trains each of them for 20 steps, picks the best one, and then creates three child runs from that. I call this algorithm beam search with k = 1. 

If I just run this algorithm, the lr decreases too quickly and reaches a local minimum. so I added a 200 step cooldown from each checkpoint so that the searched lr does not have to do the cooldown itself and can focus on generalization while the cooldown will focus on getting to the local minimum to have a fair comparison. cuz without the cooldown, high lr gets punished. the future search continues from the 20 step checkpoint, and the cooldowned checkpoint is discarded. After i did this, lr becomes too high and the final loss is also not good. 

cooldown formula
0.1 ** ((x/170) ** 0.42) where x is the step count starting from the cooldown. 

without cooldown 
lrs = [1.75, 1.05, 1.05, 1.05, 0.84, 0.672, 0.5376, 0.5376, 0.5376, 0.43008, 0.43008, 0.344064, 0.344064, 0.344064, 0.275251, 0.220201, 0.176161, 0.176161, 0.140929, 0.140929, 0.140929, 0.112743, 0.112743, 0.0901943, 0.0901943, 0.0901943, 0.0901943, 0.0721555, 0.0721555,...]
loss = 3.278825

with cooldown
[2, 2.4, 2.88, 3.456, 4.1472, 4.97664, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197, 5.97197]
loss = 2.999505

a baseline run with constant lr for 0.3 of the run and then linear decay to 0:
loss: 2.922183

idea: reduce cooldown steps
20 search steps + 20 cooldown steps
20 search steps + 40 cooldown steps
40 search steps + 40 cooldown steps

idea: search from a baseline
actual_lr = global_lr(step) * searched_multiplier
searched_multiplier in [0.8, 1.0, 1.2]

idea: every 50 steps, evaluate on the whole train set. 


change the cooldown lr to linear candidate_lr_mult * (1-x/cooldown_steps)
during cooldown, fix the momentum and weight decay to constants instead of whatever schedule they were on. 

run four experiments. 
1. BEAM_SEGMENT_STEPS = 20 and BEAM_COOLDOWN_STEPS = 20
2. BEAM_SEGMENT_STEPS = 20 and BEAM_COOLDOWN_STEPS = 10

3. BEAM_SEGMENT_STEPS = 50 and BEAM_COOLDOWN_STEPS = 50
4. BEAM_SEGMENT_STEPS = 50 and BEAM_COOLDOWN_STEPS = 25

write the code, i will run the experiments
modify only train_scheduler3.py

train with a constant lr with 1.0 or 5.0 for 1600 steps and checkpoint at 50, 200 and 1600, including the optimizer states. 
then we experiment with the cooldown to see how to cooldown the fastest. 
with fewer steps, the cooldown of 5.0 should be better. if 1600 steps, the cooldown of 1.0 should be better. 


remove weight decay 
smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
train loss is smoothedå


work under autoresearch directory. python is /venv/main/bin/python. work under the beam_search directory. modify only train_scheduler2.py.
segment_steps = 50. try cooldown with 25, 50 steps. for cooldown, try both linear decay and the exponential formula 0.1 ** (x/cooldown_steps) where x is the step count starting from the cooldown, starting from 1. have 1350 total steps, where 1350-cooldown_steps steps are the searched steps and the last cooldown steps are the cooldown. evaluate val_bpb after 1350 steps. when cooldown_steps = 25, the last searched steps is 75 steps instead of 50 to align up to the 1350 target.

 for each round of segment of 20 steps, add a 200 step lr_mult cooldown after the 20 step of fixed lr. the cooldown formula is 0.1 ** ((x/170) ** 0.42) where x is the step count starting from the cooldown. the avg3 train loss is calculated at the end of the cooldown, but the checkpoint saved should be the one after the 20 step fixed lr and before the cooldown. the checkpoint after cooldown is always discarded. have the beam search for 1150 steps so as to save 200 steps for the cooldown. 

ask questions before you implement

loss plateau


work under autoresearch directory. python is /venv/main/bin/python. work under the hyperball directory. only modify hyperball/train.py

add logging for weight norms, gradient norms. 
wte
lm_head
q,k,v
attn.c_proj
mlp.c_fc
mlp_c_proj
resid_lambdas
x0_lambdas
ve
attn.ve_gate

for a matrix A of size N x M, log the Frobenius norm of M divided by sqrt(max(N,M)). 
for a scalar s, just log the abs(s)
resid_lambdas and x0_lambdas are scalars. the others are all matrices. 

after log the L2 norm of the activations after each block (x = block(x, ve, cos_sin, self.window_sizes[i])). 

gating is important

for N x M matrix, 
if N < M, each N-dim vector should be norm 1
If N > M, each M-dim vector should be norm 1
each min(N,M) dim vector should be norm 1. 

gating params as norm scalars. e^p > 0. 

0.8**(1.2*x**0.42)

beam_search_with_cooldown.log might actually be good if we allow it to fullly cooldown over 800 steps to 0.01 lr. 

how to properly cooldown. 
absorption of data. 

neural network is dumb without enough data. 
linear search on one batch to find the lowest loss achievable in one step. that might be the cooldown? if the model has potential, then if can achieve a lot in one step. but if the model has no potential, then it can't achieve much in one step. put the trust in the river valley??s


> more_logging.log 2>&1

from train.py, add update norms. the update tensor differs from the gradient tensor in that the gradient tensor is the raw gradient whiel the update tensor is the tensor that actually gets put into the weight

For Adam, p.add_(exp_avg / denom, alpha=-step_size), update is exp_avg / denom * step_size. 

For Muon, 
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
update is lr * g. 
ignore the weight decay for the update tensor. i think one way to log these update tensors is to set p.grad = update_tensor and then check p.grad, before zeroing out p.grad again for the next step.

train2.py: migration to hyperball
train3.py: pop off


we either do per vector or per matrix norm. 

let's do per matrix norm first. 
cautious wd decays based on the updated param, traditional weight decay decays based on the old param. 

work under autoresearch directory. python is /venv/main/bin/python.
work under hyperball, and only modify visualize_weight_decay.py. 
suppose the weight vector is a 3D vector constrained to unit norm.
the update vector is also unit norm. 
we have one toggle to toggle the angle between the update vector and the weight vector. 
visualize the standard weight decay vector and the catious weight decay vector. 
cautious weight decay applies a mask (update * weight) < 0 and decays only the weights that satisfy this mask. note that this is the same as (g * weight) >= 0 mask cuz update = -g. 
standard weight decay does not have this mask




work under autoresearch directory. python is /venv/main/bin/python.
work under hyperball, and only modify precision.py.  


if i have a D dim vector and a D x D matrix. both the vector and the matrix have frobenius norm 1.  and i apply the matrix onto the vector.

try out all possible precision settings available on A100, and see the rounding error of the output when compared to the ground truth (aka the highest precision setting).
vary D and see how the rounding error changes. 
write the code, i will run it. 
running it should produce all the results in the output console.

python precision.py --dims 10000,50000 --trials 10 > precision2.log

python precision.py \
  --dims 10000,50000 \
  --matrix-norms 1,224,1000 \
  --vector-norms 1 \
  --trials 5 > precision3.log

normalization operation dtype
matrix and vector dtype
vary dim D and norm C. 
speed
relative error
compile

work under autoresearch directory. python is /venv/main/bin/python.
work under hyperball. modify only precision.py

if i have a D dim vector and a D x D matrix. 
the vector has norm A and the matrix has frobenius norm B, and I apply the matrix onto the vector. Then, the output vector is normalized to norm A. 
conduct an ablation study on the effects of dtype on the gpu memory, speed, and the relative error in the output compared to the ground truth.
try out all dtypes available on A100. reat fp32 highest precision as ground truth, no fp64. 
measure the relative L2 error and the speed. 
try out torch.compile. 
also try out varying the dtype of the normalization operation of the output_vector. 
write the code in precision.py, run the code, and report the results back to me. iterate on the code so that the result is imformative. 

you can ask me questions first before starting. 



both the vector and the matrix have frobenius norm 1.  and i apply the matrix onto the vector.

work under autoresearch directory. python is /venv/main/bin/python.
work under hyperball. do not modify any file. 


work under autoresearch directory. python is /venv/main/bin/python.
work under hyperball, and only modify train2.py.  Modify only GPT's init_weights. init according to the weight norms in weight_norm.py
The numbers in weight_norm.py are Frobenius norm divided by sqrt(max(N,M)) for matrices. convert them back to actual Frobenius norm. 
all the matrices should be initialized using a normal distribution instead of uniform or whatever. keep the init of learnable scalars unchanged
do not change this part of the code
norm_scheme = 'matrix'
assert norm_scheme in {'matrix','per_output','per_input',"per_smaller_vector"}
for w, norm_value in init_scheme:
    if norm_scheme == 'matrix':
        w.div_(w.norm()).mul_(norm_value)
        continue
    output_dim, input_dim = w.shape
    if norm_scheme == 'per_output' or (norm_scheme == 'per_smaller_vector' and output_dim >= input_dim):
        output_dim, input_dim = w.shape
        norm_value = norm_value / (output_dim ** 0.5)
        w.div_(w.norm(dim=1,keepdim=True)).mul_(norm_value)
    else:
        output_dim, input_dim = w.shape
        norm_value = norm_value / (input_dim ** 0.5)
        w.div_(w.norm(dim=0,keepdim=True)).mul_(norm_value)


work under autoresearch directory. python is /venv/main/bin/python. work under the hyperball directory. only modify hyperball/train2.py
we want more activation norm loggings
for MLP, log after c_fc, and after c_proj.
For attention, log before and after this line (v = v + gate.unsqueeze(-1) * ve) and also ve. log before and after c_proj(y). 
for lm_head, log before and after lm_head. 

do not calculate the norm over all tokens and emd_dim, only over embed_dim, and average over the tokens. 

v = v + gate.unsqueeze(-1) * ve


x = self.c_fc(x)
x = F.relu(x).square()
x = self.c_proj(x)

python train2.py > hyperball.log 2>&1


0.0006
0.06

ideas for the discrepancy: zero init, cautious wd. 
zero init is the most likely: loss curve was worse from the beginning.
try using hyperball for most weights but fallback on some weights to debug, such as the zero init weights. 
the ratio between cautious wd vector and the update vector is a problem




python train.py > hyperball_baseline2.log 2>&1
python train2.py > hyperball_better_scheduler2.log 2>&1
python train3.py > ablation.log 2>&1

ratio of activations


attn.c_proj 1.04517236479753
attn.ve_gate 1.0402585179821096
k 1.040956598836486
lm_head 1.042297123275748
mlp.c_fc 1.0398597169480592
mlp.c_proj 1.040865771345476
q 1.0396111926645204
v 1.0411021275055712
ve 1.0390418701725663
wte 1.0412029697902851


x = x + self.attn(
    norm(x), ve, cos_sin, window_size, activation_norms, prefix
)
x = x + self.mlp(norm(x), activation_norms, prefix)

for each of the two lines, calculate the norm of x and the norm of the output of the attn or mlp and divide by the sum of the norm so they sum to 1. this is to understand how much the residual path contributes to the final output. add the logging to both train files and also update visualize.py



softcap = 15
logits = self.lm_head(x)
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)
