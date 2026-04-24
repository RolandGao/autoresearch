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


work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax.py.

work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax.py.

the setup is a linear layer then softmax, with cross entropy loss. the input is normalized to norm 1. the weights corresponding to an output neuron is also normalized to norm 1. the weight is optimized using SGD but the update projects the weight to per-output norm 1 again (check train_hyperball.py to get an idea). no weight decay. this is a toy problem. the synthetic ground truth is generated with some ground truth linear layer and softmax_scale with some noise. softmax_scale is defined softmax(softmax_scale * x). softmax_scale is basically 1/temperature. try four softmax setups. 
1. the vanilla softmax, with a fixed softmax_scale. we will ablate on 5 different softmax_scale and see which one gets the best accuracy. one of the 5 should be the ground_truth softmax_scale.
2. softmax with learnable softmax_scale. softmax_scale is its own param_group and optimized with SGD with no weight decay. it does not have the norm projection that the linear layer is subject to. 
try two different setups for this. one is the exp(p) parameterization to ensure softmax_scale is never negative. the other is to clip the softmax_scale back to 1 if it falls below 1. 
3. fixed softcap. check the following code snippet. ablate over different fixed softcaps.
softcap = 15
logits = self.lm_head(x)
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)
4. learnable softcap. 


For all of them, use SGD with no momentum and no weight decay. linear weights are subject to the norm 1 constraint. 

iterate on learnable_softmax_scale_exp. try to use as few samples (batch size * steps) as possible to achieve the target clean RMSE 1.19e-5. 




we have five optimizers.
Adam is one. you can very the betas.
for SGD, we have the norm constrained optimizer on the linear weight, but for the scalar, try four variants, whether to do p.grad = sign(p.grad), whether to do update = sign(momentum). without doing any sign operations, it's the current SGD.

so there are 2 optimizers for the weights and 5 optimizers for the scalars.




try AdamW, vanilla SGD, and SGD with the norm constraint (already implemented), also try Muon (reference implementation in train_baseline.py, remove the NorMuon part). for the SGD with norm constraint, add another variant where the gradient is projected to be perpendicular to the weight before adding to the momentum, and the momentum is projected to be perpendicular to the weight before the momentum is normalized to norm lr and added to the weight. 
for AdamW 


for adam, try many betas. 
for SGD, try four variants, whether to do p.grad = sign(p.grad), whether to do update = sign(momentum). without doing any sign operations, it's the current SGD. try doing the sign operation. 

no weight decay for any of them. momentum can be 0 for SGD.







work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax.py.

first fix the scalar to 10 and train only the linear weight. 
try the following optimizers
AdamW, SGD, and Muon. when using these optimizers, weight decay does not have to be zero. weight is not normalized after every update.
also try the following normalized versions
AdamH, SGDH, and MuonH. for these optimizers, no weight decay, and weight is row-normalized to 1. 
there are two variants, call them H1 and H2.
H1 is as follows.
p = p - lr * u / norm(u)
p = p/norm(p)
H2 is 
u = projected to be perpendicular to p. 
p = p - lr * u / norm(u)
p = p/norm(p)
for SGD, there are more variants
H3 to H10: (g_projection in {True, False}) x (g_norm in {True, False}) x (nesterov in {True, False})
g = g projected to be perpendicular to p
g = g/norm(g)
m = momentum * m + g
m = m projected to be perpendicular to p
if nesterov:
    u = g + momentum * m
else:
    u = m
p = p - lr * u / norm(u)
p = p/norm(p)

norm always means row-wise norm, aka norm of weights per output neuron. 

two adam variants, two muon variants, and 10 sgd variants. 
report the best results for each optimizer variant.
we are aiming for the lowest RMSE loss given fixed num samples (batch size * steps). RMSE has to be at most 1.19e-5. 
write the code that samples 64 runs for each of the 14 optimizers. 
and run the code to report the results back to me



work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax.py.

for each of the 17 optimizers, find the minimum number of samples necessary (batch size * num steps) to reach RMSE at most 1.19e-5. reduce the number of samples necessary by as much as possible. hparams might need tuning as the number of samples are reduced. tune momentums, betas, weight_decays, lr, and lr_scheduler, and sampling_mode. keep ns at 5 for muon. 


 try your hardest for two hours, do not stop before two hours. report back the hparams and results for each of the optimizers at the end, and edit the code such that i can reproduce your results. whenever you find a better setting, write it into the code. I think it's possible to get at 30k num samples or even 20k num samples. You can also play around with the lr scheduler. try exponential decay with 0.01 ** ((t/total_steps)**p) where p is 1.0 or 0.5 and the 0.01 can be something else too. 

speed, and gpu memory


work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax.py.

write code that does full hparam search for AdamH1 and SGDH1. log all the hparams and the clean RMSE, and duration for each individual run. the search space is specified below. 

this hparam search should be the default behavior of running python train_learnable_softmax.py

for the discrete hparam spaces, iterate over all of them, for the lr_decay and lr that have continuous spaces, sample 64 of them. 

there should be 20 * 64 * 2 = 2560 runs total. 

fix num samples at 30000. 
batch_size in {4,8,16,32,64}
steps = 30000/batch_size rounded. 
sample_mode in {shuffle_cycle, fixed_cycle}
lr_schedule 'exp_power'
lr_power in {1.0, 0.5}
lr_decay in uniform(3,5.5)
lr in log_uniform(0.001,0.3)
SGDH1
momentum in {0, 0.5, 0.7, 0.8, 0.9}
AdamH1
hparams["beta1"] = rng.choice((0.0, 0.5, 0.8, 0.9, 0.95))
hparams["beta2"] = rng.choice((0.9, 0.95, 0.99, 0.999))
hparams["eps"] = rng.choice((1e-8, 1e-7, 1e-6))
20 * 64 * 2 = 2560

python train_learnable_softmax.py > softmax3.log 2>&1



work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax.py.

write code that does full hparam search for AdamH1 and SGDH1. log all the hparams and the clean RMSE, and duration for each individual run. the search space is specified below. 

this hparam search should be the default behavior of running python train_learnable_softmax.py

iterate over each batch size. each batch size gets exactly 50 runs.
given the batch size, sample the other hparams as follows. there are some optimizer-specific stuff too. be careful of that. 

there should be 10*2*50=1000 runs


fix num samples at 30000. 
batch_size in {4,8,16,32,64,128,256,512}
steps = 30000/batch_size rounded. 
sample_mode =fixed_cycle
lr_schedule 'exp_power'
lr_power = 1
lr_decay in uniform(3.5,6)
SGDH1
momentum in {0, 0.5, 0.7, 0.8, 0.9}
predicted_lr(batch_size) = 0.001409420528 * batch_size + 0.002388538963
lr in log_uniform(predicted_lr/10, predicted_lr*10)
AdamH1
hparams["beta1"] = rng.choice((0.0, 0.5, 0.8, 0.9, 0.95))
hparams["beta2"] = rng.choice((0.9, 0.95, 0.99, 0.999))
hparams["eps"] = rng.choice((1e-8, 1e-7, 1e-6))
lr(batch_size) = 0.001215494191 * batch_size + 0.009600782619
lr in log_uniform(predicted_lr/10, predicted_lr*10)


work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax2.py.
iterate over the 16 optimizer settings and the 2 batch size settings. the rest of the hparam are randomly sampled 100 times each.

there are 16*2*100=3200 runs total.

SGDH
(g_projection in {True, False}) x (g_norm in {True, False}) x (nesterov in {True, False}) x (m_projection in {True, False})
g = g projected to be perpendicular to p
g = g/norm(g)
m = momentum * m + g
m = m projected to be perpendicular to p
if nesterov:
    u = g + momentum * m
else:
    u = m
p = p - lr * u / norm(u)
p = p/norm(p)

now, do ablation study on the 16 variants of SGDH. 
(g_projection in {True, False}) x (g_norm in {True, False}) x (nesterov in {True, False}) x (m_projection in {True, False})
batch size {32,128}
steps = 30000/batch_size rounded
sample_mode =fixed_cycle
lr_schedule 'exp_power'
lr_power = 1
lr_decay in uniform(3.5,6)
momentum in {0, 0.5, 0.7, 0.8, 0.9}
predicted_lr(batch_size) = 0.001409420528 * batch_size + 0.002388538963
lr in log_uniform(predicted_lr/10, predicted_lr*10)


SGDH_H1:
lr(batch_size) = 0.001409420528 * batch_size + 0.002388538963
R^2 = 0.9951

AdamH_H1:
lr(batch_size) = 0.001215494191 * batch_size + 0.009600782619
R^2 = 0.7929


SGDH_H1:
lr = 0.00384310831113 * batch_size^0.84355406331
R^2 = 0.940723716984

AdamH_H1:
lr = 0.00300750300518 * batch_size^0.864040017614
R^2 = 0.959164237596

SGD
lr = 0.00227815808734 * batch_size - 0.025830735378
R^2 = 0.967007417256

Adam
lr = 0.000916766120626 * batch_size + 0.0371030911324
R^2 = 0.910525505187

python train_learnable_softmax2.py > sgdh_ablation.log 2>&1



batch size tolerance is important

i think the boostrap CI sampling 100 depends on how many samples you have. 

use the best found hparams to train the models again +- CI and then compare.

check the norm implementation


gnorm x momentum has some interactions, given that the update is always normed. 
if momentum > 0, gnorm = False can enable locally relative norm sizes for g. 
gnorm = True would destroy the locally relative information. 
if momentum = 0, gnorm does not matter. 
this momentum > 0 and gnorm = False interaction is similar to adaptive gnorm. perhaps we can decouple adaptive gnorm from momentum. TODO
update norm can be challenged too. perhaps some adaptive thing is better. 

gnorm is false, nesterov is true. it does not matter whether g projection and m projection exist. 
gnorm is OK when batch size is large, but bad when batch size is small. find something that works both when bs is large and small. 



work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_learnable_softmax3.py.



python train_learnable_softmax3.py > sgd_ablations3.log 2>&1

here is SGD with momentum with g norm. the problem with this is that when batch size is large, it works pretty well but when batch size is small, the g norm makes things a lot worse. i think the reason is that the norm of g actually holds information about how hard the sample is and g norm removes that information



ALSO, beta should increase over time, aka ideal batch size should increase over time. we need to capture hard samples in one batch. 
if the model becomes too good, loss too low, then we need a larger batch size to find hard samples. 

Adam is for scalars. ours is for vectors. ours is basically Adam for vectors. 
TODO: check out NorMuon. I think it might be related. 

EMA(norm_value) is more correct than sqrt(EMA(norm_value^2)) cuz we want to imitate an unnormalized layer with a specifically tuned lr.
the purpose of normalizing is so that it's two different layers can use the same lr instead of having a different lr per layer as in the unnormalized case. 

1 x 10 times and 10 x 1 time should be the same as 10 x 10 times and 100 x 1 time. and {2,3,4} should be the same as {3,3,3}. 

denom bias correction makes sense, but numerator bias correction will cause the first few gradients to count for too much. 

how do we know the ideal batch size? the goal is that a batch of the ideal batch size should almost always contain a hard example that gives high loss. 

when batch size large enough, beta2 = 0 and it goes back to u = normalize(u)

the ideal batch size for beta2 might not actually scale with loss. 
the ideal batch size for beta2 is the number samples required to capture the variance of the distribution.
if all the samples are equally easy or equally hard, beta2 is the same for both.
we can try to measure the variance of the gradient norm across batches and use that to calculate beta2. 
beta2 might be proportional to the slope of loss. 

the ideal batch size for beta depends on whether a sample of ideal batch size can capture most of the variation of the distribution of the gradient norm or loss. find a way to do this without extra forward or backward passes



beta2 = max(0.0, 1.0 - (batch_size / ideal_batch_size))

SGD2: p, grad, m, v, beta1, beta2, lr, step

m = beta1 * m + (1-beta1) * g
v = beta2 * v + (1-beta2) * norm(g)
v_hat = v / (1 - beta2 ** step)
if nesterov:
    u = (1-beta1) * g + beta1 * m
else:
    u = m
u = u / v_hat
p = p - lr * u
p = p/norm(p)


work under autoresearch directory. python is /venv/main/bin/python.
work under the hyperball directory and modify only hyperball/train_optimizers.py.

conduct hparam search using the following search space. 
5 optimizers x 3 batchsize x 100 runs = 1500 runs total.

remove old/unused code. 

AdamW, AdamH, Muon, MuonH, SGD
batchsize in {8,64,512}
step_size = 30000/batchsize rounded
predicted_lr = 0.00227815808734 * batch_size - 0.025830735378
lr in log_uniform(predicted_lr/10, predicted_lr*10)
sample_mode =fixed_cycle
lr_schedule 'exp_power'
lr_power = 1
lr_decay in uniform(3.5,6.5)
Muon, MuonH, SGD: 
momentum in uniform(0,1)
AdamW, AdamH. 
hparams["beta1"] = rng.choice((0.0, 0.5, 0.8, 0.9, 0.95))
hparams["beta2"] = rng.choice((0.9, 0.95, 0.99, 0.999))
hparams["eps"] = rng.choice((1e-8, 1e-7, 1e-6))
AdamW, Muon, SGD:
wd in log_uniform(1e-5, 1e-1)



, SGD2


1. yes
2. integer step size
3. the H variants have no wd. their implementation is as follows the non-H variant except that 
p = p - lr * u / norm(u)
p = p/norm(p)
4.


python train_optimizers.py > optimizers_logging.log 2>&1