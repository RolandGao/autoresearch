so there are a few things:

1. instead of computing dx and dw at the same time from dy, do dw first, recalculate dy, and then do dx. 
2. use normal equation for dw
3. use pseudo inverse for dx
4. cross entropy loss needs special handling cuz it's different from MSE. 


modify autoresearch/automatic_lr3/cifar_baseline2_overfit_n_search.py train only on the first batch, with batch size 500, 2000, or 10000. Given hparam N, find the best lr that minimizes the train loss on the first batch after N steps. we do this by starting from some initial_lr, and then probing initial_lr*0.8^k, where k is an integer. it probes one to the left and one to the right. the best lr is found when the two neighbours are both worse than the middle one. cache the results for each k so that we don't reevaluate the same thing twice. make sure that batchnorm is always in train mode, never in eval mode. train for 50 steps, for each of 500,2000,10000. and try N in 1,2,5,10. log the train loss after every step. also log the train loss before the first step. log the lr -> loss loss landscape for the lr search. round the lr to 2 sig figs before applying it.

no, the hparam N means that we break the 50 step training into intervals of N steps, and the lr during each interval is constant. the lr is searched for each interval. the lr search for the current interval can use the best lr for the previous interval. 

lr is always parameterized by initial_lr * 0.8^k rounded to 2 sig figs. so initial_lr and k determine the lr. 

different Ns are different runs with no dependence between them.


modify autoresearch/automatic_lr3/cifar_overfit_n_search_cooldown.py

currently, the lr search is parameterized by lr = 0.2*0.6^k, rounded to 2 sig figs, with some integer k. now we introduce two modes of the search. the "coarse" search is the currently implemented search, the "precise" search is as follows: after finding the best integer k, where k+1 and k-1 both achieve worse loss, we try k+0.5 and k-0.5 to find the best k under this finer grid, after finding the best k in this finder grid, we try k+0.25 and k-0.25 to find the best k under this even finer grid. we stop at 0.25 granularity. 

currently, we have a hparam N, that divides the train steps into intervals of N steps such that the N steps are searched together with a constant lr. now we introduce a new hparam called M. M can be either 1 or 2. after the N steps in the interval, we attach a cooldown of M steps with an independently searched constant lr. the best lr for the N steps interval is not based on the train loss after N steps but instead based on the train loss after the cooldown of M steps. after finding the best lr for the N steps by using the train loss after the M steps cooldown, we discard the cooldown steps and use the checkpoint after the N steps to keep going. for the last N steps interval where having M more steps would go over the total train steps, we won't have the cooldown.

for each fo the N steps interval and the M steps cooldown, we can choose "coarse" or "precise" for the search. instead of having just coarse or precise, use an hparam called final_k_granularity. if this hparam is 1, it's "coarse", and currently precise means this hparam is 0.25. do not use words coarse and precise but instead use final_k_granularity. 

i will introduce a few more hparams. normal_equation_dw. if normal_equation_dw is False, the current behaviour holds. if normal_equation_dw is true, set dw to minimize dy=dw*x, where dy is the output gradient and x is the input and dw is the weight and they do matrix multiplication. use the normal equation with lambda = 0 to solve for dw. this applies to only muon conv weights. 

pseudo_inverse_dx. if pseudo_inverse_dx is False, current behaviour holds. if pseudo_inverse_dx is True, set dx to minimize dy=w*dx. x is solved using pseudo inverse. applies to only muon conv weights. 

norm_inverse. if norm_inverse is False, current behaviour holds. if norm_inverse is True, the BN backwards is modified. originally, forward is (x-mean(x))/(std(x)), and backwards is kind of like (tangent(dy))/std(x). we modify the backwards to (tangent(dy))*std(x) instead. our BN norm group is per-output-channel, so we still do that. 

remove_norm. if remove_norm is False, current behaviour holds. if remove_norm is True, remove BN layers from the model. if remove_norm is True, norm_inverse does not matter.

one step means one forward pass and one backward pass. you can't do this. 

we know we can get to 1.83 in one step, so we can probably save a few steps in the beginning if we are careful. for ReLU backwards, the dx that are zero because x < 0 and dy < 0 should not be treated as true zero target. they should be "ignored" in the backward pass when going through the earlier layers. 




modify only autoresearch/automatic_lr3/cifar_baseline2_backprop4.py

your goal is to achieve a train loss of less than 0.88 in at most 8 steps. 

you can not change the architecture, or the forward pass. you can only change the backward pass. you cannot cheat.

you can try anything within the above constraints, but i think the relu backwards might need to be revised: the x < 0 and dy < 0 elements should be ignored instead of forced to zero for the few prior layers targeting this layer. And the last layer's calculations need to be revised. I was converting the CE loss into an MSE loss but the MSE loss optimal point is different from the CE loss optimal point. and the optimal point is not achievable given our model, so we need a new way to calculate dx,dw,dy of the last layer. also, i noticed that the loss first goes up for a few steps and then goes down to 2.0. we know we can get to 1.83 in one step, so we can probably save a few steps in the beginning if we are careful. 

do not stop until you achieve your goal.
one step means one forward and one backward pass. no secret extra passes. 


modify autoresearch/automatic_lr3/cifar_overfit_search.py

keep orthogonalize = True.

now we do a 2D search over muon lr and muon momentum. remove nesterov. 

m1 = g1
m2 = m*m1 + (1-m)*g2

m_t = m*m_{t-1}+(1-m)*g_t

m is the momentum hparam. g is the raw gradient, m_t is the momentum buffer. 
m goes from 0 to 0.9 with 0.1 granularity, inclusive of 0 and 0.9. 

for (lr,m), the neighbours are the four points, where m is fixed and lr shifts, or lr is fixed and m shifts to m+0.1 and m-0.1. remember that m has boundaries while lr does not. 

we do this 2D search for the main interval, and only search the lr for the cooldown, the cooldown inherits the momentum of the main interval.

modify autoresearch/automatic_lr3/cifar_overfit_search.py

train for 30 steps instead of 50.

have 3 runs, one is the current nesterov_search = True one, where it does a full search for the interval for both nesterov = True and False. have two more runs, where nesterov is fixed to False always in one run, and always True in the other run.

for the momentum hparam space, it goes from 0 to 0.9 in granularity of 0.1, including 0 and 0.9. but we now want to extend it to include 0.95 and 0.99. one step above 0.9 is 0.95, and one step above that is 0.99. the boundary of this space is 0 and 0.99.




for each interval, we currently search for a constant lr. now we change it so that we search for a start_lr and an end_lr for that interval and the lr during the interval is a linear decay from start_lr to end_lr inclusive. if N = 1, only search for start_lr. for (start_lr,end_lr), its neighbours are its eight points where start_lr and or end_lr could be *0.6 or /0.6.
for the first interval, start_lr and end_lr are initialized to 0.2. for later intervals, they are initialized to the end_lr of the previous interval. 
we have an hparam called interval_scheduler. if interval_scheduler = linear, the above algorithm is used. if interval_scheduler = constant, the original algorithm is used. 

during the neighbourhood search, if the (start_lr, end_lr) point has been run before, cache it so as to not duplicate efforts. 

If M > 0, M's start_lr and end_lr are initialized to the main interval's end_lr. If M = 1, only start_lr is used.


write code that plots autoresearch/automatic_lr3/cifar_overfit_search_momentum_ratio.log

the output summary.txt should contain all the ranking metrics in autoresearch/automatic_lr3/cifar_overfit_search_linear_plots/summary.txt

besides that, plot the lr curves, one per subplot. the momentum curves, the loss curves. use linear y scale for lr curves. 

so only a .txt and a .png file

modify autoresearch/automatic_lr3/cifar_overfit_search.py

introduce a new hparam called lr_connectedness. If lr_connectedness is "jump_allowed", then it's the current behaviour. If it's "continuous_double", then the start_lr of the next interval has to be equal to the end_lr of the previous interval. If it's "continuous_single", then the start_lr of the next interval also has to be equal to the end_lr of the previous interval. 

suppose N = 3. 

step 1 to step 3 is a linear curve. if it's jump_allowed, both start_lr and end_lr are searched and step 4 to step 6 use the searched start_lr and end_lr. if it's continuous_double, we search for only end_lr and step 4's lr = step 3's lr and it linear curve down to step 6 end_lr. but if it's continuous_single, we search for only end_lr and step 4 lr != step 3's lr and the curve linear goes down from step 3 to step 6. 

if M > 0 and and it's continuous_*, then it also only searches for end_lr


tsp          # list jobs
tsp -k 0     # terminate running job 0
tsp -r 5


best_hparams=muon_lr=0.016 muon_momentum=0.6 bias_lr=22 head_lr=800 main=0.9382, best_cooldown=0.9382

muon_lr=0.0056 muon_momentum=0.7 bias_lr=13 head_lr=1300 -> tta_val_acc=0.9388


factor setter

the point of the calibration is to find a list of values such that
any local best is the global best.
this list of values should be as precise as possible while still robustly holding the above property. 

for a center point, going 8 steps towards each side should be convex. aka monotonically getting worse. 
if this is false, double the step size. 
if this is true, halve the step size. if halving the step size makes it false, then we are done. 

it's ok if the best is no longer the best after halving the step size. we just recalculate the best after halving the step size. 

we have a new algorithm for calibration.

use N = 5 instead of N = 40.

the lrs probe using their factor = 0.1. they try middle*factor^(k*precision) for k from -4 to 4 inclusive, integer. precision = 1.0 for now. then, we check if tta val acc is monotonically decreasing on each side of the middle. if not, precision = precision * 2, and we try again. if yes, then precision = precision / 2 and we try again. we stop when we have found the boundary where precision is yes and precision/2 is no. 

for momentum, use middle+k*precision, where k is integer, from -4 to 4 inclusive. precision = 0.1 at the start. also do precision * 2 or precision / 2 as above. 

lr has the constraint that lr >= 0. momentum has the constraint that momentum is in [0,1] inclusive. 

if lr or momentum hits the boundary constraint, it's ok for |k| to stop shorter than 4 on that side. if it's monotonically decreasing with |k| smaller than 4, that's ok. 

when doing the 1-step search at the beggining, for momentum, use [0,0.5,0.9]

include zero. 
suppose we accept two peaks, then we always evaluate both peaks. 


Given a 1D function with one large peak and some noise in the output that makes some little local peaks
It is expensive to evaluate the function so we can only try a around 5 points. How do we find a robust peak?

given a 1D function with some noise in the output. how to find a robust peak?
what if it's expensive to evaluate the function f and we only can only spend time evaluating 5 points
explain how to develop the acquisition function. and how to develop the surrogate model
what if i know there's only one big peak and if i go near the boundary there's definitely no peak there


idea: just train bias_lr and head_lr and see what we get. 


autoresearch/automatic_lr4/20260630_230733_608720/cifar_search_baseline.log
autoresearch/automatic_lr4/20260702_164537_312299/cifar_search_1init.log
autoresearch/automatic_lr4/20260702_171353_931352/cifar_search_1init.log
autoresearch/automatic_lr4/20260706_174252_402210/cifar_search_1init.log
autoresearch/automatic_lr4/20260706_183636_919627/cifar_search_1init_higher_muon_lr_precision.log

python plot_cifar_search.py 20260706_183636_919627/cifar_search_1init_higher_muon_lr_precision.log
python plot_cifar_search.py 20260702_171353_931352/cifar_search_1init.log

factor=0.6 is pretty good. 

Search main interval. Search cooldown interval. Then search main interval while fixing the cooldown interval. 
main interval is more expensive than the cooldown. so find the best cooldown early on.
we need to start from arbitrary values and still be able to converge. 

main:
interval=0 muon_lr=0.2 momentum=0.6 bias_lr=62 head_lr=800 path_final_tta=0.9136
interval=1 muon_lr=0.12 momentum=0.6 bias_lr=37 head_lr=800 path_final_tta=0.931
interval=2 muon_lr=0.043 momentum=0.8 bias_lr=37 head_lr=800 path_final_tta=0.9379
interval=3 muon_lr=0.026 momentum=0.8 bias_lr=37 head_lr=800 path_final_tta=0.9412
interval=4 muon_lr=0.002 momentum=0.8 bias_lr=37 head_lr=1300 path_final_tta=0.9404

cooldown:
interval=0 muon_lr=0.043 momentum=0.6 bias_lr=8.1 head_lr=480
interval=1 muon_lr=0.026 momentum=0.6 bias_lr=22 head_lr=8.1
interval=2 muon_lr=0.0093 momentum=0.8 bias_lr=0.38 head_lr=8.1
interval=3 muon_lr=0.0056 momentum=0.8 bias_lr=2.9 head_lr=1.8
interval=4 muon_lr=0.002 momentum=0.8 bias_lr=37 head_lr=1300

interval=0 phase=main     start_step=0 steps=40 muon_lr=0.2 momentum=0.6 bias_lr=62 head_lr=800 path_final_tta=0.9136 loss=2.31->1.317
interval=1 phase=main     start_step=40 steps=40 muon_lr=0.12 momentum=0.6 bias_lr=37 head_lr=800 path_final_tta=0.931 loss=1.297->1.134
interval=2 phase=main     start_step=80 steps=40 muon_lr=0.043 momentum=0.8 bias_lr=37 head_lr=800 path_final_tta=0.9386 loss=1.154->1.033
interval=3 phase=main     start_step=120 steps=40 muon_lr=0.016 momentum=0.9 bias_lr=37 head_lr=800 path_final_tta=0.9425 loss=1.046->0.9611
interval=4 phase=main     start_step=160 steps=40 muon_lr=0.0056 momentum=0.9 bias_lr=4.9 head_lr=2200 path_final_tta=0.9425 loss=0.954->0.928
interval=0 phase=cooldown best_cooldown=0.9136 cooldown_muon_lr=0.043 cooldown_momentum=0.6 cooldown_bias_lr=8.1 cooldown_head_lr=480
interval=1 phase=cooldown best_cooldown=0.931 cooldown_muon_lr=0.026 cooldown_momentum=0.6 cooldown_bias_lr=22 cooldown_head_lr=8.1
interval=2 phase=cooldown best_cooldown=0.9386 cooldown_muon_lr=0.0093 cooldown_momentum=0.8 cooldown_bias_lr=4.9 cooldown_head_lr=3700
interval=3 phase=cooldown best_cooldown=0.9425 cooldown_muon_lr=0.0056 cooldown_momentum=0.9 cooldown_bias_lr=4.9 cooldown_head_lr=2200
interval=4 phase=cooldown best_cooldown=0.9425 cooldown_muon_lr=0.0056 cooldown_momentum=0.9 cooldown_bias_lr=4.9 cooldown_head_lr=2200

We are working in /workspace/neural_networks_optimization. Use /venv/main/bin/python.

Modify autoresearch/automatic_lr4/search_toy.py.

Goal: improve black-box peak finding under a strict 20-evaluation budget.

Important constraint:
The search algorithm must NOT use or fit against the summary curve/table directly, except through normal black-box calls to evaluate(x). Future curves may look different. Assume only a broad structural prior: in log(x)-space, the denoised function usually has one or two large peaks plus noise.

Current context:
- x spans many orders of magnitude, so search in log-space.
- Must use at most 20 evaluations per function.
- The existing robust log_coverage_sweep uses 122 evals and clears 0.99, but is over budget.
- Existing 20-eval methods miss:
  - coarse_to_fine_log stress min around 0.982
  - log_space_gp_ucb stress min around 0.979
  - smooth_interval_ucb improves stress to around min 0.985, mean 0.9945, below_0.99 around 73/1000, but still misses.
- smooth_interval_ucb currently uses 8 initial log-space points and then bisects high-scoring intervals. It does not explore below the smallest initial probe or above the largest initial probe unless changed.

Optimization target:
1. Accuracy = found_f / actual_max.
2. Try to exceed 0.99 on:
   - the fixed original + 10 transformed functions
   - fresh random stress tests
3. Keep and report fresh random stress:
   - min_accuracy
   - mean_accuracy
   - below_0.99 count
   - max_evaluations
4. Be honest if no curve-agnostic 20-eval method clears the target.

Do not overfit to the current 11 fixed functions. Do not use the parsed summary values as a template/prior. Acceptable priors include generic assumptions like smoothness in log-space, one/two broad peaks, heteroscedastic noise, boundary-safe exploration, multi-start interval refinement, or curve-agnostic Bayesian optimization.

Please:
- Inspect search_toy.py first.
- Implement one or more improved curve-agnostic algorithms, or improve existing algorithms
- Include them in the output comparison table against existing algorithms.
- Run:
  /venv/main/bin/python autoresearch/automatic_lr4/search_toy.py
- Report exact results and whether the 20-eval target is actually cleared.
- work for at least 30 minutes on this problem


modify autoresearch/automatic_lr4/cifar_search2.py

remove TTA_VAL_ACC_DIFF_THRESHOLD = 0.0005 and the component of the algorithm that uses TTA_VAL_ACC_DIFF_THRESHOLD

add a param called full_grid_search. if full_grid_search is false, current behaviour is ok. if full_grid_search is true, when searching for bias_lr and head_lr, do a full grid search of 1*0.6^k for k from 0 to -20 and pick the best one.

muon_lr can be denser. 