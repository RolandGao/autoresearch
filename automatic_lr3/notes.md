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