so there are a few things:

1. instead of computing dx and dw at the same time from dy, do dw first, recalculate dy, and then do dx. 
2. use normal equation for dw
3. use pseudo inverse for dx
4. cross entropy loss needs special handling cuz it's different from MSE. 


modify autoresearch/automatic_lr3/cifar_baseline2_overfit_n_search.py train only on the first batch, with batch size 500, 2000, or 10000. Given hparam N, find the best lr that minimizes the train loss on the first batch after N steps. we do this by starting from some initial_lr, and then probing initial_lr*0.8^k, where k is an integer. it probes one to the left and one to the right. the best lr is found when the two neighbours are both worse than the middle one. cache the results for each k so that we don't reevaluate the same thing twice. make sure that batchnorm is always in train mode, never in eval mode. train for 50 steps, for each of 500,2000,10000. and try N in 1,2,5,10. log the train loss after every step. also log the train loss before the first step. log the lr -> loss loss landscape for the lr search. round the lr to 2 sig figs before applying it.

no, the hparam N means that we break the 50 step training into intervals of N steps, and the lr during each interval is constant. the lr is searched for each interval. the lr search for the current interval can use the best lr for the previous interval. 

lr is always parameterized by initial_lr * 0.8^k rounded to 2 sig figs. so initial_lr and k determine the lr. 

different Ns are different runs with no dependence between them.