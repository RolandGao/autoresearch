search | batch_size | update | initial_lr | best_k | best_lr | val_acc | tta_val_acc | evals
---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---:
0 | 125 | row norm | 0.04 | 0 | 0.04 | 0.9316 | 0.9392 | 7
1 | 125 | Newton-Schulz | 0.04 | -2 | 0.062 | 0.9284 | 0.9406 | 9
2 | 500 | row norm | 0.079 | -0.25 | 0.084 | 0.9267 | 0.9387 | 7
3 | 500 | Newton-Schulz | 0.079 | -1 | 0.099 | 0.9317 | 0.9404 | 8
4 | 2000 | row norm | 0.095 | 2 | 0.061 | 0.9020 | 0.9151 | 9
5 | 2000 | Newton-Schulz | 0.19 | -0.75 | 0.22 | 0.9310 | 0.9409 | 8

Best overall:
search=5 batch_size=2000 update=Newton-Schulz k=-0.75 lr=0.22 tta_val_acc=0.9409
