run | batch_size | update | train25_loss | val_acc | tta_val_acc | seconds
--- | ---: | --- | ---: | ---: | ---: | ---:
0 | 125 | row norm | 0.9347 | 0.9316 | 0.9392 | 24.96
1 | 125 | Newton-Schulz | 0.9160 | 0.9296 | 0.9384 | 24.43
2 | 500 | row norm | 0.9333 | 0.9296 | 0.9361 | 10.80
3 | 500 | Newton-Schulz | 0.9079 | 0.9302 | 0.9374 | 11.31
4 | 2000 | row norm | 1.1223 | 0.8551 | 0.8640 | 10.19
5 | 2000 | Newton-Schulz | 0.8986 | 0.9304 | 0.9379 | 10.35

Best TTA:
run=0 batch_size=125 update=row norm tta_val_acc=0.9392
