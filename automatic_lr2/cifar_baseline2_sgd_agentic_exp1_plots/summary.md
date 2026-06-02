rank | search | update | initial_lr | best_k | best_lr | val_acc | tta_val_acc | train25_loss | evals
---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:
1 | 18 | Newton-Schulz steps=4 | 0.22 | 0 | 0.22 | 0.9310 | 0.9420 | 0.9022 | 3
2 | 19 | Newton-Schulz centered | 0.22 | 1 | 0.18 | 0.9215 | 0.9338 | 0.9165 | 4
3 | 16 | QR row orthogonal | 0.22 | 1 | 0.18 | 0.9226 | 0.9325 | 0.9102 | 4
4 | 17 | Newton-Schulz steps=1 | 0.22 | -2 | 0.34 | 0.9091 | 0.9174 | 0.9838 | 5
5 | 15 | row inv-sqrt weighted | 0.061 | -2 | 0.095 | 0.9068 | 0.9166 | 0.9828 | 5
6 | 12 | factorized RMS | 0.061 | 0 | 0.061 | 0.9038 | 0.9163 | 0.9841 | 3
7 | 3 | sinkhorn columns first | 0.061 | 0 | 0.061 | 0.9047 | 0.9157 | 0.9831 | 3
8 | 2 | sinkhorn rows first | 0.061 | -1 | 0.076 | 0.9029 | 0.9148 | 0.9749 | 3
9 | 10 | softsign | 0.061 | 0 | 0.061 | 0.9022 | 0.9143 | 0.9915 | 3
10 | 0 | column norm | 0.061 | -1 | 0.076 | 0.9036 | 0.9138 | 0.9852 | 4
11 | 7 | signed sqrt | 0.061 | 0 | 0.061 | 0.9048 | 0.9129 | 0.9933 | 3
12 | 8 | signed cuberoot | 0.061 | -1 | 0.076 | 0.9005 | 0.9129 | 0.9877 | 4
13 | 14 | row sqrt weighted | 0.061 | 0 | 0.061 | 0.9036 | 0.9128 | 0.9983 | 3
14 | 11 | tanh | 0.061 | 0 | 0.061 | 0.9023 | 0.9124 | 0.9918 | 3
15 | 1 | column norm max | 0.061 | 4 | 0.025 | 0.8999 | 0.9113 | 0.9857 | 7
16 | 4 | row centered rows | 0.061 | 0 | 0.061 | 0.8935 | 0.9013 | 1.0518 | 3
17 | 5 | column centered columns | 0.061 | 1 | 0.049 | 0.8908 | 0.9009 | 1.0470 | 4
18 | 13 | inverse factorized RMS | 0.061 | 1 | 0.049 | 0.8863 | 0.8971 | 1.0623 | 4
19 | 9 | signed square | 0.061 | 0 | 0.061 | 0.8848 | 0.8953 | 1.0644 | 3
20 | 6 | double centered matrix | 0.061 | 2 | 0.039 | 0.8780 | 0.8852 | 1.1035 | 5

Best overall:
search=18 update=Newton-Schulz steps=4 lr=0.22 tta_val_acc=0.9420
