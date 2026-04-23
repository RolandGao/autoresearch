
Overall ranking across batch sizes
rank | mean_rank | worst_rank | wins | mean_best_rmse | batch_8 | batch_32 | batch_128 | batch_512 | variant
   1 |      4.00 |          7 |    0 | 1.010311411e-05 | 1.009771107e-05 | 9.991401686e-06 | 9.955245141e-06 | 1.036809854e-05 | g_projection=T,g_norm=F,nesterov=T,m_projection=T
   2 |      4.75 |          8 |    1 | 1.006102437e-05 | 9.928222936e-06 | 1.004874434e-05 | 9.983339951e-06 | 1.028379027e-05 | g_projection=F,g_norm=F,nesterov=T,m_projection=T
   3 |      5.00 |          6 |    0 | 1.010031695e-05 | 1.010374136e-05 | 9.991899422e-06 | 9.977930807e-06 | 1.032769622e-05 | g_projection=F,g_norm=F,nesterov=T,m_projection=F
   4 |      6.00 |          8 |    0 | 1.017156988e-05 | 1.040628016e-05 | 9.960510712e-06 | 9.977976236e-06 | 1.034151242e-05 | g_projection=T,g_norm=F,nesterov=T,m_projection=F
   5 |      6.25 |         16 |    0 | 1.01300285e-05 | 1.013206872e-05 | 9.956468603e-06 | 1.014954307e-05 | 1.02820336e-05 | g_projection=F,g_norm=F,nesterov=F,m_projection=F
   6 |      6.50 |         10 |    1 | 1.020005886e-05 | 1.034340268e-05 | 1.009420295e-05 | 9.947427553e-06 | 1.041520225e-05 | g_projection=T,g_norm=F,nesterov=F,m_projection=F
   7 |      6.75 |         15 |    1 | 1.022039233e-05 | 1.027193751e-05 | 9.937950885e-06 | 9.96310351e-06 | 1.070857743e-05 | g_projection=F,g_norm=F,nesterov=F,m_projection=T
   8 |      8.25 |         12 |    0 | 1.08223694e-05 | 1.242141315e-05 | 1.051772331e-05 | 9.962393615e-06 | 1.038794753e-05 | g_projection=T,g_norm=T,nesterov=T,m_projection=T
   9 |      9.00 |         16 |    0 | 1.029704153e-05 | 1.013055267e-05 | 9.986095177e-06 | 1.005682215e-05 | 1.101469613e-05 | g_projection=T,g_norm=F,nesterov=F,m_projection=T
  10 |      9.25 |         14 |    0 | 1.083671986e-05 | 1.237760227e-05 | 1.057828113e-05 | 9.959235276e-06 | 1.043176077e-05 | g_projection=T,g_norm=T,nesterov=F,m_projection=F
  11 |      9.75 |         15 |    1 | 1.082680531e-05 | 1.24350045e-05 | 1.052951635e-05 | 1.008014068e-05 | 1.026255971e-05 | g_projection=F,g_norm=T,nesterov=T,m_projection=T
  12 |     10.25 |         16 |    0 | 1.088310549e-05 | 1.265180718e-05 | 1.054976956e-05 | 1.000680249e-05 | 1.032404272e-05 | g_projection=F,g_norm=T,nesterov=T,m_projection=F
  13 |     11.50 |         15 |    0 | 1.085987969e-05 | 1.240088201e-05 | 1.061107941e-05 | 1.003611029e-05 | 1.039144706e-05 | g_projection=T,g_norm=T,nesterov=F,m_projection=T
  14 |     12.00 |         13 |    0 | 1.08613466e-05 | 1.242309933e-05 | 1.056218032e-05 | 1.002182493e-05 | 1.043828181e-05 | g_projection=F,g_norm=T,nesterov=F,m_projection=T
  15 |     12.50 |         15 |    0 | 1.092983562e-05 | 1.239118921e-05 | 1.054431751e-05 | 1.009596692e-05 | 1.068786881e-05 | g_projection=T,g_norm=T,nesterov=T,m_projection=F
  16 |     14.25 |         16 |    0 | 1.089458637e-05 | 1.242604504e-05 | 1.06246456e-05 | 1.008900783e-05 | 1.0438647e-05 | g_projection=F,g_norm=T,nesterov=F,m_projection=F


SUMMARY g_projection=F,g_norm=F,nesterov=F,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.010735961e-05 +- 7.639759473e-08 final_train_loss=9.157161694e-05 +- 3.111242077e-07
SUMMARY   batch_size=32: clean_rmse=1.009972671e-05 +- 6.603043536e-08 final_train_loss=9.14039003e-05 +- 1.565356614e-07
SUMMARY   batch_size=128: clean_rmse=1.000618761e-05 +- 5.009553771e-08 final_train_loss=9.130411781e-05 +- 5.61724416e-08
SUMMARY   batch_size=512: clean_rmse=1.072995212e-05 +- 2.746757127e-08 final_train_loss=9.143646021e-05 +- 6.674004335e-08
SUMMARY g_projection=F,g_norm=F,nesterov=F,m_projection=T
SUMMARY   batch_size=8: clean_rmse=1.041758254e-05 +- 7.171060927e-08 final_train_loss=9.16240504e-05 +- 3.186397777e-07
SUMMARY   batch_size=32: clean_rmse=1.00715909e-05 +- 7.648328376e-08 final_train_loss=9.140104376e-05 +- 1.562345468e-07
SUMMARY   batch_size=128: clean_rmse=1.000095224e-05 +- 5.100470395e-08 final_train_loss=9.130478429e-05 +- 5.596941945e-08
SUMMARY   batch_size=512: clean_rmse=1.12577165e-05 +- 2.513166762e-08 final_train_loss=9.150123806e-05 +- 6.826623438e-08
SUMMARY g_projection=F,g_norm=F,nesterov=T,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.011743176e-05 +- 7.729540925e-08 final_train_loss=9.15830431e-05 +- 3.13383685e-07
SUMMARY   batch_size=32: clean_rmse=1.007668748e-05 +- 7.154223715e-08 final_train_loss=9.139412286e-05 +- 1.573220765e-07
SUMMARY   batch_size=128: clean_rmse=1.018225673e-05 +- 5.568345374e-08 final_train_loss=9.132036066e-05 +- 5.866674997e-08
SUMMARY   batch_size=512: clean_rmse=1.053129964e-05 +- 2.87082193e-08 final_train_loss=9.141566843e-05 +- 6.624330156e-08
SUMMARY g_projection=F,g_norm=F,nesterov=T,m_projection=T
SUMMARY   batch_size=8: clean_rmse=9.955978924e-06 +- 6.463776166e-08 final_train_loss=9.15790195e-05 +- 3.081350221e-07
SUMMARY   batch_size=32: clean_rmse=1.023484105e-05 +- 9.453747548e-08 final_train_loss=9.13981552e-05 +- 1.551916206e-07
SUMMARY   batch_size=128: clean_rmse=1.005055427e-05 +- 5.668315935e-08 final_train_loss=9.130458202e-05 +- 5.76385972e-08
SUMMARY   batch_size=512: clean_rmse=1.068410411e-05 +- 3.667393127e-08 final_train_loss=9.14328426e-05 +- 6.775828209e-08
SUMMARY g_projection=F,g_norm=T,nesterov=F,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.255277971e-05 +- 1.541904148e-07 final_train_loss=9.181971691e-05 +- 3.052970177e-07
SUMMARY   batch_size=32: clean_rmse=1.425853555e-05 +- 1.178883284e-07 final_train_loss=9.19532642e-05 +- 1.386557358e-07
SUMMARY   batch_size=128: clean_rmse=1.020958887e-05 +- 5.083754672e-08 final_train_loss=9.132236737e-05 +- 5.670002529e-08
SUMMARY   batch_size=512: clean_rmse=1.052804304e-05 +- 5.700501793e-08 final_train_loss=9.141397022e-05 +- 6.826395131e-08
SUMMARY g_projection=F,g_norm=T,nesterov=F,m_projection=T
SUMMARY   batch_size=8: clean_rmse=1.28084755e-05 +- 2.256178469e-07 final_train_loss=9.179347835e-05 +- 3.153210432e-07
SUMMARY   batch_size=32: clean_rmse=1.084782916e-05 +- 1.25805106e-07 final_train_loss=9.143030911e-05 +- 1.518376381e-07
SUMMARY   batch_size=128: clean_rmse=1.015814526e-05 +- 5.451710894e-08 final_train_loss=9.131639672e-05 +- 5.650292513e-08
SUMMARY   batch_size=512: clean_rmse=1.061353569e-05 +- 3.796009125e-08 final_train_loss=9.142347553e-05 +- 6.693623601e-08
SUMMARY g_projection=F,g_norm=T,nesterov=T,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.267012453e-05 +- 1.357169238e-07 final_train_loss=9.183124494e-05 +- 3.085816841e-07
SUMMARY   batch_size=32: clean_rmse=1.067534512e-05 +- 1.296959892e-07 final_train_loss=9.141687042e-05 +- 1.54443951e-07
SUMMARY   batch_size=128: clean_rmse=1.006100451e-05 +- 5.420330354e-08 final_train_loss=9.130439284e-05 +- 5.883991918e-08
SUMMARY   batch_size=512: clean_rmse=1.08382598e-05 +- 2.098855695e-07 final_train_loss=9.145022486e-05 +- 8.838927963e-08
SUMMARY g_projection=F,g_norm=T,nesterov=T,m_projection=T
SUMMARY   batch_size=8: clean_rmse=1.264858356e-05 +- 1.140813425e-07 final_train_loss=9.184575611e-05 +- 3.041514263e-07
SUMMARY   batch_size=32: clean_rmse=1.064969206e-05 +- 1.292207978e-07 final_train_loss=9.141202318e-05 +- 1.543653726e-07
SUMMARY   batch_size=128: clean_rmse=1.029574386e-05 +- 5.429657407e-08 final_train_loss=9.133152926e-05 +- 5.614757477e-08
SUMMARY   batch_size=512: clean_rmse=1.037550382e-05 +- 3.600503303e-08 final_train_loss=9.139683971e-05 +- 6.560194017e-08
SUMMARY g_projection=T,g_norm=F,nesterov=F,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.02746386e-05 +- 8.580084564e-08 final_train_loss=9.15867713e-05 +- 3.17487951e-07
SUMMARY   batch_size=32: clean_rmse=1.018404532e-05 +- 9.320588588e-08 final_train_loss=9.138856549e-05 +- 1.555596494e-07
SUMMARY   batch_size=128: clean_rmse=1.006295732e-05 +- 4.606697489e-08 final_train_loss=9.131644183e-05 +- 5.426151929e-08
SUMMARY   batch_size=512: clean_rmse=1.098327482e-05 +- 3.995541217e-08 final_train_loss=9.146195953e-05 +- 6.803885356e-08
SUMMARY g_projection=T,g_norm=F,nesterov=F,m_projection=T
SUMMARY   batch_size=8: clean_rmse=9.975267237e-06 +- 6.494040798e-08 final_train_loss=9.157213499e-05 +- 3.079491031e-07
SUMMARY   batch_size=32: clean_rmse=1.025016242e-05 +- 9.5684794e-08 final_train_loss=9.140140173e-05 +- 1.551232637e-07
SUMMARY   batch_size=128: clean_rmse=1.006287448e-05 +- 4.783427791e-08 final_train_loss=9.131451807e-05 +- 5.54938588e-08
SUMMARY   batch_size=512: clean_rmse=1.099398512e-05 +- 2.283907688e-08 final_train_loss=9.147144301e-05 +- 6.668573147e-08
SUMMARY g_projection=T,g_norm=F,nesterov=T,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.049174536e-05 +- 9.856947034e-08 final_train_loss=9.158231987e-05 +- 3.135441739e-07
SUMMARY   batch_size=32: clean_rmse=1.144570644e-05 +- 1.049334832e-07 final_train_loss=9.156947199e-05 +- 1.54491211e-07
SUMMARY   batch_size=128: clean_rmse=1.05600148e-05 +- 5.239218138e-08 final_train_loss=9.136649169e-05 +- 5.683933941e-08
SUMMARY   batch_size=512: clean_rmse=1.055753922e-05 +- 4.164790923e-08 final_train_loss=9.141523333e-05 +- 6.785757892e-08
SUMMARY g_projection=T,g_norm=F,nesterov=T,m_projection=T
SUMMARY   batch_size=8: clean_rmse=1.049819577e-05 +- 1.028613735e-07 final_train_loss=9.157085733e-05 +- 3.084989243e-07
SUMMARY   batch_size=32: clean_rmse=1.029383797e-05 +- 6.166934169e-08 final_train_loss=9.142934578e-05 +- 1.496353617e-07
SUMMARY   batch_size=128: clean_rmse=1.00962881e-05 +- 4.966411892e-08 final_train_loss=9.131459956e-05 +- 5.555609191e-08
SUMMARY   batch_size=512: clean_rmse=1.065671224e-05 +- 3.526376485e-08 final_train_loss=9.142725612e-05 +- 6.646068158e-08
SUMMARY g_projection=T,g_norm=T,nesterov=F,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.278334859e-05 +- 2.167183322e-07 final_train_loss=9.179317858e-05 +- 3.166734241e-07
SUMMARY   batch_size=32: clean_rmse=1.090184025e-05 +- 1.186969093e-07 final_train_loss=9.144654759e-05 +- 1.491053115e-07
SUMMARY   batch_size=128: clean_rmse=1.024042739e-05 +- 5.274636132e-08 final_train_loss=9.132526757e-05 +- 5.666245911e-08
SUMMARY   batch_size=512: clean_rmse=1.047294441e-05 +- 5.139358419e-08 final_train_loss=9.140609473e-05 +- 6.712730021e-08
SUMMARY g_projection=T,g_norm=T,nesterov=F,m_projection=T
SUMMARY   batch_size=8: clean_rmse=1.284446863e-05 +- 2.126148545e-07 final_train_loss=9.179279004e-05 +- 3.156012152e-07
SUMMARY   batch_size=32: clean_rmse=1.070597555e-05 +- 1.302647449e-07 final_train_loss=9.141110058e-05 +- 1.549449634e-07
SUMMARY   batch_size=128: clean_rmse=1.018616039e-05 +- 5.293618974e-08 final_train_loss=9.132001142e-05 +- 5.631357432e-08
SUMMARY   batch_size=512: clean_rmse=1.095917034e-05 +- 3.663150007e-08 final_train_loss=9.146522207e-05 +- 6.745111915e-08
SUMMARY g_projection=T,g_norm=T,nesterov=T,m_projection=F
SUMMARY   batch_size=8: clean_rmse=1.249649153e-05 +- 1.396927643e-07 final_train_loss=9.181722562e-05 +- 3.134966302e-07
SUMMARY   batch_size=32: clean_rmse=1.386569242e-05 +- 1.170353646e-07 final_train_loss=9.188418626e-05 +- 1.414092638e-07
SUMMARY   batch_size=128: clean_rmse=1.00668139e-05 +- 5.689715654e-08 final_train_loss=9.130702092e-05 +- 5.774281917e-08
SUMMARY   batch_size=512: clean_rmse=1.068033698e-05 +- 4.036713461e-08 final_train_loss=9.143021016e-05 +- 6.81351414e-08
SUMMARY g_projection=T,g_norm=T,nesterov=T,m_projection=T
SUMMARY   batch_size=8: clean_rmse=1.249968913e-05 +- 1.356514858e-07 final_train_loss=9.181563073e-05 +- 3.141698476e-07
SUMMARY   batch_size=32: clean_rmse=1.064557563e-05 +- 1.261732021e-07 final_train_loss=9.141217743e-05 +- 1.538487827e-07
SUMMARY   batch_size=128: clean_rmse=1.015931739e-05 +- 5.400608358e-08 final_train_loss=9.131753322e-05 +- 5.656916891e-08
SUMMARY   batch_size=512: clean_rmse=1.048205652e-05 +- 3.878506525e-08 final_train_loss=9.140727198e-05 +- 6.72967986e-08


Here’s the table with each optimizer’s `clean_rmse` at all four batch sizes, its rank within each batch size, and `worst rank = max(r@8, r@32, r@128, r@512)`. The overall row order is still by mean `clean_rmse` across the four batch sizes.

| Rank | Optimizer | bs=8 | r@8 | bs=32 | r@32 | bs=128 | r@128 | bs=512 | r@512 | Worst rank | Mean clean_rmse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `g_projection=F, g_norm=F, nesterov=T, m_projection=F` | `1.011743176e-05` | 4 | `1.007668748e-05` | 2 | `1.018225673e-05` | 11 | `1.053129964e-05` | 5 | 11 | `1.022691890250e-05` |

| 2 | `g_projection=F, g_norm=F, nesterov=T, m_projection=T` | `9.955978924e-06` | 1 | `1.023484105e-05` | 5 | `1.005055427e-05` | 3 | `1.068410411e-05` | 10 | 10 | `1.023136958850e-05` |
| 3 | `g_projection=F, g_norm=F, nesterov=F, m_projection=F` | `1.010735961e-05` | 3 | `1.009972671e-05` | 3 | `1.000618761e-05` | 2 | `1.072995212e-05` | 11 | 11 | `1.023580651250e-05` |
| 4 | `g_projection=T, g_norm=F, nesterov=F, m_projection=T` | `9.975267237e-06` | 2 | `1.025016242e-05` | 6 | `1.006287448e-05` | 5 | `1.099398512e-05` | 15 | 15 | `1.032057231425e-05` |
| 5 | `g_projection=T, g_norm=F, nesterov=F, m_projection=F` | `1.027463860e-05` | 5 | `1.018404532e-05` | 4 | `1.006295732e-05` | 6 | `1.098327482e-05` | 14 | 14 | `1.037622901500e-05` |


| 6 | `g_projection=T, g_norm=F, nesterov=T, m_projection=T` | `1.049819577e-05` | 8 | `1.029383797e-05` | 7 | `1.009628810e-05` | 8 | `1.065671224e-05` | 8 | 8 | `1.038625852000e-05` |
| 7 | `g_projection=F, g_norm=F, nesterov=F, m_projection=T` | `1.041758254e-05` | 6 | `1.007159090e-05` | 1 | `1.000095224e-05` | 1 | `1.125771650e-05` | 16 | 16 | `1.043696054500e-05` |

| 8 | `g_projection=T, g_norm=F, nesterov=T, m_projection=F` | `1.049174536e-05` | 7 | `1.144570644e-05` | 14 | `1.056001480e-05` | 16 | `1.055753922e-05` | 6 | 16 | `1.076375145500e-05` |

| 9 | `g_projection=T, g_norm=T, nesterov=T, m_projection=T` | `1.249968913e-05` | 10 | `1.064557563e-05` | 8 | `1.015931739e-05` | 10 | `1.048205652e-05` | 3 | 10 | `1.094665966750e-05` |
| 10 | `g_projection=F, g_norm=T, nesterov=T, m_projection=T` | `1.264858356e-05` | 12 | `1.064969206e-05` | 9 | `1.029574386e-05` | 15 | `1.037550382e-05` | 1 | 15 | `1.099238082500e-05` |
| 11 | `g_projection=F, g_norm=T, nesterov=T, m_projection=F` | `1.267012453e-05` | 13 | `1.067534512e-05` | 10 | `1.006100451e-05` | 4 | `1.083825980e-05` | 12 | 13 | `1.106118349000e-05` |
| 12 | `g_projection=T, g_norm=T, nesterov=F, m_projection=F` | `1.278334859e-05` | 14 | `1.090184025e-05` | 13 | `1.024042739e-05` | 14 | `1.047294441e-05` | 2 | 14 | `1.109964016000e-05` |
| 13 | `g_projection=F, g_norm=T, nesterov=F, m_projection=T` | `1.280847550e-05` | 15 | `1.084782916e-05` | 12 | `1.015814526e-05` | 9 | `1.061353569e-05` | 7 | 15 | `1.110699640250e-05` |
| 14 | `g_projection=T, g_norm=T, nesterov=F, m_projection=T` | `1.284446863e-05` | 16 | `1.070597555e-05` | 11 | `1.018616039e-05` | 12 | `1.095917034e-05` | 13 | 16 | `1.117394372750e-05` |
| 15 | `g_projection=T, g_norm=T, nesterov=T, m_projection=F` | `1.249649153e-05` | 9 | `1.386569242e-05` | 15 | `1.006681390e-05` | 7 | `1.068033698e-05` | 9 | 15 | `1.177733370750e-05` |
| 16 | `g_projection=F, g_norm=T, nesterov=F, m_projection=F` | `1.255277971e-05` | 11 | `1.425853555e-05` | 16 | `1.020958887e-05` | 13 | `1.052804304e-05` | 4 | 16 | `1.188723679250e-05` |

A quick read: the most robust by this `worst rank` column is `g_projection=T, g_norm=F, nesterov=T, m_projection=T`, whose worst rank is only `8`.

| 1 | `g_projection=F, g_norm=F, nesterov=T, m_projection=F` | `1.011743176e-05` | 4 | `1.007668748e-05` | 2 | `1.018225673e-05` | 11 | `1.053129964e-05` | 5 | 11 | `1.022691890250e-05` |
| 10 | `g_projection=F, g_norm=T, nesterov=T, m_projection=T` | `1.264858356e-05` | 12 | `1.064969206e-05` | 9 | `1.029574386e-05` | 15 | `1.037550382e-05` | 1 | 15 | `1.099238082500e-05` |
| 3 | `g_projection=F, g_norm=F, nesterov=F, m_projection=F` | `1.010735961e-05` | 3 | `1.009972671e-05` | 3 | `1.000618761e-05` | 2 | `1.072995212e-05` | 11 | 11 | `1.023580651250e-05` |
| 16 | `g_projection=F, g_norm=T, nesterov=F, m_projection=F` | `1.255277971e-05` | 11 | `1.425853555e-05` | 16 | `1.020958887e-05` | 13 | `1.052804304e-05` | 4 | 16 | `1.188723679250e-05` |
