
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


batch_size=8, num_samples=2048
  AdamW_row__beta1=0.9__flush_last=True: best_sse=3.060151563e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.108723822e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=3.109285972e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=3.138195804e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=3.143969583e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=3.144036509e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=3.155800802e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=3.157521763e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=3.167512538e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=3.167640525e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=3.175041635e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=3.197498542e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=3.197498542e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=3.297232045e-08

batch_size=8, num_samples=4096
  AdamW_row__beta1=0.95__flush_last=True: best_sse=1.579697603e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=1.619579528e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.638807638e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.656245616e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.691474551e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.698368699e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.71331484e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.71694306e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.724976961e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.727121213e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.732186499e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.733528743e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.747464748e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.747464748e-08

batch_size=8, num_samples=8192
  AdamW_row__beta1=0.95__flush_last=True: best_sse=9.384847105e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=9.484579808e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.005060944e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.009776224e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.013426767e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.013426767e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.014021967e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.014167713e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.014473304e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.014880656e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.015301773e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.016231434e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.017180507e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.019433154e-08

batch_size=8, num_samples=16384
  AdamW_row__beta1=0__flush_last=False: best_sse=6.313209875e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.313209875e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=6.3239366e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=6.325860007e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=6.327572843e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=6.330335426e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=6.333154321e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=6.337075797e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=6.343287691e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=6.349085917e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=6.36775635e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=6.379895854e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=6.407285802e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=6.436248388e-09

batch_size=8, num_samples=32768
  AdamW_row__beta1=0.9__flush_last=True: best_sse=4.347600035e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=4.347926426e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.34870102e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.349435229e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.350140542e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=4.350721887e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.351063618e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.351465122e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.351800578e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.352051455e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.352707943e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=4.353390121e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.353390121e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.36279898e-09

batch_size=16, num_samples=2048
  AdamW_row__beta1=0.95__flush_last=True: best_sse=3.15728048e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=3.274648013e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=3.274648013e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=3.288189194e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=3.298003194e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=3.307509736e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=3.31820335e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=3.329276184e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=3.370248484e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.374817073e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=3.470086843e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=3.495613964e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=3.733649022e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=3.980284606e-08

batch_size=16, num_samples=4096
  AdamW_row__beta1=0__flush_last=False: best_sse=1.660546203e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.660546203e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.662592231e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.663945298e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.666630327e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.666758006e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.670417355e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.674589112e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.677392249e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.693823287e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.716786874e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.764409126e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=1.937154996e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=2.062091611e-08

batch_size=16, num_samples=8192
  AdamW_row__beta1=0.8__flush_last=True: best_sse=9.274894698e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=9.28865796e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=9.29171315e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=9.300745144e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=9.310012986e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=9.311478895e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=9.313105284e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=9.317018635e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=9.320817981e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=9.335374943e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=9.335374943e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=9.407594947e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=9.530432256e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=9.792117028e-09

batch_size=16, num_samples=16384
  AdamW_row__beta1=0.95__flush_last=True: best_sse=5.972444205e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=5.9834176e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=6.007606862e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=6.009675265e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=6.019664935e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=6.021418786e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=6.027165068e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=6.028324682e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=6.029065268e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=6.032072649e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=6.03319326e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=6.03572582e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=6.04607607e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.04607607e-09

batch_size=16, num_samples=32768
  AdamW_row__beta1=0.95__flush_last=True: best_sse=4.1977743e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.218996378e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=4.250460089e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.259996998e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=4.287394468e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.291490529e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.3016339e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.304332299e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.309322564e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.311446852e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.314305496e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.316051521e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=4.326454825e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.326454825e-09

batch_size=32, num_samples=2048
  AdamW_row__beta1=0.8__flush_last=True: best_sse=2.942891e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=3.087034729e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=3.214413521e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=3.30125009e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=3.391889592e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=3.417461858e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=3.475609616e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=3.563103916e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=3.602868783e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=3.665707979e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=3.665707979e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=3.957769483e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=7.486176267e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=1.09367347e-07

batch_size=32, num_samples=4096
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.443792257e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.521244752e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.830246179e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.830246179e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.842899346e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.849908467e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.854467178e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.854676417e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.862300887e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.867762921e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.890619825e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.890709676e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=2.094343187e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=2.546059313e-08

batch_size=32, num_samples=8192
  AdamW_row__beta1=0.95__flush_last=True: best_sse=7.115171078e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=7.512190632e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=1.007460971e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.007460971e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.009864318e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.011550889e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.011887255e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.01474028e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.015262265e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.021471647e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.021838521e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.034682797e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.044882541e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.075976634e-08

batch_size=32, num_samples=16384
  AdamW_row__beta1=0__flush_last=False: best_sse=6.272094727e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.272094727e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=6.281099218e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=6.285576107e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=6.285743151e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=6.292688515e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=6.29311966e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=6.304903726e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=6.309571707e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=6.332061889e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=6.369830173e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=6.430273243e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=6.511646264e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=6.660859791e-09

batch_size=32, num_samples=32768
  AdamW_row__beta1=0__flush_last=False: best_sse=4.264027264e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.264027264e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.265701902e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.266501435e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.267454081e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.268089106e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.269234375e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=4.27193191e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.272837418e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.281377672e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=4.28664874e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.311207348e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=4.322685906e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.375492002e-09

batch_size=64, num_samples=2048
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.100338102e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=3.247465465e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=3.319983938e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=3.352490221e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=3.442387824e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=3.449489353e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=3.451628033e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=3.454923108e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=3.476205893e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=3.476205893e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=4.223591195e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=6.649476139e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=6.825689476e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=1.836851465e-07

batch_size=64, num_samples=4096
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.550930459e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.592599098e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.666643126e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.675518539e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.712838336e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.714082252e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.742734942e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.74747125e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.766974389e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.834702786e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.844352913e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.844352913e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=3.400583003e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=6.381698792e-08

batch_size=64, num_samples=8192
  AdamW_row__beta1=0.9__flush_last=True: best_sse=7.887087516e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=8.033580555e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=8.306021997e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=9.282847774e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=9.485255221e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=9.944373717e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.006589015e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.034382059e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=1.038152319e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.04219778e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.061033045e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.066127023e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.122536942e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.122536942e-08

batch_size=64, num_samples=16384
  AdamW_row__beta1=0.95__flush_last=True: best_sse=4.470599382e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.655648051e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=5.823625456e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=5.932897087e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=6.819829468e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=6.87042002e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.87042002e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=6.876766292e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=6.884372632e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=6.889440139e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=6.892594896e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=6.903608945e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=6.907746751e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=6.930335144e-09

batch_size=64, num_samples=32768
  AdamW_row__beta1=0.95__flush_last=True: best_sse=3.928956539e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=3.99904441e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=4.519952768e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.519952768e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.524768641e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.52765089e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.527970653e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.532947689e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.533951873e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=4.544979705e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.544996947e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.56912378e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=4.583556476e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.641421663e-09

batch_size=128, num_samples=2048
  AdamW_row__beta1=0.9__flush_last=True: best_sse=3.798099697e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.806198301e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=3.81100951e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=3.81100951e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.152979272e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.240750595e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.302456091e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.400191525e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.488664499e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.651048154e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.757247716e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=6.733564411e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=8.15815593e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=1.667033272e-07

batch_size=128, num_samples=4096
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.619121476e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.840809368e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.840809368e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.847646216e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.878499829e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.886449342e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.892424213e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.931466331e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.95051501e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=2.037020137e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=2.050088897e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=2.222712668e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=2.627153575e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=1.004109804e-07

batch_size=128, num_samples=8192
  AdamW_row__beta1=0.9__flush_last=True: best_sse=9.249312643e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=9.282674735e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=9.604142482e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=9.91438695e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=9.945984417e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.002444352e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.010023153e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.011293039e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.013311136e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.017908462e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=1.034236257e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.034236257e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.056588044e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=2.398261466e-08

batch_size=128, num_samples=16384
  AdamW_row__beta1=0.95__flush_last=False: best_sse=5.646635301e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=5.851681703e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=6.245181559e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=6.34755639e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=6.351901622e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=6.407244746e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=6.452785419e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=6.485701761e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=6.521083568e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=6.529184376e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=6.549190258e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=6.723383603e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.723383603e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=6.809943701e-09

batch_size=128, num_samples=32768
  AdamW_row__beta1=0.9__flush_last=True: best_sse=4.248412984e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.337928241e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=4.389599068e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.432774161e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.497455711e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.523075957e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.563011506e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.578907177e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.606337863e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.61626286e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=4.702553758e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.702553758e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=4.736723326e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.980706315e-09

batch_size=256, num_samples=2048
  AdamW_row__beta1=0.9__flush_last=False: best_sse=6.87504859e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=7.479071974e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=8.388282143e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=9.533118173e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=9.609283419e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=9.704996179e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.009175136e-07
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.039583925e-07
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.081339816e-07
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.087425416e-07
  AdamW_row__beta1=0__flush_last=False: best_sse=1.115440899e-07
  AdamW_row__beta1=0__flush_last=True: best_sse=1.115440899e-07
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.412928226e-07
  AdamW_row__beta1=0.95__flush_last=True: best_sse=6.962282797e-07

batch_size=256, num_samples=4096
  AdamW_row__beta1=0__flush_last=False: best_sse=2.027347574e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=2.027347574e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=2.427922236e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=2.569858495e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=2.728995723e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=2.787866976e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.026551661e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=3.055015065e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=3.093506609e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=3.142867554e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=3.422182908e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=3.618182221e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=8.724828452e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=9.134834424e-08

batch_size=256, num_samples=8192
  AdamW_row__beta1=0__flush_last=False: best_sse=1.079518004e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.079518004e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.10360292e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.126329903e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.149302229e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.152030178e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=1.18758741e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.227552993e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=1.279231965e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=1.287321267e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=1.356042347e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=1.419047843e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=1.469663575e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=5.529371617e-08

batch_size=256, num_samples=16384
  AdamW_row__beta1=0.5__flush_last=True: best_sse=6.661084293e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=6.673668471e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=6.696244373e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=6.736655799e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=6.74584525e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.74584525e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=6.783803936e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=6.902243763e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=7.382989512e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=7.431640643e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=7.742507538e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=8.096771724e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=8.287648939e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=9.493423795e-09

batch_size=256, num_samples=32768
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.502885076e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.521475914e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=4.525423644e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.541939446e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.554069941e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.554774637e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.563778729e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.597921115e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=4.606931376e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.606931376e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=4.612216018e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=5.351217726e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=5.406726086e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=5.677250677e-09

batch_size=512, num_samples=2048
  AdamW_row__beta1=0__flush_last=False: best_sse=5.519895688e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=5.519895688e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.107199223e-07
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.553081596e-07
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.64842933e-07
  AdamW_row__beta1=0.7__flush_last=True: best_sse=2.213787018e-07
  AdamW_row__beta1=0.6__flush_last=False: best_sse=2.378004271e-07
  AdamW_row__beta1=0.7__flush_last=False: best_sse=2.850859712e-07
  AdamW_row__beta1=0.8__flush_last=False: best_sse=3.06471508e-07
  AdamW_row__beta1=0.9__flush_last=False: best_sse=3.563035073e-07
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.59820925e-07
  AdamW_row__beta1=0.95__flush_last=False: best_sse=3.92567945e-07
  AdamW_row__beta1=0.95__flush_last=True: best_sse=4.328918677e-07
  AdamW_row__beta1=0.9__flush_last=True: best_sse=9.492708195e-07

batch_size=512, num_samples=4096
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.171244949e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=4.417668309e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=6.402622863e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=8.129537728e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=8.55006702e-08
  AdamW_row__beta1=0__flush_last=False: best_sse=8.89894276e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=8.89894276e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=8.965125702e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=9.004634655e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=9.005021857e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=9.314299641e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=9.950797218e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.476795179e-07
  AdamW_row__beta1=0.95__flush_last=True: best_sse=7.65396836e-07

batch_size=512, num_samples=8192
  AdamW_row__beta1=0__flush_last=False: best_sse=1.190105295e-08
  AdamW_row__beta1=0__flush_last=True: best_sse=1.190105295e-08
  AdamW_row__beta1=0.5__flush_last=True: best_sse=1.485434917e-08
  AdamW_row__beta1=0.5__flush_last=False: best_sse=1.573132457e-08
  AdamW_row__beta1=0.6__flush_last=True: best_sse=1.921027892e-08
  AdamW_row__beta1=0.9__flush_last=False: best_sse=1.966894224e-08
  AdamW_row__beta1=0.6__flush_last=False: best_sse=2.150985509e-08
  AdamW_row__beta1=0.8__flush_last=False: best_sse=2.599371761e-08
  AdamW_row__beta1=0.7__flush_last=True: best_sse=2.630330494e-08
  AdamW_row__beta1=0.7__flush_last=False: best_sse=2.960002348e-08
  AdamW_row__beta1=0.8__flush_last=True: best_sse=3.086386993e-08
  AdamW_row__beta1=0.9__flush_last=True: best_sse=3.727797781e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=5.736373751e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=1.156970745e-07

batch_size=512, num_samples=16384
  AdamW_row__beta1=0__flush_last=False: best_sse=6.981085304e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=6.981085304e-09
  AdamW_row__beta1=0.5__flush_last=True: best_sse=7.194139865e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=7.273140651e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=7.570361322e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=7.753412721e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=8.535186112e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=8.923180637e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=9.073074231e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=9.517825909e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=9.913452247e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=1.501924468e-08
  AdamW_row__beta1=0.95__flush_last=True: best_sse=1.674720036e-08
  AdamW_row__beta1=0.95__flush_last=False: best_sse=3.613875864e-08

batch_size=512, num_samples=32768
  AdamW_row__beta1=0.5__flush_last=True: best_sse=4.676768602e-09
  AdamW_row__beta1=0.6__flush_last=True: best_sse=4.694724627e-09
  AdamW_row__beta1=0.5__flush_last=False: best_sse=4.703763284e-09
  AdamW_row__beta1=0__flush_last=False: best_sse=4.705418326e-09
  AdamW_row__beta1=0__flush_last=True: best_sse=4.705418326e-09
  AdamW_row__beta1=0.6__flush_last=False: best_sse=4.737831121e-09
  AdamW_row__beta1=0.7__flush_last=True: best_sse=4.776666488e-09
  AdamW_row__beta1=0.7__flush_last=False: best_sse=4.852727723e-09
  AdamW_row__beta1=0.9__flush_last=False: best_sse=4.998751812e-09
  AdamW_row__beta1=0.8__flush_last=True: best_sse=5.296652432e-09
  AdamW_row__beta1=0.8__flush_last=False: best_sse=5.532381582e-09
  AdamW_row__beta1=0.9__flush_last=True: best_sse=6.414421644e-09
  AdamW_row__beta1=0.95__flush_last=False: best_sse=8.322691489e-09
  AdamW_row__beta1=0.95__flush_last=True: best_sse=1.236318343e-08
