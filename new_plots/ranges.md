# Adaptive grid LR ranges (contiguous TTA ≥ 93% around each momentum's best; BS 2000 baseline, seed 0)

## Head SGD
start LR 1300 / momentum 0.85; tested LR 1–46000; best 94.20% at LR 6000, momentum 0.7
- momentum 0: LR 2.8–28000 (1e+04x)
- momentum 0.5: LR 13–28000 (2.15e+03x)
- momentum 0.7: LR 13–28000 (2.15e+03x)
- momentum 0.8: LR 1.7–28000 (1.65e+04x)
- momentum 0.9: LR 2.8–28000 (1e+04x)

## Head Lion
start LR 0.22 / momentum 0.9; tested LR 0.00017–13; best 94.18% at LR 0.079, momentum 0.5
- momentum 0: LR 0.0037–7.9 (2.14e+03x)
- momentum 0.5: LR 0.0037–7.9 (2.14e+03x)
- momentum 0.7: LR 0.0022–4.7 (2.14e+03x)
- momentum 0.8: LR 0.0062–4.7 (758x)
- momentum 0.9: LR 0.00048–4.7 (9.79e+03x)

## Head input-conditioned
start LR 10000 / momentum 0.8; tested LR 36–130000; best 94.16% at LR 10000, momentum 0.8
- momentum 0: LR 280–77000 (275x)
- momentum 0.5: LR 780–77000 (98.7x)
- momentum 0.7: LR 780–77000 (98.7x)
- momentum 0.8: LR 780–77000 (98.7x)
- momentum 0.9: LR 470–77000 (164x)

## Conv Muon2
start LR 0.22 / momentum 0.7; tested LR 0.048–1; best 94.04% at LR 0.22, momentum 0.5
- momentum 0: LR 0.13–0.61 (4.69x)
- momentum 0.5: LR 0.13–0.61 (4.69x)
- momentum 0.7: LR 0.079–0.61 (7.72x)
- momentum 0.8: LR 0.079–0.37 (4.68x)
- momentum 0.9: LR 0.079–0.22 (2.78x)

## Norm bias SGD
start LR 99 / momentum 0.8; tested LR 0.0036–9800; best 94.19% at LR 460, momentum 0.8
- momentum 0: LR 0.0036–2100 (5.83e+05x)
- momentum 0.5: LR 0.0036–2100 (5.83e+05x)
- momentum 0.7: LR 0.0036–3500 (9.72e+05x)
- momentum 0.8: LR 0.0036–3500 (9.72e+05x)
- momentum 0.9: LR 0.0036–5900 (1.64e+06x)

## Norm bias Lion
start LR 0.017 / momentum 0.9; tested LR 6.2e-07–1.7; best 94.12% at LR 0.028, momentum 0.7
- momentum 0: LR 6.2e-07–1 (1.61e+06x)
- momentum 0.5: LR 6.2e-07–1 (1.61e+06x)
- momentum 0.7: LR 6.2e-07–0.61 (9.84e+05x)
- momentum 0.8: LR 6.2e-07–0.61 (9.84e+05x)
- momentum 0.9: LR 6.2e-07–0.61 (9.84e+05x)

## Whitening bias SGD
start LR 36 / momentum 0.8; tested LR 0.0013–590000; best 94.20% at LR 0.078, momentum 0.7
- momentum 0: LR 0.0013–77000 (5.92e+07x)
- momentum 0.5: LR 0.0013–77000 (5.92e+07x)
- momentum 0.7: LR 0.0013–130000 (1e+08x)
- momentum 0.8: LR 0.0013–77000 (5.92e+07x)
- momentum 0.9: LR 0.0013–28000 (2.15e+07x)

## Whitening bias Lion
start LR 0.047 / momentum 0.9; tested LR 1.7e-06–22; best 94.26% at LR 0.00047, momentum 0.7
- momentum 0: LR 1.7e-06–7.8 (4.59e+06x)
- momentum 0.5: LR 1.7e-06–7.8 (4.59e+06x)
- momentum 0.7: LR 1.7e-06–4.7 (2.76e+06x)
- momentum 0.8: LR 1.7e-06–13 (7.65e+06x)
- momentum 0.9: LR 1.7e-06–13 (7.65e+06x)
