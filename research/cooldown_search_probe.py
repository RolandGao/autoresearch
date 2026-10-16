"""Which one-interval (40-step at BS 2000) lookahead ranks candidates like the truth? (F1)

Scenario A probe: conv is searched; head, norm_bias and whiten_bias follow the baseline.
For interval k (length L = T/5) and candidate constant conv LRs c:
  truth        baseline until kL, c for L steps, then the baseline schedule to the end
  none         TTA right after the interval
  lin_conv     L-step lookahead: conv decays linearly c -> 0, the other groups continue
  lin_all      L-step lookahead: every group decays linearly to 0 (conv from c, the others
               from their current value)
  shared       L-step lookahead, one searched constant cooldown multiplier m for all four
               groups (conv at m x c, each other group at m x its mean LR over the interval):
               the best m on this seed
  conv_only    L-step lookahead, one searched constant multiplier for conv only (m x c); the
               other groups follow their baseline (ground-truth) schedules: the best m
  per_group    L-step lookahead, four searched multipliers (one per group): coordinate search
               from the best shared m, one group at a time, until no group changes (<= 3 passes)
Every evaluated cooldown is logged, so other selections can be recomputed.
"""
import argparse

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--interval", type=int, required=True)
parser.add_argument("--seeds", default="0,1,2,3")
parser.add_argument("--offsets", default="-3,-2,-1,0,1,2,3")
parser.add_argument("--mults", default="1,0.6,0.36,0.22,0.13,0.078,0.047,0")
args = parser.parse_args()
log = open_log(f"cooldown_search_probe_bs{args.batch_size}_k{args.interval}", args, __file__)

config = base_config(args.batch_size, None)
mults = [float(m) for m in args.mults.split(",")]
groups = ("conv", "head", "norm_bias", "whiten_bias")
model = new_model()
candidates = None
for seed in [int(s) for s in args.seeds.split(",")]:
    t.TRAINING_SEED = seed
    trainer = t.SegmentTrainer(model, config)
    base = trainer.base_schedules
    total = trainer.total_steps
    length = total // 5
    start = args.interval * length
    cool_start = start + length
    if cool_start + length > total:
        raise SystemExit("the last interval has no room for a lookahead")
    if candidates is None:
        centre = t.lr_grid_index(t.window_mean(base["conv.initial_lr"], start, length), 0.6)
        candidates = [t.round_hparam(0.6 ** (centre + o)) for o in map(int, args.offsets.split(","))]
        emit(log, kind="setup", total=total, length=length, start=start, candidates=candidates)
    means = {g: t.window_mean(base[f"{g}.initial_lr"], start, length) for g in groups}
    trainer.train(base, start)
    prefix = trainer.snapshot()
    for c in candidates:
        trainer.restore(prefix)
        main = dict(base)
        main["conv.initial_lr"] = t.constant_window(base["conv.initial_lr"], start, length, c)
        ok = trainer.train(main, length)
        main_state = trainer.snapshot()
        emit(log, kind="none", seed=seed, c=c, score=trainer.score("tta_val_acc") if ok else None)
        if ok:
            ok = trainer.train(main, total - cool_start)
        emit(log, kind="truth", seed=seed, c=c, score=trainer.score("tta_val_acc") if ok else None)
        cache = {}

        def lookahead(kind, ms):
            """Score of the L-step lookahead; ms maps group -> constant multiplier."""
            key = (kind, tuple(ms[g] for g in groups) if ms else None)
            if key in cache:
                return cache[key]
            trainer.restore(main_state)
            sched = dict(main)
            for g in groups:
                path = f"{g}.initial_lr"
                level = c if g == "conv" else means[g]
                if kind in ("lin_conv", "const_conv") and g != "conv":
                    continue  # The other groups continue their schedules.
                if kind in ("lin_conv", "lin_all"):
                    first = c if g == "conv" else t.schedule_value_at_step(base[path], cool_start)
                    sched[path] = t.linear_window(main[path], cool_start, length, first, 0.0)
                else:
                    sched[path] = t.constant_window(main[path], cool_start, length, ms[g] * level)
            ok = trainer.train(sched, length)
            value = trainer.score("tta_val_acc") if ok else float("-inf")
            cache[key] = value
            emit(log, kind="eval", seed=seed, c=c, mode=kind,
                 mults=None if ms is None else [ms[g] for g in groups], score=value)
            return value

        emit(log, kind="lin_conv", seed=seed, c=c, score=lookahead("lin_conv", None))
        emit(log, kind="lin_all", seed=seed, c=c, score=lookahead("lin_all", None))
        shared = {m: lookahead("const", {g: m for g in groups}) for m in mults}
        best_m = max(mults, key=lambda m: shared[m])
        emit(log, kind="shared", seed=seed, c=c, m=best_m, score=shared[best_m])
        conv_only = {m: lookahead("const_conv", {g: m for g in groups}) for m in mults}
        best_conv = max(mults, key=lambda m: conv_only[m])
        emit(log, kind="conv_only", seed=seed, c=c, m=best_conv, score=conv_only[best_conv])
        current = {g: best_m for g in groups}
        best = shared[best_m]
        for _ in range(3):
            moved = False
            for g in groups:
                for m in mults:
                    trial = dict(current, **{g: m})
                    value = lookahead("const", trial)
                    if value > best:
                        current, best, moved = trial, value, True
            if not moved:
                break
        emit(log, kind="per_group", seed=seed, c=c, mults=[current[g] for g in groups], score=best)
emit(log, kind="done", train_steps=trainer.train_steps)
