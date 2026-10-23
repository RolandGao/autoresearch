"""Which one-interval lookahead ranks interval candidates like the truth, for each of the four LRs?

The plan is the baseline run itself (every group on its configured schedule). The run is split into
5 intervals. For a main group g and interval k, the baseline is followed up to interval k; there,
g's LR is replaced by a constant candidate c (the x0.6 grid value nearest the baseline's mean of g
over interval k, moved by grid offsets; whitening bias, which the baseline has at 0 from step 75,
uses its grid value nearest 36 there), trained for one interval from the same state, then scored by:
  truth      the rest of the run on the baseline (every group, g included)
  lin_main   one interval with g decaying linearly from c to 0, the others on the baseline
  lin_all    one interval with every group decaying linearly to 0 (g from c, the others from their
             baseline value at the interval's start)
  s_main     one interval with g constant at m x c, m searched, the others on the baseline
  s_tied     one interval with g at m x c and the other three groups at m_o x their baseline mean
             over interval k (one tied m_o), m and m_o searched
  s_each     one interval with every group at its own searched m (g: m x c, others: m x their
             baseline mean over interval k)
Searches: multipliers on the 0.6^k grid (2 significant figures), unbounded, starting at 0.22; a
coordinate search that moves a multiplier to its best neighbour among k+-1, k+-2 while that improves
the score, until every multiplier beats its k+-1 and k+-2 (lr grid protocol). Groups whose baseline
mean over interval k is 0 have nothing to scale and are not searched. Every evaluated point is logged
(kind=probe), and the searched multipliers with each eval.
"""
import argparse

from common import base_config, emit, new_model, open_log, t

GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--groups", default=",".join(GROUPS))
parser.add_argument("--intervals", type=int, default=5)
parser.add_argument("--offsets", default="-2,-1,0,1,2")
parser.add_argument("--start-mult", type=float, default=0.22)
parser.add_argument("--tag", default="")
args = parser.parse_args()
log = open_log(f"hparam_lookahead{args.tag}_bs{args.batch_size}_s{args.seed}", args, __file__)

config = base_config(args.batch_size)
offsets = [int(o) for o in args.offsets.split(",")]
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
base = trainer.base_schedules
total = trainer.total_steps
length = total // args.intervals
assert length * args.intervals == total
key = {g: f"{g}.initial_lr" for g in GROUPS}


def value_at(g, step):
    return t.schedule_value_at_step(base[key[g]], step)


def mean_over(g, start, steps):
    return sum(value_at(g, s) for s in range(start, start + steps)) / steps


def window(lines, start, line):
    """lines with [start, start + line[0]) replaced by line (constant (n, v) or linear (n, v0, v1))."""
    return t.slice_schedule(lines, 0, start) + [line] + t.slice_schedule(lines, start + line[0], total)


def grid(mult_index):
    return t.round_hparam(0.6 ** mult_index)


def run_with(lines_by_group, steps):
    sched = dict(base)
    for g, lines in lines_by_group.items():
        sched[key[g]] = lines
    ok = trainer.train(sched, steps)
    return 100 * trainer.score("tta_val_acc") if ok else 0.0


def neighbour_search(names, score):
    """Coordinate search over grid indices (one per name), unbounded. Returns (best indices, score, probes)."""
    cache = {}

    def evaluate(point):
        if point not in cache:
            cache[point] = score(dict(zip(names, (grid(i) for i in point))))
        return cache[point]

    current = tuple(t.lr_grid_index(args.start_mult, 0.6) for _ in names)
    current_score = evaluate(current)
    while True:
        moved = False
        for j in range(len(names)):
            options = [current[:j] + (current[j] + d,) + current[j + 1:] for d in (-2, -1, 1, 2)]
            best = max(options, key=evaluate)
            if evaluate(best) > current_score:
                current, current_score, moved = best, evaluate(best), True
        if not moved:
            probes = [dict(zip(names, (grid(i) for i in p)), score=s) for p, s in cache.items()]
            return dict(zip(names, (grid(i) for i in current))), current_score, probes


emit(log, kind="setup", seed=args.seed, total=total, length=length, groups=args.groups.split(","),
     baseline={g: base[key[g]] for g in GROUPS})
start = trainer.snapshot()
for g in args.groups.split(","):
    for k in range(args.intervals):
        s0, s1 = k * length, (k + 1) * length
        trainer.restore(start)
        trainer.train(dict(base), s0)
        prefix_state = trainer.snapshot()
        center = mean_over(g, s0, length)
        center_index = t.lr_grid_index(center if center > 0 else 36.0, 0.6)
        others = [o for o in GROUPS if o != g]
        other_mean = {o: mean_over(o, s0, length) for o in others}
        for offset in offsets:
            c = grid(center_index - offset)
            main_lines = window(base[key[g]], s0, (length, c))
            trainer.restore(prefix_state)
            run_with({g: main_lines}, length)
            mid = trainer.snapshot()
            scores, searched = {}, {}
            scores["truth"] = run_with({g: main_lines}, total - s1) if s1 < total else 100 * trainer.score("tta_val_acc")
            if s1 < total:
                def lookahead(lines_by_group):
                    trainer.restore(mid)
                    return run_with(lines_by_group, length)

                scores["lin_main"] = lookahead({g: window(main_lines, s1, (length, c, 0.0))})
                lin_all = {o: window(base[key[o]], s1, (length, value_at(o, s1), 0.0)) for o in others}
                scores["lin_all"] = lookahead(dict(lin_all, **{g: window(main_lines, s1, (length, c, 0.0))}))

                def constant_cooldown(mults):
                    """mults: {group: m}; groups without an m stay on the baseline (g on its candidate)."""
                    lines = {g: window(main_lines, s1, (length, mults[g] * c))}
                    for o in others:
                        if o in mults:
                            lines[o] = window(base[key[o]], s1, (length, mults[o] * other_mean[o]))
                    return lookahead(lines)

                live = [o for o in others if other_mean[o] > 0]
                variants = {
                    "s_main": ([g], lambda m: constant_cooldown(m)),
                    "s_tied": ([g, "others"] if live else [g],
                               lambda m: constant_cooldown(dict({g: m[g]}, **{o: m.get("others", 1.0) for o in live}))),
                    "s_each": ([g] + live, lambda m: constant_cooldown(m)),
                }
                for name, (names, fn) in variants.items():
                    best, best_score, probes = neighbour_search(names, fn)
                    scores[name] = best_score
                    searched[name] = best
                    emit(log, kind="probe", group=g, k=k, offset=offset, c=c, variant=name, probes=probes)
            emit(log, kind="eval", group=g, k=k, offset=offset, c=c, center=center,
                 **{n: v / 100 for n, v in scores.items()}, searched=searched)
emit(log, kind="done", train_steps=trainer.train_steps)
