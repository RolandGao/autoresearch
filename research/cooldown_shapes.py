"""More cooldown shapes and lengths for one lookahead cell (default: conv, interval 2, BS 2000, no transform),
where every one-interval cooldown of research/hparam_lookahead.py picked 0.078 against the truth's 0.13.

Same plan and candidates as hparam_lookahead.py: the baseline run is followed up to interval k, conv is set
to the constant candidate c (x0.6 grid around the baseline's interval mean, moved by --offset) for one
interval, then scored by a cooldown of --length steps (t = i / length, i = 0 .. length-1; other groups on
the baseline unless stated):
  lin_main   c (1 - t)                                  (= pow with p = 1)
  lin_all    lin_main, and every other group linear from its baseline value to 0
  s_main     m c constant, m searched
  s_tied     m c, the other three at m_o x their baseline mean over interval k (one tied m_o)
  s_each     every group at its own searched m (others: x their baseline mean over interval k)
  pow        c (1 - t)^p, p searched
  exp        c q^t (from c to q c), q searched
  powc       c (q + (1 - q)(1 - t)^p), p and q searched jointly
and truth = the rest of the run on the plan. Searches: 0.6^k grid (2 significant figures), unbounded
coordinate search moving each parameter to its best of k+-1, k+-2 while that improves (lr grid protocol);
p starts at 1, every multiplier at 0.22. Every probe is logged.

python research/cooldown_shapes.py --offset 0 --length 40
"""
import argparse

from common import base_config, emit, new_model, open_log, t

GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--group", default="conv")
parser.add_argument("--k", type=int, default=1)
parser.add_argument("--offset", type=int, required=True)
parser.add_argument("--length", type=int, required=True)
parser.add_argument("--intervals", type=int, default=5)
args = parser.parse_args()
log = open_log(f"cooldown_shapes_{args.group}_k{args.k}_L{args.length}_o{args.offset}_bs{args.batch_size}_s{args.seed}",
               args, __file__)

model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, base_config(args.batch_size))
base = dict(trainer.base_schedules)
total = trainer.total_steps
interval = total // args.intervals
g, L = args.group, args.length
s0, s1 = args.k * interval, (args.k + 1) * interval
assert s1 + L <= total
key = {o: f"{o}.initial_lr" for o in GROUPS}
others = [o for o in GROUPS if o != g]


def value_at(o, step):
    return t.schedule_value_at_step(base[key[o]], step)


def mean_over(o, start, steps):
    return sum(value_at(o, s) for s in range(start, start + steps)) / steps


def window(lines, start, new, n):
    """lines with [start, start + n) replaced by the lines new (covering n steps)."""
    return t.slice_schedule(lines, 0, start) + new + t.slice_schedule(lines, start + n, total)


def grid(index):
    return t.round_hparam(0.6 ** index)


def run_with(lines_by_group, steps):
    sched = dict(base)
    for o, lines in lines_by_group.items():
        sched[key[o]] = lines
    ok = trainer.train(sched, steps)
    return 100 * trainer.score("tta_val_acc") if ok else 0.0


def neighbour_search(start, score):
    """Coordinate search over grid indices, unbounded; start: {name: value}. Returns (best, score, probes)."""
    names = list(start)
    cache = {}

    def evaluate(point):
        if point not in cache:
            cache[point] = score(dict(zip(names, (grid(i) for i in point))))
        return cache[point]

    current = tuple(t.lr_grid_index(start[n], 0.6) for n in names)
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


trainer.train(dict(base), s0)
center = mean_over(g, s0, interval)
c = grid(t.lr_grid_index(center, 0.6) - args.offset)
main_lines = window(base[key[g]], s0, [(interval, c)], interval)
run_with({g: main_lines}, interval)
mid = trainer.snapshot()
other_mean = {o: mean_over(o, s0, interval) for o in others}
live = [o for o in others if other_mean[o] > 0]
emit(log, kind="setup", seed=args.seed, group=g, k=args.k, offset=args.offset, c=c, center=center, length=L,
     other_mean=other_mean)
scores, searched = {"truth": run_with({g: main_lines}, total - s1)}, {}


def cooldown(main_shape, other_lines=None):
    """main_shape(t) = multiplier of c at fraction t of the cooldown; other_lines: {group: lines over L steps}."""
    trainer.restore(mid)
    lines = {g: window(main_lines, s1, [(1, c * main_shape(i / L)) for i in range(L)], L)}
    for o, new in (other_lines or {}).items():
        lines[o] = window(base[key[o]], s1, new, L)
    return run_with(lines, L)


def constant(mults):
    return cooldown(lambda _: mults[g], {o: [(L, mults[o] * other_mean[o])] for o in others if o in mults})


scores["lin_main"] = cooldown(lambda x: 1 - x)
scores["lin_all"] = cooldown(lambda x: 1 - x, {o: [(L, value_at(o, s1), 0.0)] for o in others})
searches = {
    "s_main": ({g: 0.22}, constant),
    "s_tied": (dict({g: 0.22}, others=0.22) if live else {g: 0.22},
               lambda m: constant(dict({g: m[g]}, **{o: m.get("others", 1.0) for o in live}))),
    "s_each": ({o: 0.22 for o in [g] + live}, constant),
    "pow": ({"p": 1.0}, lambda m: cooldown(lambda x: (1 - x) ** m["p"])),
    "exp": ({"q": 0.22}, lambda m: cooldown(lambda x: m["q"] ** x)),
    "powc": ({"p": 1.0, "q": 0.22}, lambda m: cooldown(lambda x: m["q"] + (1 - m["q"]) * (1 - x) ** m["p"])),
}
for name, (start, fn) in searches.items():
    best, best_score, probes = neighbour_search(start, fn)
    scores[name], searched[name] = best_score, best
    emit(log, kind="probe", variant=name, probes=probes)
emit(log, kind="eval", group=g, k=args.k, offset=args.offset, c=c, center=center, length=L,
     **{n: v / 100 for n, v in scores.items()}, searched=searched)
emit(log, kind="done", train_steps=trainer.train_steps)
