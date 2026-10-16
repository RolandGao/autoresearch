"""Final comparison table: held-out TTA (mean over search seeds) and search seconds for the
main algorithms per scenario, batch size and transform. python research/final_table.py"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = subprocess.run([sys.executable, os.path.join(REPO, "research", "summarize_auto.py")],
                     capture_output=True, text=True).stdout
rows = {}
for line in out.split("\n"):
    parts = line.split()
    if len(parts) >= 6 and parts[0].startswith("auto_") and parts[1].isdigit():
        rows[parts[0]] = dict(n=int(parts[1]), seconds=float(parts[2]), heldout=float(parts[5]))

ALGOS = ["v1", "v1p", "v1x2", "v1x2m3", "v1x2m3t", "v1x2m2t", "v1x3m2t"]
for scenario, algos in (("a", ALGOS), ("b", ["v1fast", "v1fas", "v1psfast", "v1x2m3fas", "v1x2m3tfas"]),
                        ("ahead", ["v1x2m3t"]), ("a2", ["v1x2m3t"])):
    for bs in (2000, 500, 125):
        lines = []
        for algo in algos:
            cells = []
            for transform in ("", "_ramp", "_wave"):
                r = rows.get(f"auto_{scenario}_{algo}_bs{bs}{transform}")
                cells.append(f"{r['heldout']:.2f} ({r['n']}, {r['seconds']:.0f}s)" if r else "-")
            if any(c != "-" for c in cells):
                lines.append(f"  {algo:11s} " + "  ".join(f"{c:18s}" for c in cells))
        if lines:
            print(f"scenario {scenario}, BS {bs}:   none                ramp                wave")
            print("\n".join(lines))
