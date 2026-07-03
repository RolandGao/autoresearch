#!/usr/bin/env python3

from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
import random
import re
from pathlib import Path


SUMMARY_PATH = Path(__file__).parent / "20260626_022201_204962" / "summary.txt"
NUM_TRANSFORMED_FUNCTIONS = 10
RANDOM_SEED = 0
PROBE_COUNT = 122
PROBE_MIN = 1e-3
PROBE_MAX = 1e6
EXTRA_STRESS_FUNCTIONS = 1000
EXTRA_STRESS_SEED = 1


@dataclass
class SearchResult:
    x: float
    fx: float
    evaluations: int


@dataclass
class FixedFunction:
    name: str
    ground_truth: "GroundTruth"


def parse_summary(path: Path) -> list[tuple[float, float]]:
    pattern = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*->\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*$", re.IGNORECASE)
    points = []

    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            points.append((float(match.group(1)), float(match.group(2))))

    if len(points) < 2:
        raise ValueError(f"Expected at least two points in {path}")

    points.sort()
    return points


class GroundTruth:
    def __init__(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.xs = [x for x, _ in points]
        self.ys = [y for _, y in points]

    def lookup(self, x: float) -> tuple[float, str]:
        index = bisect.bisect_left(self.xs, x)

        if index < len(self.xs) and math.isclose(self.xs[index], x, rel_tol=0.0, abs_tol=1e-12):
            return self.ys[index], "table"
        if index > 0 and math.isclose(self.xs[index - 1], x, rel_tol=0.0, abs_tol=1e-12):
            return self.ys[index - 1], "table"

        if index == 0:
            return self.ys[0], "clamped"
        elif index == len(self.xs):
            return self.ys[-1], "clamped"
        else:
            left_index, right_index = index - 1, index
            source = "interpolated"

        left_x = self.xs[left_index]
        right_x = self.xs[right_index]
        left_y = self.ys[left_index]
        right_y = self.ys[right_index]
        fraction = (x - left_x) / (right_x - left_x)
        return left_y + fraction * (right_y - left_y), source

    def max_value(self) -> float:
        return max(self.ys)


class TransformedGroundTruth(GroundTruth):
    def __init__(
        self,
        points: list[tuple[float, float]],
        value_multiplier: float,
        y_shift: int,
        x_multiplier: float,
    ) -> None:
        shifted_points = self.shift_values(points, y_shift)
        scaled_points = [(x, y * value_multiplier) for x, y in shifted_points]
        super().__init__(scaled_points)
        self.value_multiplier = value_multiplier
        self.y_shift = y_shift
        self.x_multiplier = x_multiplier

    @staticmethod
    def shift_values(points: list[tuple[float, float]], shift: int) -> list[tuple[float, float]]:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        shift %= len(ys)
        shifted_ys = ys[-shift:] + ys[:-shift] if shift else ys
        return list(zip(xs, shifted_ys))

    def lookup(self, x: float) -> tuple[float, str]:
        return super().lookup(self.x_multiplier * x)


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_transforms(points: list[tuple[float, float]], count: int) -> list[TransformedGroundTruth]:
    rng = random.Random(RANDOM_SEED)
    transforms = []

    for _ in range(count):
        transforms.append(
            TransformedGroundTruth(
                points=points,
                value_multiplier=log_uniform(rng, 0.1, 10.0),
                y_shift=rng.randrange(len(points)),
                x_multiplier=log_uniform(rng, 0.1, 10.0),
            )
        )

    return transforms


def make_fixed_functions(points: list[tuple[float, float]]) -> list[FixedFunction]:
    functions = [FixedFunction("original", GroundTruth(points))]
    functions.extend(
        FixedFunction(f"transformed #{index}", ground_truth)
        for index, ground_truth in enumerate(sample_transforms(points, NUM_TRANSFORMED_FUNCTIONS), start=1)
    )
    return functions


class EvaluationCounter:
    def __init__(self, ground_truth: GroundTruth) -> None:
        self.ground_truth = ground_truth
        self.count = 0

    def evaluate(self, x: float) -> float:
        self.count += 1
        fx, _ = self.ground_truth.lookup(x)
        return fx


class LogCoverageSearch:
    def __init__(self, probe_count: int, probe_min: float, probe_max: float) -> None:
        log_min = math.log(probe_min)
        log_max = math.log(probe_max)
        self.probes = [
            math.exp(log_min + (log_max - log_min) * index / (probe_count - 1))
            for index in range(probe_count)
        ]

    def search(self, evaluator: EvaluationCounter) -> SearchResult:
        best_x = self.probes[0]
        best_fx = evaluator.evaluate(best_x)

        for x in self.probes[1:]:
            fx = evaluator.evaluate(x)
            if fx > best_fx:
                best_x = x
                best_fx = fx

        return SearchResult(best_x, best_fx, evaluator.count)


def fmt_x(value: float) -> str:
    return f"{value:.2g}"


def print_accuracy_table(rows: list[dict[str, float | str]]) -> None:
    headers = ["function", "evals", "found x", "found f", "actual max", "accuracy"]
    table = [headers]

    for row in rows:
        table.append(
            [
                str(row["function"]),
                str(row["evals"]),
                f"{row['found_x']:.4g}",
                f"{row['found_f']:.6f}",
                f"{row['actual_max']:.4f}",
                f"{row['accuracy']:.6f}",
            ]
        )

    widths = [max(len(table_row[column]) for table_row in table) for column in range(len(headers))]

    print("accuracy summary")
    for index, table_row in enumerate(table):
        print("  " + "  ".join(value.rjust(widths[column]) for column, value in enumerate(table_row)))
        if index == 0:
            print("  " + "  ".join("-" * width for width in widths))


def sample_extra_functions(points: list[tuple[float, float]], count: int) -> list[FixedFunction]:
    rng = random.Random(EXTRA_STRESS_SEED)
    functions = []
    for index in range(1, count + 1):
        functions.append(
            FixedFunction(
                f"stress #{index}",
                TransformedGroundTruth(
                    points=points,
                    value_multiplier=log_uniform(rng, 0.1, 10.0),
                    y_shift=rng.randrange(len(points)),
                    x_multiplier=log_uniform(rng, 0.1, 10.0),
                ),
            )
        )
    return functions


def run_search(function: FixedFunction, search: LogCoverageSearch) -> dict[str, float | str]:
    actual_max = function.ground_truth.max_value()
    evaluator = EvaluationCounter(function.ground_truth)
    result = search.search(evaluator)
    return {
        "function": function.name,
        "evals": result.evaluations,
        "found_x": result.x,
        "found_f": result.fx,
        "actual_max": actual_max,
        "accuracy": result.fx / actual_max,
    }


def main() -> None:
    points = parse_summary(SUMMARY_PATH)
    functions = make_fixed_functions(points)
    search = LogCoverageSearch(PROBE_COUNT, PROBE_MIN, PROBE_MAX)

    print(f"summary={SUMMARY_PATH}")
    print(f"fixed transformed functions={NUM_TRANSFORMED_FUNCTIONS}")
    print(f"algorithm=log_coverage_sweep")
    print(f"probe_count={PROBE_COUNT}")
    print(f"probe_range=[{PROBE_MIN:g}, {PROBE_MAX:g}]\n")

    accuracy_rows = []
    for function in functions:
        ground_truth = function.ground_truth
        print(function.name)
        if isinstance(ground_truth, TransformedGroundTruth):
            print(
                "  "
                f"value_multiplier={ground_truth.value_multiplier:.4g}, "
                f"y_shift={ground_truth.y_shift}, "
                f"x_multiplier={ground_truth.x_multiplier:.4g}"
            )
        print()

        row = run_search(function, search)
        print(
            f"  found_x={fmt_x(row['found_x'])} "
            f"found_f={row['found_f']:.6f} "
            f"actual_max={row['actual_max']:.6f} "
            f"accuracy={row['accuracy']:.6f} "
            f"evaluations={row['evals']}\n"
        )
        accuracy_rows.append(row)

    print_accuracy_table(accuracy_rows)

    stress_rows = [run_search(function, search) for function in sample_extra_functions(points, EXTRA_STRESS_FUNCTIONS)]
    stress_accuracies = [row["accuracy"] for row in stress_rows]
    print()
    print("fresh random stress")
    print(f"  functions={EXTRA_STRESS_FUNCTIONS}")
    print(f"  evaluations_each={PROBE_COUNT}")
    print(f"  min_accuracy={min(stress_accuracies):.6f}")
    print(f"  mean_accuracy={sum(stress_accuracies) / len(stress_accuracies):.6f}")
    print(f"  below_0.99={sum(accuracy < 0.99 for accuracy in stress_accuracies)}")


if __name__ == "__main__":
    main()
