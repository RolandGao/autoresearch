#!/usr/bin/env python3

from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
import random
import re
from pathlib import Path

import numpy as np


SUMMARY_PATH = Path(__file__).parent / "20260626_022201_204962" / "summary.txt"
NUM_TRANSFORMED_FUNCTIONS = 10
RANDOM_SEED = 0
PROBE_COUNT = 122
PROBE_MIN = 1e-3
PROBE_MAX = 1e6
SHORT_BUDGET = 20
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


class CoarseToFineLogSearch:
    def __init__(
        self,
        budget: int,
        probe_min: float,
        probe_max: float,
        coarse_points: int = 8,
        top_regions: int = 4,
    ) -> None:
        self.budget = budget
        self.log_min = math.log(probe_min)
        self.log_max = math.log(probe_max)
        self.coarse_points = coarse_points
        self.top_regions = top_regions

    def search(self, evaluator: EvaluationCounter) -> SearchResult:
        observations: list[tuple[float, float, float]] = []
        seen_logs = set()

        def evaluate_log(log_x: float) -> None:
            if evaluator.count >= self.budget:
                return
            log_x = max(self.log_min, min(self.log_max, log_x))
            key = round(log_x, 12)
            if key in seen_logs:
                return
            seen_logs.add(key)
            x = math.exp(log_x)
            observations.append((log_x, x, evaluator.evaluate(x)))

        coarse_step = (self.log_max - self.log_min) / (self.coarse_points - 1)
        for index in range(self.coarse_points):
            evaluate_log(self.log_min + coarse_step * index)

        top_observations = sorted(observations, key=lambda row: row[2], reverse=True)[: self.top_regions]
        for log_x, _, _ in top_observations:
            evaluate_log(log_x - coarse_step / 2)
            evaluate_log(log_x + coarse_step / 2)

        offsets = (coarse_step / 4, -coarse_step / 4, coarse_step / 8, -coarse_step / 8)
        while evaluator.count < self.budget:
            added = False
            for log_x, _, _ in sorted(observations, key=lambda row: row[2], reverse=True)[:3]:
                for offset in offsets:
                    before = evaluator.count
                    evaluate_log(log_x + offset)
                    added = added or evaluator.count > before
                    if evaluator.count >= self.budget:
                        break
                if evaluator.count >= self.budget:
                    break
            if not added:
                break

        best_log_x, best_x, best_fx = max(observations, key=lambda row: row[2])
        return SearchResult(best_x, best_fx, evaluator.count)


class LogSpaceGPUCBSearch:
    def __init__(
        self,
        budget: int,
        probe_min: float,
        probe_max: float,
        initial_points: int = 7,
        candidate_points: int = 300,
        length_scale: float = 0.3,
        beta: float = 0.5,
        noise: float = 1e-5,
    ) -> None:
        self.budget = budget
        self.log_min = math.log(probe_min)
        self.log_max = math.log(probe_max)
        self.initial_points = initial_points
        self.candidate_logs = np.linspace(self.log_min, self.log_max, candidate_points)
        self.length_scale = length_scale
        self.beta = beta
        self.noise = noise

    def search(self, evaluator: EvaluationCounter) -> SearchResult:
        observed_logs: list[float] = []
        observed_values: list[float] = []
        seen_candidates = set()

        def evaluate_log(log_x: float) -> None:
            if evaluator.count >= self.budget:
                return
            log_x = max(self.log_min, min(self.log_max, log_x))
            candidate_index = int(round((log_x - self.log_min) / (self.log_max - self.log_min) * (len(self.candidate_logs) - 1)))
            candidate_index = max(0, min(len(self.candidate_logs) - 1, candidate_index))
            if candidate_index in seen_candidates:
                return
            seen_candidates.add(candidate_index)
            log_x = float(self.candidate_logs[candidate_index])
            observed_logs.append(log_x)
            observed_values.append(evaluator.evaluate(math.exp(log_x)))

        for log_x in np.linspace(self.log_min, self.log_max, self.initial_points):
            evaluate_log(float(log_x))

        while evaluator.count < self.budget:
            x_train = np.asarray(observed_logs)
            y_train = np.asarray(observed_values)
            y_std = y_train.std()
            if y_std == 0:
                y_std = max(abs(float(y_train.mean())), 1.0)
            y_norm = (y_train - y_train.mean()) / y_std

            distances = x_train[:, None] - x_train[None, :]
            kernel = np.exp(-0.5 * distances * distances / (self.length_scale * self.length_scale))
            kernel += self.noise * np.eye(len(x_train))

            try:
                cholesky = np.linalg.cholesky(kernel)
                alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, y_norm))
                cross_kernel = np.exp(
                    -0.5
                    * ((self.candidate_logs[:, None] - x_train[None, :]) ** 2)
                    / (self.length_scale * self.length_scale)
                )
                mean = cross_kernel @ alpha
                uncertainty_basis = np.linalg.solve(cholesky, cross_kernel.T)
                variance = np.maximum(1.0 - np.sum(uncertainty_basis * uncertainty_basis, axis=0), 0.0)
            except np.linalg.LinAlgError:
                break

            acquisition = mean + self.beta * np.sqrt(variance)
            for candidate_index in seen_candidates:
                acquisition[candidate_index] = -np.inf
            evaluate_log(float(self.candidate_logs[int(np.argmax(acquisition))]))

        best_index = max(range(len(observed_values)), key=lambda index: observed_values[index])
        return SearchResult(math.exp(observed_logs[best_index]), observed_values[best_index], evaluator.count)


class SmoothIntervalUCBSearch:
    def __init__(
        self,
        budget: int,
        probe_min: float,
        probe_max: float,
        initial_points: int = 8,
        initial_phase: float = 0.75,
        exploration_weight: float = 0.05,
        continuity_weight: float = 0.0,
    ) -> None:
        self.budget = budget
        self.log_min = math.log(probe_min)
        self.log_max = math.log(probe_max)
        self.initial_points = initial_points
        self.initial_phase = initial_phase
        self.exploration_weight = exploration_weight
        self.continuity_weight = continuity_weight

    def search(self, evaluator: EvaluationCounter) -> SearchResult:
        observed_logs: list[float] = []
        observed_xs: list[float] = []
        observed_values: list[float] = []
        seen_logs = set()

        def evaluate_log(log_x: float) -> bool:
            if evaluator.count >= self.budget:
                return False
            log_x = max(self.log_min, min(self.log_max, float(log_x)))
            key = round(log_x, 12)
            if key in seen_logs:
                return False

            seen_logs.add(key)
            x = math.exp(log_x)
            fx = evaluator.evaluate(x)
            observed_logs.append(log_x)
            observed_xs.append(x)
            observed_values.append(fx)
            return True

        search_width = self.log_max - self.log_min
        initial_step = search_width / self.initial_points
        for index in range(self.initial_points):
            evaluate_log(self.log_min + initial_step * (index + self.initial_phase))

        while evaluator.count < self.budget:
            ordered = sorted(zip(observed_logs, observed_xs, observed_values))
            min_value = min(observed_values)
            max_value = max(observed_values)
            value_scale = max(max_value - min_value, abs(max_value), 1e-9)
            best_score = -math.inf
            best_log_x = None

            for left, right in zip(ordered, ordered[1:]):
                left_log, _, left_value = left
                right_log, _, right_value = right
                candidate_log_x = (left_log + right_log) / 2.0
                if round(candidate_log_x, 12) in seen_logs:
                    continue

                interval_width = right_log - left_log
                score = (
                    max(left_value, right_value)
                    + self.continuity_weight * min(left_value, right_value)
                    + self.exploration_weight * value_scale * interval_width / search_width
                )
                if score > best_score:
                    best_score = score
                    best_log_x = candidate_log_x

            if best_log_x is None:
                break
            evaluate_log(best_log_x)

        best_index = max(range(len(observed_values)), key=lambda index: observed_values[index])
        return SearchResult(observed_xs[best_index], observed_values[best_index], evaluator.count)


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


def print_stress_summary(rows: list[dict[str, float | str]]) -> None:
    accuracies = [row["accuracy"] for row in rows]
    evals = [row["evals"] for row in rows]
    print("fresh random stress")
    print(f"  functions={len(rows)}")
    print(f"  max_evaluations={max(evals)}")
    print(f"  min_accuracy={min(accuracies):.6f}")
    print(f"  mean_accuracy={sum(accuracies) / len(accuracies):.6f}")
    print(f"  below_0.99={sum(accuracy < 0.99 for accuracy in accuracies)}")


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


def run_search(function: FixedFunction, search) -> dict[str, float | str]:
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
    stress_functions = sample_extra_functions(points, EXTRA_STRESS_FUNCTIONS)
    searches = [
        ("log_coverage_sweep", LogCoverageSearch(PROBE_COUNT, PROBE_MIN, PROBE_MAX)),
        ("coarse_to_fine_log", CoarseToFineLogSearch(SHORT_BUDGET, PROBE_MIN, PROBE_MAX)),
        ("log_space_gp_ucb", LogSpaceGPUCBSearch(SHORT_BUDGET, PROBE_MIN, PROBE_MAX)),
        ("smooth_interval_ucb", SmoothIntervalUCBSearch(SHORT_BUDGET, PROBE_MIN, PROBE_MAX)),
    ]

    print(f"summary={SUMMARY_PATH}")
    print(f"fixed transformed functions={NUM_TRANSFORMED_FUNCTIONS}")
    print(f"probe_range=[{PROBE_MIN:g}, {PROBE_MAX:g}]")
    print(f"stress_functions={EXTRA_STRESS_FUNCTIONS}\n")

    for name, search in searches:
        print(f"algorithm={name}")
        accuracy_rows = [run_search(function, search) for function in functions]
        print_accuracy_table(accuracy_rows)
        print()
        stress_rows = [run_search(function, search) for function in stress_functions]
        print_stress_summary(stress_rows)
        print()


if __name__ == "__main__":
    main()
