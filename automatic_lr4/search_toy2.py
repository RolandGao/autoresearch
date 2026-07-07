from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATHS = (
    HERE
    / "20260706_235846_088906"
    / "cifar_search_improved_plots"
    / "summary.txt",
    HERE
    / "20260707_065827_213717"
    / "cifar_search_bias_split_plots"
    / "summary.txt",
)

DEFAULT_BUDGET = 46
MAIN_ONLY_COST = 1
NEW_MAIN_COST = 2
SAME_MAIN_COOLDOWN_COST = 1
DEFAULT_SHIFT_MIN = -10
DEFAULT_SHIFT_MAX = 10

# The current interval's selected_train_hparams are intentionally not used: in
# these summaries they are effectively the answer for the current matrix.
# No probe rule is keyed by attr name or search-space bounds. Probes are integer
# coordinates in the 0.6^k lattice. A hidden augmentation shift maps requested
# coordinate i to source matrix index clamp(i + shift).
INITIAL_MAIN_COORDS = (-6, 0, 6, 12)
INITIAL_COOLDOWN_COORD = 0
FOLLOWUP_MAIN_COORD_OFFSETS = (0, -1, 1)
COOLDOWN_REFINEMENT_COORD_OFFSETS = (0, 6, -6, 12, -12, 2, -2, 1, -1)
PORTFOLIO_MAIN_COORD_START = -2
PORTFOLIO_MAIN_COORD_COUNT = 23
PORTFOLIO_COOLDOWN_RELATIVE_OFFSET = -4


@dataclass(frozen=True)
class MatrixSummary:
    source: Path
    interval: int
    attr: str
    global_peak: dict[str, float]
    main_values: list[float]
    cooldown_values: list[float]
    matrix: list[list[float]]
    main_before_cooldown: list[float]


@dataclass(frozen=True)
class CellEval:
    main_coord: int
    cooldown_coord: int
    value: float
    objective_value: float | None
    step_cost: int
    total_cost: int
    phase: str


@dataclass(frozen=True)
class SearchResult:
    summary: MatrixSummary
    shift: int
    prior_cell: tuple[int, int] | None
    screen_cooldown_coord: int
    main_probe_coords: list[int]
    best_eval: CellEval
    evaluations: list[CellEval]

    @property
    def global_value(self) -> float:
        return self.summary.global_peak["tta_val_acc"]

    @property
    def gap(self) -> float:
        return self.global_value - objective_score(self.best_eval)

    @property
    def spent_cost(self) -> int:
        return self.evaluations[-1].total_cost


def parse_number(text: str) -> float:
    return float(text)


def parse_key_values(payload: str) -> dict[str, float]:
    values = {}
    for token in payload.split():
        key, raw_value = token.split("=", 1)
        values[key] = parse_number(raw_value)
    return values


def parse_summary_file(path: Path) -> list[MatrixSummary]:
    lines = path.read_text().splitlines()
    summaries: list[MatrixSummary] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.startswith("interval="):
            index += 1
            continue

        fields = dict(token.split("=", 1) for token in line.split())
        interval = int(fields["interval"])
        attr = fields["attr"]
        global_peak: dict[str, float] = {}
        index += 1

        while index < len(lines) and lines[index] != "matrix:":
            line = lines[index]
            if line.startswith("global_peak="):
                global_peak = parse_key_values(line.split("=", 1)[1])
            index += 1

        if index >= len(lines):
            raise ValueError(f"{path}: interval={interval} attr={attr} has no matrix")
        if not global_peak:
            raise ValueError(
                f"{path}: interval={interval} attr={attr} has no global peak"
            )

        index += 1
        header = lines[index].split("\t")
        main_values = [parse_number(value) for value in header[1:]]
        index += 1

        cooldown_values: list[float] = []
        matrix: list[list[float]] = []
        while (
            index < len(lines)
            and lines[index].strip()
            and not lines[index].startswith("main_tta_val_acc_before_cooldown:")
        ):
            row = lines[index].split("\t")
            cooldown_values.append(parse_number(row[0]))
            matrix.append([parse_number(value) for value in row[1:]])
            index += 1

        expected_shape = (len(cooldown_values), len(main_values))
        actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{path}: interval={interval} attr={attr} expected shape "
                f"{expected_shape}, got {actual_shape}"
            )

        main_before_cooldown = [float("nan")] * len(main_values)
        if (
            index < len(lines)
            and lines[index].startswith("main_tta_val_acc_before_cooldown:")
        ):
            index += 1
            if index >= len(lines):
                raise ValueError(
                    f"{path}: interval={interval} attr={attr} has no main table"
                )
            index += 1
            main_value_to_index = {
                value: value_index for value_index, value in enumerate(main_values)
            }
            while index < len(lines) and lines[index].strip():
                row = lines[index].split("\t")
                if len(row) != 2:
                    break
                main_value = parse_number(row[0])
                main_index = main_value_to_index.get(main_value)
                if main_index is None:
                    raise ValueError(
                        f"{path}: interval={interval} attr={attr} unknown "
                        f"main before cooldown value {main_value}"
                    )
                main_before_cooldown[main_index] = parse_number(row[1])
                index += 1

        summaries.append(
            MatrixSummary(
                source=path,
                interval=interval,
                attr=attr,
                global_peak=global_peak,
                main_values=main_values,
                cooldown_values=cooldown_values,
                matrix=matrix,
                main_before_cooldown=main_before_cooldown,
            )
        )

    return summaries


def parse_summary_files(paths: list[Path]) -> list[MatrixSummary]:
    summaries: list[MatrixSummary] = []
    for path in paths:
        summaries.extend(parse_summary_file(path))
    return summaries


def log_distance(left: float, right: float) -> float:
    if left > 0.0 and right > 0.0:
        return abs(math.log(left) - math.log(right))
    return abs(left - right)


def nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda index: log_distance(values[index], target))


def clamp_index(index: int, size: int) -> int:
    return min(max(index, 0), size - 1)


def shifted_index(coord: int, shift: int, size: int) -> int:
    return clamp_index(coord + shift, size)


def coord_value(coord: int) -> float:
    return float(f"{0.6 ** (-coord):.2g}")


def matrix_value(
    summary: MatrixSummary,
    main_coord: int,
    cooldown_coord: int,
    shift: int,
) -> float:
    main_index = shifted_index(main_coord, shift, len(summary.main_values))
    cooldown_index = shifted_index(
        cooldown_coord, shift, len(summary.cooldown_values)
    )
    return summary.matrix[cooldown_index][main_index]


def main_before_cooldown_value(
    summary: MatrixSummary,
    main_coord: int,
    shift: int,
) -> float:
    main_index = shifted_index(main_coord, shift, len(summary.main_values))
    return summary.main_before_cooldown[main_index]


def objective_score(evaluation: CellEval) -> float:
    if evaluation.objective_value is None:
        return float("-inf")
    return evaluation.objective_value


def unique_coords(coords: list[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for coord in coords:
        if coord in seen:
            continue
        seen.add(coord)
        unique.append(coord)
    return unique


def main_probe_coords(prior_cell: tuple[int, int] | None) -> list[int]:
    if prior_cell is None:
        return list(INITIAL_MAIN_COORDS)

    prior_coord = prior_cell[0]
    return unique_coords(
        [prior_coord + offset for offset in FOLLOWUP_MAIN_COORD_OFFSETS]
    )


def screen_cooldown_coord(prior_cell: tuple[int, int] | None) -> int:
    if prior_cell is not None:
        return prior_cell[1]
    return INITIAL_COOLDOWN_COORD


def cooldown_refinement_coords(screen_coord: int) -> list[int]:
    return unique_coords(
        [
            screen_coord + offset
            for offset in COOLDOWN_REFINEMENT_COORD_OFFSETS
        ]
    )


def portfolio_matrix_coords() -> list[tuple[int, int]]:
    return [
        (main_coord, main_coord + PORTFOLIO_COOLDOWN_RELATIVE_OFFSET)
        for main_coord in range(
            PORTFOLIO_MAIN_COORD_START,
            PORTFOLIO_MAIN_COORD_START + PORTFOLIO_MAIN_COORD_COUNT,
        )
    ]


def portfolio_search_cost() -> int:
    # Portfolio coordinates deliberately use one cooldown per main coordinate,
    # so each probe starts a new main run in this toy cost model.
    return len(portfolio_matrix_coords()) * NEW_MAIN_COST


def evaluate_cell(
    summary: MatrixSummary,
    shift: int,
    main_coord: int,
    cooldown_coord: int,
    evaluated_mains: set[int],
    total_cost: int,
    phase: str,
) -> CellEval:
    if phase == "main_screen":
        step_cost = MAIN_ONLY_COST
        value = main_before_cooldown_value(summary, main_coord, shift)
        objective_value = None
    else:
        step_cost = (
            SAME_MAIN_COOLDOWN_COST if main_coord in evaluated_mains else NEW_MAIN_COST
        )
        value = matrix_value(summary, main_coord, cooldown_coord, shift)
        objective_value = value
    return CellEval(
        main_coord=main_coord,
        cooldown_coord=cooldown_coord,
        value=value,
        objective_value=objective_value,
        step_cost=step_cost,
        total_cost=total_cost + step_cost,
        phase=phase,
    )


def add_evaluation(
    summary: MatrixSummary,
    shift: int,
    main_coord: int,
    cooldown_coord: int,
    budget: int,
    evaluations: list[CellEval],
    evaluated_cells: set[tuple[int, int]],
    evaluated_mains: set[int],
    phase: str,
) -> bool:
    if phase != "main_screen" and (main_coord, cooldown_coord) in evaluated_cells:
        return True

    total_cost = evaluations[-1].total_cost if evaluations else 0
    evaluation = evaluate_cell(
        summary,
        shift=shift,
        main_coord=main_coord,
        cooldown_coord=cooldown_coord,
        evaluated_mains=evaluated_mains,
        total_cost=total_cost,
        phase=phase,
    )
    if evaluation.total_cost > budget:
        return False

    evaluated_mains.add(main_coord)
    if phase != "main_screen":
        evaluated_cells.add((main_coord, cooldown_coord))
    evaluations.append(evaluation)
    return True


def portfolio_matrix_search(
    summary: MatrixSummary,
    shift: int,
    prior_cell: tuple[int, int] | None,
    budget: int,
) -> SearchResult:
    evaluations: list[CellEval] = []
    evaluated_cells: set[tuple[int, int]] = set()
    evaluated_mains: set[int] = set()

    for main_coord, cooldown_coord in portfolio_matrix_coords():
        if not add_evaluation(
            summary,
            shift,
            main_coord,
            cooldown_coord,
            budget,
            evaluations,
            evaluated_cells,
            evaluated_mains,
            phase="scale_portfolio",
        ):
            break

    if not evaluations:
        raise ValueError(f"budget {budget} did not allow any portfolio evaluations")

    return SearchResult(
        summary=summary,
        shift=shift,
        prior_cell=prior_cell,
        screen_cooldown_coord=INITIAL_COOLDOWN_COORD,
        main_probe_coords=[main_coord for main_coord, _ in portfolio_matrix_coords()],
        best_eval=max(evaluations, key=objective_score),
        evaluations=evaluations,
    )


def budgeted_matrix_search(
    summary: MatrixSummary,
    shift: int,
    prior_cell: tuple[int, int] | None,
    budget: int = DEFAULT_BUDGET,
) -> SearchResult:
    if budget < NEW_MAIN_COST:
        raise ValueError(f"budget must be at least {NEW_MAIN_COST}")
    if budget >= portfolio_search_cost():
        return portfolio_matrix_search(
            summary=summary,
            shift=shift,
            prior_cell=prior_cell,
            budget=budget,
        )

    main_coords = main_probe_coords(prior_cell)
    initial_cooldown_coord = screen_cooldown_coord(prior_cell)
    evaluations: list[CellEval] = []
    evaluated_cells: set[tuple[int, int]] = set()
    evaluated_mains: set[int] = set()

    for main_coord in main_coords:
        if not add_evaluation(
            summary,
            shift,
            main_coord,
            initial_cooldown_coord,
            budget,
            evaluations,
            evaluated_cells,
            evaluated_mains,
            phase="main_screen",
        ):
            break

    if not evaluations:
        raise ValueError(f"budget {budget} did not allow any evaluations")

    screen_best = max(evaluations, key=lambda evaluation: evaluation.value)
    for cooldown_coord in cooldown_refinement_coords(initial_cooldown_coord):
        if not add_evaluation(
            summary,
            shift,
            screen_best.main_coord,
            cooldown_coord,
            budget,
            evaluations,
            evaluated_cells,
            evaluated_mains,
            phase="cooldown_refine",
        ):
            break

    cooldown_evaluations = [
        evaluation for evaluation in evaluations if evaluation.objective_value is not None
    ]
    best_eval = (
        max(cooldown_evaluations, key=objective_score)
        if cooldown_evaluations
        else screen_best
    )

    return SearchResult(
        summary=summary,
        shift=shift,
        prior_cell=prior_cell,
        screen_cooldown_coord=initial_cooldown_coord,
        main_probe_coords=main_coords,
        best_eval=best_eval,
        evaluations=evaluations,
    )


def source_key(summary: MatrixSummary) -> Path:
    return summary.source.parent


def run_interval_searches(
    summaries: list[MatrixSummary],
    budget: int = DEFAULT_BUDGET,
    shifts: range = range(DEFAULT_SHIFT_MIN, DEFAULT_SHIFT_MAX + 1),
) -> list[SearchResult]:
    results: list[SearchResult] = []
    summaries_by_source: dict[Path, list[MatrixSummary]] = {}
    for summary in summaries:
        summaries_by_source.setdefault(source_key(summary), []).append(summary)

    for source in sorted(summaries_by_source):
        source_summaries = sorted(
            summaries_by_source[source], key=lambda item: (item.interval, item.attr)
        )
        for shift in shifts:
            prior_by_attr: dict[str, tuple[int, int]] = {}
            intervals = sorted({summary.interval for summary in source_summaries})
            for interval in intervals:
                interval_summaries = [
                    summary
                    for summary in source_summaries
                    if summary.interval == interval
                ]
                interval_updates: dict[str, tuple[int, int]] = {}
                for summary in interval_summaries:
                    result = budgeted_matrix_search(
                        summary,
                        shift=shift,
                        prior_cell=prior_by_attr.get(summary.attr),
                        budget=budget,
                    )
                    best = result.best_eval
                    interval_updates[summary.attr] = (
                        best.main_coord,
                        best.cooldown_coord,
                    )
                    results.append(result)
                prior_by_attr.update(interval_updates)

    return results


def format_number(value: float) -> str:
    return f"{value:g}"


def format_score(value: float) -> str:
    return f"{value:.4f}"


def format_coord(coord: int) -> str:
    return format_number(coord_value(coord))


def format_optional_coord(coord: int | None) -> str:
    return "none" if coord is None else format_coord(coord)


def global_main_coord(result: SearchResult) -> int:
    summary = result.summary
    base_index = nearest_index(
        summary.main_values, summary.global_peak[f"main_{summary.attr}"]
    )
    return base_index - result.shift


def global_cooldown_coord(result: SearchResult) -> int:
    summary = result.summary
    base_index = nearest_index(
        summary.cooldown_values, summary.global_peak[f"cooldown_{summary.attr}"]
    )
    return base_index - result.shift


CONTEXT_FIELD_ORDER = (
    "source",
    "shift",
    "interval",
    "attr",
    "global",
    "gap",
)

PROBE_FIELD_ORDER = (
    "probe",
    "phase",
    "prior",
    "eval",
    "cost",
    "acc",
)


def context_row(result: SearchResult) -> dict[str, str]:
    summary = result.summary
    return {
        "source": f"source={summary.source.parent.name}",
        "shift": f"shift={result.shift}",
        "interval": f"interval={summary.interval}",
        "attr": f"attr={summary.attr}",
        "global": (
            "global=("
            f"main={format_coord(global_main_coord(result))},"
            f"cooldown={format_coord(global_cooldown_coord(result))},"
            f"acc={format_score(result.global_value)})"
        ),
        "gap": f"gap={result.gap:.4g}",
    }


def probe_rows(result: SearchResult, budget: int) -> list[dict[str, str]]:
    summary = result.summary
    prior_main = result.prior_cell[0] if result.prior_cell is not None else None
    prior_cooldown = result.prior_cell[1] if result.prior_cell is not None else None
    rows = []

    for probe_index, evaluation in enumerate(result.evaluations, start=1):
        cooldown_text = (
            "none"
            if evaluation.phase == "main_screen"
            else format_coord(evaluation.cooldown_coord)
        )
        rows.append(
            {
                "probe": f"probe={probe_index:02d}",
                "phase": f"phase={evaluation.phase}",
                "prior": (
                    "prior=("
                    f"{format_optional_coord(prior_main)}, "
                    f"{format_optional_coord(prior_cooldown)})"
                ),
                "eval": (
                    "eval=("
                    f"{format_coord(evaluation.main_coord)}, "
                    f"{cooldown_text})"
                ),
                "cost": f"cost={evaluation.total_cost}/{budget}",
                "acc": f"acc={format_score(evaluation.value)}",
            }
        )
    return rows


def aligned_probe_lines(results: list[SearchResult], budget: int) -> list[str]:
    context_rows = [context_row(result) for result in results]
    probe_rows_by_result = [probe_rows(result, budget) for result in results]
    probe_rows_flat = [row for rows in probe_rows_by_result for row in rows]
    context_widths = {
        field: max(len(row[field]) for row in context_rows)
        for field in CONTEXT_FIELD_ORDER
    }
    probe_widths = {
        field: max(len(row[field]) for row in probe_rows_flat)
        for field in PROBE_FIELD_ORDER
    }
    lines = []
    for row, probe_rows_for_result in zip(context_rows, probe_rows_by_result):
        context_fields = [
            row[field].ljust(context_widths[field])
            for field in CONTEXT_FIELD_ORDER[:-1]
        ]
        context_fields.append(row[CONTEXT_FIELD_ORDER[-1]])
        lines.append("  ".join(context_fields))
        for probe_row in probe_rows_for_result:
            probe_fields = [
                probe_row[field].ljust(probe_widths[field])
                for field in PROBE_FIELD_ORDER[:-1]
            ]
            probe_fields.append(probe_row[PROBE_FIELD_ORDER[-1]])
            lines.append("  ".join(probe_fields))
    return lines


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def performance_line(results: list[SearchResult]) -> str:
    gaps = [result.gap for result in results]
    found_accs = [objective_score(result.best_eval) for result in results]
    global_accs = [result.global_value for result in results]
    costs = [result.spent_cost for result in results]
    total_probes = sum(len(result.evaluations) for result in results)
    case_count = len(results)

    exact_count = sum(abs(gap) <= 1e-12 for gap in gaps)
    within_1e4 = sum(gap <= 0.0001 for gap in gaps)
    within_5e4 = sum(gap <= 0.0005 for gap in gaps)
    within_1e3 = sum(gap <= 0.001 for gap in gaps)

    return " ".join(
        [
            "performance",
            f"cases={case_count}",
            f"probes={total_probes}",
            f"exact={exact_count}/{case_count}",
            f"within_0.0001={within_1e4}/{case_count}",
            f"within_0.0005={within_5e4}/{case_count}",
            f"within_0.001={within_1e3}/{case_count}",
            f"mean_gap={sum(gaps) / case_count:.6g}",
            f"median_gap={percentile(gaps, 0.5):.6g}",
            f"p90_gap={percentile(gaps, 0.9):.6g}",
            f"worst_gap={max(gaps):.6g}",
            f"mean_found_acc={sum(found_accs) / case_count:.4f}",
            f"mean_global_acc={sum(global_accs) / case_count:.4f}",
            f"mean_cost={sum(costs) / case_count:.3g}",
            f"max_cost={max(costs)}",
        ]
    )


def print_report(results: list[SearchResult], budget: int) -> None:
    for line in aligned_probe_lines(results, budget):
        print(line)
    print(performance_line(results))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cost-limited toy search over CIFAR cooldown heatmap summaries."
        )
    )
    parser.add_argument(
        "summary_paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SUMMARY_PATHS),
        help="summary.txt files to parse",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="maximum search cost per matrix",
    )
    parser.add_argument(
        "--shift-min",
        type=int,
        default=DEFAULT_SHIFT_MIN,
        help="minimum hidden integer shift to evaluate",
    )
    parser.add_argument(
        "--shift-max",
        type=int,
        default=DEFAULT_SHIFT_MAX,
        help="maximum hidden integer shift to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shift_min > args.shift_max:
        raise SystemExit("--shift-min must be <= --shift-max")
    summaries = parse_summary_files(args.summary_paths)
    shifts = range(args.shift_min, args.shift_max + 1)
    results = run_interval_searches(summaries, budget=args.budget, shifts=shifts)
    print_report(results, budget=args.budget)


if __name__ == "__main__":
    main()
