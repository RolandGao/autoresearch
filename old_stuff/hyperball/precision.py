import argparse
import csv
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch


DEVICE = "cuda"
GROUND_TRUTH_NAME = "fp32"


@dataclass(frozen=True)
class ComputeCase:
    name: str
    storage_dtype: torch.dtype
    allow_tf32: bool = False


@dataclass(frozen=True)
class BenchResult:
    d: int
    compute: str
    norm: str
    compiled: bool
    supported: bool
    reason: str
    rel_l2_error: float | None = None
    output_norm: float | None = None
    ms: float | None = None
    iters: int | None = None
    input_mib: float | None = None
    peak_mib: float | None = None
    case_peak_mib: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate CUDA dtype effects for y = normalize(W @ x), where "
            "||x||_2 = A and ||W||_F = B."
        )
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--main-dim", type=int, default=4096)
    parser.add_argument("--a", type=float, default=1.0, help="Target vector norm.")
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Target matrix Frobenius norm. Defaults to sqrt(D) for each D.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--min-bench-seconds",
        type=float,
        default=0.35,
        help="Increase repetitions until each timing pass lasts at least this long.",
    )
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".log"),
        help="Write human-readable results here.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_suffix(".csv"),
        help="Write machine-readable results here.",
    )
    return parser.parse_args()


@contextmanager
def tf32_mode(enabled: bool):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def available_compute_cases() -> list[ComputeCase]:
    cases = [
        ComputeCase("fp32", torch.float32, allow_tf32=False),
        ComputeCase("tf32", torch.float32, allow_tf32=True),
        ComputeCase("fp16", torch.float16, allow_tf32=False),
        ComputeCase("bf16", torch.bfloat16, allow_tf32=False),
    ]
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            cases.append(ComputeCase(name.replace("float8_", "fp8_"), dtype, allow_tf32=False))
    return cases


def available_norm_dtypes() -> dict[str, torch.dtype | None]:
    return {
        "same": None,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }


def make_inputs(d: int, a: float, b: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed + d)

    x = torch.randn(d, device=DEVICE, dtype=torch.float32, generator=generator)
    x = x * (a / torch.linalg.vector_norm(x))

    w = torch.randn(d, d, device=DEVICE, dtype=torch.float32, generator=generator)
    w = w * (b / torch.linalg.vector_norm(w))
    return w.contiguous(), x.contiguous()


def normalize_matvec(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    target_norm: float,
    norm_dtype: torch.dtype | None,
) -> torch.Tensor:
    y = matrix @ vector
    work = y if norm_dtype is None else y.to(norm_dtype)
    norm = torch.linalg.vector_norm(work)
    normalized = work * (target_norm / norm)
    return normalized.to(torch.float32)


def make_runner(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    target_norm: float,
    norm_dtype: torch.dtype | None,
):
    if norm_dtype is None:
        def run():
            y = matrix @ vector
            norm = torch.linalg.vector_norm(y)
            return (y * (target_norm / norm)).to(torch.float32)
        return run

    def run():
        y = matrix @ vector
        work = y.to(norm_dtype)
        norm = torch.linalg.vector_norm(work)
        return (work * (target_norm / norm)).to(torch.float32)

    return run


def relative_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(output - reference)
    denominator = torch.linalg.vector_norm(reference)
    return (numerator / denominator).item()


def output_norm(output: torch.Tensor) -> float:
    return torch.linalg.vector_norm(output.float()).item()


def dtype_nbytes(dtype: torch.dtype) -> float:
    if dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }:
        return 1.0
    return float(torch.empty((), dtype=dtype).element_size())


def input_storage_mib(d: int, dtype: torch.dtype) -> float:
    return ((d * d + d) * dtype_nbytes(dtype)) / (1024.0**2)


def cuda_time_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def choose_iters(fn, warmup: int, min_seconds: float, max_iters: int) -> int:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    iters = 10
    while True:
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds or iters >= max_iters:
            return min(iters, max_iters)
        multiplier = max(2, math.ceil(min_seconds / max(elapsed, 1e-9)))
        iters = min(max_iters, iters * multiplier)


def benchmark_case(
    *,
    d: int,
    compute: ComputeCase,
    norm_name: str,
    norm_dtype: torch.dtype | None,
    compiled: bool,
    w32: torch.Tensor,
    x32: torch.Tensor,
    reference: torch.Tensor,
    target_norm: float,
    warmup: int,
    min_bench_seconds: float,
    max_iters: int,
) -> BenchResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with tf32_mode(compute.allow_tf32):
            baseline_mib = torch.cuda.memory_allocated() / (1024.0**2)
            matrix = w32.to(dtype=compute.storage_dtype, copy=True)
            vector = x32.to(dtype=compute.storage_dtype, copy=True)
            torch.cuda.synchronize()

            run = make_runner(matrix, vector, target_norm, norm_dtype)
            if compiled and hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            timed_fn = torch.compile(run, fullgraph=True) if compiled else run

            # Compile and warm once before measuring so timings are steady-state.
            candidate = timed_fn()
            torch.cuda.synchronize()
            err = relative_l2(candidate, reference)
            norm = output_norm(candidate)

            iters = choose_iters(timed_fn, warmup, min_bench_seconds, max_iters)

            torch.cuda.reset_peak_memory_stats()
            samples = [cuda_time_ms(timed_fn, iters) for _ in range(5)]
            peak_mib = torch.cuda.max_memory_allocated() / (1024.0**2)
            case_peak_mib = peak_mib - baseline_mib

        return BenchResult(
            d=d,
            compute=compute.name,
            norm=norm_name,
            compiled=compiled,
            supported=True,
            reason="",
            rel_l2_error=err,
            output_norm=norm,
            ms=statistics.median(samples),
            iters=iters,
            input_mib=input_storage_mib(d, compute.storage_dtype),
            peak_mib=peak_mib,
            case_peak_mib=case_peak_mib,
        )
    except Exception as exc:
        torch.cuda.synchronize()
        return BenchResult(
            d=d,
            compute=compute.name,
            norm=norm_name,
            compiled=compiled,
            supported=False,
            reason=f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            input_mib=input_storage_mib(d, compute.storage_dtype),
        )


def ground_truth(w32: torch.Tensor, x32: torch.Tensor, target_norm: float) -> torch.Tensor:
    with tf32_mode(False):
        return normalize_matvec(w32, x32, target_norm, torch.float32)


def should_run_full_grid(d: int, main_dim: int) -> bool:
    return d == main_dim


def run_experiment(args: argparse.Namespace) -> list[BenchResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.set_float32_matmul_precision("highest")
    results: list[BenchResult] = []
    compute_cases = available_compute_cases()
    norm_dtypes = available_norm_dtypes()

    for d in args.dims:
        b = math.sqrt(d) if args.b is None else args.b
        w32, x32 = make_inputs(d, args.a, b, args.seed)
        reference = ground_truth(w32, x32, args.a)
        torch.cuda.synchronize()

        full_grid = should_run_full_grid(d, args.main_dim)
        for compute in compute_cases:
            norm_items = norm_dtypes.items() if full_grid else [("fp32", torch.float32)]
            for norm_name, norm_dtype in norm_items:
                eager_result = benchmark_case(
                    d=d,
                    compute=compute,
                    norm_name=norm_name,
                    norm_dtype=norm_dtype,
                    compiled=False,
                    w32=w32,
                    x32=x32,
                    reference=reference,
                    target_norm=args.a,
                    warmup=args.warmup,
                    min_bench_seconds=args.min_bench_seconds,
                    max_iters=args.max_iters,
                )
                results.append(eager_result)
                print(format_result(eager_result), flush=True)

                if not full_grid or not args.compile:
                    continue

                if not eager_result.supported:
                    result = BenchResult(
                        d=d,
                        compute=compute.name,
                        norm=norm_name,
                        compiled=True,
                        supported=False,
                        reason=f"not attempted after eager failed: {eager_result.reason}",
                        input_mib=input_storage_mib(d, compute.storage_dtype),
                    )
                else:
                    result = benchmark_case(
                        d=d,
                        compute=compute,
                        norm_name=norm_name,
                        norm_dtype=norm_dtype,
                        compiled=True,
                        w32=w32,
                        x32=x32,
                        reference=reference,
                        target_norm=args.a,
                        warmup=args.warmup,
                        min_bench_seconds=args.min_bench_seconds,
                        max_iters=args.max_iters,
                    )
                results.append(result)
                print(format_result(result), flush=True)

        del w32, x32, reference
        torch.cuda.empty_cache()

    return results


def format_float(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def format_result(result: BenchResult) -> str:
    compile_name = "compile" if result.compiled else "eager"
    if not result.supported:
        return (
            f"D={result.d:<5} {result.compute:<13} norm={result.norm:<4} "
            f"{compile_name:<7} SKIP {result.reason}"
        )
    return (
        f"D={result.d:<5} {result.compute:<13} norm={result.norm:<4} {compile_name:<7} "
        f"err={format_float(result.rel_l2_error, 3):>10} "
        f"ms={format_float(result.ms, 4):>9} "
        f"input={format_float(result.input_mib, 2):>8} MiB "
        f"case_peak={format_float(result.case_peak_mib, 2):>8} MiB "
        f"process_peak={format_float(result.peak_mib, 2):>8} MiB"
    )


def write_csv(results: list[BenchResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "d",
        "compute",
        "norm",
        "compiled",
        "supported",
        "reason",
        "rel_l2_error",
        "output_norm",
        "ms",
        "iters",
        "input_mib",
        "peak_mib",
        "case_peak_mib",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def summarize(results: list[BenchResult]) -> str:
    supported = [r for r in results if r.supported]
    skipped = [r for r in results if not r.supported]
    lines: list[str] = []

    lines.append("Environment")
    lines.append(f"torch={torch.__version__}")
    lines.append(f"device={torch.cuda.get_device_name(0)}")
    lines.append(f"capability={torch.cuda.get_device_capability(0)}")
    lines.append("")

    lines.append("Supported steady-state results")
    for result in supported:
        lines.append(format_result(result))
    lines.append("")

    if skipped:
        lines.append("Skipped / unsupported")
        seen: set[tuple[str, str]] = set()
        for result in skipped:
            key = (result.compute, result.reason)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{result.compute}: {result.reason}")
        lines.append("")

    counts_by_d = {d: sum(1 for r in supported if r.d == d) for d in {r.d for r in supported}}
    main_d = max(counts_by_d, key=lambda d: counts_by_d[d])
    main_dim_results = [r for r in supported if r.d == main_d]
    if main_dim_results:
        best_speed = min(main_dim_results, key=lambda r: r.ms if r.ms is not None else float("inf"))
        non_ground_truth = [
            r for r in main_dim_results
            if not (
                r.compute == GROUND_TRUTH_NAME
                and r.norm in {"same", "fp32"}
                and not r.compiled
                and r.rel_l2_error == 0
            )
        ]
        best_error = min(
            non_ground_truth or main_dim_results,
            key=lambda r: r.rel_l2_error if r.rel_l2_error is not None else float("inf"),
        )
        lines.append("Quick read")
        lines.append(
            "Fastest largest-D case: "
            f"{best_speed.compute}, norm={best_speed.norm}, "
            f"{'compile' if best_speed.compiled else 'eager'}, "
            f"{format_float(best_speed.ms, 4)} ms, "
            f"rel_err={format_float(best_speed.rel_l2_error, 3)}."
        )
        lines.append(
            "Most accurate largest-D non-ground-truth case: "
            f"{best_error.compute}, norm={best_error.norm}, "
            f"{'compile' if best_error.compiled else 'eager'}, "
            f"rel_err={format_float(best_error.rel_l2_error, 3)}."
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = run_experiment(args)
    summary = summarize(results)
    print("\n" + summary)
    args.out.write_text(summary + "\n")
    write_csv(results, args.csv)
    print(f"\nWrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
