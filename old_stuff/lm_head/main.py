import argparse
import json
import mmap
import re
import struct
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_MODEL_IDS = [
    "moonshotai/Kimi-K2.6",
    "deepseek-ai/DeepSeek-V4-Flash-Base",
    "deepseek-ai/DeepSeek-V4-Flash",
]
OUTPUT_HEAD_TENSOR_NAMES = [
    "lm_head.weight",
    "language_model.lm_head.weight",
    "head.weight",
]
OUTPUT_DIR = Path(__file__).parent
RANDOM_INPUT_DIM = 1024
RANDOM_OUTPUT_DIM = 32768
RANDOM_MEAN = 0.0
RANDOM_STD = 1.0

SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def model_slug(model_id):
    model_name = model_id.split("/", maxsplit=1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", model_name).strip("_")


def default_output_paths(model_id, output_dir):
    slug = model_slug(model_id)
    return (
        output_dir / f"{slug}_lm_head_l2_norms.png",
        output_dir / f"{slug}_lm_head_l2_norms_histogram.png",
    )


def random_output_paths(output_dir):
    stem = (
        f"random_gaussian_{RANDOM_INPUT_DIM}_to_{RANDOM_OUTPUT_DIM}"
        f"_mean_{RANDOM_MEAN:g}_std_{RANDOM_STD:g}"
    )
    return (
        output_dir / f"{stem}_lm_head_l2_norms.png",
        output_dir / f"{stem}_lm_head_l2_norms_histogram.png",
    )


def find_output_head_file(model_id):
    files = list_repo_files(model_id)
    index_files = [
        name
        for name in files
        if name.endswith(".safetensors.index.json")
    ]

    for index_file in index_files:
        index_path = hf_hub_download(model_id, index_file)
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        for tensor_name in OUTPUT_HEAD_TENSOR_NAMES:
            if tensor_name in weight_map:
                return tensor_name, weight_map[tensor_name]

    if "model.safetensors" in files:
        return OUTPUT_HEAD_TENSOR_NAMES[0], "model.safetensors"

    safetensor_files = sorted(name for name in files if name.endswith(".safetensors"))
    if safetensor_files:
        raise RuntimeError(
            "Could not find an output head tensor in a safetensors index. "
            f"Tried: {OUTPUT_HEAD_TENSOR_NAMES}. "
            f"Available safetensors files: {safetensor_files}"
        )

    raise RuntimeError(f"No safetensors weights found for {model_id}.")


def read_safetensors_metadata(path):
    with open(path, "rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise RuntimeError(f"{path} is too small to be a safetensors file.")

        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = json.loads(handle.read(header_size))

    return header_size, header


def load_tensor_from_safetensors(path, tensor_name):
    header_size, header = read_safetensors_metadata(path)
    if tensor_name not in header:
        available = sorted(name for name in header if name != "__metadata__")
        raise RuntimeError(
            f"{tensor_name!r} was not found in {path}. "
            f"Available tensors: {available}"
        )

    metadata = header[tensor_name]
    dtype = SAFETENSORS_DTYPES.get(metadata["dtype"])
    if dtype is None:
        raise RuntimeError(f"Unsupported dtype in safetensors file: {metadata['dtype']}")

    start, end = metadata["data_offsets"]
    shape = tuple(metadata["shape"])
    data_start = 8 + header_size

    handle = open(path, "rb")
    mapped_file = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    tensor_bytes = memoryview(mapped_file)[data_start + start:data_start + end]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        tensor = torch.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)

    return tensor, mapped_file, handle


def calculate_row_l2_norms(weight, chunk_size):
    norms = []
    for start in range(0, weight.shape[0], chunk_size):
        rows = weight[start:start + chunk_size].to(torch.float32)
        norms.append(torch.linalg.vector_norm(rows, ord=2, dim=1).cpu())
    return torch.cat(norms)


def plot_l2_norms(norms, model_id, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(norms.numpy(), linewidth=0.45)
    ax.set_title(f"{model_id} lm_head row L2 norms")
    ax.set_xlabel("Output neuron / vocabulary row index")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_l2_norm_histogram(norms, model_id, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(norms.numpy(), bins=100, color="#2f6f8f", edgecolor="white", linewidth=0.25)
    ax.set_title(f"{model_id} lm_head row L2 norm histogram")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Output neuron count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_model(model_id, output_path, histogram_output_path, chunk_size):
    tensor_name, shard_name = find_output_head_file(model_id)
    shard_path = hf_hub_download(model_id, shard_name)
    weight, mapped_file, handle = load_tensor_from_safetensors(shard_path, tensor_name)
    try:
        norms = calculate_row_l2_norms(weight, chunk_size)
    finally:
        del weight
        mapped_file.close()
        handle.close()

    plot_l2_norms(norms, model_id, output_path)
    plot_l2_norm_histogram(norms, model_id, histogram_output_path)

    print(f"Model: {model_id}")
    print(f"Loaded tensor: {tensor_name}")
    print(f"Weight shard: {shard_path}")
    print(f"Tensor rows: {norms.numel()}")
    print(f"Min L2 norm: {norms.min().item():.6f}")
    print(f"Max L2 norm: {norms.max().item():.6f}")
    print(f"Mean L2 norm: {norms.mean().item():.6f}")
    print(f"Saved plot: {output_path}")
    print(f"Saved histogram: {histogram_output_path}")


def process_random_lm_head(output_path, histogram_output_path, seed):
    if seed is not None:
        torch.manual_seed(seed)

    weight = torch.empty((RANDOM_OUTPUT_DIM, RANDOM_INPUT_DIM), dtype=torch.float32)
    weight.normal_(mean=RANDOM_MEAN, std=RANDOM_STD)
    norms = torch.linalg.vector_norm(weight, ord=2, dim=1).cpu()

    label = (
        f"Random Gaussian lm_head "
        f"({RANDOM_INPUT_DIM} -> {RANDOM_OUTPUT_DIM}, "
        f"mean={RANDOM_MEAN:g}, std={RANDOM_STD:g})"
    )
    plot_l2_norms(norms, label, output_path)
    plot_l2_norm_histogram(norms, label, histogram_output_path)

    print(label)
    print(f"Weight shape: [{RANDOM_OUTPUT_DIM}, {RANDOM_INPUT_DIM}]")
    print(f"Seed: {seed if seed is not None else 'not set'}")
    print(f"Tensor rows: {norms.numel()}")
    print(f"Min L2 norm: {norms.min().item():.6f}")
    print(f"Max L2 norm: {norms.max().item():.6f}")
    print(f"Mean L2 norm: {norms.mean().item():.6f}")
    print(f"Saved plot: {output_path}")
    print(f"Saved histogram: {histogram_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a random Gaussian lm_head, or download one or more "
            "models' lm_head weights, compute "
            "per-output-neuron L2 norms, and save line/histogram plots."
        )
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Process Hugging Face models instead of the random Gaussian lm_head.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help=(
            "Hugging Face model ID to process. May be repeated. Defaults to "
            "the requested Kimi and DeepSeek models."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for the Gaussian lm_head.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for default plot paths. Defaults to {OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Line plot path. Only valid when processing one model.",
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        help="Histogram plot path. Only valid when processing one model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of lm_head rows to process at a time.",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    if not args.hf:
        output_path, histogram_output_path = random_output_paths(args.output_dir)
        if args.output:
            output_path = args.output
        if args.histogram_output:
            histogram_output_path = args.histogram_output

        process_random_lm_head(output_path, histogram_output_path, args.seed)
        return

    model_ids = args.model_ids or DEFAULT_MODEL_IDS
    if len(model_ids) > 1 and (args.output or args.histogram_output):
        raise ValueError("--output and --histogram-output are only valid for one model.")

    for index, model_id in enumerate(model_ids):
        if index:
            print()

        output_path, histogram_output_path = default_output_paths(model_id, args.output_dir)
        if args.output:
            output_path = args.output
        if args.histogram_output:
            histogram_output_path = args.histogram_output

        process_model(model_id, output_path, histogram_output_path, args.chunk_size)


if __name__ == "__main__":
    main()
