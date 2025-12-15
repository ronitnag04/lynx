"""
Generate predictions for every parameter combination using trained checkpoints.

This script:
- Loads serializer and deserializer checkpoints from the local `checkpoints/` dir
- Reads benchmark features and parameter sweep definitions
- Builds inputs in the same layout as `generate_dataset.py`
- Streams predictions to JSONL files under `predictions/`
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
import torch_xla.core.xla_model as xm

from model import LynxMLModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lynx MLP predictions over full parameter sweep.")
    parser.add_argument(
        "--parameter-sweep",
        type=Path,
        default=Path(__file__).parent.parent / "sample_protoacc_model" / "parameter_sweep.json",
        help="Path to parameter_sweep.json.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path(__file__).parent.parent / "analytical_model" / "extracted_features.json",
        help="Path to extracted_features.json containing benchmark features.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).parent / "checkpoints",
        help="Directory containing serializer_checkpoint.pt and deserializer_checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "predictions",
        help="Directory to write JSONL prediction files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of samples to evaluate per forward pass.",
    )
    return parser.parse_args()


def flatten_benchmark_features(benchmark_features: Dict) -> torch.Tensor:
    features: List[float] = []
    features.append(benchmark_features["total_size_bytes"])
    features.append(benchmark_features["num_messages"])
    features.append(benchmark_features["min_size_bytes"])
    features.append(benchmark_features["max_size_bytes"])
    features.append(benchmark_features["avg_size_bytes"])
    features.append(benchmark_features["min_total_fields"])
    features.append(benchmark_features["max_total_fields"])
    features.append(benchmark_features["avg_total_fields"])
    features.append(benchmark_features["min_nested_message_count"])
    features.append(benchmark_features["max_nested_message_count"])
    features.append(benchmark_features["avg_nested_message_count"])
    features.append(benchmark_features["min_depth"])
    features.append(benchmark_features["max_depth"])
    features.append(benchmark_features["avg_depth"])

    features.extend(benchmark_features["size_bytes_distribution"])
    features.extend(benchmark_features["total_fields_distribution"])
    features.extend(benchmark_features["nested_message_count_distribution"])
    features.extend(benchmark_features["depth_counter_list"])
    return torch.tensor(features, dtype=torch.float32)


def flatten_parameter_vector(parameter_vector: Dict, param_order: Iterable[str]) -> torch.Tensor:
    return torch.tensor([parameter_vector[name] for name in param_order], dtype=torch.float32)


def parameter_combinations(parameter_sweep: Dict[str, List]) -> Iterator[Dict[str, float]]:
    names = list(parameter_sweep.keys())
    values = [parameter_sweep[name] for name in names]
    for combo in itertools.product(*values):
        yield {name: value for name, value in zip(names, combo)}


def batched(iterable: Iterator[Dict[str, float]], batch_size: int) -> Iterator[List[Dict[str, float]]]:
    batch: List[Dict[str, float]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_model(op_type: str, checkpoint_dir: Path, input_size: int, device: torch.device) -> LynxMLModel:
    checkpoint_path = checkpoint_dir / f"{op_type}_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = LynxMLModel(input_size=input_size, hidden_dims=(64, 32, 16), output_size=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_predictions_for_op(
    op_type: str,
    parameter_sweep: Dict[str, List],
    benchmark_features: Dict[str, Dict],
    output_dir: Path,
    checkpoint_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    param_order = list(parameter_sweep.keys())
    bench_names = list(benchmark_features.keys())

    # Determine input dimensionality (param vector + flattened benchmark features).
    sample_bench = bench_names[0]
    bench_flat = flatten_benchmark_features(benchmark_features[sample_bench])
    input_size = len(param_order) + bench_flat.numel()

    model = load_model(op_type, checkpoint_dir, input_size, device)
    total_param_combos = math.prod(len(values) for values in parameter_sweep.values())
    total_samples = total_param_combos * len(bench_names)
    processed_samples = 0

    output_path = output_dir / f"{op_type}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for bench_idx, bench_name in enumerate(bench_names, start=1):
            bench_tensor = flatten_benchmark_features(benchmark_features[bench_name]).to(device)
            combos_iter = parameter_combinations(parameter_sweep)
            bench_processed = 0
            bench_total = total_param_combos
            print(f"[{op_type}] Benchmark {bench_idx}/{len(bench_names)}: {bench_name} ({bench_total} combos)")

            for batch in batched(combos_iter, batch_size):
                inputs = []
                for param_vec in batch:
                    param_tensor = flatten_parameter_vector(param_vec, param_order).to(device)
                    feature_tensor = torch.cat([param_tensor, bench_tensor], dim=0)
                    inputs.append(feature_tensor)

                input_batch = torch.stack(inputs)
                with torch.no_grad():
                    preds = model(input_batch).squeeze(-1)
                    xm.mark_step()
                    preds = preds.to("cpu").tolist()

                for param_vec, pred in zip(batch, preds):
                    record = {
                        "op_type": op_type,
                        "benchmark": bench_name,
                        "parameters": param_vec,
                        "prediction": float(pred),
                    }
                    f.write(json.dumps(record) + "\n")

                bench_processed += len(batch)
                processed_samples += len(batch)
                # Periodic progress update (per ~5% of benchmark or on completion)
                progress_step = max(1, bench_total // 20)
                if bench_processed % progress_step == 0 or bench_processed == bench_total:
                    bench_pct = (bench_processed / bench_total) * 100
                    total_pct = (processed_samples / total_samples) * 100
                    print(
                        f"[{op_type}] {bench_name}: {bench_processed}/{bench_total} "
                        f"({bench_pct:.1f}%) | overall {processed_samples}/{total_samples} ({total_pct:.1f}%)"
                    )

    print(
        f"[{op_type}] Wrote {total_samples} predictions "
        f"({total_param_combos} param combos x {len(bench_names)} benchmarks) -> {output_path}"
    )


def main() -> None:
    args = parse_args()
    # Enforce XLA/Neuron usage; CPU inference is disallowed.
    device = 'cpu'
    print(f"Using device: {device}")

    with args.parameter_sweep.open("r", encoding="utf-8") as f:
        sweep = json.load(f)
    with args.features_file.open("r", encoding="utf-8") as f:
        benchmark_features = json.load(f)

    for op_type in ["deserializer", "serializer"]:
        start_time = time.time()
        run_predictions_for_op(
            op_type=op_type,
            parameter_sweep=sweep[op_type],
            benchmark_features=benchmark_features,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoints_dir,
            batch_size=args.batch_size,
            device=device,
        )
        end_time = time.time()
        print(f"[{op_type}] Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

