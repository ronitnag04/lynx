import os
import re
from typing import Dict, List, Tuple

import numpy as np
import itertools


def find_log_files(root: str) -> List[Tuple[str, int]]:
    """Find all benchmark .log files and extract data length from filename.

    Only files matching 'sha3-rocc-benchmark-<len>-k[0|1].log' are used, since
    the length is encoded there and these contain detailed performance counters.
    """
    log_entries: List[Tuple[str, int]] = []
    pattern = re.compile(r"sha3-rocc-benchmark-(\d+)-k[0|1]\.log$")

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            match = pattern.match(fname)
            if match is None:
                continue
            length = int(match.group(1))
            full_path = os.path.join(dirpath, fname)
            log_entries.append((full_path, length))

    return sorted(log_entries)


def parse_log(path: str) -> List[Dict[str, float]]:
    """Parse a single .log file into a list of per-test-vector metrics dicts.

    Each dict maps metric name -> value for one "Running test vector" block.
    """
    metrics_per_vector: List[Dict[str, float]] = []
    current: Dict[str, float] = {}
    in_block = False

    metric_re = re.compile(r"^\s*([^:]+?)\s*:\s*(-?\d+)\s*$")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("Running test vector"):
                # Start of a new test vector block
                if in_block and current:
                    metrics_per_vector.append(current)
                current = {}
                in_block = True
                continue

            if in_block and line.startswith("Test vector") and "passed" in line:
                # End of current test vector block
                if current:
                    metrics_per_vector.append(current)
                current = {}
                in_block = False
                continue

            if not in_block:
                continue

            m = metric_re.match(line)
            if m:
                name = m.group(1).strip()
                try:
                    value = float(m.group(2))
                except ValueError:
                    continue
                current[name] = value

    # In case file ended without a closing "Test vector" line
    if in_block and current:
        metrics_per_vector.append(current)

    return metrics_per_vector


def aggregate_metrics(vectors: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each metric across all test vectors in a log file."""
    if not vectors:
        return {}

    sums: Dict[str, float] = {}
    count = len(vectors)

    for vec in vectors:
        for k, v in vec.items():
            sums[k] = sums.get(k, 0.0) + v

    return {k: v / count for k, v in sums.items()}


def parse_config_from_path(path: str) -> Dict[str, float]:
    """Extract SHA3 config features from the parent directory name.

    Expected pattern (inside the directory name):
      Sha3RocketW64S(NUM_STAGES)Fm(IS_FAST_MEM)Bs(IS_BUFFER_SRAM)K(IS_KECCAT)
    """
    dirname = os.path.basename(os.path.dirname(path))

    # Look for the config substring anywhere in the directory name
    m = re.search(r"Sha3RocketW64S(\d+)Fm(\d+)Bs(\d+)K(\d+)", dirname)
    if not m:
        # If the pattern is not found, return an empty dict so we don't break.
        raise ValueError(f"Could not find config pattern in directory name: {dirname}")

    num_stages, is_fast_mem, is_buffer_sram, is_keccat = map(float, m.groups())

    return {
        "num_stages": num_stages,
        "is_fast_mem": is_fast_mem,
        "is_buffer_sram": is_buffer_sram,
        "is_keccat": is_keccat,
    }


def build_dataset(root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature and label arrays from all matching log files.

    Features: averaged performance counters (excluding 'cycles')
              + data length
              + config features (num_stages, is_fast_mem, is_buffer_sram, is_keccat).
    Labels:   averaged 'cycles'.
    """
    log_files = find_log_files(root)
    for path in log_files:
        print(f"Processing log file: {path}")

    print(f"Found {len(log_files)} log files")
    feature_dicts: List[Dict[str, float]] = []
    labels: List[float] = []
    all_feature_names = set()

    for path, data_len in log_files:
        vectors = parse_log(path)
        agg = aggregate_metrics(vectors)
        if not agg:
            # Skip files we couldn't parse
            continue

        if "cycles" not in agg:
            # Without cycles there is no label; skip
            continue

        avg_cycles = agg.pop("cycles")

        # Performance counters (excluding 'cycles')
        feat: Dict[str, float] = {k: v for k, v in agg.items()}

        # Data length from filename
        feat["data_length"] = float(data_len)

        # Config features from directory name
        config_feats = parse_config_from_path(path)
        feat.update(config_feats)

        feature_dicts.append(feat)
        labels.append(float(avg_cycles))
        all_feature_names.update(feat.keys())

    if not feature_dicts:
        raise RuntimeError("No valid benchmark log files were parsed. Check the log format and paths.")

    # Create a consistent feature order:
    #   - all counters alphabetically
    #   - then config features
    #   - finally data_length
    special_order = ["num_stages", "is_fast_mem", "is_buffer_sram", "is_keccat", "data_length"]
    feature_names = sorted(name for name in all_feature_names if name not in special_order)
    for name in special_order:
        if name in all_feature_names:
            feature_names.append(name)

    print(feature_names)

    print(feature_dicts[0])
    print(labels[0])
    print(log_files[0][0])


    X = np.array(
        [
            [fd.get(name, 0.0) for name in feature_names]
            for fd in feature_dicts
        ],
        dtype=np.float32,
    )
    y = np.array(labels, dtype=np.float32)[:, np.newaxis]

    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/test split with the given ratio.

    Ensures that, when there are at least 2 samples, both splits are non-empty.
    """
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Empty dataset; cannot perform train/test split.")

    if n_samples == 1:
        # Degenerate case: everything is 'train', empty test
        return X, y, np.empty((0, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)

    rng = np.random.RandomState()
    indices = rng.permutation(n_samples)

    n_train = int(round(train_ratio * n_samples))
    # Make sure both sets are non-empty
    n_train = max(1, min(n_train, n_samples - 1))

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = script_dir  # process all .log files under sha_successful_workloads

    X, y = build_dataset(root)
    train_X, train_y, test_X, test_y = train_test_split(X, y, train_ratio=0.75)

    data_dir = os.path.join(script_dir, "sha3_dataset")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "train_features.npy"), train_X)
    np.save(os.path.join(data_dir, "train_labels.npy"), train_y)
    np.save(os.path.join(data_dir, "test_features.npy"), test_X)
    np.save(os.path.join(data_dir, "test_labels.npy"), test_y)

    print(f"Saved train/test splits with shapes: ")
    print(f"  train_features: {train_X.shape}, train_labels: {train_y.shape}")
    print(f"  test_features:  {test_X.shape}, test_labels:  {test_y.shape}")


if __name__ == "__main__":
    main()
