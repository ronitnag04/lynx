# Analytical Model

This directory contains analytical tools for analyzing protobuf workloads and estimating performance characteristics.

## Protobuf Analyzer

A comprehensive Python tool (`protobuf_analyzer.py`) for analyzing Protocol Buffers message definitions from HyperProtoBench benchmarks. The analyzer extracts detailed metadata, statistics, and configurations from protobuf `.proto` and `.inc` files.

### Features

- **Message Analysis**: Extracts metadata for all messages including:
  - Message names and hierarchy
  - Field counts and field numbers
  - Nesting depth and parent-child relationships
  - Estimated message sizes

- **Field Analysis**: Detailed field information including:
  - Field types (int32, int64, string, bytes, message, enum, etc.)
  - Cardinality (optional, required, repeated)
  - Wire types (VARINT, I32, I64, LEN)
  - Field sizes in bytes extracted from .inc files

### Usage

```bash
# Basic usage - analyze all benchmarks and save to JSON
python3 protobuf_analyzer.py --hyperprotobench-path $HYPERPROTOBENCH

# Print summary to console
python3 protobuf_analyzer.py --summary --hyperprotobench-path $HYPERPROTOBENCH 

# Specify custom output file
python3 protobuf_analyzer.py --output my_analysis.json --hyperprotobench-path $HYPERPROTOBENCH
```

#### Command Line Arguments

- `--hyperprotobench-path`: Path to HyperProtoBench directory
- `--output`: Output JSON file path (default: `protobuf_analysis.json`)
- `--summary`: Print summary statistics to console
- `--verilator-bench-profile`: Cap the serialized-size computation to match what the Verilator-resident ProtoAcc bench actually exercises (skip repeated fields, cap per-field string/bytes at 1024 B, treat submessages at nesting depth ≥ 5 as zero). Use this when the downstream ML model should see features that line up with the measured cycle counts from `sims/verilator`.
- `--max-string-len N`: Cap string/bytes payload length (overrides the bench profile for this single knob).
- `--max-nested-depth N`: Messages at nesting depth ≥ N contribute zero (overrides bench profile).
- `--skip-repeated`: Treat all repeated fields as absent (overrides bench profile).

#### Workload profiles

The analyzer supports a ``WorkloadProfile`` that caps the size calculation to
reflect what a specific execution environment actually runs. The canonical
profile is ``WorkloadProfile.verilator_bench_default()``, which matches the
defaults used by
`generators/protoacc/software/verilator-bench/proto_to_accel.py` in the
Chipyard tree:

| Cap                 | Value  | Why                                                                                          |
|---------------------|--------|----------------------------------------------------------------------------------------------|
| ``max_string_len``  | 1024 B | Real HPB strings reach MBs; the Verilator bench caps at 1 KB for feasibility. |
| ``max_nested_depth``| 5      | Bench2's schema nests up to depth 13; Verilator bench tops out at 5.  |
| ``skip_repeated``   | True   | Generator skips `repeated` fields (about 3% of HPB fields).               |

When the Verilator-bench profile is used, `total_size_bytes` drops roughly
10–100× vs. the full-HPB figure (e.g. bench2: 29 MB → 22 KB), landing in the
same order of magnitude as the on-wire bytes measured by
`generators/protoacc/software/verilator-bench/benchmark_results.json`.

Typical two-output workflow — produce both the full-HPB feature set and the
Verilator-bench-scoped one side by side:

```bash
# Full HPB features (for reference / offline comparisons)
python3 protobuf_analyzer.py --output protobuf_analysis.json
python3 extract_features.py --input protobuf_analysis.json --output extracted_features_full.json

# Verilator-bench-scoped features (what the ML model should train on)
python3 protobuf_analyzer.py --verilator-bench-profile \
    --output protobuf_analysis_verilator_bench.json
python3 extract_features.py --input protobuf_analysis_verilator_bench.json \
    --output extracted_features.json
```

## Synthetic-bench analyzer (`analyze_synth.py`)

`protobuf_analyzer.analyze_hyperprotobench()` is tightly coupled to HPB's
`benchmark.inc` format — it regex-parses `<Msg>_Set_F1(...)` bodies whose
field-name regex stops at underscore, so it won't consume fleetbench's
`access_message<N>.cc` files (`set_f_0(...)` etc.). Rather than patch the HPB
path, `analyze_synth.py` is a sibling that:

1. Reuses `ProtobufAnalyzer` to parse each synthetic bench's `benchmark.proto`
   (produced by `verilator-bench/gen_synth_proto.py` — a concatenation of 5
   random fleetbench `Message<N>.proto` schemas). All structural features
   (`depth`, `total_fields`, `nested_message_count`) come from that shared
   parser, so synth and HPB rows are directly comparable.
2. Reads the sibling `runtime_lengths.json` (emitted by `fleetbench_runtime.py`
   when the synth bench is generated) and uses the median observed length
   per string/bytes field as that field's representative size.
3. Applies the same `WorkloadProfile` caps as the HPB side
   (`max_string_len=1024`, `max_nested_depth=5`, `skip_repeated=True`) so the
   `serialized_size_bytes` for a synth message reflects what `proto_to_accel.py`
   actually emits into the Verilator bench's cpp_obj layout.
4. Merges its results into an existing
   `protobuf_analysis_verilator_bench.json`, so running the HPB analyzer
   first and then `analyze_synth.py` yields one combined JSON that
   `extract_features.py` consumes without modification.

```bash
# HPB benches 0-5 (writes bench0..bench5 entries):
python3 protobuf_analyzer.py --verilator-bench-profile \
    --output protobuf_analysis_verilator_bench.json

# Synthetic benches 6+ (appends bench6..bench<N> entries in place):
python3 analyze_synth.py \
    --synth-root ../../verilator-bench/gen/synth \
    --in-json protobuf_analysis_verilator_bench.json \
    --out-json protobuf_analysis_verilator_bench.json

# Extract feature vectors for every bench in the merged JSON:
python3 extract_features.py \
    --input protobuf_analysis_verilator_bench.json \
    --output extracted_features.json
```

### Known caveats

- **Size is a median, not a sample.** `proto_to_accel.py` samples a random
  length per string occurrence; `analyze_synth.py` uses the median of the
  observed distribution. `total_size_bytes` is therefore a close proxy for
  the measured wire bytes, not an exact match. Same order of fidelity as the
  HPB analyzer (which also estimates from `.inc`).
- **Schema depth is unclipped.** `max_depth` reports the raw protobuf
  nesting depth (17 is common for fleetbench), while serialized size respects
  the 5-depth cap. This is intentional so the ML model sees "how ambitious
  the schema is" separately from "how much of it hits the pipe" — but if
  `max_depth` dominates feature importance, consider clipping it at the
  profile's cap before training.
- **Simple-name collisions.** When a synth bench combines two fleetbench
  `Message<K>` schemas that both have a nested `M1`, their runtime-length
  observations are merged and the first `M1` seen during the walk wins for
  sizing purposes. Since any realistic-looking length distribution works for
  string sampling, this is benign — but it means the `runtime_lengths.json`
  in the synth bench dir is the authoritative audit trail if you ever need
  to trace which Message<K> contributed which observation.

#### Output Format

The tool generates two types of output:

1. **Console Summary** (with `--summary` flag):
   - Per-benchmark statistics
   - Field type, cardinality, and wire type distributions
   - Top messages by field count
   - Nested message statistics

2. **JSON Report** (always generated):
   - Complete analysis data for all benchmarks
   - Detailed message and field information
   - Nested message hierarchies
   - All statistics and distributions

## Extract Features
The feature extraction script (`extract_features.py`) reads `protobuf_analysis.json` and extracts features for ML model training:

- **Input**: `protobuf_analysis.json` (generated by the analytical model analysis)
- **Output**: `extracted_features.json` containing feature vectors for each benchmark

**Extracted Features**:
- `total_size_bytes`: Total serialized size for each benchmark
- `num_messages`: Number of messages (including nested) in the benchmark
- **Frequency distributions** (10-bin normalized histograms):
  - `size_bytes_distribution`: Distribution of message serialized sizes
  - `total_fields_distribution`: Distribution of total fields per message
  - `nested_message_count_distribution`: Distribution of nested message counts
- **Min/Max/Avg values**:
  - `min_size_bytes`, `max_size_bytes`, `avg_size_bytes`
  - `min_total_fields`, `max_total_fields`, `avg_total_fields`
  - `min_nested_message_count`, `max_nested_message_count`, `avg_nested_message_count`
  - `min_depth`, `max_depth`, `avg_depth`
- **Counter lists** (15-bin):
  - `depth_counter_list`: Count of messages at each depth level

## Plot Features
The plotting script (`plot_features.py`) reads `extracted_features.json` and generates distribution plots across all benchmarks.

- **Input**: `extracted_features.json` (generated by `extract_features.py`)
- **Output**: PNG plots saved to the `plots/` directory (one per feature)

### Usage

```bash
# Plot all distribution features
python3 plot_features.py

# Plot a single feature
python3 plot_features.py --feature size_bytes_distribution

# Specify custom input and output paths
python3 plot_features.py --input extracted_features.json --output plots
```

#### Command Line Arguments

- `--feature`: Distribution feature to plot. One of `size_bytes_distribution`, `total_fields_distribution`, `nested_message_count_distribution`, `depth_counter_list`. Defaults to plotting all features.
- `--input`: Path to `extracted_features.json` (default: same directory as script)
- `--output`: Output directory for plots (default: `plots`)

## References

- [Protocol Buffers Documentation](https://protobuf.dev/)
- ProtoAcc MICRO 2021 Paper: "A Hardware Accelerator for Protocol Buffers"
- HyperProtoBench: Representative protobuf workloads from Google's datacenter fleet
