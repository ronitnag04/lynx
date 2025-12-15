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

## References

- [Protocol Buffers Documentation](https://protobuf.dev/)
- ProtoAcc MICRO 2021 Paper: "A Hardware Accelerator for Protocol Buffers"
- HyperProtoBench: Representative protobuf workloads from Google's datacenter fleet
