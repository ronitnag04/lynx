# Analytical Model

This directory contains analytical tools for analyzing protobuf workloads and estimating performance characteristics.

## Protobuf Analyzer

A comprehensive Python tool (`protobuf_analyzer.py`) for analyzing Protocol Buffers message definitions from HyperProtoBench benchmarks. The analyzer extracts detailed metadata, statistics, and configurations from protobuf `.proto` files.

### Features

- **Message Analysis**: Extracts metadata for all messages including:
  - Message names and hierarchy
  - Field counts and field numbers
  - Nesting depth and parent-child relationships
  - Estimated message sizes

- **Field Analysis**: Detailed field information including:
  - Field types (int32, int64, string, bytes, message, enum, etc.)
  - Cardinality (optional, required, repeated)
  - Wire types (VARINT, FIXED32, FIXED64, LENGTH_DELIMITED)
  - Field numbers and estimated sizes

- **Structure Analysis**:
  - Recursive parsing of nested messages
  - Enum detection (top-level and nested)
  - Maximum nesting depth calculation
  - Repeated field identification

- **Statistics Generation**:
  - Field type distribution across all messages
  - Cardinality distribution (optional/required/repeated)
  - Wire type distribution
  - Per-benchmark summaries


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

#### Notes

- The analyzer uses regex-based parsing and handles nested messages recursively
- Estimated sizes are approximations based on fixed-size types only
- Variable-length types (string, bytes, messages) are marked with `null` estimated size
- The tool correctly handles proto2 syntax with optional/required/repeated fields