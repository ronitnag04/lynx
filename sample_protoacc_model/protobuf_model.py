import os
import numpy
import json
import itertools
import random
import math
from pathlib import Path
from typing import Dict, List, Any

default_benchmark_results = {
    "deserializer": {
        "bench0": {"throughput": 38},
        "bench1": {"throughput": 28},
        "bench2": {"throughput": 38},
        "bench3": {"throughput": 39},
        "bench4": {"throughput": 39},
        "bench5": {"throughput": 39},
    },
    "serializer": {
        "bench0": {"throughput": 75},
        "bench1": {"throughput": 85},
        "bench2": {"throughput": 38},
        "bench3": {"throughput": 95},
        "bench4": {"throughput": 100},
        "bench5": {"throughput": 40},
    }
}

# Load benchmark features from extracted features JSON
script_dir = Path(__file__).parent
features_path = script_dir.parent / "analytical_model" / "extracted_features.json"
if features_path.exists():
    with open(features_path, "r") as f:
        default_benchmark_features = json.load(f)
else:
    # Fallback: empty dict if file doesn't exist
    default_benchmark_features = {}

default_parameters = {
    "deserializer": {
        "top_descriptor_reqs": 4,
        "top_memloader_reqs": 64,
        "cr_rocc_commands": 2,
        "dth_l1_reqs": 4,
        "dth_fd_reqs": 4,
        "dth_fd_resps": 4,
        "fw_l1_reqs": 4,
        "ml_buf_info_q": 16,
        "ml_load_info_q": 256,
    },
    "serializer": {
        "top_num_field_handlers": 6,
        "cr_rocc_commands": 2,
        "dth_hasbits_reqs": 4,
        "dth_descriptor_reqs": 4,
        "dth_reg_resps": 10,
        "dth_reqs_meta": 4,
        "dth_fh_outputs": 4,
        "mw_write_input": 4,
        "mw_write_inject": 4,
        "mw_write_ptrs": 10,
    }
}

# Load parameter sweep from parameter_sweep.json
sweep_path = script_dir / "parameter_sweep.json"
with open(sweep_path, "r") as f:
    parameter_sweep = json.load(f)
    serializer_parameter_sweep = parameter_sweep["serializer"]
    deserializer_parameter_sweep = parameter_sweep["deserializer"]

def estimated_throughput(parameters: Dict, benchmark_name: str = None) -> float:
    """
    Estimate the throughput of the protobuf model given the parameters.
    
    Parameters should contain:
    - Either "deserializer" or "serializer" key with parameter values
    
    Args:
        parameters: Dictionary containing deserializer/serializer parameters
        benchmark_name: Name of the benchmark (e.g., "bench0", "bench1", etc.)
                       If None, uses "geomean" for default throughput
    """
    # Determine if this is deserializer or serializer
    if "deserializer" in parameters:
        op_type = "deserializer"
        op_params = parameters["deserializer"]
        default_params = default_parameters["deserializer"]
        # Use benchmark-specific throughput if available, otherwise geomean
        if benchmark_name and benchmark_name in default_benchmark_results["deserializer"]:
            default_throughput = default_benchmark_results["deserializer"][benchmark_name]["throughput"]
        else:
            raise ValueError(f"Benchmark {benchmark_name} not found in default benchmark results")
    elif "serializer" in parameters:
        op_type = "serializer"
        op_params = parameters["serializer"]
        default_params = default_parameters["serializer"]
        if benchmark_name and benchmark_name in default_benchmark_results["serializer"]:
            default_throughput = default_benchmark_results["serializer"][benchmark_name]["throughput"]
        else:
            raise ValueError(f"Benchmark {benchmark_name} not found in default benchmark results")
    else:
        raise ValueError("Parameters must contain either 'deserializer' or 'serializer' key")
    
    # Get benchmark features from default_benchmark_features
    if benchmark_name and benchmark_name in default_benchmark_features:
        benchmark_features = default_benchmark_features[benchmark_name]
        
        # Extract aggregate statistics
        total_size_bytes = benchmark_features["total_size_bytes"]
        num_messages = benchmark_features["num_messages"]
        min_size_bytes = benchmark_features["min_size_bytes"]
        max_size_bytes = benchmark_features["max_size_bytes"]
        avg_size_bytes = benchmark_features["avg_size_bytes"]
        min_total_fields = benchmark_features["min_total_fields"]
        max_total_fields = benchmark_features["max_total_fields"]
        avg_total_fields = benchmark_features["avg_total_fields"]
        min_nested_message_count = benchmark_features["min_nested_message_count"]
        max_nested_message_count = benchmark_features["max_nested_message_count"]
        avg_nested_message_count = benchmark_features["avg_nested_message_count"]
        min_depth = benchmark_features["min_depth"]
        max_depth = benchmark_features["max_depth"]
        avg_depth = benchmark_features["avg_depth"]
    else:
        raise ValueError(f"Benchmark {benchmark_name} not found in default benchmark features")
    
    estimated = float(default_throughput)
    
    if op_type == "deserializer":
        # top_descriptor_reqs: Top descriptor request queue
        default_val = default_params["top_descriptor_reqs"]
        new_val = op_params["top_descriptor_reqs"]
        if new_val < default_val:
            # Impact is proportional to nested message count
            total_nested_messages = avg_nested_message_count * num_messages
            # Normalize by comparing total nested messages to max possible (num_messages * max_nested_message_count)
            max_possible_nested = num_messages * max(max_nested_message_count, 1.0)
            nested_stress = min(total_nested_messages / max(max_possible_nested, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * nested_stress * 0.3
            estimated *= (1 - penalty)
        elif new_val > default_val:
            # Benefit is smaller, diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.1, 0.1)
            estimated *= (1 + benefit)
        
        # top_memloader_reqs: Controls outstanding memory loader requests
        # Lower values bottleneck with large messages
        default_val = default_params["top_memloader_reqs"]
        new_val = op_params["top_memloader_reqs"]
        if new_val < default_val:
            # Impact is proportional to message size (normalize to max size in benchmark)
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            penalty = (default_val - new_val) / default_val * size_stress * 0.2
            estimated *= (1 - penalty)
        elif new_val > default_val:
            benefit = min((new_val - default_val) / default_val * 0.05, 0.05)
            estimated *= (1 + benefit)
        
        # dth_l1_reqs: Descriptor table handler L1 request queue
        # Lower values bottleneck with many descriptor lookups (nested messages)
        default_val = default_params["dth_l1_reqs"]
        new_val = op_params["dth_l1_reqs"]
        if new_val < default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            penalty = (default_val - new_val) / default_val * nested_stress * 0.15
            estimated *= (1 - penalty)
        elif new_val > default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            benefit = min((new_val - default_val) / default_val * nested_stress * 0.05, 0.05)
            estimated *= (1 + benefit)
        
        # dth_fd_reqs: Field destination request queue
        # Lower values bottleneck with many fields
        default_val = default_params["dth_fd_reqs"]
        new_val = op_params["dth_fd_reqs"]
        if new_val < default_val:
            # Normalize by typical field count (higher fields = more stress)
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)  # Normalize to typical benchmark
            penalty = (default_val - new_val) / default_val * fields_stress * 0.1
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.03, 0.03)
            estimated *= (1 + benefit)
        
        # dth_fd_resps: Field destination response queue
        # Lower values bottleneck with many fields
        default_val = default_params["dth_fd_resps"]
        new_val = op_params["dth_fd_resps"]
        if new_val < default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.1
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.03, 0.03)
            estimated *= (1 + benefit)
        
        # fw_l1_reqs: Fixed writer L1 request queue
        # Lower values bottleneck with many fixed-width fields
        default_val = default_params["fw_l1_reqs"]
        new_val = op_params["fw_l1_reqs"]
        if new_val < default_val:
            # Fixed writer is used for fixed-width fields, approximate by total fields
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.08
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.02, 0.02)
            estimated *= (1 + benefit)
        
        # ml_buf_info_q: Memloader buffer info queue
        # Lower values bottleneck with many messages
        default_val = default_params["ml_buf_info_q"]
        new_val = op_params["ml_buf_info_q"]
        if new_val < default_val:
            # Impact depends on number of messages (using benchmark's message count as baseline)
            # Since we're comparing within the same benchmark, use 1.0 for normalization
            messages_ratio = 1.0
            penalty = (default_val - new_val) / default_val * messages_ratio * 0.1
            estimated *= (1 - penalty)
        elif new_val > default_val:
            messages_ratio = 1.0
            benefit = min((new_val - default_val) / default_val * messages_ratio * 0.03, 0.03)
            estimated *= (1 + benefit)
        
        # ml_load_info_q: Memloader load info queue
        # Lower values bottleneck with large messages requiring many loads
        default_val = default_params["ml_load_info_q"]
        new_val = op_params["ml_load_info_q"]
        if new_val < default_val:
            # Impact depends on message size (more loads for larger messages)
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            penalty = (default_val - new_val) / default_val * size_stress * 0.12
            estimated *= (1 - penalty)
        elif new_val > default_val:
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            benefit = min((new_val - default_val) / default_val * size_stress * 0.04, 0.04)
            estimated *= (1 + benefit)
        
        # cr_rocc_commands: Command router ROCC commands queue
        # Lower values bottleneck with high message throughput
        default_val = default_params["cr_rocc_commands"]
        new_val = op_params["cr_rocc_commands"]
        if new_val < default_val:
            # Impact is generally small but can affect high-throughput scenarios
            penalty = (default_val - new_val) / default_val * 0.05
            estimated *= (1 - penalty)
        elif new_val > default_val:
            # Benefit is small, diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.02, 0.02)
            estimated *= (1 + benefit)
    
    else:  # serializer
        # SERIALIZER PARAMETERS
        
        # top_num_field_handlers: Number of parallel field handlers
        # Lower values bottleneck with many fields (most critical for serializer)
        default_val = default_params["top_num_field_handlers"]
        new_val = op_params["top_num_field_handlers"]
        if new_val < default_val:
            # Impact is proportional to number of fields (field handlers process fields in parallel)
            total_fields = avg_total_fields * num_messages
            fields_ratio = total_fields / max(num_messages * default_val, 1)
            # More severe penalty when fewer handlers than needed
            penalty = (default_val - new_val) / default_val * min(fields_ratio, 1.0) * 0.4
            estimated *= (1 - penalty)
        elif new_val > default_val:
            # Benefit from more parallelism, but diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.15, 0.2)
            estimated *= (1 + benefit)
        
        # dth_hasbits_reqs: Hasbits request queue
        # Lower values bottleneck with messages that have many optional fields
        default_val = default_params["dth_hasbits_reqs"]
        new_val = op_params["dth_hasbits_reqs"]
        if new_val < default_val:
            # Hasbits are used for optional fields, approximate by total fields
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.12
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.04, 0.04)
            estimated *= (1 + benefit)
        
        # dth_descriptor_reqs: Descriptor request queue
        # Lower values bottleneck with nested messages
        default_val = default_params["dth_descriptor_reqs"]
        new_val = op_params["dth_descriptor_reqs"]
        if new_val < default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            penalty = (default_val - new_val) / default_val * nested_stress * 0.15
            estimated *= (1 - penalty)
        elif new_val > default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            benefit = min((new_val - default_val) / default_val * nested_stress * 0.05, 0.05)
            estimated *= (1 + benefit)
        
        # dth_reg_resps: Register response queue
        # Lower values bottleneck with many field operations
        default_val = default_params["dth_reg_resps"]
        new_val = op_params["dth_reg_resps"]
        if new_val < default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.1
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.03, 0.03)
            estimated *= (1 + benefit)
        
        # dth_reqs_meta: Request metadata queue
        # Lower values bottleneck with many requests
        default_val = default_params["dth_reqs_meta"]
        new_val = op_params["dth_reqs_meta"]
        if new_val < default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.08
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.02, 0.02)
            estimated *= (1 + benefit)
        
        # dth_fh_outputs: Field handler outputs queue
        # Lower values bottleneck with many field handlers producing output
        default_val = default_params["dth_fh_outputs"]
        new_val = op_params["dth_fh_outputs"]
        if new_val < default_val:
            # Impact depends on number of field handlers and fields
            handlers_ratio = op_params["top_num_field_handlers"] / default_params["top_num_field_handlers"]
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * handlers_ratio * fields_stress * 0.1
            estimated *= (1 - penalty)
        elif new_val > default_val:
            handlers_ratio = op_params["top_num_field_handlers"] / default_params["top_num_field_handlers"]
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * handlers_ratio * fields_stress * 0.03, 0.03)
            estimated *= (1 + benefit)
        
        # mw_write_input: Memwriter write input queue
        # Lower values bottleneck with high write throughput
        default_val = default_params["mw_write_input"]
        new_val = op_params["mw_write_input"]
        if new_val < default_val:
            # Impact depends on message size (more writes for larger messages)
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            penalty = (default_val - new_val) / default_val * size_stress * 0.12
            estimated *= (1 - penalty)
        elif new_val > default_val:
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            benefit = min((new_val - default_val) / default_val * size_stress * 0.04, 0.04)
            estimated *= (1 + benefit)
        
        # mw_write_inject: Memwriter write inject queue
        # Lower values bottleneck with nested messages (size injection)
        default_val = default_params["mw_write_inject"]
        new_val = op_params["mw_write_inject"]
        if new_val < default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            penalty = (default_val - new_val) / default_val * nested_stress * 0.1
            estimated *= (1 - penalty)
        elif new_val > default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            benefit = min((new_val - default_val) / default_val * nested_stress * 0.03, 0.03)
            estimated *= (1 + benefit)
        
        # mw_write_ptrs: Memwriter write pointers queue
        # Lower values bottleneck with string/bytes fields (pointer writes)
        default_val = default_params["mw_write_ptrs"]
        new_val = op_params["mw_write_ptrs"]
        if new_val < default_val:
            # Approximate by total fields (some fields may be strings/bytes)
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.08
            estimated *= (1 - penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.02, 0.02)
            estimated *= (1 + benefit)
        
        # cr_rocc_commands: Command router ROCC commands queue
        default_val = default_params["cr_rocc_commands"]
        new_val = op_params["cr_rocc_commands"]
        if new_val < default_val:
            penalty = (default_val - new_val) / default_val * 0.05
            estimated *= (1 - penalty)
        elif new_val > default_val:
            # Benefit is small, diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.02, 0.02)
            estimated *= (1 + benefit)
    
    # Ensure non-negative throughput
    return max(estimated, 0.0)
