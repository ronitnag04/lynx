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

        # Extract the distribution statistics
        size_bytes_distribution = benchmark_features["size_bytes_distribution"]
        total_fields_distribution = benchmark_features["total_fields_distribution"]
        nested_message_count_distribution = benchmark_features["nested_message_count_distribution"]
        depth_counter_list = benchmark_features["depth_counter_list"]
    else:
        raise ValueError(f"Benchmark {benchmark_name} not found in default benchmark features")
    
    estimated = float(default_throughput)
    
    def apply_penalty(amount: float):
        nonlocal estimated
        estimated *= (1 - amount * random.uniform(0.95, 1.05))
    
    def apply_benefit(amount: float):
        nonlocal estimated
        estimated *= (1 + amount * random.uniform(0.95, 1.05))
    
    if op_type == "deserializer":
        # Precompute distribution-driven stress ratios
        size_small_ratio = min(sum(size_bytes_distribution[:3]) if size_bytes_distribution else 0.0, 1.0)
        size_large_ratio = min(sum(size_bytes_distribution[-3:]) if size_bytes_distribution else 0.0, 1.0)
        fields_light_ratio = min(sum(total_fields_distribution[:3]) if total_fields_distribution else 0.0, 1.0)
        fields_heavy_ratio = min(sum(total_fields_distribution[-3:]) if total_fields_distribution else 0.0, 1.0)
        nested_low_ratio = min(sum(nested_message_count_distribution[:3]) if nested_message_count_distribution else 0.0, 1.0)
        nested_high_ratio = min(sum(nested_message_count_distribution[-3:]) if nested_message_count_distribution else 0.0, 1.0)
        depth_shallow_ratio = min(sum(depth_counter_list[:3]) / max(sum(depth_counter_list), 1) if depth_counter_list else 0.0, 1.0)
        depth_deep_ratio = min(sum(depth_counter_list[-3:]) / max(sum(depth_counter_list), 1) if depth_counter_list else 0.0, 1.0)
        
        # top_descriptor_reqs: Top descriptor request queue
        default_val = default_params["top_descriptor_reqs"]
        new_val = op_params["top_descriptor_reqs"]
        # Small messages trigger more descriptor traffic, so scale penalties when
        # the size histogram is skewed toward the smallest bins.
        small_message_ratio = min(sum(size_bytes_distribution[:3]) if size_bytes_distribution else 0.0, 1.0)
        small_message_penalty_factor = 1 + small_message_ratio  # 1.0–2.0 multiplier
        if new_val < default_val:
            # Impact is proportional to nested message count
            total_nested_messages = avg_nested_message_count * num_messages
            # Normalize by comparing total nested messages to max possible (num_messages * max_nested_message_count)
            max_possible_nested = num_messages * max(max_nested_message_count, 1.0)
            nested_stress = min(total_nested_messages / max(max_possible_nested, 1.0), 1.0)
            penalty = (default_val - new_val) / default_val * nested_stress * 0.3
            penalty *= small_message_penalty_factor
            apply_penalty(penalty)
        elif new_val > default_val:
            # Benefit is smaller, diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.1, 0.1)
            benefit *= max(0.3, 1 - 0.7 * small_message_ratio)  # Small-message heavy workloads see less uplift
            apply_benefit(benefit)
        
        # top_memloader_reqs: Controls outstanding memory loader requests
        # Lower values bottleneck with large messages
        default_val = default_params["top_memloader_reqs"]
        new_val = op_params["top_memloader_reqs"]
        if new_val < default_val:
            # Impact is proportional to message size (normalize to max size in benchmark)
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            size_stress *= (0.6 + 0.8 * size_large_ratio)  # heavier impact when large-message bins dominate
            penalty = (default_val - new_val) / default_val * size_stress * 0.2
            apply_penalty(penalty)
        elif new_val > default_val:
            benefit = min((new_val - default_val) / default_val * 0.05, 0.05)
            benefit *= max(0.4, 1 - 0.6 * size_small_ratio)  # small-message heavy workloads see less uplift
            apply_benefit(benefit)
        
        # dth_l1_reqs: Descriptor table handler L1 request queue
        # Lower values bottleneck with many descriptor lookups (nested messages)
        default_val = default_params["dth_l1_reqs"]
        new_val = op_params["dth_l1_reqs"]
        if new_val < default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            nested_stress *= (0.6 + 0.8 * nested_high_ratio)
            penalty = (default_val - new_val) / default_val * nested_stress * 0.15
            apply_penalty(penalty)
        elif new_val > default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            nested_stress *= (0.6 + 0.8 * nested_high_ratio)
            benefit = min((new_val - default_val) / default_val * nested_stress * 0.05, 0.05)
            benefit *= max(0.4, 1 - 0.6 * depth_shallow_ratio)  # deep messages benefit more
            apply_benefit(benefit)
        
        # dth_fd_reqs: Field destination request queue
        # Lower values bottleneck with many fields
        default_val = default_params["dth_fd_reqs"]
        new_val = op_params["dth_fd_reqs"]
        if new_val < default_val:
            # Normalize by typical field count (higher fields = more stress)
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.1
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.03, 0.03)
            benefit *= max(0.4, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # dth_fd_resps: Field destination response queue
        # Lower values bottleneck with many fields
        default_val = default_params["dth_fd_resps"]
        new_val = op_params["dth_fd_resps"]
        if new_val < default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.1
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.03, 0.03)
            benefit *= max(0.4, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # fw_l1_reqs: Fixed writer L1 request queue
        # Lower values bottleneck with many fixed-width fields
        default_val = default_params["fw_l1_reqs"]
        new_val = op_params["fw_l1_reqs"]
        if new_val < default_val:
            # Fixed writer is used for fixed-width fields, approximate by total fields
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.08
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.02, 0.02)
            benefit *= max(0.5, 1 - 0.4 * fields_light_ratio)
            apply_benefit(benefit)
        
        # ml_buf_info_q: Memloader buffer info queue
        # Lower values bottleneck with many messages
        default_val = default_params["ml_buf_info_q"]
        new_val = op_params["ml_buf_info_q"]
        if new_val < default_val:
            # Impact depends on number of messages (using benchmark's message count as baseline)
            # Since we're comparing within the same benchmark, use 1.0 for normalization
            messages_ratio = 1.0
            penalty = (default_val - new_val) / default_val * messages_ratio * (0.1 + 0.05 * size_small_ratio)
            apply_penalty(penalty)
        elif new_val > default_val:
            messages_ratio = 1.0
            benefit = min((new_val - default_val) / default_val * messages_ratio * 0.03, 0.03)
            benefit *= max(0.5, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # ml_load_info_q: Memloader load info queue
        # Lower values bottleneck with large messages requiring many loads
        default_val = default_params["ml_load_info_q"]
        new_val = op_params["ml_load_info_q"]
        if new_val < default_val:
            # Impact depends on message size (more loads for larger messages)
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            size_stress *= (0.6 + 0.8 * size_large_ratio)
            penalty = (default_val - new_val) / default_val * size_stress * 0.12
            apply_penalty(penalty)
        elif new_val > default_val:
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            size_stress *= (0.6 + 0.8 * size_large_ratio)
            benefit = min((new_val - default_val) / default_val * size_stress * 0.04, 0.04)
            benefit *= max(0.4, 1 - 0.6 * size_small_ratio)
            apply_benefit(benefit)

        # Correlation: memloader front-end requests vs internal load queues
        memloader_ratio = op_params["top_memloader_reqs"] / max(default_params["top_memloader_reqs"], 1e-9)
        ml_load_ratio = op_params["ml_load_info_q"] / max(default_params["ml_load_info_q"], 1e-9)
        ml_buf_ratio = op_params["ml_buf_info_q"] / max(default_params["ml_buf_info_q"], 1e-9)
        memloader_chain_bottleneck = min(memloader_ratio, ml_load_ratio, ml_buf_ratio)
        if memloader_chain_bottleneck < 1.0:
            chain_stress = (0.6 + 0.6 * size_large_ratio) * (0.6 + 0.4 * depth_deep_ratio)
            apply_penalty((1 - memloader_chain_bottleneck) * chain_stress * 0.06)

        # ml_buf_info_q and top_memloader_reqs are correlated, so we need to adjust the estimated throughput accordingly
        # The more messages, the more buffers are needed
        messages_ratio = op_params["ml_buf_info_q"] / default_params["ml_buf_info_q"]
        memloader_ratio = op_params["top_memloader_reqs"] / default_params["top_memloader_reqs"]
        apply_benefit(messages_ratio * memloader_ratio * 0.05)
        
        # cr_rocc_commands: Command router ROCC commands queue
        # Lower values bottleneck with high message throughput
        default_val = default_params["cr_rocc_commands"]
        new_val = op_params["cr_rocc_commands"]
        if new_val < default_val:
            # Impact is generally small but can affect high-throughput scenarios
            penalty = (default_val - new_val) / default_val * 0.05
            apply_penalty(penalty)
        elif new_val > default_val:
            # Benefit is small, diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.02, 0.02)
            apply_benefit(benefit)

        # Correlation: descriptor fetch chain (top descriptor reqs -> DTH queues)
        top_desc_ratio = op_params["top_descriptor_reqs"] / max(default_params["top_descriptor_reqs"], 1e-9)
        dth_l1_ratio = op_params["dth_l1_reqs"] / max(default_params["dth_l1_reqs"], 1e-9)
        dth_fd_ratio = op_params["dth_fd_reqs"] / max(default_params["dth_fd_reqs"], 1e-9)
        dth_fd_resp_ratio = op_params["dth_fd_resps"] / max(default_params["dth_fd_resps"], 1e-9)
        descriptor_chain_bottleneck = min(top_desc_ratio, dth_l1_ratio, dth_fd_ratio, dth_fd_resp_ratio)
        if descriptor_chain_bottleneck < 1.0:
            chain_stress = (0.6 + 0.6 * nested_high_ratio) * (0.6 + 0.4 * depth_deep_ratio)
            apply_penalty((1 - descriptor_chain_bottleneck) * chain_stress * 0.07)
    
    else:  # serializer
        # SERIALIZER PARAMETERS
        size_small_ratio = min(sum(size_bytes_distribution[:3]) if size_bytes_distribution else 0.0, 1.0)
        size_large_ratio = min(sum(size_bytes_distribution[-3:]) if size_bytes_distribution else 0.0, 1.0)
        fields_light_ratio = min(sum(total_fields_distribution[:3]) if total_fields_distribution else 0.0, 1.0)
        fields_heavy_ratio = min(sum(total_fields_distribution[-3:]) if total_fields_distribution else 0.0, 1.0)
        nested_low_ratio = min(sum(nested_message_count_distribution[:3]) if nested_message_count_distribution else 0.0, 1.0)
        nested_high_ratio = min(sum(nested_message_count_distribution[-3:]) if nested_message_count_distribution else 0.0, 1.0)
        depth_shallow_ratio = min(sum(depth_counter_list[:3]) / max(sum(depth_counter_list), 1) if depth_counter_list else 0.0, 1.0)
        depth_deep_ratio = min(sum(depth_counter_list[-3:]) / max(sum(depth_counter_list), 1) if depth_counter_list else 0.0, 1.0)
        
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
            penalty *= (0.6 + 0.8 * fields_heavy_ratio)
            apply_penalty(penalty)
        elif new_val > default_val:
            # Benefit from more parallelism, but diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.15, 0.2)
            benefit *= max(0.4, 1 - 0.6 * fields_light_ratio)
            apply_benefit(benefit)
        
        # dth_hasbits_reqs: Hasbits request queue
        # Lower values bottleneck with messages that have many optional fields
        default_val = default_params["dth_hasbits_reqs"]
        new_val = op_params["dth_hasbits_reqs"]
        if new_val < default_val:
            # Hasbits are used for optional fields, approximate by total fields
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.12
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.04, 0.04)
            benefit *= max(0.4, 1 - 0.6 * fields_light_ratio)
            apply_benefit(benefit)
        
        # dth_descriptor_reqs: Descriptor request queue
        # Lower values bottleneck with nested messages
        default_val = default_params["dth_descriptor_reqs"]
        new_val = op_params["dth_descriptor_reqs"]
        if new_val < default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            nested_stress *= (0.6 + 0.8 * nested_high_ratio)
            penalty = (default_val - new_val) / default_val * nested_stress * 0.15
            apply_penalty(penalty)
        elif new_val > default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            nested_stress *= (0.6 + 0.8 * nested_high_ratio)
            benefit = min((new_val - default_val) / default_val * nested_stress * 0.05, 0.05)
            benefit *= max(0.4, 1 - 0.6 * depth_shallow_ratio)
            apply_benefit(benefit)
        
        # dth_reg_resps: Register response queue
        # Lower values bottleneck with many field operations
        default_val = default_params["dth_reg_resps"]
        new_val = op_params["dth_reg_resps"]
        if new_val < default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.1
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.03, 0.03)
            benefit *= max(0.5, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # dth_reqs_meta: Request metadata queue
        # Lower values bottleneck with many requests
        default_val = default_params["dth_reqs_meta"]
        new_val = op_params["dth_reqs_meta"]
        if new_val < default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.08
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.02, 0.02)
            benefit *= max(0.5, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # dth_fh_outputs: Field handler outputs queue
        # Lower values bottleneck with many field handlers producing output
        default_val = default_params["dth_fh_outputs"]
        new_val = op_params["dth_fh_outputs"]
        if new_val < default_val:
            # Impact depends on number of field handlers and fields
            handlers_ratio = op_params["top_num_field_handlers"] / default_params["top_num_field_handlers"]
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * handlers_ratio * fields_stress * 0.1
            apply_penalty(penalty)
        elif new_val > default_val:
            handlers_ratio = op_params["top_num_field_handlers"] / default_params["top_num_field_handlers"]
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * handlers_ratio * fields_stress * 0.03, 0.03)
            benefit *= max(0.5, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # mw_write_input: Memwriter write input queue
        # Lower values bottleneck with high write throughput
        default_val = default_params["mw_write_input"]
        new_val = op_params["mw_write_input"]
        if new_val < default_val:
            # Impact depends on message size (more writes for larger messages)
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            size_stress *= (0.6 + 0.8 * size_large_ratio)
            penalty = (default_val - new_val) / default_val * size_stress * 0.12
            apply_penalty(penalty)
        elif new_val > default_val:
            size_stress = min(avg_size_bytes / max(max_size_bytes, 1.0), 1.0) if avg_size_bytes > 0 else 0.0
            size_stress *= (0.6 + 0.8 * size_large_ratio)
            benefit = min((new_val - default_val) / default_val * size_stress * 0.04, 0.04)
            benefit *= max(0.4, 1 - 0.6 * size_small_ratio)
            apply_benefit(benefit)

        # Correlation: descriptor-to-fieldhandler pipeline (hasbits -> descriptor -> meta -> reg resps -> fh outputs)
        hasbits_ratio = op_params["dth_hasbits_reqs"] / max(default_params["dth_hasbits_reqs"], 1e-9)
        descr_ratio = op_params["dth_descriptor_reqs"] / max(default_params["dth_descriptor_reqs"], 1e-9)
        meta_ratio = op_params["dth_reqs_meta"] / max(default_params["dth_reqs_meta"], 1e-9)
        reg_resp_ratio = op_params["dth_reg_resps"] / max(default_params["dth_reg_resps"], 1e-9)
        fh_out_ratio = op_params["dth_fh_outputs"] / max(default_params["dth_fh_outputs"], 1e-9)
        pipeline_bottleneck = min(hasbits_ratio, descr_ratio, meta_ratio, reg_resp_ratio, fh_out_ratio)
        if pipeline_bottleneck < 1.0:
            pipe_stress = (0.6 + 0.6 * nested_high_ratio) * (0.6 + 0.6 * depth_deep_ratio)
            pipe_stress *= (0.6 + 0.6 * fields_heavy_ratio)
            apply_penalty((1 - pipeline_bottleneck) * pipe_stress * 0.08)

        # Correlation: field handlers fan-out vs memwriter queues
        handler_ratio = op_params["top_num_field_handlers"] / max(default_params["top_num_field_handlers"], 1e-9)
        mw_input_ratio = op_params["mw_write_input"] / max(default_params["mw_write_input"], 1e-9)
        mw_inject_ratio = op_params["mw_write_inject"] / max(default_params["mw_write_inject"], 1e-9)
        mw_ptr_ratio = op_params["mw_write_ptrs"] / max(default_params["mw_write_ptrs"], 1e-9)
        mw_chain_bottleneck = min(handler_ratio, mw_input_ratio, mw_inject_ratio, mw_ptr_ratio)
        if mw_chain_bottleneck < 1.0:
            mw_stress = (0.6 + 0.6 * size_large_ratio) * (0.6 + 0.4 * fields_heavy_ratio)
            apply_penalty((1 - mw_chain_bottleneck) * mw_stress * 0.08)
        
        # mw_write_inject: Memwriter write inject queue
        # Lower values bottleneck with nested messages (size injection)
        default_val = default_params["mw_write_inject"]
        new_val = op_params["mw_write_inject"]
        if new_val < default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            nested_stress *= (0.6 + 0.8 * nested_high_ratio)
            penalty = (default_val - new_val) / default_val * nested_stress * 0.1
            apply_penalty(penalty)
        elif new_val > default_val:
            nested_stress = min(avg_nested_message_count / max(max_nested_message_count, 1.0), 1.0) if avg_nested_message_count > 0 else 0.0
            nested_stress *= (0.6 + 0.8 * nested_high_ratio)
            benefit = min((new_val - default_val) / default_val * nested_stress * 0.03, 0.03)
            benefit *= max(0.4, 1 - 0.6 * nested_low_ratio)
            apply_benefit(benefit)
        
        # mw_write_ptrs: Memwriter write pointers queue
        # Lower values bottleneck with string/bytes fields (pointer writes)
        default_val = default_params["mw_write_ptrs"]
        new_val = op_params["mw_write_ptrs"]
        if new_val < default_val:
            # Approximate by total fields (some fields may be strings/bytes)
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            penalty = (default_val - new_val) / default_val * fields_stress * 0.08
            apply_penalty(penalty)
        elif new_val > default_val:
            fields_stress = min(avg_total_fields / max(max_total_fields, 1.0), 1.0)
            fields_stress *= (0.6 + 0.8 * fields_heavy_ratio)
            benefit = min((new_val - default_val) / default_val * fields_stress * 0.02, 0.02)
            benefit *= max(0.5, 1 - 0.5 * fields_light_ratio)
            apply_benefit(benefit)
        
        # cr_rocc_commands: Command router ROCC commands queue
        default_val = default_params["cr_rocc_commands"]
        new_val = op_params["cr_rocc_commands"]
        if new_val < default_val:
            penalty = (default_val - new_val) / default_val * 0.05
            apply_penalty(penalty)
        elif new_val > default_val:
            # Benefit is small, diminishing returns
            benefit = min((new_val - default_val) / default_val * 0.02, 0.02)
            apply_benefit(benefit)

    # Add random noise to the estimated throughput
    estimated += random.uniform(-0.5, 0.5)
    
    # Ensure non-negative throughput
    return max(estimated, 0.0)
