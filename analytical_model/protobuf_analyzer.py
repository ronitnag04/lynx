#!/usr/bin/env python3
"""
Protobuf Workload Analyzer for HyperProtoBench

This script analyzes protobuf message definitions from HyperProtoBench benchmarks,
extracting metadata, sizes, configurations, and patterns from each protobuf message.

Based on Protocol Buffers specification: https://protobuf.dev/
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class EnumValue:
    """Represents an enum constant value"""
    name: str
    value: int


@dataclass
class Field:
    """Represents a protobuf field"""
    name: str
    field_number: int
    field_type: str  # int32, int64, string, bytes, message, enum, etc.
    cardinality: str  # optional, required, repeated
    is_nested_message: bool
    nested_message_name: Optional[str] = None
    is_enum: bool = False
    enum_name: Optional[str] = None
    wire_type: Optional[str] = None
    estimated_size_bytes: Optional[int] = None


@dataclass
class Message:
    """Represents a protobuf message"""
    name: str
    fields: List[Field]
    nested_messages: List['Message']
    enums: List[Dict]
    total_fields: int
    max_field_number: int
    has_repeated_fields: bool
    has_nested_messages: bool
    has_enums: bool
    estimated_size_bytes: Optional[int] = None
    depth: int = 0  # nesting depth


@dataclass
class RuntimeData:
    """Actual runtime data extracted from .inc files"""
    message_name: str
    string_lengths: List[int]  # Actual string field lengths
    bytes_lengths: List[int]   # Actual bytes field lengths
    numeric_values: Dict[str, List]  # Field name -> list of values
    operation_counts: Dict[str, int]  # Operation type -> count
    total_string_bytes: int
    total_bytes_bytes: int


@dataclass
class BenchmarkConfig:
    """Runtime configuration from .cc files"""
    working_set_size: int
    iterations: int
    messages_used: List[str]  # Messages actually used in benchmark


@dataclass
class BenchmarkAnalysis:
    """Analysis results for a single benchmark"""
    benchmark_name: str
    proto_file_path: str
    syntax_version: str
    messages: List[Message]
    total_messages: int
    total_fields: int
    unique_field_types: Set[str]
    field_type_distribution: Dict[str, int]
    cardinality_distribution: Dict[str, int]
    wire_type_distribution: Dict[str, int]
    max_nesting_depth: int
    repeated_field_count: int
    enum_count: int
    nested_message_count: int
    # Enhanced data from generated files
    runtime_data: Optional[Dict[str, RuntimeData]] = None
    benchmark_config: Optional[BenchmarkConfig] = None
    estimated_serialized_sizes: Optional[Dict[str, int]] = None  # Message name -> estimated size
    operation_statistics: Optional[Dict[str, int]] = None  # Operation type -> total count


class ProtobufAnalyzer:
    """Analyzes protobuf .proto files and extracts metadata"""
    
    # Wire type mappings based on protobuf encoding
    WIRE_TYPES = {
        'int32': 'VARINT',
        'int64': 'VARINT',
        'uint32': 'VARINT',
        'uint64': 'VARINT',
        'sint32': 'VARINT',
        'sint64': 'VARINT',
        'bool': 'VARINT',
        'enum': 'VARINT',
        'fixed32': 'FIXED32',
        'sfixed32': 'FIXED32',
        'float': 'FIXED32',
        'fixed64': 'FIXED64',
        'sfixed64': 'FIXED64',
        'double': 'FIXED64',
        'string': 'LENGTH_DELIMITED',
        'bytes': 'LENGTH_DELIMITED',
        'message': 'LENGTH_DELIMITED',
    }
    
    # Estimated sizes for different types (in bytes)
    TYPE_SIZES = {
        'int32': 4,
        'int64': 8,
        'uint32': 4,
        'uint64': 8,
        'sint32': 4,
        'sint64': 8,
        'bool': 1,
        'fixed32': 4,
        'sfixed32': 4,
        'float': 4,
        'fixed64': 8,
        'sfixed64': 8,
        'double': 8,
        'string': None,  # Variable length
        'bytes': None,  # Variable length
        'message': None,  # Variable length
        'enum': 1,
    }
    
    def __init__(self, proto_file_path: str):
        self.proto_file_path = proto_file_path
        self.content = self._read_file()
        self.syntax_version = self._extract_syntax()
        self.messages = []
        self.enums = []
        # Paths to related files
        self.bench_dir = Path(proto_file_path).parent
        self.inc_file = self.bench_dir / 'benchmark.inc'
        self.cc_file = self.bench_dir / 'benchmark.cc'
        
    def _read_file(self) -> str:
        """Read the .proto file content"""
        with open(self.proto_file_path, 'r') as f:
            return f.read()
    
    def _extract_syntax(self) -> str:
        """Extract syntax version (proto2 or proto3)"""
        match = re.search(r'syntax\s*=\s*["\']([^"\']+)["\']', self.content)
        return match.group(1) if match else 'proto2'
    
    def _parse_enums(self) -> List[Dict]:
        """Parse all enum definitions"""
        enums = []
        enum_pattern = r'enum\s+(\w+)\s*\{([^}]+)\}'
        
        for match in re.finditer(enum_pattern, self.content, re.DOTALL):
            enum_name = match.group(1)
            enum_body = match.group(2)
            
            enum_values = []
            value_pattern = r'(\w+)\s*=\s*([-\d]+)'
            for value_match in re.finditer(value_pattern, enum_body):
                enum_values.append({
                    'name': value_match.group(1),
                    'value': int(value_match.group(2))
                })
            
            enums.append({
                'name': enum_name,
                'values': enum_values
            })
        
        return enums
    
    def _find_matching_brace(self, text: str, start_pos: int) -> int:
        """Find the matching closing brace for an opening brace at start_pos"""
        depth = 0
        i = start_pos
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1
    
    def _parse_message(self, message_content: str, depth: int = 0, all_enums: List[Dict] = None) -> Optional[Message]:
        """Parse a single message definition with proper nested message handling"""
        if all_enums is None:
            all_enums = self.enums
        
        # Extract message name
        name_match = re.search(r'message\s+(\w+)', message_content)
        if not name_match:
            return None
        message_name = name_match.group(1)
        
        # Find the message body (handle nested braces properly)
        brace_start = message_content.find('{')
        if brace_start == -1:
            return None
        
        brace_end = self._find_matching_brace(message_content, brace_start)
        if brace_end == -1:
            return None
        
        message_body = message_content[brace_start + 1:brace_end]
        
        fields = []
        nested_messages = []
        enums = []
        
        # First, extract nested messages and enums to avoid parsing them as fields
        nested_message_pattern = r'message\s+(\w+)\s*\{'
        nested_enum_pattern = r'enum\s+(\w+)\s*\{'
        
        # Track positions of nested structures
        nested_positions = []
        i = 0
        while i < len(message_body):
            # Check for nested message
            msg_match = re.search(nested_message_pattern, message_body[i:])
            if msg_match:
                msg_start = i + msg_match.start()
                msg_name = msg_match.group(1)
                brace_pos = message_body.find('{', msg_start)
                if brace_pos != -1:
                    nested_end = self._find_matching_brace(message_body, brace_pos)
                    if nested_end != -1:
                        nested_content = message_body[msg_start:nested_end + 1]
                        nested_msg = self._parse_message(nested_content, depth + 1, all_enums)
                        if nested_msg:
                            nested_messages.append(nested_msg)
                        nested_positions.append((msg_start, nested_end + 1))
                        i = nested_end + 1
                        continue
            
            # Check for nested enum
            enum_match = re.search(nested_enum_pattern, message_body[i:])
            if enum_match:
                enum_start = i + enum_match.start()
                enum_name = enum_match.group(1)
                brace_pos = message_body.find('{', enum_start)
                if brace_pos != -1:
                    enum_end = self._find_matching_brace(message_body, brace_pos)
                    if enum_end != -1:
                        enums.append({'name': enum_name, 'nested': True})
                        nested_positions.append((enum_start, enum_end + 1))
                        i = enum_end + 1
                        continue
            
            i += 1
        
        # Now parse fields, excluding nested message/enum regions
        field_pattern = r'(optional|required|repeated)?\s*(\w+)\s+(\w+)\s*=\s*(\d+)(?:\s*\[.*?\])?;'
        
        for field_match in re.finditer(field_pattern, message_body):
            # Check if this field is inside a nested structure
            field_start = field_match.start()
            field_end = field_match.end()
            is_in_nested = any(start <= field_start < end for start, end in nested_positions)
            
            if is_in_nested:
                continue
            
            cardinality = field_match.group(1) or 'optional'
            field_type = field_match.group(2).strip()
            field_name = field_match.group(3)
            field_number = int(field_match.group(4))
            
            is_nested_message = False
            nested_message_name = None
            is_enum = False
            enum_name = None
            
            # Check if field type is a nested message (by name)
            for nested_msg in nested_messages:
                if field_type == nested_msg.name:
                    is_nested_message = True
                    nested_message_name = nested_msg.name
                    break
            
            # Check if field type is an enum
            for enum in all_enums + enums:
                if field_type == enum['name']:
                    is_enum = True
                    enum_name = enum['name']
                    break
            
            # Determine wire type
            wire_type = self.WIRE_TYPES.get(field_type, 'LENGTH_DELIMITED')
            if is_nested_message:
                wire_type = 'LENGTH_DELIMITED'
            elif is_enum:
                wire_type = 'VARINT'
            
            # Estimate size
            estimated_size = self.TYPE_SIZES.get(field_type)
            
            field = Field(
                name=field_name,
                field_number=field_number,
                field_type=field_type,
                cardinality=cardinality,
                is_nested_message=is_nested_message,
                nested_message_name=nested_message_name,
                is_enum=is_enum,
                enum_name=enum_name,
                wire_type=wire_type,
                estimated_size_bytes=estimated_size
            )
            fields.append(field)
        
        # Calculate message statistics
        max_field_number = max([f.field_number for f in fields]) if fields else 0
        has_repeated = any(f.cardinality == 'repeated' for f in fields)
        has_nested = len(nested_messages) > 0
        has_enums = len(enums) > 0
        
        # Estimate message size (rough approximation)
        estimated_size = sum(f.estimated_size_bytes or 0 for f in fields if f.estimated_size_bytes)
        
        return Message(
            name=message_name,
            fields=fields,
            nested_messages=nested_messages,
            enums=enums,
            total_fields=len(fields),
            max_field_number=max_field_number,
            has_repeated_fields=has_repeated,
            has_nested_messages=has_nested,
            has_enums=has_enums,
            estimated_size_bytes=estimated_size if estimated_size > 0 else None,
            depth=depth
        )
    
    def analyze(self) -> BenchmarkAnalysis:
        """Perform complete analysis of the protobuf file"""
        # Parse enums first
        self.enums = self._parse_enums()
        
        # Parse all top-level messages (handle nested braces properly)
        i = 0
        while i < len(self.content):
            msg_match = re.search(r'message\s+(\w+)', self.content[i:])
            if not msg_match:
                break
            
            msg_start = i + msg_match.start()
            brace_pos = self.content.find('{', msg_start)
            if brace_pos == -1:
                break
            
            brace_end = self._find_matching_brace(self.content, brace_pos)
            if brace_end == -1:
                break
            
            message_content = self.content[msg_start:brace_end + 1]
            message = self._parse_message(message_content, depth=0, all_enums=self.enums)
            if message:
                self.messages.append(message)
            
            i = brace_end + 1
        
        # Calculate statistics
        all_fields = []
        for msg in self.messages:
            all_fields.extend(msg.fields)
            # Also include nested message fields
            for nested in msg.nested_messages:
                all_fields.extend(nested.fields)
        
        field_type_dist = defaultdict(int)
        cardinality_dist = defaultdict(int)
        wire_type_dist = defaultdict(int)
        
        for field in all_fields:
            field_type_dist[field.field_type] += 1
            cardinality_dist[field.cardinality] += 1
            if field.wire_type:
                wire_type_dist[field.wire_type] += 1
        
        # Calculate max depth by recursively checking all messages
        def get_max_depth(msg: Message) -> int:
            if not msg.nested_messages:
                return msg.depth
            return max([msg.depth] + [get_max_depth(nested) for nested in msg.nested_messages])
        
        max_depth = max([get_max_depth(msg) for msg in self.messages] + [0])
        
        # Count all nested messages recursively
        def count_nested(msg: Message) -> int:
            return len(msg.nested_messages) + sum(count_nested(nested) for nested in msg.nested_messages)
        
        nested_count = sum(count_nested(msg) for msg in self.messages)
        repeated_count = sum(1 for f in all_fields if f.cardinality == 'repeated')
        enum_count = len(self.enums)
        
        # Extract runtime data from generated files
        runtime_data = self._extract_runtime_data()
        benchmark_config = self._extract_benchmark_config()
        estimated_sizes = self._calculate_serialized_sizes(runtime_data)
        operation_stats = self._extract_operation_statistics()
        
        return BenchmarkAnalysis(
            benchmark_name=Path(self.proto_file_path).parent.name,
            proto_file_path=self.proto_file_path,
            syntax_version=self.syntax_version,
            messages=self.messages,
            total_messages=len(self.messages),
            total_fields=len(all_fields),
            unique_field_types=set(field_type_dist.keys()),
            field_type_distribution=dict(field_type_dist),
            cardinality_distribution=dict(cardinality_dist),
            wire_type_distribution=dict(wire_type_dist),
            max_nesting_depth=max_depth,
            repeated_field_count=repeated_count,
            enum_count=enum_count,
            nested_message_count=nested_count,
            runtime_data=runtime_data,
            benchmark_config=benchmark_config,
            estimated_serialized_sizes=estimated_sizes,
            operation_statistics=operation_stats
        )
    
    def _extract_runtime_data(self) -> Optional[Dict[str, RuntimeData]]:
        """Extract actual runtime data from benchmark.inc file"""
        if not self.inc_file.exists():
            return None
        
        try:
            with open(self.inc_file, 'r') as f:
                inc_content = f.read()
        except Exception:
            return None
        
        runtime_data = {}
        
        # Find all Set_F1 functions and extract data
        # Use a more robust pattern that handles nested braces
        set_pattern = r'int\s+(\w+)_Set_F1[^{]*\{'
        
        i = 0
        while i < len(inc_content):
            match = re.search(set_pattern, inc_content[i:])
            if not match:
                break
            
            start_pos = i + match.end() - 1  # Position of opening brace
            message_name = match.group(1)
            
            # Find matching closing brace
            brace_end = self._find_matching_brace(inc_content, start_pos)
            if brace_end == -1:
                i = start_pos + 1
                continue
            
            function_body = inc_content[start_pos + 1:brace_end]
            i = brace_end + 1
            
            string_lengths = []
            bytes_lengths = []
            numeric_values = defaultdict(list)
            
            # Extract string literals and their lengths
            string_pattern = r'set_[a-z0-9]+\(["\']([^"\']+)["\']\)'
            for str_match in re.finditer(string_pattern, function_body):
                string_val = str_match.group(1)
                string_lengths.append(len(string_val.encode('utf-8')))
            
            # Extract bytes (they're also strings in the code)
            # Look for set_f calls with long strings (likely bytes)
            bytes_pattern = r'set_f\d+\(["\']([^"\']{50,})["\']\)'  # Strings > 50 chars likely bytes
            for bytes_match in re.finditer(bytes_pattern, function_body):
                bytes_val = bytes_match.group(1)
                bytes_lengths.append(len(bytes_val.encode('utf-8')))
            
            # Extract numeric values
            numeric_pattern = r'set_[a-z0-9]+\(([0-9xU\.]+)\)'
            for num_match in re.finditer(numeric_pattern, function_body):
                num_str = num_match.group(1)
                try:
                    if 'U' in num_str or '0x' in num_str:
                        # Hex or unsigned
                        num_val = int(num_str.replace('U', '').replace('0x', ''), 16) if '0x' in num_str else int(num_str.replace('U', ''))
                    elif '.' in num_str:
                        num_val = float(num_str)
                    else:
                        num_val = int(num_str)
                    numeric_values['numeric'].append(num_val)
                except ValueError:
                    pass
            
            # Count operations for this message
            operation_counts = {
                'Create': len(re.findall(rf'{message_name}_Create_F1', inc_content)),
                'Set': len(re.findall(rf'{message_name}_Set_F1', inc_content)),
                'Serialize': len(re.findall(rf'{message_name}_Serialize_F1', inc_content)),
                'Deserialize': len(re.findall(rf'{message_name}_Deserialize_F1', inc_content)),
                'Get': len(re.findall(rf'{message_name}_Get_F1', inc_content)),
                'Destroy': len(re.findall(rf'{message_name}_Destroy_F1', inc_content)),
            }
            
            runtime_data[message_name] = RuntimeData(
                message_name=message_name,
                string_lengths=string_lengths,
                bytes_lengths=bytes_lengths,
                numeric_values=dict(numeric_values),
                operation_counts=operation_counts,
                total_string_bytes=sum(string_lengths),
                total_bytes_bytes=sum(bytes_lengths)
            )
        
        return runtime_data if runtime_data else None
    
    def _extract_benchmark_config(self) -> Optional[BenchmarkConfig]:
        """Extract runtime configuration from benchmark.cc file"""
        if not self.cc_file.exists():
            return None
        
        try:
            with open(self.cc_file, 'r') as f:
                cc_content = f.read()
        except Exception:
            return None
        
        # Extract WORKING_SET_SIZE
        working_set_match = re.search(r'#define\s+WORKING_SET_SIZE\s+(\d+)', cc_content)
        working_set_size = int(working_set_match.group(1)) if working_set_match else 1
        
        # Extract iterations
        iters_match = re.search(r'int\s+iters\s*=\s*(\d+)', cc_content)
        iterations = int(iters_match.group(1)) if iters_match else 1
        
        # Extract messages used in BenchmarkIteration
        # Look for message type declarations in the function
        messages_used = []
        # Find all message type patterns (e.g., M1*, M10*, etc.)
        message_pattern = r'\b(M\d+)_\w+\['
        for match in re.finditer(message_pattern, cc_content):
            msg_name = match.group(1)
            if msg_name not in messages_used:
                messages_used.append(msg_name)
        
        # Also check benchmark.inc for messages used
        if self.inc_file.exists():
            try:
                with open(self.inc_file, 'r') as f:
                    inc_content = f.read()
                # Find all message operations
                inc_message_pattern = r'\b(M\d+)_(?:Create|Set|Serialize|Deserialize|Get|Destroy)_F1'
                for match in re.finditer(inc_message_pattern, inc_content):
                    msg_name = match.group(1)
                    if msg_name not in messages_used:
                        messages_used.append(msg_name)
            except Exception:
                pass
        
        return BenchmarkConfig(
            working_set_size=working_set_size,
            iterations=iterations,
            messages_used=sorted(messages_used)
        )
    
    def _calculate_serialized_sizes(self, runtime_data: Optional[Dict[str, RuntimeData]]) -> Optional[Dict[str, int]]:
        """Calculate more accurate serialized sizes using actual data"""
        if not runtime_data:
            return None
        
        estimated_sizes = {}
        
        for msg_name, msg in [(m.name, m) for m in self.messages]:
            if msg_name not in runtime_data:
                continue
            
            rt_data = runtime_data[msg_name]
            total_size = 0
            
            # Base overhead for message (varint tag for each field)
            field_tag_overhead = len(msg.fields) * 2  # Rough estimate: 1-2 bytes per field tag
            
            # Add actual string/bytes sizes
            total_size += rt_data.total_string_bytes
            total_size += rt_data.total_bytes_bytes
            
            # Add estimated sizes for numeric fields
            numeric_field_count = sum(1 for f in msg.fields 
                                     if f.field_type in ['int32', 'int64', 'uint32', 'uint64', 
                                                         'fixed32', 'fixed64', 'float', 'double', 
                                                         'bool', 'enum'])
            
            # Estimate varint encoding (1-10 bytes per varint, average ~3-5)
            varint_fields = sum(1 for f in msg.fields 
                              if f.wire_type == 'VARINT')
            varint_size = varint_fields * 4  # Average 4 bytes per varint
            
            # Fixed-size fields
            fixed_fields = sum(1 for f in msg.fields 
                             if f.wire_type in ['FIXED32', 'FIXED64'])
            fixed_size = (sum(1 for f in msg.fields if f.wire_type == 'FIXED32') * 4 +
                         sum(1 for f in msg.fields if f.wire_type == 'FIXED64') * 8)
            
            # Length-delimited overhead (1-5 bytes for length prefix)
            length_delim_fields = sum(1 for f in msg.fields 
                                    if f.wire_type == 'LENGTH_DELIMITED')
            length_overhead = length_delim_fields * 3  # Average 3 bytes per length prefix
            
            # Nested messages (recursive, but simplified)
            nested_size = 0
            for nested in msg.nested_messages:
                if nested.name in runtime_data:
                    nested_rt = runtime_data[nested.name]
                    nested_size += nested_rt.total_string_bytes + nested_rt.total_bytes_bytes
                    nested_size += len(nested.fields) * 3  # Rough estimate
            
            total_size += field_tag_overhead + varint_size + fixed_size + length_overhead + nested_size
            
            estimated_sizes[msg_name] = total_size
        
        return estimated_sizes if estimated_sizes else None
    
    def _extract_operation_statistics(self) -> Optional[Dict[str, int]]:
        """Extract operation statistics from benchmark.inc"""
        if not self.inc_file.exists():
            return None
        
        try:
            with open(self.inc_file, 'r') as f:
                inc_content = f.read()
        except Exception:
            return None
        
        # Count total operations by type
        operations = {
            'total_create': len(re.findall(r'_Create_F1', inc_content)),
            'total_set': len(re.findall(r'_Set_F1', inc_content)),
            'total_serialize': len(re.findall(r'_Serialize_F1', inc_content)),
            'total_deserialize': len(re.findall(r'_Deserialize_F1', inc_content)),
            'total_get': len(re.findall(r'_Get_F1', inc_content)),
            'total_destroy': len(re.findall(r'_Destroy_F1', inc_content)),
        }
        
        # Count unique messages used
        unique_messages = set()
        message_pattern = r'\b(M\d+)_(?:Create|Set|Serialize|Deserialize|Get|Destroy)_F1'
        for match in re.finditer(message_pattern, inc_content):
            unique_messages.add(match.group(1))
        
        operations['unique_messages_used'] = len(unique_messages)
        
        return operations


def analyze_hyperprotobench(base_path: str) -> Dict:
    """Analyze all benchmarks in HyperProtoBench directory"""
    base_path = Path(base_path)
    results = {}
    
    # Find all benchmark directories
    for bench_dir in sorted(base_path.glob('bench*')):
        if not bench_dir.is_dir():
            continue
        
        proto_file = bench_dir / 'benchmark.proto'
        if not proto_file.exists():
            print(f"Warning: {proto_file} not found, skipping {bench_dir.name}")
            continue
        
        print(f"Analyzing {bench_dir.name}...")
        try:
            analyzer = ProtobufAnalyzer(str(proto_file))
            analysis = analyzer.analyze()
            
            # Convert to serializable format
            results[bench_dir.name] = {
                'benchmark_name': analysis.benchmark_name,
                'proto_file_path': analysis.proto_file_path,
                'syntax_version': analysis.syntax_version,
                'statistics': {
                    'total_messages': analysis.total_messages,
                    'total_fields': analysis.total_fields,
                    'max_nesting_depth': analysis.max_nesting_depth,
                    'repeated_field_count': analysis.repeated_field_count,
                    'enum_count': analysis.enum_count,
                    'nested_message_count': analysis.nested_message_count,
                },
                'field_type_distribution': analysis.field_type_distribution,
                'cardinality_distribution': analysis.cardinality_distribution,
                'wire_type_distribution': analysis.wire_type_distribution,
                'messages': []
            }
            
            # Add runtime configuration if available
            if analysis.benchmark_config:
                results[bench_dir.name]['runtime_config'] = {
                    'working_set_size': analysis.benchmark_config.working_set_size,
                    'iterations': analysis.benchmark_config.iterations,
                    'messages_used': analysis.benchmark_config.messages_used,
                    'messages_used_count': len(analysis.benchmark_config.messages_used)
                }
            
            # Add operation statistics if available
            if analysis.operation_statistics:
                results[bench_dir.name]['operation_statistics'] = analysis.operation_statistics
            
            # Add estimated serialized sizes if available
            if analysis.estimated_serialized_sizes:
                results[bench_dir.name]['estimated_serialized_sizes'] = analysis.estimated_serialized_sizes
                results[bench_dir.name]['total_estimated_bytes'] = sum(analysis.estimated_serialized_sizes.values())
            
            # Helper function to recursively add messages
            def add_message(msg: Message, parent_name: str = None):
                msg_data = {
                    'name': msg.name,
                    'parent': parent_name,
                    'total_fields': msg.total_fields,
                    'max_field_number': msg.max_field_number,
                    'depth': msg.depth,
                    'has_repeated_fields': msg.has_repeated_fields,
                    'has_nested_messages': msg.has_nested_messages,
                    'has_enums': msg.has_enums,
                    'estimated_size_bytes': msg.estimated_size_bytes,
                    'nested_message_count': len(msg.nested_messages),
                    'fields': [],
                    'nested_messages': []
                }
                
                for field in msg.fields:
                    msg_data['fields'].append({
                        'name': field.name,
                        'field_number': field.field_number,
                        'field_type': field.field_type,
                        'cardinality': field.cardinality,
                        'wire_type': field.wire_type,
                        'is_nested_message': field.is_nested_message,
                        'nested_message_name': field.nested_message_name,
                        'is_enum': field.is_enum,
                        'enum_name': field.enum_name,
                        'estimated_size_bytes': field.estimated_size_bytes,
                    })
                
                # Recursively add nested messages
                for nested_msg in msg.nested_messages:
                    nested_data = add_message(nested_msg, msg.name)
                    msg_data['nested_messages'].append(nested_data['name'])
                    results[bench_dir.name]['messages'].append(nested_data)
                
                # Add runtime data if available
                if analysis.runtime_data and msg.name in analysis.runtime_data:
                    rt_data = analysis.runtime_data[msg.name]
                    msg_data['runtime_data'] = {
                        'total_string_bytes': rt_data.total_string_bytes,
                        'total_bytes_bytes': rt_data.total_bytes_bytes,
                        'string_count': len(rt_data.string_lengths),
                        'bytes_count': len(rt_data.bytes_lengths),
                        'avg_string_length': sum(rt_data.string_lengths) / len(rt_data.string_lengths) if rt_data.string_lengths else 0,
                        'avg_bytes_length': sum(rt_data.bytes_lengths) / len(rt_data.bytes_lengths) if rt_data.bytes_lengths else 0,
                        'max_string_length': max(rt_data.string_lengths) if rt_data.string_lengths else 0,
                        'max_bytes_length': max(rt_data.bytes_lengths) if rt_data.bytes_lengths else 0,
                        'operation_counts': rt_data.operation_counts
                    }
                
                # Add estimated serialized size if available
                if analysis.estimated_serialized_sizes and msg.name in analysis.estimated_serialized_sizes:
                    msg_data['estimated_serialized_size_bytes'] = analysis.estimated_serialized_sizes[msg.name]
                
                return msg_data
            
            # Add all messages (top-level and nested)
            for msg in analysis.messages:
                msg_data = add_message(msg)
                results[bench_dir.name]['messages'].append(msg_data)
            
        except Exception as e:
            print(f"Error analyzing {bench_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def print_summary(results: Dict):
    """Print a summary of the analysis results"""
    print("\n" + "="*80)
    print("HYPERPROTOBENCH ANALYSIS SUMMARY")
    print("="*80)
    
    for bench_name, data in results.items():
        print(f"\n{bench_name.upper()}")
        print("-" * 80)
        print(f"Syntax: {data['syntax_version']}")
        print(f"Total Messages: {data['statistics']['total_messages']}")
        print(f"Total Fields: {data['statistics']['total_fields']}")
        print(f"Max Nesting Depth: {data['statistics']['max_nesting_depth']}")
        print(f"Repeated Fields: {data['statistics']['repeated_field_count']}")
        print(f"Enums: {data['statistics']['enum_count']}")
        print(f"Nested Messages: {data['statistics']['nested_message_count']}")
        
        print("\nField Type Distribution:")
        for ftype, count in sorted(data['field_type_distribution'].items()):
            print(f"  {ftype:20s}: {count:4d}")
        
        print("\nCardinality Distribution:")
        for card, count in sorted(data['cardinality_distribution'].items()):
            print(f"  {card:20s}: {count:4d}")
        
        print("\nWire Type Distribution:")
        for wtype, count in sorted(data['wire_type_distribution'].items()):
            print(f"  {wtype:20s}: {count:4d}")
        
        # Show runtime configuration if available
        if 'runtime_config' in data:
            config = data['runtime_config']
            print(f"\nRuntime Configuration:")
            print(f"  Working Set Size: {config['working_set_size']}")
            print(f"  Iterations: {config['iterations']}")
            print(f"  Messages Used: {config['messages_used_count']} ({', '.join(config['messages_used'][:10])}{'...' if len(config['messages_used']) > 10 else ''})")
        
        # Show operation statistics if available
        if 'operation_statistics' in data:
            ops = data['operation_statistics']
            print(f"\nOperation Statistics:")
            print(f"  Serialize Operations: {ops.get('total_serialize', 0)}")
            print(f"  Deserialize Operations: {ops.get('total_deserialize', 0)}")
            print(f"  Set Operations: {ops.get('total_set', 0)}")
            print(f"  Get Operations: {ops.get('total_get', 0)}")
            print(f"  Create Operations: {ops.get('total_create', 0)}")
            print(f"  Destroy Operations: {ops.get('total_destroy', 0)}")
            print(f"  Unique Messages Used: {ops.get('unique_messages_used', 0)}")
        
        # Show estimated sizes if available
        if 'estimated_serialized_sizes' in data:
            sizes = data['estimated_serialized_sizes']
            print(f"\nEstimated Serialized Sizes:")
            sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            for msg_name, size in sorted_sizes:
                print(f"  {msg_name:20s}: {size:8,d} bytes")
            if 'total_estimated_bytes' in data:
                print(f"  {'TOTAL':20s}: {data['total_estimated_bytes']:8,d} bytes")
        
        print(f"\nTop Messages (by field count):")
        sorted_messages = sorted(
            data['messages'],
            key=lambda m: m['total_fields'],
            reverse=True
        )[:5]
        for msg in sorted_messages:
            nested_info = ""
            if msg['has_nested_messages']:
                nested_info = f", {len([m for m in data['messages'] if m.get('parent') == msg['name']])} nested msgs"
            size_info = ""
            if 'estimated_serialized_size_bytes' in msg:
                size_info = f", ~{msg['estimated_serialized_size_bytes']:,} bytes"
            runtime_info = ""
            if 'runtime_data' in msg:
                rt = msg['runtime_data']
                runtime_info = f", {rt['total_string_bytes'] + rt['total_bytes_bytes']:,} str/bytes"
            print(f"  {msg['name']:20s}: {msg['total_fields']:3d} fields, "
                  f"max field #: {msg['max_field_number']:3d}, "
                  f"depth: {msg['depth']}{nested_info}{size_info}{runtime_info}")
        
        # Show nested message statistics
        nested_msg_count = sum(1 for m in data['messages'] if m['depth'] > 0)
        if nested_msg_count > 0:
            print(f"\nNested Message Statistics:")
            depth_dist = {}
            for msg in data['messages']:
                if msg['depth'] > 0:
                    depth_dist[msg['depth']] = depth_dist.get(msg['depth'], 0) + 1
            for depth, count in sorted(depth_dist.items()):
                print(f"  Depth {depth}: {count} messages")


def save_json_report(results: Dict, output_path: str):
    """Save analysis results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze protobuf workloads in HyperProtoBench'
    )
    parser.add_argument(
        '--hyperprotobench-path',
        type=str,
        help='Path to HyperProtoBench directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='protobuf_analysis.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary to console'
    )
    
    args = parser.parse_args()
    
    # Resolve path relative to script location
    script_dir = Path(__file__).parent
    hyperprotobench_path = script_dir / args.hyperprotobench_path
    
    if not hyperprotobench_path.exists():
        print(f"Error: HyperProtoBench path not found: {hyperprotobench_path}")
        return
    
    print(f"Analyzing protobuf workloads in: {hyperprotobench_path}")
    
    # Analyze all benchmarks
    results = analyze_hyperprotobench(str(hyperprotobench_path))
    
    if not results:
        print("No benchmarks found or analyzed.")
        return
    
    # Print summary
    if args.summary:
        print_summary(results)
    
    # Save JSON report
    output_path = script_dir / args.output
    save_json_report(results, str(output_path))
    
    print(f"\nAnalysis complete! Analyzed {len(results)} benchmarks.")


if __name__ == '__main__':
    main()

