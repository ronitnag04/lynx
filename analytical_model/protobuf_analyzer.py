#!/usr/bin/env python3
"""
Protobuf Workload Analyzer for HyperProtoBench

This script analyzes protobuf message definitions from HyperProtoBench benchmarks,
extracting metadata, sizes, configurations, and patterns from each protobuf message.

Based on Protocol Buffers specification: https://protobuf.dev/
"""

import os
import time
import re
import math
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
    size_bytes: Optional[int] = None  # Size determined by type or runtime values, None if unknown


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
    size_bytes: Optional[int] = None  # Size determined by type or runtime values, None if unknown
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
    field_string_lengths: Optional[Dict[str, List[int]]] = None  # field_name -> [lengths]
    field_bytes_lengths: Optional[Dict[str, List[int]]] = None  # field_name -> [lengths]


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
    serialized_sizes: Optional[Dict[str, int]] = None  # Message name -> serialized size (from runtime or type)
    operation_statistics: Optional[Dict[str, int]] = None  # Operation type -> total count


class ProtobufAnalyzer:
    """Analyzes protobuf .proto files and extracts metadata"""
    
    @staticmethod
    def _calculate_tag_size(field_number: int) -> int:
        """Calculate the size of a tag varint.
        
        Tag is encoded as: (field_number << 3) | wire_type
        The tag itself is a varint, so its size depends on the encoded value.
        Per Protocol Buffers encoding spec: https://protobuf.dev/programming-guides/encoding/
        
        Args:
            field_number: The field number from the .proto definition
            
        Returns:
            Size in bytes of the tag varint (1-5 bytes)
        """
        # Tag value = (field_number << 3) | wire_type
        # Wire type is 0-5 (3 bits), so max tag value is (field_number << 3) | 5
        # For practical purposes, field numbers are typically < 536870912 (2^29)
        tag_value = (field_number << 3) | 5  # Use max wire_type (5) for worst case
        
        # Calculate varint size (varints use 7 bits per byte, MSB is continuation bit)
        if tag_value < 128:
            return 1
        elif tag_value < 16384:
            return 2
        elif tag_value < 2097152:
            return 3
        elif tag_value < 268435456:
            return 4
        else:
            return 5  # Max varint size (up to 10 bytes for 64-bit values, but tags are smaller)
    
    @staticmethod
    def _calculate_varint_size(value: int) -> int:
        """Calculate the size of a varint-encoded value.
        
        Per Protocol Buffers encoding, varints use 7 bits per byte with MSB as continuation bit.
        Values can be 1-10 bytes depending on magnitude.
        
        Args:
            value: The integer value to encode
            
        Returns:
            Size in bytes of the varint encoding (1-10 bytes)
        """
        if value < 128:
            return 1
        elif value < 16384:
            return 2
        elif value < 2097152:
            return 3
        elif value < 268435456:
            return 4
        elif value < 34359738368:
            return 5
        elif value < 4398046511104:
            return 6
        elif value < 562949953421312:
            return 7
        elif value < 72057594037927936:
            return 8
        elif value < 9223372036854775808:
            return 9
        else:
            return 10  # Max varint size for 64-bit values
    
    # Wire type mappings based on protobuf encoding (https://protobuf.dev/programming-guides/encoding/)
    # Wire types: VARINT (0), I64 (1), LEN (2), SGROUP (3), EGROUP (4), I32 (5)
    WIRE_TYPES = {
        'int32': 'VARINT',      # Wire type 0
        'int64': 'VARINT',      # Wire type 0
        'uint32': 'VARINT',     # Wire type 0
        'uint64': 'VARINT',     # Wire type 0
        'sint32': 'VARINT',     # Wire type 0 (ZigZag encoded)
        'sint64': 'VARINT',     # Wire type 0 (ZigZag encoded)
        'bool': 'VARINT',       # Wire type 0
        'enum': 'VARINT',       # Wire type 0
        'fixed32': 'I32',       # Wire type 5
        'sfixed32': 'I32',      # Wire type 5
        'float': 'I32',         # Wire type 5
        'fixed64': 'I64',       # Wire type 1
        'sfixed64': 'I64',      # Wire type 1
        'double': 'I64',        # Wire type 1
        'string': 'LEN',        # Wire type 2
        'bytes': 'LEN',         # Wire type 2
        'message': 'LEN',       # Wire type 2
    }
    
    def __init__(self, proto_file_path: str, messages_to_analyze: Optional[List[str]] = None):
        self.proto_file_path = proto_file_path
        self.content = self._read_file()
        self.syntax_version = self._extract_syntax()
        self.messages = []
        self.enums = []
        self.messages_to_analyze = messages_to_analyze  # Only analyze these messages
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
            original_field_type = field_match.group(2).strip()
            field_name = field_match.group(3)
            field_number = int(field_match.group(4))
            
            is_nested_message = False
            nested_message_name = None
            is_enum = False
            enum_name = None
            
            # Check if field type is a nested message (by name)
            for nested_msg in nested_messages:
                if original_field_type == nested_msg.name:
                    is_nested_message = True
                    nested_message_name = nested_msg.name
                    break
            
            # Check if field type is an enum
            for enum in all_enums + enums:
                if original_field_type == enum['name']:
                    is_enum = True
                    enum_name = enum['name']
                    break
            
            # Normalize field_type: M[0-9]+ -> 'message', E[0-9]+ -> 'enum'
            # Keep original for checks above, normalize for storage
            field_type = original_field_type
            if re.match(r'^M\d+$', original_field_type):
                field_type = 'message'
            elif re.match(r'^E\d+$', original_field_type):
                field_type = 'enum'
            
            # Determine wire type
            wire_type = self.WIRE_TYPES.get(field_type, 'LEN')
            if is_nested_message:
                wire_type = 'LEN'  # Embedded messages use LEN wire type
            elif is_enum:
                wire_type = 'VARINT'
            
            # Set size based on type (including tag for fixed-size types)
            # Per Protocol Buffers encoding: tag = (field_number << 3) | wire_type
            # Tag is encoded as a varint, followed by the value
            # For fixed-size types, we can determine the serialized size: tag + value
            # For variable-size types (VARINT with variable values, LEN), size is None until runtime values are available
            size_bytes = None
            tag_size = self._calculate_tag_size(field_number)
            
            if wire_type == 'I32':
                # I32 wire type (fixed32, sfixed32, float): tag + 4 bytes
                # Per spec: 4-byte little-endian encoding
                size_bytes = tag_size + 4
            elif wire_type == 'I64':
                # I64 wire type (fixed64, sfixed64, double): tag + 8 bytes
                # Per spec: 8-byte little-endian encoding
                size_bytes = tag_size + 8
            elif wire_type == 'VARINT':
                # VARINT wire type: tag + varint value (1-10 bytes)
                if field_type == 'bool' or is_enum:
                    # For bool and enum, they are encoded as if they were int32s
                    size_bytes = tag_size + 1
                else:
                    # Other VARINT types (int32, int64, etc.) - size depends on value, cannot determine from type
                    # Varint values can be 1-10 bytes depending on the number
                    size_bytes = None
            elif wire_type == 'LEN':
                # LEN wire type (string, bytes, embedded messages): tag + length varint + payload
                # Size = tag + length_varint_size + payload_size
                # Cannot determine without runtime values
                size_bytes = None
            
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
                size_bytes=size_bytes
            )
            fields.append(field)
        
        # Calculate message statistics
        max_field_number = max([f.field_number for f in fields]) if fields else 0
        has_repeated = any(f.cardinality == 'repeated' for f in fields)
        has_nested = len(nested_messages) > 0
        has_enums = len(enums) > 0
        
        # Calculate message size from type-based field sizes (only for fixed-size fields)
        # Variable-size fields (strings, bytes, nested messages) will be set from runtime data
        size_bytes = sum(f.size_bytes or 0 for f in fields if f.size_bytes)
        
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
            size_bytes=size_bytes if size_bytes > 0 else None,
            depth=depth
        )
    
    def analyze(self) -> BenchmarkAnalysis:
        """Perform complete analysis of the protobuf file"""
        # Parse enums first
        self.enums = self._parse_enums()
        
        # Parse all messages (we need all of them to handle nested messages properly)
        # We'll filter later when building output
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
                # Only add top-level messages to self.messages
                # Nested messages are already part of their parents
                if message.depth == 0:
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
        serialized_sizes = self._calculate_serialized_sizes(runtime_data)
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
            serialized_sizes=serialized_sizes,
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
            # Map field names to their string/bytes lengths
            field_string_lengths = defaultdict(list)  # field_name -> [lengths]
            field_bytes_lengths = defaultdict(list)    # field_name -> [lengths]
            
            # Extract string literals and their lengths, mapping to field names
            # Pattern: set_f<number> or set_<fieldname>(string_value)
            # Also handle nested: v6->set_f1(...)
            string_pattern = r'(?:->|\.)?set_([a-z0-9]+)\(["\']([^"\']+)["\']\)'
            for str_match in re.finditer(string_pattern, function_body):
                field_name = str_match.group(1)
                string_val = str_match.group(2)
                str_len = len(string_val.encode('utf-8'))
                string_lengths.append(str_len)
                field_string_lengths[field_name].append(str_len)
            
            # Extract bytes (they're also strings in the code)
            # Look for set_f calls with long strings (likely bytes)
            # Pattern: set_f<number>("very long string...")
            bytes_pattern = r'(?:->|\.)?set_(f\d+)\(["\']([^"\']{50,})["\']\)'  # Strings > 50 chars likely bytes
            for bytes_match in re.finditer(bytes_pattern, function_body):
                field_name = bytes_match.group(1)
                bytes_val = bytes_match.group(2)
                bytes_len = len(bytes_val.encode('utf-8'))
                bytes_lengths.append(bytes_len)
                field_bytes_lengths[field_name].append(bytes_len)
            
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
                total_bytes_bytes=sum(bytes_lengths),
                field_string_lengths=dict(field_string_lengths) if field_string_lengths else None,
                field_bytes_lengths=dict(field_bytes_lengths) if field_bytes_lengths else None
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
        """Calculate serialized sizes using runtime data when available
        
        Note: This method is deprecated in favor of calculate_serialized_size_from_set_function
        which provides more accurate sizes from actual Set_F1 function analysis.
        Returns None as sizes should be calculated from Set_F1 functions.
        """
        # Sizes are now calculated from Set_F1 functions, not approximated here
        return None
    
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


def extract_messages_from_benchmark_iteration(inc_file_path: str) -> Set[str]:
    """Extract message names that are serialized/deserialized in BenchmarkIteration"""
    if not Path(inc_file_path).exists():
        return set()
    
    try:
        with open(inc_file_path, 'r') as f:
            content = f.read()
    except Exception:
        return set()
    
    # Find BenchmarkIteration function
    bench_iter_match = re.search(r'inline\s+int\s+BenchmarkIteration', content)
    if not bench_iter_match:
        return set()
    
    # Find the function body
    start_pos = bench_iter_match.end()
    brace_start = content.find('{', start_pos)
    if brace_start == -1:
        return set()
    
    # Find matching closing brace
    depth = 0
    brace_end = brace_start
    for i in range(brace_start, len(content)):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                brace_end = i
                break
    
    function_body = content[brace_start + 1:brace_end]
    
    # Extract messages that are serialized/deserialized
    messages = set()
    
    # Find all _Serialize_F1 and _Deserialize_F1 calls
    serialize_pattern = r'(\w+)_Serialize_F1'
    deserialize_pattern = r'(\w+)_Deserialize_F1'
    
    for match in re.finditer(serialize_pattern, function_body):
        messages.add(match.group(1))
    
    for match in re.finditer(deserialize_pattern, function_body):
        messages.add(match.group(1))
    
    return messages


def find_nested_message_by_path(root_message: Message, message_path: List[str]) -> Optional[Message]:
    """Find a nested message by following a path like ['M15', 'M17', 'M18']
    
    The path represents the type hierarchy. If root_message is M15, then:
    - ['M15', 'M17', 'M18'] means: find M17 in M15, then find M18 in M17
    """
    current = root_message
    
    # If the first element matches root, skip it
    start_idx = 1 if message_path and message_path[0] == root_message.name else 0
    
    for msg_name in message_path[start_idx:]:
        # Find nested message in current message
        found = None
        for nested in current.nested_messages:
            if nested.name == msg_name:
                found = nested
                break
        if found:
            current = found
        else:
            return None
    return current


def get_message_for_var(var_name: str, var_to_message: Dict, message: Message, all_messages: Dict[str, Message]) -> Optional[Message]:
    """Get the message object for a given variable name"""
    if var_name == "message":
        return message
    elif var_name in var_to_message:
        msg_path, target_msg_name, _ = var_to_message[var_name]
        # Try to find from message structure first
        nested_msg = find_nested_message_by_path(message, msg_path)
        if nested_msg:
            return nested_msg
        # Fall back to all_messages dict
        if target_msg_name in all_messages:
            return all_messages[target_msg_name]
    return None


def calculate_serialized_size_from_set_function(msg_name: str, set_function_body: str, message: Message, all_messages: Dict[str, Message]) -> Tuple[int, Dict[str, int]]:
    """Calculate serialized size from runtime Set_F1 function and update field sizes
    
    Processes Set_F1 function line-by-line in a single pass:
    1. Track nested message variable assignments (e.g., M15::M17::M18* v11 = v9->mutable_f1();)
    2. Process set/add operations to determine sizes for variable-size fields from runtime values
    3. Update field.size_bytes directly in the message structure (for both top-level and nested messages)
    4. After processing, calculate nested message sizes and update parent field sizes
    5. Return total serialized size for the message (including nested messages) and nested message sizes
    
    Returns:
        Tuple of (total_size, nested_message_sizes) where nested_message_sizes maps message names to their sizes
    """
    total_size = 0
    
    # Track variable assignments: var_name -> (message_path_list, target_message_name, field_name)
    # Example: v11 -> (['M15', 'M17', 'M18'], 'M18', 'f1')
    var_to_message = {}  # var_name -> (message_path_list, target_msg_name, field_name)
    var_to_field = {}  # var_name -> (parent_message, field_name) for tracking which field contains the nested message
    
    # Pattern: M15::M17::M18* v11 = v9->mutable_f1();
    mutable_pattern = r'(\w+(?:::\w+)*)\*\s+(\w+)\s*=\s*(\w+)->mutable_(\w+)\(\)'
    
    # Pattern: var->set_field(...) or var->add_field(...)
    set_pattern = r'(\w+)->set_([a-z0-9]+)\('
    add_pattern = r'(\w+)->add_([a-z0-9]+)\('
    
    # Track repeated field values: (var_name, field_name) -> list of values
    repeated_field_values = {}  # (var_name, field_name) -> list of values
    
    # Process function body line by line
    lines = set_function_body.split('\n')
    
    # Single pass: track mutable assignments and process set/add operations
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line contains "=" - could be a mutable assignment
        if '=' in line:
            match = re.search(mutable_pattern, line)
            if match:
                full_msg_path = match.group(1)  # e.g., "M15::M17::M18"
                var_name = match.group(2)  # e.g., "v11"
                parent_var = match.group(3)  # e.g., "v9"
                field_name = match.group(4)  # e.g., "f1"
                
                # Split the full path into message names
                message_path = full_msg_path.split('::')
                target_msg_name = message_path[-1]  # e.g., "M18"
                var_to_message[var_name] = (message_path, target_msg_name, field_name)
                
                # Track which field in the parent message this nested message belongs to
                if parent_var == "message":
                    var_to_field[var_name] = (message, field_name)
                elif parent_var in var_to_message:
                    # Find the parent nested message
                    parent_msg = get_message_for_var(parent_var, var_to_message, message, all_messages)
                    if parent_msg:
                        var_to_field[var_name] = (parent_msg, field_name)
            continue

        # Check if line is a set or add operation
        match = None
        is_add_operation = False
        set_add_line = line[:line.find('(')+1]
        if 'set' in set_add_line:
            match = re.search(set_pattern, line)
        elif 'add' in set_add_line:
            match = re.search(add_pattern, line)
            is_add_operation = True

        if match:
            var_name = match.group(1)  # e.g., "message" or "v11"
            field_name = match.group(2)  # e.g., "f1"
            
            # Get the target message (could be top-level or nested)
            target_msg = get_message_for_var(var_name, var_to_message, message, all_messages)
            if not target_msg:
                continue
            
            # Find the field in the target message
            target_field = next((f for f in target_msg.fields if f.name == field_name), None)
            if not target_field:
                continue
            
            # For repeated fields, track values separately (both set and add operations)
            if target_field.cardinality == 'repeated':
                key = (var_name, field_name)
                if key not in repeated_field_values:
                    repeated_field_values[key] = []
                # Store the line for later processing
                repeated_field_values[key].append(line)
                continue  # Process repeated fields in a separate pass
            
            # Process non-repeated field: calculate size and update directly
            value_start = match.end()
            if value_start < len(line):
                first_char = line[value_start]
                
                if first_char == '"' or first_char == "'":
                    # String/bytes value - calculate size from line length
                    line_len = len(line)
                    str_len = line_len - (value_start + 1) - 3  # +1 to skip opening quote, -3 for ");"
                    
                    # Set size for LEN fields from runtime values
                    # Per Protocol Buffers encoding: LEN = tag + length_varint + payload
                    if target_field.wire_type == 'LEN':
                        if target_field.field_type == 'string' or target_field.field_type == 'bytes':
                            # Calculate size: tag + length_varint + payload
                            tag_size = ProtobufAnalyzer._calculate_tag_size(target_field.field_number)
                            length_varint_size = ProtobufAnalyzer._calculate_varint_size(str_len)
                            field_size = tag_size + length_varint_size + str_len
                            target_field.size_bytes = field_size

                # Check if the value is numeric (VARINT wire type)
                elif target_field.wire_type == 'VARINT' and target_field.field_type not in ['bool', 'enum']:
                    # Extract numeric value from the line
                    # Pattern: set_field(value) or add_field(value);
                    # Value could be: 123, 0x123, 123U, 123.0, etc.
                    value_match = re.search(r'(?:set_|add_)\w+\(([^)]+)\)', line)
                    if value_match:
                        num_str = value_match.group(1).strip()
                        try:
                            # Parse numeric value
                            if 'U' in num_str or 'u' in num_str:
                                # Unsigned integer
                                num_str_clean = num_str.replace('U', '').replace('u', '')
                                if '0x' in num_str_clean or '0X' in num_str_clean:
                                    num_val = int(num_str_clean, 16)
                                else:
                                    num_val = int(num_str_clean)
                            elif '0x' in num_str or '0X' in num_str:
                                # Hex integer
                                num_val = int(num_str, 16)
                            elif '.' in num_str:
                                # Float - for VARINT types this shouldn't happen, but handle it
                                num_val = int(float(num_str))
                            else:
                                # Regular integer
                                num_val = int(num_str)
                            
                            # Handle sint32/sint64 ZigZag encoding
                            # Per spec: (n << 1) ^ (n >> 31) for sint32, (n << 1) ^ (n >> 63) for sint64
                            if target_field.field_type in ['sint32', 'sint64']:
                                if num_val < 0:
                                    # ZigZag encode negative numbers
                                    if target_field.field_type == 'sint32':
                                        encoded_val = (num_val << 1) ^ (num_val >> 31)
                                    else:  # sint64
                                        encoded_val = (num_val << 1) ^ (num_val >> 63)
                                else:
                                    encoded_val = num_val << 1  # Positive numbers: 2 * p
                            elif target_field.field_type in ['int32', 'int64']:
                                # For int32/int64 with negative values, use two's complement
                                # Convert to unsigned for size calculation
                                if num_val < 0:
                                    if target_field.field_type == 'int32':
                                        # Two's complement of 32-bit negative number
                                        encoded_val = num_val & 0xFFFFFFFF
                                    else:  # int64
                                        # Two's complement of 64-bit negative number
                                        encoded_val = num_val & 0xFFFFFFFFFFFFFFFF
                                else:
                                    encoded_val = num_val
                            else:
                                # For uint32, uint64, bool, enum - values are already unsigned/positive
                                encoded_val = num_val
                            
                            # Calculate VARINT size: tag + varint value
                            tag_size = ProtobufAnalyzer._calculate_tag_size(target_field.field_number)
                            varint_size = ProtobufAnalyzer._calculate_varint_size(encoded_val)
                            field_size = tag_size + varint_size
                            target_field.size_bytes = field_size
                        except (ValueError, AttributeError):
                            # If we can't parse the value, leave size_bytes as None
                            raise ValueError(f"Failed to parse {line}, expected VARINT wire type for {target_field.field_type}")

    # Process repeated fields: accumulate values and calculate sizes
    for (var_name, field_name), value_lines in repeated_field_values.items():
        # Get the target message (could be top-level or nested)
        target_msg = get_message_for_var(var_name, var_to_message, message, all_messages)
        if not target_msg:
            continue
        
        # Find the field in the target message
        target_field = next((f for f in target_msg.fields if f.name == field_name), None)
        if not target_field or target_field.cardinality != 'repeated':
            continue
        
        # Determine if field is packable (primitive numeric types)
        # Per spec: only repeated fields of primitive numeric types can be packed
        # These are types that would normally use VARINT, I32, or I64 wire types
        is_packable = target_field.wire_type in ['VARINT', 'I32', 'I64']
        # Strings, bytes, and messages are not packable
        is_packable = is_packable and target_field.field_type not in ['string', 'bytes', 'message']
        
        if is_packable:
            # Packed repeated field: single LEN record with all values concatenated
            # Size = tag + length_varint + (sum of all element sizes)
            tag_size = ProtobufAnalyzer._calculate_tag_size(target_field.field_number)
            total_payload_size = 0
            
            for value_line in value_lines:
                # Extract value from set_field(value) or add_field(value)
                value_match = re.search(r'(?:set_|add_)\w+\(([^)]+)\)', value_line)
                if not value_match:
                    continue
                
                num_str = value_match.group(1).strip()
                try:
                    # Parse numeric value (same logic as VARINT processing)
                    if 'U' in num_str or 'u' in num_str:
                        num_str_clean = num_str.replace('U', '').replace('u', '')
                        if '0x' in num_str_clean or '0X' in num_str_clean:
                            num_val = int(num_str_clean, 16)
                        else:
                            num_val = int(num_str_clean)
                    elif '0x' in num_str or '0X' in num_str:
                        num_val = int(num_str, 16)
                    elif '.' in num_str:
                        num_val = int(float(num_str))
                    else:
                        num_val = int(num_str)
                    
                    # Encode value based on wire type
                    if target_field.wire_type == 'VARINT':
                        # Handle encoding (ZigZag, two's complement, etc.)
                        if target_field.field_type in ['sint32', 'sint64']:
                            if num_val < 0:
                                if target_field.field_type == 'sint32':
                                    encoded_val = (num_val << 1) ^ (num_val >> 31)
                                else:
                                    encoded_val = (num_val << 1) ^ (num_val >> 63)
                            else:
                                encoded_val = num_val << 1
                        elif target_field.field_type in ['int32', 'int64']:
                            if num_val < 0:
                                if target_field.field_type == 'int32':
                                    encoded_val = num_val & 0xFFFFFFFF
                                else:
                                    encoded_val = num_val & 0xFFFFFFFFFFFFFFFF
                            else:
                                encoded_val = num_val
                        else:
                            encoded_val = num_val
                        
                        varint_size = ProtobufAnalyzer._calculate_varint_size(encoded_val)
                        total_payload_size += varint_size
                    elif target_field.wire_type == 'I32':
                        total_payload_size += 4  # Fixed 4 bytes
                    elif target_field.wire_type == 'I64':
                        total_payload_size += 8  # Fixed 8 bytes
                except (ValueError, AttributeError):
                    continue
            
            if total_payload_size > 0:
                length_varint_size = ProtobufAnalyzer._calculate_varint_size(total_payload_size)
                field_size = tag_size + length_varint_size + total_payload_size
                target_field.size_bytes = field_size
        else:
            # Non-packed repeated field: multiple records (tag + value for each)
            # For strings/bytes: each is a LEN record
            # For messages: each is a LEN record
            total_size = 0
            tag_size = ProtobufAnalyzer._calculate_tag_size(target_field.field_number)
            
            for value_line in value_lines:
                if target_field.field_type in ['string', 'bytes']:
                    # For strings/bytes, find the opening quote to avoid slow regex on long strings
                    # Pattern: var->set_field("...") or var->add_field("...")
                    # Find the opening parenthesis and then the quote
                    paren_pos = value_line.find('(')
                    if paren_pos == -1:
                        continue
                    
                    # Look for quote after the opening parenthesis
                    quote_pos = -1
                    for i in range(paren_pos + 1, len(value_line)):
                        if value_line[i] == '"' or value_line[i] == "'":
                            quote_pos = i
                            break
                    
                    if quote_pos != -1:
                        # Calculate string length: line length - opening quote position - 1 - 3 for ");"
                        line_len = len(value_line)
                        str_len = line_len - (quote_pos + 1) - 3  # +1 to skip opening quote, -3 for ");"
                        length_varint_size = ProtobufAnalyzer._calculate_varint_size(str_len)
                        total_size += tag_size + length_varint_size + str_len
                elif target_field.field_type == 'message':
                    # Messages: tag + length_varint + message_size for each
                    # Note: We can't determine nested message size here without more context
                    # This will be handled by the nested message size calculation
                    # For now, we'll need to track this separately or handle it differently
                    pass  # TODO: Handle repeated nested messages
                else:
                    # For other types, use regex (shouldn't be common for non-packed repeated)
                    value_match = re.search(r'(?:set_|add_)\w+\(([^)]+)\)', value_line)
                    if value_match:
                        # Handle other types if needed
                        pass
            
            if total_size > 0:
                target_field.size_bytes = total_size

    # Calculate nested message sizes by summing their field sizes
    # Process bottom-up (deepest first) so parent fields can use nested message sizes
    nested_message_sizes = {}  # var_name -> size
    
    # Sort nested messages by depth (deepest first) to calculate bottom-up
    nested_vars_by_depth = []
    for var_name, (msg_path, target_msg_name, _) in var_to_message.items():
        nested_msg = get_message_for_var(var_name, var_to_message, message, all_messages)
        if nested_msg:
            depth = len(msg_path) - 1  # Depth in the path
            nested_vars_by_depth.append((depth, var_name, nested_msg))
    
    # Sort by depth descending (deepest first)
    nested_vars_by_depth.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate sizes for nested messages from deepest to shallowest
    for depth, var_name, nested_msg in nested_vars_by_depth:
        # Calculate the size of this nested message by summing its field sizes
        nested_size = 0
        for field in nested_msg.fields:
            if field.size_bytes is not None:
                nested_size += field.size_bytes
            elif field.is_nested_message and field.nested_message_name:
                # Check if we have a pre-calculated size for this nested-nested message
                # Find the variable that sets this nested message
                nested_var = None
                for var, (var_msg_path, var_target_msg_name, _) in var_to_message.items():
                    if var_target_msg_name == field.nested_message_name:
                        # Check if this variable's parent is our current var_name
                        if var in var_to_field:
                            parent_msg, parent_field_name = var_to_field[var]
                            if parent_msg == nested_msg:
                                nested_var = var
                                break
                
                if nested_var and nested_var in nested_message_sizes:
                    # Use pre-calculated size for nested message
                    # Per Protocol Buffers encoding: LEN = tag + length_varint + payload
                    nested_nested_size = nested_message_sizes[nested_var]
                    tag_size = ProtobufAnalyzer._calculate_tag_size(field.field_number)
                    length_varint_size = ProtobufAnalyzer._calculate_varint_size(nested_nested_size)
                    field_total_size = tag_size + length_varint_size + nested_nested_size
                    nested_size += field_total_size
                    field.size_bytes = field_total_size
        
        nested_message_sizes[var_name] = nested_size
        
        # Update the parent field's size_bytes to include the nested message
        if var_name in var_to_field:
            parent_msg, field_name = var_to_field[var_name]
            parent_field = next((f for f in parent_msg.fields if f.name == field_name), None)
            if parent_field and parent_field.is_nested_message:
                # Calculate size for nested message field
                # Per Protocol Buffers encoding: LEN = tag + length_varint + payload
                tag_size = ProtobufAnalyzer._calculate_tag_size(parent_field.field_number)
                length_varint_size = ProtobufAnalyzer._calculate_varint_size(nested_size)
                parent_field.size_bytes = tag_size + length_varint_size + nested_size
   

    # Calculate total size from all fields (only include fields with determined sizes)
    for field in message.fields:
        if field.size_bytes is not None:
            # Size determined from runtime values or type
            total_size += field.size_bytes
        # If size_bytes is None, the field size cannot be determined and is not included
        # This ensures we only count sizes we actually know

    # Map nested message sizes from variable names to message names
    nested_sizes_by_name = {}  # message_name -> size
    for var_name, size in nested_message_sizes.items():
        if var_name in var_to_message:
            _, target_msg_name, _ = var_to_message[var_name]
            nested_sizes_by_name[target_msg_name] = size

    return total_size, nested_sizes_by_name

# Print detailed field and size information for each message recursively
def print_message_details(msg: Message, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}Message: {msg.name} (depth: {msg.depth})")
    
    # Print message-level info
    print(f"{prefix}  Fields ({len(msg.fields)}):")
    for field in msg.fields:
        field_info = f"{prefix}    {field.name} (#{field.field_number}): {field.cardinality} {field.field_type}"
        if field.wire_type:
            field_info += f" [wire: {field.wire_type}]"
        if field.size_bytes is not None:
            field_info += f" [size: {field.size_bytes} bytes]"
        if field.is_nested_message and field.nested_message_name:
            field_info += f" -> {field.nested_message_name}"
        if field.is_enum and field.enum_name:
            field_info += f" -> enum {field.enum_name}"
        print(field_info)
    
    # Print nested messages recursively
    if msg.nested_messages:
        print(f"{prefix}  Nested messages ({len(msg.nested_messages)}):")
        for nested_msg in msg.nested_messages:
            print_message_details(nested_msg, indent + 2)
    
    # Print enums if any
    if msg.enums:
        print(f"{prefix}  Enums ({len(msg.enums)}):")
        for enum in msg.enums:
            print(f"{prefix}    {enum.get('name', 'Unknown')} with {len(enum.get('values', []))} values")


def analyze_hyperprotobench(base_path: str) -> Dict:
    """Analyze all benchmarks in HyperProtoBench directory"""
    base_path = Path(base_path)
    results = {}
    
    # Find all benchmark directories, skipping -ser and -deser versions
    bench_dirs = []
    
    for bench_dir in sorted(base_path.glob('bench*')):
        if not bench_dir.is_dir():
            continue
        
        # Skip -ser and -deser versions (calculated sizes are the same)
        if bench_dir.name.endswith('-ser') or bench_dir.name.endswith('-deser'):
            continue
        
        bench_dirs.append(bench_dir)
    
    # Sort benchmark directories
    bench_dirs.sort()

    print(f"Found {len(bench_dirs)} benchmarks to analyze: {', '.join([bench_dir.name for bench_dir in bench_dirs])}")
    
    for bench_dir in bench_dirs:
        proto_file = bench_dir / 'benchmark.proto'
        if not proto_file.exists():
            print(f"Warning: {proto_file} not found, skipping {bench_dir.name}")
            continue
        
        print(f"Analyzing {bench_dir.name}...")
        try:
            # First, extract which messages are actually used from BenchmarkIteration
            inc_file = bench_dir / 'benchmark.inc'
            print(f"  Extracting messages used from {inc_file}")
            messages_used = extract_messages_from_benchmark_iteration(str(inc_file))
            
            if not messages_used:
                print(f"  Warning: No messages found in BenchmarkIteration for {bench_dir.name}")
                continue
            
            print(f"  Found {len(messages_used)} messages: {', '.join(sorted(messages_used))}")
            
            # Now analyze only those messages
            analyzer = ProtobufAnalyzer(str(proto_file), messages_to_analyze=list(messages_used))
            print(f"  Initialized ProtobufAnalyzer for {bench_dir.name}")
            analysis = analyzer.analyze()
            
            # Extract Set_F1 function bodies and calculate actual sizes
            if not inc_file.exists():
                print(f"  Warning: {inc_file} not found")
                continue
            
            with open(inc_file, 'r') as f:
                inc_content = f.read()

            # Build map of all messages (including nested) for size calculation
            all_messages_map = {}
            def add_to_map(msg: Message):
                all_messages_map[msg.name] = msg
                for nested in msg.nested_messages:
                    add_to_map(nested)
            
            for msg in analysis.messages:
                add_to_map(msg)
            
            # Extract Set_F1 functions and calculate sizes
            message_sizes = {}  # message_name -> size (includes both top-level and nested)
            set_pattern = r'int\s+(\w+)_Set_F1[^{]*\{'
            
            i = 0
            while i < len(inc_content):
                match = re.search(set_pattern, inc_content[i:])
                if not match:
                    break
                
                start_pos = i + match.end() - 1
                msg_name = match.group(1)
                
                if msg_name not in messages_used:
                    i = start_pos + 1
                    continue
                
                # Find matching closing brace
                brace_end = analyzer._find_matching_brace(inc_content, start_pos)
                if brace_end == -1:
                    i = start_pos + 1
                    continue
                
                function_body = inc_content[start_pos + 1:brace_end]
                i = brace_end + 1
                
                # Find the message in our analysis
                msg_obj = None
                for msg in analysis.messages:
                    if msg.name == msg_name:
                        msg_obj = msg
                        break
                
                if msg_obj:
                    # Calculate actual serialized size and nested message sizes
                    size, nested_sizes = calculate_serialized_size_from_set_function(
                        msg_name, function_body, msg_obj, all_messages_map
                    )
                    message_sizes[msg_name] = size
                    # Add nested message sizes to the dictionary
                    for nested_msg_name, nested_size in nested_sizes.items():
                        message_sizes[nested_msg_name] = nested_size
                    # print_message_details(msg_obj)

            print(f"  Finished calculating serialized sizes for {bench_dir.name}")        
            
            # Convert to serializable format - only include used messages
            results[bench_dir.name] = {
                'benchmark_name': analysis.benchmark_name,
                'proto_file_path': analysis.proto_file_path,
                'syntax_version': analysis.syntax_version,
                'total_size_bytes': sum(message_sizes.values()),
                'top_level_messages': sorted(list(messages_used)),
                'messages': [],
                'message_sizes_bytes': message_sizes,
            }
            
            
            # Helper function to recursively add messages (only used ones and their nested messages)
            def add_message(msg: Message, parent_name: str = None, is_used: bool = False):
                # is_used indicates if this message or an ancestor is in messages_used
                msg_data = {
                    'name': msg.name,
                    'parent': parent_name,
                    'total_fields': msg.total_fields,
                    'max_field_number': msg.max_field_number,
                    'depth': msg.depth,
                    'has_repeated_fields': msg.has_repeated_fields,
                    'has_nested_messages': msg.has_nested_messages,
                    'has_enums': msg.has_enums,
                    'nested_message_count': len(msg.nested_messages),
                    'serialized_size_bytes': None,  
                    'fields': [],
                    'nested_messages': []
                }
                
                # Add size if available (for both top-level and nested messages)
                if msg.name in message_sizes:
                    msg_data['serialized_size_bytes'] = message_sizes[msg.name]
                
                for field in msg.fields:
                    field_data = {
                        'name': field.name,
                        'field_number': field.field_number,
                        'field_type': field.field_type,
                        'cardinality': field.cardinality,
                        'wire_type': field.wire_type,
                        'is_nested_message': field.is_nested_message,
                        'nested_message_name': field.nested_message_name,
                        'is_enum': field.is_enum,
                        'enum_name': field.enum_name,
                        'size_bytes': field.size_bytes,  # Include size_bytes for all fields (can be None)
                    }
                    
                    msg_data['fields'].append(field_data)
                
                # Recursively add nested messages (they're part of the serialized data)
                for nested_msg in msg.nested_messages:
                    nested_data = add_message(nested_msg, msg.name, is_used=True)
                    msg_data['nested_messages'].append(nested_data['name'])
                    # Add nested messages to the list
                    results[bench_dir.name]['messages'].append(nested_data)
                
                return msg_data
            
            # Add only used messages (and their nested messages will be added recursively)
            for msg in analysis.messages:
                if msg.name in messages_used:
                    msg_data = add_message(msg, is_used=True)
                    results[bench_dir.name]['messages'].append(msg_data)
            
            print(f"  Finished analyzing {bench_dir.name}")
            
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
        
        # Show serialized sizes if available
        if 'serialized_sizes' in data:
            sizes = data['serialized_sizes']
            print(f"\nSerialized Sizes:")
            sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            for msg_name, size in sorted_sizes:
                print(f"  {msg_name:20s}: {size:8,d} bytes")
            if 'total_size_bytes' in data:
                print(f"  {'TOTAL':20s}: {data['total_size_bytes']:8,d} bytes")
        
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
            if 'serialized_size_bytes' in msg:
                size_info = f", {msg['serialized_size_bytes']:,} bytes"
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
    
    script_dir = Path(__file__).parent
    # Resolve path - can be absolute or relative to script location
    if args.hyperprotobench_path:
        if Path(args.hyperprotobench_path).is_absolute():
            hyperprotobench_path = Path(args.hyperprotobench_path)
        else:
            hyperprotobench_path = script_dir / args.hyperprotobench_path
    
    if not hyperprotobench_path.exists():
        print(f"Error: HyperProtoBench path not found: {hyperprotobench_path}")
        return
    
    print(f"Analyzing protobuf workloads in: {hyperprotobench_path}")
    
    # Analyze all benchmarks
    start_time = time.time()
    results = analyze_hyperprotobench(str(hyperprotobench_path))
    end_time = time.time()
    print(f"Analysis complete in {end_time - start_time} seconds")
    
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

