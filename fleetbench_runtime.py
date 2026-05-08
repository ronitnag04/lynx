#!/usr/bin/env python3
"""
fleetbench_runtime.py — extract per-field string/bytes lengths from fleetbench
`access_message<N>.cc` files.

Fleetbench ships with 20 proto schemas (``fleetbench/proto/Message<N>.proto``)
and alongside each one, an auto-generated ``access_message<N>.cc`` that
populates those messages with realistic runtime values inside a set of
``Message<N>_Set_K`` functions. Unlike HyperProtoBench, there is no
``benchmark.inc`` — the set/get bodies live directly in these .cc files.

This module parses those .cc files to produce the same shape that
``proto_to_accel.py`` expects from HPB's ``.inc`` runtime data:

    {<message_simple_name>: {<field_name>: [<observed_string_lengths>, ...]}}

so the synthetic bench generator can feed realistic per-field string/bytes
lengths straight into the existing serializer/deserializer emission path.

Typical source pattern we parse:

    void Message0_Set_1(Message0* message, std::string* s) {
      Message0::M1* v0_0 = message->add_f_4();
      Message0::M1::M3::M5::M6* v3 = v2->mutable_f_3();
      v3->set_f_0(s->substr(0, 3));     // string/bytes, length 3
      v3->set_f_8(0x64);                // numeric — ignored
      message->set_f_2("literal");      // literal string, length 7
    }

`message` is the implicit top-level variable whose type is the file's
``Message<N>`` class; local variables are mapped to their declared leaf type
(``M6`` in the example above).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# `Message0_Set_K(Message0* message, std::string* s)` — the function header
# also tells us the top-level type, which `message->set_*` calls reference.
_FUNC_HDR_RE = re.compile(
    r"void\s+(Message\d+)_(?:Set|Get|Create|Destroy|Serialize|Deserialize)"
    r"_(\d+)\s*\(\s*\1\s*\*\s*message"
)

# Variable declaration: `Message0::M1::M3* v3 = ...;` — capture (leaf type, var).
# We accept any depth of `::` qualification; the leaf is the simple class name.
# Variable names are `v<digits>` or `v<digits>_<digits>` (the _0 suffix is used
# when the initializer is an `add_repeated()` call).
_VAR_DECL_RE = re.compile(
    r"(?:Message\d+)(?:::[A-Za-z_]\w*)*::([A-Za-z_]\w*)\s*\*\s*(v\d+(?:_\d+)?)\s*="
)

# `var->set_fieldname(...)` or `var.set_fieldname(...)` with substr or literal
# string arg. We handle two forms:
#   (1) s->substr(0, N)                 → length N
#   (2) "literal string"                → len(literal) (as utf-8 bytes)
# Numeric / enum / bool args are ignored — those aren't string/bytes fields.
#
# Cost of anchoring to `var->` vs. bare `set_f_N`: the fleetbench files contain
# `(void)` casts and comments but nothing else shaped like a method call, so
# this stays unambiguous.
_SET_SUBSTR_RE = re.compile(
    r"(message|v\d+(?:_\d+)?)\s*(?:->|\.)\s*set_([A-Za-z_]\w*)\s*"
    r"\(\s*s\s*->\s*substr\s*\(\s*0\s*,\s*(\d+)\s*\)\s*\)"
)
_SET_LITERAL_RE = re.compile(
    r"(message|v\d+(?:_\d+)?)\s*(?:->|\.)\s*set_([A-Za-z_]\w*)\s*"
    r"\(\s*\"((?:[^\"\\]|\\.)*)\"\s*\)"
)


@dataclass
class MessageRuntime:
    """Per-message field length observations."""
    # field_name -> list of observed string/bytes lengths
    field_lengths: Dict[str, List[int]]


def _decode_c_string_literal(lit: str) -> bytes:
    """Decode the contents of a C++ double-quoted string literal.

    The regex already stripped the outer quotes. We only need to handle the
    standard C escapes fleetbench uses (\\, \", \n, \t, \\xHH, \\0, ...).
    Anything exotic falls back to treating the raw bytes as UTF-8 — the
    downstream consumer only cares about the byte length.
    """
    try:
        # Python's 'unicode_escape' handles \x, \n, \t, \\, \" etc.
        return lit.encode("latin-1").decode("unicode_escape").encode("utf-8")
    except Exception:
        return lit.encode("utf-8", errors="replace")


def _parse_function_body(body: str, top_msg: str) -> Dict[str, Dict[str, List[int]]]:
    """Parse one Message<N>_Set_K function body.

    Returns partial observations: {leaf_msg_name: {field_name: [lengths...]}}.
    """
    # Build var -> leaf class table. `message` itself maps to the top-level.
    var_types: Dict[str, str] = {"message": top_msg}
    for m in _VAR_DECL_RE.finditer(body):
        leaf = m.group(1)
        var = m.group(2)
        # Last-decl-wins if a name is reused (doesn't happen in practice,
        # but a safe default). Leaf is already the simple class name.
        var_types[var] = leaf

    result: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for m in _SET_SUBSTR_RE.finditer(body):
        var, field, length_s = m.group(1), m.group(2), m.group(3)
        owner = var_types.get(var)
        if owner is None:
            continue
        result[owner][field].append(int(length_s))

    for m in _SET_LITERAL_RE.finditer(body):
        var, field, literal = m.group(1), m.group(2), m.group(3)
        owner = var_types.get(var)
        if owner is None:
            continue
        result[owner][field].append(len(_decode_c_string_literal(literal)))

    # Freeze defaultdicts before returning.
    return {owner: dict(fields) for owner, fields in result.items()}


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Return the index of the '}' that matches the '{' at open_pos, or -1."""
    depth = 0
    i = open_pos
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def extract_from_access_cc(cc_path: Path) -> Dict[str, Dict[str, List[int]]]:
    """Extract per-message field length observations from one access_message<N>.cc.

    Returns {<msg_simple_name>: {<field_name>: [<lengths>, ...]}}.
    """
    text = cc_path.read_text()
    out: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for hdr in _FUNC_HDR_RE.finditer(text):
        top_msg = hdr.group(1)
        # Find the start of the function body `{` after the header match.
        brace_open = text.find("{", hdr.end())
        if brace_open == -1:
            continue
        brace_close = _find_matching_brace(text, brace_open)
        if brace_close == -1:
            continue
        body = text[brace_open + 1:brace_close]

        partial = _parse_function_body(body, top_msg)
        for owner, fields in partial.items():
            for fname, lengths in fields.items():
                out[owner][fname].extend(lengths)

    return {owner: dict(fields) for owner, fields in out.items()}


def extract_for_messages(
    proto_dir: Path,
    message_ids: List[int],
) -> Dict[str, Dict[str, List[int]]]:
    """Union runtime data across a set of Message<id>.proto files.

    For each id, reads ``access_message<id>.cc`` from ``proto_dir`` and merges
    the per-message observations. Fleetbench nested types use simple names
    (``M1``, ``M2``, ...) scoped inside ``Message<N>``; these can collide
    across different top-level messages (e.g. ``Message0::M1`` vs
    ``Message3::M1``). For the purpose of driving string-length sampling this
    is fine — the consumer looks up by simple name and any realistic-looking
    length distribution works — so collisions are merged silently.
    """
    merged: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for mid in message_ids:
        cc = proto_dir / f"access_message{mid}.cc"
        if not cc.is_file():
            raise FileNotFoundError(f"Expected access .cc at {cc}")
        per_msg = extract_from_access_cc(cc)
        for owner, fields in per_msg.items():
            for fname, lengths in fields.items():
                merged[owner][fname].extend(lengths)
    return {owner: dict(fields) for owner, fields in merged.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proto-dir", type=Path,
                    default=Path(__file__).resolve().parent / "fleetbench" / "fleetbench" / "proto",
                    help="Directory holding Message<N>.proto + access_message<N>.cc.")
    ap.add_argument("--messages", type=str, required=True,
                    help="Comma-separated message ids (e.g. '0,6,13,9,17') "
                         "or 'all' to dump every Message<N>.")
    ap.add_argument("--output", type=Path,
                    help="Write merged JSON to this path instead of stdout.")
    args = ap.parse_args()

    if args.messages == "all":
        ids = sorted(int(p.stem.replace("access_message", ""))
                     for p in args.proto_dir.glob("access_message*.cc"))
    else:
        ids = [int(x) for x in args.messages.split(",") if x.strip()]

    merged = extract_for_messages(args.proto_dir, ids)
    payload = json.dumps(merged, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload)
        n_fields = sum(len(v) for v in merged.values())
        print(f"Wrote {args.output}: {len(merged)} messages, {n_fields} fields",
              file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
