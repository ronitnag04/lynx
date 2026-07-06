"""
Structural (RTL-derived) features for the ProtoAcc area cost model.

Each feature is a physically-motivated quantity extracted from the Chisel
sources under ``generators/protoacc/src/main/scala``. The feature set is
designed so that:

  * a hierarchical additive area model
    ``area = bias + sum_submodule f_submodule(features_of_that_submodule)``
    can be fit per-submodule from a Sky130 synthesis area report;
  * the feature values extrapolate cleanly to the extremes of the design
    space (Pareto search targets those corners);
  * per-parameter monotonicity is preserved (larger depth -> larger
    feature -> larger predicted area).

Every constant here (entry widths, replication counts, fixed submodule
sizes) is annotated with the RTL site it came from so downstream users
can verify against the current Chisel sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Global structural constants pulled from the RTL.
# ---------------------------------------------------------------------------

# MAX_NESTED_LEVELS = 25 (util.scala:96) drives the depth of every
# per-submessage Reg(Vec) stack in the serializer's descriptor table
# handler and in the field handlers. Not a config knob today but affects
# fixed-cost sizes below.
MAX_NESTED_LEVELS = 25

# L1MemHelper per-source response FIFO shape. Each outstanding request gets
# a dedicated Queue(L1RespInternal, 4, flow=true) indexed by TL source id.
# See l1shim_interface.scala.
L1MEMHELPER_RESP_ENTRY_BITS = 128
L1MEMHELPER_RESP_DEPTH = 4

# L1MemHelper has fixed-depth input request queue + response queue
# regardless of the outstanding-req knob (l1shim_interface.scala, depth 4).
L1MEMHELPER_FIXED_REQQ_BITS = 4 * 198
L1MEMHELPER_FIXED_RESPQ_BITS = 4 * 128
L1MEMHELPER_FIXED_BITS = L1MEMHELPER_FIXED_REQQ_BITS + L1MEMHELPER_FIXED_RESPQ_BITS

# ---------------------------------------------------------------------------
# Per-parameter entry widths (bits per FIFO entry). Derived from the Bundle
# definitions in the Chisel sources.
# ---------------------------------------------------------------------------

# Deserializer entry widths.
DES_ENTRY_WIDTH_BITS: dict[str, int] = {
    # Two queues: proto_parse_info_out_queue + do_proto_parse_out_queue
    # (des/commandrouter.scala:54-55). Each RoCCCommand ~= 192 bits.
    "des_cr_rocc_commands": 192,
    # descriptortablehandler_des.scala:209 -- Queue(L1ReqInternal, D)
    "des_dth_l1_reqs": 198,
    # descriptortablehandler_des.scala:212 -- Queue(DescriptorRequest, D)
    "des_dth_fd_reqs": 192,
    # descriptortablehandler_des.scala:218 + 221 -- two queues share depth.
    # Sum widths so a single "queue_bits" feature captures the joint cost.
    "des_dth_fd_resps": 198 + 128,
    # des/fixedwriter.scala:29 -- Queue(L1ReqInternal, D)
    "des_fw_l1_reqs": 198,
    # des/memloader.scala:55 -- Queue(BufInfoBundle, D)
    "des_ml_buf_info_q": 192,
    # des/memloader.scala:57 -- Queue(LoadInfoBundle, D). Very narrow!
    "des_ml_load_info_q": 8,
}

# Two of the deserializer knobs drive L1MemHelper replication rather than
# a single queue depth. Handled specially in feature extraction.
DES_L1MEMHELPER_KNOBS = (
    "des_top_descriptor_reqs",
    "des_top_memloader_reqs",
)

# The CommandRouter emits TWO queues at the same depth (protoacc.scala
# CommandRouter). Model the joint storage cost.
DES_CR_QUEUE_MULTIPLICITY: dict[str, int] = {
    "des_cr_rocc_commands": 2,
}

# Serializer entry widths.
SER_ENTRY_WIDTH_BITS: dict[str, int] = {
    # ser/commandrouter.scala:54-55 -- two Queue(RoCCCommand, D)
    "ser_cr_rocc_commands": 192,
    # descriptortablehandler_ser.scala:142 -- HasBitsRequestMetaBundle
    "ser_dth_hasbits_reqs": 229,
    # descriptortablehandler_ser.scala:234 -- DescrRequestUopBundle
    "ser_dth_descriptor_reqs": 230,
    # descriptortablehandler_ser.scala:322 -- Queue(Bool, D). 1 bit per entry.
    "ser_dth_reg_resps": 1,
    # descriptortablehandler_ser.scala:325 -- DescrRequestUopBundle (mirror)
    "ser_dth_reqs_meta": 230,
    # descriptortablehandler_ser.scala:522 -- DescrToHandlerBundle
    "ser_dth_fh_outputs": 108,
    # ser/memwriter.scala:30 -- Queue(WriterBundle, D)
    "ser_mw_write_input": 141,
    # ser/memwriter.scala:33 -- Queue(WriterBundle, D)
    "ser_mw_write_inject": 141,
    # ser/memwriter.scala:113 -- Queue(UInt(64.W), D)
    "ser_mw_write_ptrs": 64,
}

SER_CR_QUEUE_MULTIPLICITY: dict[str, int] = {
    "ser_cr_rocc_commands": 2,
}

# ---------------------------------------------------------------------------
# Per-field-handler internal cost (in "structural bit" units).
#
# This is the total flip-flop-bit weight of everything inside a single
# SerFieldHandler that gets replicated when ProtoAccelSerFieldHandlers
# grows. It is a *feature*, not an area estimate -- the model learns the
# gate-per-bit ratio during fitting.
# ---------------------------------------------------------------------------

# WriterBundle output queue inside each SerFieldHandler (fixed depth 4).
_SER_FH_OUTPUT_QUEUE_BITS = 141 * 4
# 3 varint encoders per handler (key/size/data) -- rough combinational
# equivalent expressed as bits so it lives on the same scale.
_SER_FH_VARINT_STRUCT_BITS = 3 * 80
# Per-handler L1MemHelper (protoacc_serializer.scala:38-42): fixed queues
# + a response FIFO vector whose size actually tracks the shared outstanding
# knob (via L1MemHelper's numOutstandingReqs) -- but the SER side does not
# expose that knob today, so use the L1MemHelper default of 4.
_SER_FH_L1MEMHELPER_DEFAULT_OUTSTANDING = 4
_SER_FH_L1MEMHELPER_BITS = (
    L1MEMHELPER_FIXED_BITS
    + _SER_FH_L1MEMHELPER_DEFAULT_OUTSTANDING
    * L1MEMHELPER_RESP_DEPTH
    * L1MEMHELPER_RESP_ENTRY_BITS
)
# Per-handler control state (order of magnitude; ~200 register bits).
_SER_FH_CONTROL_BITS = 200

PER_SER_FIELD_HANDLER_STRUCT_BITS = (
    _SER_FH_OUTPUT_QUEUE_BITS
    + _SER_FH_VARINT_STRUCT_BITS
    + _SER_FH_L1MEMHELPER_BITS
    + _SER_FH_CONTROL_BITS
)

# ---------------------------------------------------------------------------
# Fixed-cost bit counts per submodule (parameter-independent).
# These become the per-submodule bias features in the model.
# ---------------------------------------------------------------------------

# DES: MemLoader byte-lane FIFOs (16 x 64-entry x 8-bit, memloader.scala:143).
DES_FIXED_ML_BYTELANE_BITS = 16 * 64 * 8
# DES: HasbitsWriter buffer (depth 10 x ~97 bits, hasbits_writer.scala:33).
DES_FIXED_HB_WRITER_BITS = 10 * 97
# DES: FieldHandler nested stacks (5 vec regs of MAX_NESTED_LEVELS x 64 b
# average; fieldhandler.scala:60-64). Rough aggregate.
DES_FIXED_FH_STACK_BITS = 5 * MAX_NESTED_LEVELS * 64
# DES: TLBs + PTW: not exposed here as bits because area is macro-heavy;
# constant per-side bias absorbs it.

# SER: SerDescriptorTableHandler nested stacks (6 vec regs of 25 x 64 b
# average; descriptortablehandler_ser.scala).
SER_FIXED_DTH_STACK_BITS = 6 * MAX_NESTED_LEVELS * 64
# SER: SerMemwriter nested size stack (1 vec reg of 25 x 64 b).
SER_FIXED_MW_STACK_BITS = MAX_NESTED_LEVELS * 64
# SER: 16 per-byte write queues inside SerMemwriter (fixed, not parameterized).
SER_FIXED_MW_BYTELANE_BITS = 16 * 64 * 8

# ---------------------------------------------------------------------------
# Feature names -- kept in canonical order so downstream vectors align.
#
# Convention: features that live inside submodule X are prefixed x_. The
# hierarchical cost model can slice the vector by prefix to fit each
# submodule's sub-model on just its own features.
# ---------------------------------------------------------------------------

# Bucket -> list of Yosys top module names to run ``stat -top <mod>`` on.
# Every cell reachable from these tops (transitively) is attributed to the
# bucket. Buckets are non-overlapping when the Chisel hierarchy is a strict
# tree, which it is for ProtoAcc.
#
# The parser looks these up to know which ``stat -top <mod>`` invocations to
# issue and how to bucket the ``ram_<D>x<W>`` children.
#
# NOTE: L1MemHelper instances share a class in the Chisel source, so firtool
# emits them as ``L1MemHelper``, ``L1MemHelper_1``, ``L1MemHelper_2``. The
# parser matches them to the ``descr_l1`` / ``ml_l1`` / ``fw_l1`` buckets by
# probing each instance's response-FIFO-vector size and comparing against the
# config's ``des_top_descriptor_reqs`` and ``des_top_memloader_reqs``.
# L1MemHelper instance ordering is stable: firtool emits them in Chisel-source
# order. In protoacc.scala they are declared as (mem_descr, mem_memloader,
# mem_hasbits, mem_fixedwriter), which becomes:
#   L1MemHelper          -> mem_descr        (descr_l1)
#   L1MemHelper_1        -> mem_memloader    (ml_l1)
#   L1MemHelper_2        -> mem_hasbits      (hbw_l1)
#   L1MemHelperWriteFast -> mem_fixedwriter  (fw_l1, different class)
DES_BUCKET_TO_YOSYS_TOPS: dict[str, tuple[str, ...]] = {
    "top":      ("ProtoAccel",),
    "cr":       ("CommandRouter",),
    "dth":      ("DescriptorTableHandler",),
    "fh":       ("FieldHandler",),
    "fw":       ("FixedWriter",),
    "hbw":      ("HasBitsWriter",),
    "ml":       ("MemLoader",),
    "descr_l1": ("L1MemHelper",),
    "ml_l1":    ("L1MemHelper_1",),
    "hbw_l1":   ("L1MemHelper_2",),
    "fw_l1":    ("L1MemHelperWriteFast",),
    "tlb":      ("DTLB_2", "PMAChecker", "PMPChecker_s3", "OptimizationBarrier_TLBEntryData"),
    "tl_glue":  ("TLBuffer", "TLXbar", "TLWidthWidget", "Repeater"),
}

SER_BUCKET_TO_YOSYS_TOPS: dict[str, tuple[str, ...]] = {
    "top":     ("ProtoAccelSerializer",),
    "cr":      ("CommandRouterSerializer",),
    "dth":     ("SerDescriptorTableHandler",),
    "mfh":     ("MultiFieldHandler", "FieldDispatchRouter", "MemWriteArbiter"),
    "fh":      ("SerFieldHandler",),
    "mw":      ("SerMemwriter",),
    "varint":  ("CombinationalVarintEncode",),
    # Same L1MemHelper ambiguity as DES; on SER side we have per-handler
    # helpers + one write-fast for the memwriter path.
    "fh_l1":   (),
    "mw_l1":   ("L1MemHelperWriteFast",),
    "tlb":     ("DTLB_2", "PMAChecker", "PMPChecker_s3", "OptimizationBarrier_TLBEntryData"),
    "tl_glue": ("TLBuffer", "TLXbar", "TLWidthWidget", "Repeater"),
}

DES_SUBMODULES: tuple[str, ...] = tuple(DES_BUCKET_TO_YOSYS_TOPS.keys())
SER_SUBMODULES: tuple[str, ...] = tuple(SER_BUCKET_TO_YOSYS_TOPS.keys())


def bucket_to_yosys_tops(side: str) -> dict[str, tuple[str, ...]]:
    if side == "des":
        return DES_BUCKET_TO_YOSYS_TOPS
    if side == "ser":
        return SER_BUCKET_TO_YOSYS_TOPS
    raise ValueError(f"Unknown side: {side!r}")


@dataclass(frozen=True)
class QueueFeatureSpec:
    """Per-queue feature recipe.

    A queue's storage bit count is ``multiplicity * depth * entry_width``.
    ``addr_bits`` (log2 of depth) is included as a separate feature because
    it correlates with control-logic gate count, not storage.
    """
    knob: str            # parameter key that sets the depth
    entry_bits: int      # bits per FIFO entry (constant from RTL)
    multiplicity: int    # how many parallel queues share this depth (usually 1)
    submodule: str       # which submodule bucket this queue belongs to


DES_QUEUES: tuple[QueueFeatureSpec, ...] = (
    QueueFeatureSpec("des_cr_rocc_commands", 192, 2, "cr"),
    QueueFeatureSpec("des_dth_l1_reqs",      198, 1, "dth"),
    QueueFeatureSpec("des_dth_fd_reqs",      192, 1, "dth"),
    QueueFeatureSpec("des_dth_fd_resps",     198 + 128, 1, "dth"),
    QueueFeatureSpec("des_fw_l1_reqs",       198, 1, "fw"),
    QueueFeatureSpec("des_ml_buf_info_q",    192, 1, "ml"),
    QueueFeatureSpec("des_ml_load_info_q",     8, 1, "ml"),
)

SER_QUEUES: tuple[QueueFeatureSpec, ...] = (
    QueueFeatureSpec("ser_cr_rocc_commands",     192, 2, "cr"),
    QueueFeatureSpec("ser_dth_hasbits_reqs",     229, 1, "dth"),
    QueueFeatureSpec("ser_dth_descriptor_reqs",  230, 1, "dth"),
    QueueFeatureSpec("ser_dth_reg_resps",          1, 1, "dth"),
    QueueFeatureSpec("ser_dth_reqs_meta",        230, 1, "dth"),
    QueueFeatureSpec("ser_dth_fh_outputs",       108, 1, "dth"),
    QueueFeatureSpec("ser_mw_write_input",       141, 1, "mw"),
    QueueFeatureSpec("ser_mw_write_inject",      141, 1, "mw"),
    QueueFeatureSpec("ser_mw_write_ptrs",         64, 1, "mw"),
)

# Which knob drives which bucket. Used by the fitter to slice feature columns
# into per-bucket submodels.
DES_BUCKET_KNOBS: dict[str, tuple[str, ...]] = {
    "cr":       ("des_cr_rocc_commands",),
    "dth":      ("des_dth_l1_reqs", "des_dth_fd_reqs", "des_dth_fd_resps"),
    "fh":       (),
    "fw":       ("des_fw_l1_reqs",),
    "hbw":      (),
    "ml":       ("des_ml_buf_info_q", "des_ml_load_info_q"),
    "descr_l1": ("des_top_descriptor_reqs",),
    "ml_l1":    ("des_top_memloader_reqs",),
    "hbw_l1":   (),
    "fw_l1":    (),
    "top":      (),
    "tlb":      (),
    "tl_glue":  (),
}

SER_BUCKET_KNOBS: dict[str, tuple[str, ...]] = {
    "cr":      ("ser_cr_rocc_commands",),
    "dth":     ("ser_dth_hasbits_reqs", "ser_dth_descriptor_reqs",
                "ser_dth_reg_resps", "ser_dth_reqs_meta", "ser_dth_fh_outputs"),
    "fh":      ("ser_field_handlers",),
    "mfh":     ("ser_field_handlers",),
    "mw":      ("ser_mw_write_input", "ser_mw_write_inject", "ser_mw_write_ptrs"),
    "varint":  ("ser_field_handlers",),
    "fh_l1":   ("ser_field_handlers",),
    "mw_l1":   (),
    "top":     (),
    "tlb":     (),
    "tl_glue": (),
}


def bucket_knobs(side: str) -> dict[str, tuple[str, ...]]:
    if side == "des":
        return DES_BUCKET_KNOBS
    if side == "ser":
        return SER_BUCKET_KNOBS
    raise ValueError(f"Unknown side: {side!r}")


def _log2(x: float | int) -> float:
    return float(np.log2(max(int(x), 1)))


def _queue_features(spec: QueueFeatureSpec, depth: int) -> dict[str, float]:
    total_bits = float(spec.multiplicity * depth * spec.entry_bits)
    return {
        f"{spec.knob}__bits":     total_bits,
        f"{spec.knob}__depth":    float(depth),
        f"{spec.knob}__addrbits": _log2(depth),
        # SRAM-vs-flops indicators; two thresholds so the fit can pick
        # whichever matches the technology library's cost curve.
        f"{spec.knob}__is_sram_1k": 1.0 if total_bits >= 1024 else 0.0,
        f"{spec.knob}__is_sram_2k": 1.0 if total_bits >= 2048 else 0.0,
    }


def _l1memhelper_features(prefix: str, n_outstanding: int) -> dict[str, float]:
    """Structural features for an L1MemHelper instance whose outstanding-req
    depth is ``n_outstanding``.

    The dominant storage term is the per-source response FIFO vector:
    ``n_outstanding * 4 * 128`` bits. Everything else is small but non-zero.
    """
    resp_bits = float(n_outstanding * L1MEMHELPER_RESP_DEPTH * L1MEMHELPER_RESP_ENTRY_BITS)
    outstanding_addr_bits = float(n_outstanding * 4 * (4 + _log2(n_outstanding)))
    tag_queue_bits = float(n_outstanding * 2 * _log2(n_outstanding))
    total = resp_bits + outstanding_addr_bits + tag_queue_bits + L1MEMHELPER_FIXED_BITS
    return {
        f"{prefix}__resp_bits":       resp_bits,
        f"{prefix}__meta_bits":       outstanding_addr_bits + tag_queue_bits,
        f"{prefix}__tag_bits":        _log2(n_outstanding),
        f"{prefix}__outstanding":     float(n_outstanding),
        f"{prefix}__is_sram_2k":      1.0 if resp_bits >= 2048 else 0.0,
        f"{prefix}__total_bits":      total,
    }


def des_features(cfg: Mapping[str, int]) -> dict[str, float]:
    """Structural features for one deserializer configuration.

    ``cfg`` must contain every ``des_*`` key. Returns a dict with:
      * one block per submodule (queue bits/depths/addrbits/sram flags)
      * two L1MemHelper feature blocks (descr and ml)
      * a ``fixed__const_bits`` bias term for parameter-independent storage
    """
    feats: dict[str, float] = {}

    feats.update(_l1memhelper_features("descr_l1", int(cfg["des_top_descriptor_reqs"])))
    feats.update(_l1memhelper_features("ml_l1",    int(cfg["des_top_memloader_reqs"])))
    # hbw_l1 and fw_l1 use the L1MemHelper default outstanding depth (32 for
    # plain L1MemHelper, 4 for L1MemHelperWriteFast). Numbers cross-checked
    # against a real synth.log; treat as constants attached to their buckets.
    feats.update(_l1memhelper_features("hbw_l1", 32))
    feats.update(_l1memhelper_features("fw_l1", 4))

    for spec in DES_QUEUES:
        feats.update(_queue_features(spec, int(cfg[spec.knob])))

    # Fixed submodule bit-count features, attributed to their true buckets.
    feats["ml__bytelane_bits"] = float(DES_FIXED_ML_BYTELANE_BITS)
    feats["hbw__buffer_bits"]  = float(DES_FIXED_HB_WRITER_BITS)
    feats["fh__stack_bits"]    = float(DES_FIXED_FH_STACK_BITS)
    return feats


def ser_features(cfg: Mapping[str, int]) -> dict[str, float]:
    """Structural features for one serializer configuration.

    Handles the multiplicative ``ser_field_handlers`` term explicitly:

      * ``fh__handlers``, ``fh__handlers_log2``,
        ``fh__handlers_x_log2`` (n log n arbitration term),
      * ``fh__per_handler_bits`` = N * PER_SER_FIELD_HANDLER_STRUCT_BITS
        (the total flop-bit weight replicated across handlers).

    Queue-depth knobs inside the SerDescriptorTableHandler are NOT
    replicated by handler count -- the DTH is a single instance. So the
    ``dth`` submodule features are independent of handler count.
    """
    feats: dict[str, float] = {}

    n_handlers = int(cfg["ser_field_handlers"])
    feats["fh__handlers"]            = float(n_handlers)
    feats["fh__handlers_log2"]       = _log2(n_handlers)
    feats["fh__handlers_x_log2"]     = float(n_handlers) * _log2(n_handlers)
    feats["fh__per_handler_bits"]    = float(n_handlers) * PER_SER_FIELD_HANDLER_STRUCT_BITS

    # MultiFieldHandler contains FieldDispatchRouter + MemWriteArbiter --
    # arbitration structures that scale with handler count.
    feats["mfh__linear"]             = float(n_handlers)
    feats["mfh__log"]                = _log2(n_handlers)
    feats["mfh__handlers_x_log2"]    = float(n_handlers) * _log2(n_handlers)

    # Per-handler L1MemHelper is replicated N times inside the field-handler
    # bank; its bucket is fh_l1.
    feats["fh_l1__handlers"]         = float(n_handlers)
    feats["fh_l1__handlers_x_log2"]  = float(n_handlers) * _log2(n_handlers)

    # Per-handler varint encoder count (3 per handler + 1 in the memwriter).
    feats["varint__count"]           = float(3 * n_handlers + 1)

    for spec in SER_QUEUES:
        feats.update(_queue_features(spec, int(cfg[spec.knob])))

    feats["dth__stack_bits"]        = float(SER_FIXED_DTH_STACK_BITS)
    feats["mw__stack_bits"]         = float(SER_FIXED_MW_STACK_BITS)
    feats["mw__bytelane_bits"]      = float(SER_FIXED_MW_BYTELANE_BITS)
    return feats


def features_for_side(cfg: Mapping[str, int], side: str) -> dict[str, float]:
    if side == "des":
        return des_features(cfg)
    if side == "ser":
        return ser_features(cfg)
    raise ValueError(f"Unknown side: {side!r}")


def feature_names_for_side(side: str) -> tuple[str, ...]:
    """Canonical, sorted feature name order for a side."""
    from .defaults import DEFAULT_CONFIG_BY_SIDE
    return tuple(sorted(features_for_side(DEFAULT_CONFIG_BY_SIDE[side], side).keys()))


def features_matrix(
    cfgs: Sequence[Mapping[str, int]], side: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return (X, feature_names) for a batch of configs on one side.

    ``feature_names`` are sorted for determinism; the same order is
    returned across calls with the same ``side``.
    """
    names = feature_names_for_side(side)
    if not cfgs:
        return np.empty((0, len(names)), dtype=np.float64), names
    X = np.empty((len(cfgs), len(names)), dtype=np.float64)
    for i, c in enumerate(cfgs):
        d = features_for_side(c, side)
        for j, k in enumerate(names):
            X[i, j] = d[k]
    return X, names
