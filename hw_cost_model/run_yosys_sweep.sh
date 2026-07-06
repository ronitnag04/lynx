#!/usr/bin/env bash
# run_yosys_sweep.sh -- sweep ProtoAcc hardware configs through Yosys and
# append per-bucket cell/ram_bits rows to a CSV consumed by
# hw_cost_model/fit_from_yosys.py.
#
# Uses its own S3 build cache, disjoint from run_sweep.sh's:
#   s3://ronitnag04-lynx/synth_build_files/<cls>.zip
# Only exact class-name matches are considered -- no cross-lookup into
# run_sweep.sh's verilator_build_files/ prefix, no Sweep-counterpart
# fallback. Elaborate misses run a local ``make verilog`` under
# $VERILATOR_DIR; those complete in a few minutes and the added
# complexity of a two-prefix search isn't worth it for the low observed
# hit rate.
#
# Per-config worker (mirrors run_sweep.sh's pattern):
#   1. Try _s3_pull_build; on miss/failure, run ``make verilog
#      CONFIG=<cls>`` via chipyard's sims/verilator flow.
#   2. Run zachjs-sv2v -DSYNTHESIS on the resulting .sv files.
#   3. Run yosys: read_verilog + hierarchy -top <TOP> + synth + stat.
#   4. Parse the synth.log with hw_cost_model.parse_yosys_stat and append
#      one CSV row under flock.
#   5. Optionally _s3_upload_build (--push-s3-builds), then unless
#      --keep-artifacts always delete the per-config generated-src tree
#      to bound disk usage.
#
# Prereqs (already installed in the chipyard conda env):
#   - yosys              (conda-forge)
#   - zachjs-sv2v (sv2v) (litex-hub)
#   - awscli             (for S3 cache; profile "lynx")
#   - unzip, zip
#   - python (with the hw_cost_model package importable)
#
# Usage:
#   ./run_yosys_sweep.sh \
#       --side des \
#       --sweep-csv /path/to/sweep_configs.csv \
#       --output    /path/to/yosys_sweep_results.csv \
#       --workers 32 \
#       [--pull-s3-builds] [--push-s3-builds] [--keep-artifacts]
#
# The sweep-csv must have the columns (config_name, side, <des_* or ser_*
# param columns>) -- exactly what gen_sweep_configs.py already writes to
# yosys_synth_sweep_configs.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYNX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHIPYARD_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
VERILATOR_DIR="$CHIPYARD_ROOT/sims/verilator"
VERILATOR_GEN_ROOT="$VERILATOR_DIR/generated-src"
CLASSPATH_JAR="$CHIPYARD_ROOT/.classpath_cache/chipyard.jar"

# S3 build cache. Distinct from run_sweep.sh's verilator_build_files/ so
# a synth-only zip (generated-src, no simulator binary) never masks a
# full Verilator zip if run_sweep.sh runs the same cls later.
S3_BUCKET="ronitnag04-lynx"
S3_BUILD_KEY_PREFIX="synth_build_files"
S3_BUILD_PREFIX="s3://${S3_BUCKET}/${S3_BUILD_KEY_PREFIX}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-lynx}"

usage() {
  cat <<EOF
Usage: $0 --side {des|ser} --sweep-csv PATH --output PATH [flags]

Required:
  --side {des|ser}       Which top module to synthesize (ProtoAccel or
                         ProtoAccelSerializer respectively).
  --sweep-csv PATH       Input CSV of configs (from gen_sweep_configs.py).
  --output PATH          Output CSV appended to under flock.

Optional:
  --workers N            GNU parallel jobs (default: nproc/2).
  --pull-s3-builds       Fetch generated-src from S3 cache when
                         s3://.../synth_build_files/<cls>.zip exists.
                         Exact-name match only; falls back to a local
                         elaborate on miss.
  --push-s3-builds       After a fresh elaborate, zip generated-src and
                         upload to s3://.../synth_build_files/<cls>.zip.
                         Skipped when the config was pulled from cache
                         or when the destination key already exists.
                         Non-fatal: upload failures log a warning and
                         the sweep continues.
  --keep-artifacts       Don't delete generated-src / yosys workdir after.
  --config-filter REGEX  Only process configs whose config_name matches.
  --limit-configs N      Only process the first N post-filter rows for the
                         selected side. 0 (default) = no limit. Applies to
                         both dry-run and real execution.
  --dry-run              List what would run; do not synthesize.
  -h, --help             Show this help.
EOF
}

SIDE=""
SWEEP_CSV=""
OUTPUT=""
WORKERS="$(( $(nproc) / 2 ))"
PULL_S3=0
PUSH_S3=0
KEEP_ARTIFACTS=0
CONFIG_FILTER=""
DRY_RUN=0
LIMIT_CONFIGS=0
while (($#)); do
  case "$1" in
    --side) SIDE=$2; shift 2 ;;
    --sweep-csv) SWEEP_CSV=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --pull-s3-builds) PULL_S3=1; shift ;;
    --push-s3-builds) PUSH_S3=1; shift ;;
    --keep-artifacts) KEEP_ARTIFACTS=1; shift ;;
    --config-filter) CONFIG_FILTER=$2; shift 2 ;;
    --limit-configs) LIMIT_CONFIGS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Guard --limit-configs: non-negative integer.
if ! [[ $LIMIT_CONFIGS =~ ^[0-9]+$ ]]; then
  echo "--limit-configs must be a non-negative integer (got '$LIMIT_CONFIGS')" >&2
  exit 2
fi

[[ -z $SIDE || -z $SWEEP_CSV || -z $OUTPUT ]] && { usage; exit 2; }
[[ $SIDE == des || $SIDE == ser ]] || { echo "--side must be des or ser" >&2; exit 2; }
[[ -f $SWEEP_CSV ]] || { echo "sweep-csv not found: $SWEEP_CSV" >&2; exit 2; }

# Resolve the yosys top module name from the side flag.
TOP="ProtoAccel"; [[ $SIDE == ser ]] && TOP="ProtoAccelSerializer"

# --- Pre-flight tool checks ----------------------------------------------
# Core tools -- the sweep can't do anything useful without these. Hard-exit.
for tool in yosys zachjs-sv2v python3 make; do
  if ! command -v "$tool" >/dev/null; then
    echo "[preflight] required tool '$tool' not on PATH -- aborting." >&2
    exit 1
  fi
done

# Optional-feature tools -- disable the feature with a warning if missing,
# matching run_sweep.sh's policy.
if (( PULL_S3 == 1 )); then
  if ! command -v unzip >/dev/null; then
    echo "[preflight] 'unzip' not on PATH; disabling --pull-s3-builds." >&2
    PULL_S3=0
  elif ! command -v aws >/dev/null; then
    echo "[preflight] 'aws' CLI not on PATH; disabling --pull-s3-builds." >&2
    PULL_S3=0
  fi
fi
if (( PUSH_S3 == 1 )); then
  if ! command -v zip >/dev/null; then
    echo "[preflight] 'zip' not on PATH; disabling --push-s3-builds." >&2
    PUSH_S3=0
  elif ! command -v aws >/dev/null; then
    echo "[preflight] 'aws' CLI not on PATH; disabling --push-s3-builds." >&2
    PUSH_S3=0
  fi
fi

# --- Interrupt handling --------------------------------------------------
# On Ctrl+C / SIGTERM: kill descendants bottom-up (so cc1plus / yosys / sbt /
# firtool exit before their parent bash so nothing orphans to PID 1), then
# sweep any Chipyard stragglers on this UID. Same pattern as run_sweep.sh.
_kill_descendants_bottom_up() {
  local sig=$1 parent=$2 child
  for child in $(pgrep -P "$parent" 2>/dev/null || true); do
    [[ "$child" == "$$" ]] && continue
    _kill_descendants_bottom_up "$sig" "$child"
    kill -"$sig" "$child" 2>/dev/null || true
  done
}
_pkill_chipyard_stragglers() {
  local sig=$1 u=${EUID:-$(id -u)} pid rest
  pkill -"$sig" -u "$u" -f 'chipyard\.Generator' 2>/dev/null || true
  while read -r pid rest; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    case "$rest" in
      *"$CHIPYARD_ROOT"*|*"$VERILATOR_DIR"*)
        case "$rest" in
          *yosys*|*zachjs-sv2v*|*firtool*|*"chipyard.Generator"*|*sbt*)
            kill -"$sig" "$pid" 2>/dev/null || true
            ;;
        esac
        ;;
    esac
  done < <(pgrep -u "$u" -af . 2>/dev/null || true)
}
cleanup_tmp() {
  [[ -n "${CSV_LOCK:-}" ]] && rm -f "$CSV_LOCK" 2>/dev/null || true
}
on_interrupt() {
  trap '' INT TERM   # ignore re-entry while we tear down
  echo "" >&2
  echo "[abort] interrupt received -- terminating sweep subprocesses..." >&2
  _kill_descendants_bottom_up TERM $$
  sleep 3
  _kill_descendants_bottom_up KILL $$
  _pkill_chipyard_stragglers TERM
  sleep 1
  _pkill_chipyard_stragglers KILL
  cleanup_tmp
  exit 130
}
trap on_interrupt INT TERM
trap cleanup_tmp EXIT

# Column order in the sweep-csv (matches gen_sweep_configs.py). May include
# a ``sample_group`` column when the CSV was produced by ``-t synth-training``.
mapfile -t CSV_HEADER < <(head -1 "$SWEEP_CSV" | tr ',' '\n')

# Emit an initial log summary.
mkdir -p "$(dirname "$OUTPUT")"
echo "[sweep] side=$SIDE top=$TOP workers=$WORKERS pull_s3=$PULL_S3 push_s3=$PUSH_S3"
echo "[sweep] sweep_csv=$SWEEP_CSV"
echo "[sweep] output=$OUTPUT"

# ---------------------------------------------------------------------------
# S3 build cache helpers. Function shapes match run_sweep.sh's
# _s3_upload_build / _s3_pull_build so behavior is identical for anyone
# cross-reading the two scripts.
# ---------------------------------------------------------------------------

# Upload the per-config generated-src tree to s3://.../synth_build_files/
# under the SAME archive layout as run_sweep.sh -- paths relative to
# $VERILATOR_DIR, so ``unzip`` from that cwd restores
# ``generated-src/chipyard.harness.TestHarness.<cls>``.
#
# Head-objects the destination first to avoid redundant uploads. Non-fatal
# on any failure -- the sweep continues.
_s3_upload_build() {
  local cls=$1
  local gen_src="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"
  local key="$S3_BUILD_KEY_PREFIX/$cls.zip"

  if [[ ! -d $gen_src ]]; then
    echo "[upload-skip] $cls: gen-src dir missing at $gen_src" >&2
    return 0
  fi

  if aws --profile "$AWS_PROFILE_NAME" s3api head-object \
      --bucket "$S3_BUCKET" --key "$key" >/dev/null 2>&1; then
    echo "[upload-skip] $S3_BUILD_PREFIX/$cls.zip already exists" >&2
    return 0
  fi

  local zip_tmp zip_log
  zip_tmp=$(mktemp -t "build_${cls}.XXXXXX.zip")
  rm -f "$zip_tmp"  # zip wants to create, not append
  zip_log=$(mktemp -t "build_${cls}.XXXXXX.ziplog")

  if ! (cd "$VERILATOR_DIR" && zip -r -q "$zip_tmp" \
        "generated-src/chipyard.harness.TestHarness.$cls") \
      >"$zip_log" 2>&1; then
    echo "[upload-fail] $cls zip: see tail" >&2
    tail -c 400 "$zip_log" >&2
    rm -f "$zip_tmp" "$zip_log"
    return 0
  fi

  if aws --profile "$AWS_PROFILE_NAME" s3 cp --only-show-errors \
      "$zip_tmp" "$S3_BUILD_PREFIX/$cls.zip" >"$zip_log" 2>&1; then
    local sz
    sz=$(stat -c %s "$zip_tmp" 2>/dev/null || echo ?)
    echo "[upload-ok]  $S3_BUILD_PREFIX/$cls.zip (${sz}B)" >&2
  else
    echo "[upload-fail] $S3_BUILD_PREFIX/$cls.zip: see tail" >&2
    tail -c 400 "$zip_log" >&2
  fi
  rm -f "$zip_tmp" "$zip_log"
}

# Reverse of _s3_upload_build. Head-objects first (cheap miss probe), then
# downloads + unzips under $VERILATOR_DIR to restore
# ``generated-src/chipyard.harness.TestHarness.<cls>``. Returns 0 on a
# usable restore, nonzero otherwise -- the caller falls back to a local
# ``make verilog``.
_s3_pull_build() {
  local cls=$1
  local key="$S3_BUILD_KEY_PREFIX/$cls.zip"
  local gen_dir="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"

  if ! aws --profile "$AWS_PROFILE_NAME" s3api head-object \
      --bucket "$S3_BUCKET" --key "$key" >/dev/null 2>&1; then
    echo "[pull-miss] $S3_BUILD_PREFIX/$cls.zip" >&2
    return 1
  fi

  local zip_tmp pull_log
  zip_tmp=$(mktemp -t "pull_${cls}.XXXXXX.zip")
  pull_log=$(mktemp -t "pull_${cls}.XXXXXX.log")

  if ! aws --profile "$AWS_PROFILE_NAME" s3 cp --only-show-errors \
      "$S3_BUILD_PREFIX/$cls.zip" "$zip_tmp" >"$pull_log" 2>&1; then
    echo "[pull-fail] $S3_BUILD_PREFIX/$cls.zip download: see tail" >&2
    tail -c 400 "$pull_log" >&2
    rm -f "$zip_tmp" "$pull_log"
    return 1
  fi

  # Matches _s3_upload_build's archive layout: paths are relative to
  # $VERILATOR_DIR, so unzip from that cwd. ``-o`` overwrites stale
  # partial artifacts rather than prompting.
  if ! (cd "$VERILATOR_DIR" && unzip -o -q "$zip_tmp") >"$pull_log" 2>&1; then
    echo "[pull-fail] $cls unzip: see tail" >&2
    tail -c 400 "$pull_log" >&2
    rm -f "$zip_tmp" "$pull_log"
    return 1
  fi

  rm -f "$zip_tmp" "$pull_log"

  if [[ ! -d "$gen_dir/gen-collateral" ]]; then
    echo "[pull-fail] $cls: gen-collateral missing after unzip" >&2
    return 1
  fi

  local sz
  sz=$(du -sh "$gen_dir" 2>/dev/null | cut -f1)
  echo "[pull-ok]  $S3_BUILD_PREFIX/$cls.zip -> gen-collateral (${sz})" >&2
  return 0
}

# Full per-config work. Isolated in a function so `parallel` can call it.
# ALWAYS returns 0 -- any failure logs the reason and returns cleanly so
# GNU parallel doesn't tally spurious failures and (with --halt-on-error 0)
# the sweep keeps making forward progress on the remaining configs.
worker_one_config() {
  local row_csv=$1
  local cls side_col
  cls=$(echo "$row_csv" | awk -F',' '{print $1}')
  side_col=$(echo "$row_csv" | awk -F',' '{print $2}')
  if [[ -n $CONFIG_FILTER && ! $cls =~ $CONFIG_FILTER ]]; then
    return 0
  fi
  if [[ $side_col != "$SIDE" && $side_col != "joint" ]]; then
    return 0
  fi

  # Resume-friendly: if the output CSV already has a row for this cls,
  # skip re-synthesis. Matches run_sweep.sh's grep-based skip.
  if [[ -f "$OUTPUT" ]] && grep -q "^${cls}," "$OUTPUT" 2>/dev/null; then
    echo "[skip-done] $cls" >&2
    return 0
  fi

  local gen_dir="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"
  local pulled_from_s3=0
  local elab_wall=0

  # 1. Try S3 cache pull (exact-name only). Falls back to a local
  # elaborate on miss/failure.
  if (( PULL_S3 )) && _s3_pull_build "$cls"; then
    pulled_from_s3=1
  fi

  # 2. Elaborate locally if the pull didn't produce the gen-collateral dir.
  if [[ ! -d "$gen_dir/gen-collateral" ]]; then
    local t0=$(date +%s)
    # Chipyard verilog-only target. Runs Chisel + firtool but skips the
    # full Verilator C++ build; enough for the .sv artifacts we need.
    (cd "$VERILATOR_DIR" && \
       make CONFIG=$cls verilog >/dev/null) \
      || { echo "[elab-fail] $cls" >&2
           (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
           return 0; }
    elab_wall=$(( $(date +%s) - t0 ))
  fi

  # 3. zachjs-sv2v + yosys under a per-config workdir.
  local work="$gen_dir/yosys"
  mkdir -p "$work"
  local flat="$work/${TOP}_flat.v"

  if ! (cd "$work" && zachjs-sv2v -DSYNTHESIS --write="$flat" "$gen_dir"/gen-collateral/*.sv) \
       >"$work/sv2v.log" 2>&1; then
    echo "[sv2v-fail] $cls" >&2
    tail -c 400 "$work/sv2v.log" >&2
    (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
    return 0
  fi
  cp -f "$gen_dir/gen-collateral/plusarg_reader.v" "$work/"
  cp -f "$gen_dir/gen-collateral/EICG_wrapper.v" "$work/"

  local synth_log="$work/synth.log"
  local t1=$(date +%s)
  # -Q suppresses only the banner; the log file carries the full pass
  # output including the per-module 'stat' blocks that parse_yosys_stat.py
  # depends on. Do NOT pass -q -- it silences ``stat`` too, which yields
  # a two-line log of just parser warnings and every downstream row
  # comes out as zeros.
  if ! (cd "$work" && yosys -Q -l "$synth_log" -p "
      read_verilog -DSYNTHESIS ${TOP}_flat.v plusarg_reader.v EICG_wrapper.v
      hierarchy -check -top $TOP
      synth -top $TOP
      stat
  ") >/dev/null 2>&1; then
    echo "[yosys-fail] $cls" >&2
    tail -c 400 "$synth_log" >&2
    (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
    return 0
  fi
  local synth_wall=$(( $(date +%s) - t1 ))

  # Sanity: parse fails silently if the log has no ``=== <mod> ===`` block
  # (e.g. someone changes the yosys invocation and drops ``stat``). Bail
  # early with a clear message rather than emitting an all-zero row.
  if ! grep -q "^=== " "$synth_log"; then
    echo "[yosys-no-stat] $cls: log has no '=== <mod> ===' blocks; see $synth_log" >&2
    (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
    return 0
  fi

  # 4. Parse and append CSV row under flock.
  # Split CSV_ROW into JSON params + sample_group tag (may be empty if
  # the input CSV was not produced by ``-t synth-training``).
  local extract
  extract=$(CSV_ROW="$row_csv" python3 -c '
import json, os
header = os.environ["CSV_HEADER_STR"].split(",")
row    = os.environ["CSV_ROW"].split(",")
d = dict(zip(header, row))
sg = d.get("sample_group", "")
params = {}
for k, v in d.items():
    if k in ("config_name", "side", "sample_group"):
        continue
    try:
        params[k] = int(v)
    except (TypeError, ValueError):
        params[k] = v
print(json.dumps({"params": params, "sample_group": sg}))
')
  local params_json
  local sample_group
  params_json=$(echo "$extract" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["params"]))')
  sample_group=$(echo "$extract" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sample_group"])')

  if ! python3 -m hw_cost_model.parse_yosys_stat \
      --synth-log "$synth_log" \
      --side "$SIDE" \
      --config-name "$cls" \
      --config-params-json "$params_json" \
      --sample-group "$sample_group" \
      --csv-append "$OUTPUT" 2>>"$work/parse.log"; then
    echo "[parse-fail] $cls (see $work/parse.log)" >&2
    (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
    return 0
  fi

  echo "[ok] $cls  s3_pull=$pulled_from_s3 elab=${elab_wall}s synth=${synth_wall}s"

  # 5. Push to S3 if requested and we elaborated from scratch. Skip
  # when the config was pulled from cache (the zip is already there).
  # _s3_upload_build is non-fatal on failure (matches run_sweep.sh's
  # policy) and additionally skips its own upload if the destination
  # key already exists.
  if (( PUSH_S3 )) && (( pulled_from_s3 == 0 )); then
    _s3_upload_build "$cls" || true
  fi

  # 6. Cleanup.
  if (( KEEP_ARTIFACTS == 0 )); then
    rm -rf -- "$gen_dir"
  fi
  return 0
}

export -f worker_one_config _s3_pull_build _s3_upload_build
export SIDE TOP VERILATOR_DIR VERILATOR_GEN_ROOT OUTPUT
export S3_BUCKET AWS_PROFILE_NAME
export S3_BUILD_KEY_PREFIX S3_BUILD_PREFIX
export PULL_S3 PUSH_S3 KEEP_ARTIFACTS CONFIG_FILTER
# Make the CSV header available to the worker.
CSV_HEADER_STR="$(head -1 "$SWEEP_CSV")"
export CSV_HEADER_STR
# Ensure Python inside workers can find the hw_cost_model package.
export PYTHONPATH="$LYNX_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Source the Chipyard env once so every worker's ``make verilog`` sees
# sbt / firtool / $RISCV / etc. via inherited env. env.sh's conda-activate
# hooks reference unset vars, so relax nounset just for the source.
set +u
# shellcheck disable=SC1091
source "$CHIPYARD_ROOT/env.sh"
set -u

# Serialize the sbt assembly step. Without this, the first parallel worker
# to hit ``make verilog`` runs ``sbt assembly`` while later workers
# race in and try to read a half-written chipyard.jar -- producing the
# infamous "Could not find or load main class chipyard.Generator" and
# "Waiting for lock on .sbt.ivy.lock" errors. Running the jar target once
# up-front (even if fresh) forces the race to resolve deterministically.
echo "[prebuild] ensuring $CLASSPATH_JAR is up to date..." >&2
make -C "$VERILATOR_DIR" CONFIG=ProtoAccelRocketConfig "$CLASSPATH_JAR" >/dev/null

# Build the filtered work list: keep rows whose ``side`` column matches
# --side (or ``joint``) and whose config_name matches --config-filter, then
# cap with --limit-configs when set. Doing this up-front (rather than
# inside the worker) makes --limit-configs count actual synth jobs, not
# input-CSV rows skipped by other filters.
_filtered_rows() {
  tail -n +2 "$SWEEP_CSV" | awk -F',' -v side="$SIDE" -v re="$CONFIG_FILTER" '
    {
      if ($2 != side && $2 != "joint") next
      if (re != "" && $1 !~ re)         next
      print
    }
  '
}

if (( LIMIT_CONFIGS > 0 )); then
  # Materialize into a temp file first so we don't have to wrestle with
  # SIGPIPE from ``head`` closing early under ``set -o pipefail``.
  _rows_tmp=$(mktemp -t "rows.XXXXXX")
  _filtered_rows > "$_rows_tmp"
  FILTERED=$(head -n "$LIMIT_CONFIGS" "$_rows_tmp")
  rm -f "$_rows_tmp"
  echo "[sweep] limit-configs=$LIMIT_CONFIGS (kept $(printf '%s\n' "$FILTERED" | grep -c .) rows)"
else
  FILTERED=$(_filtered_rows)
fi

if command -v parallel >/dev/null; then
  if (( DRY_RUN )); then
    echo "$FILTERED" | awk -F',' '{print "[dry-run] " $1 " " $2}'
  else
    # --halt-on-error 0: keep going on any worker failure. Workers already
    # return 0 on failure (logging the reason), so this is defense in depth
    # against parallel deciding a signal-driven exit warrants a halt.
    # --termseq: on SIGINT, give workers 1s to react to INT, then 2s to
    # TERM, then KILL. Matches run_sweep.sh.
    echo "$FILTERED" \
      | parallel --line-buffer -j "$WORKERS" \
          --halt-on-error 0 \
          --termseq INT,1000,TERM,2000,KILL,25 \
          worker_one_config {}
  fi
else
  echo "[warn] GNU parallel not on PATH; falling back to serial." >&2
  echo "$FILTERED" | while IFS= read -r row; do
    [[ -z $row ]] && continue
    if (( DRY_RUN )); then
      echo "[dry-run] $(echo "$row" | awk -F',' '{print $1, $2}')"
    else
      worker_one_config "$row" || true
    fi
  done
fi

echo "[sweep] done -> $OUTPUT"
