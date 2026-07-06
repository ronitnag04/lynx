#!/usr/bin/env bash
# run_combined_sweep.sh -- sweep ProtoAcc hardware configs through BOTH Yosys
# synthesis and the Verilator HPB simulator in a single per-config pass, so
# we collect synth cell/ram_bits rows AND per-bench throughput rows against
# the same generated-src tree without paying to elaborate the design twice.
#
# Combines the per-config work of:
#   hw_cost_model/run_yosys_sweep.sh          (elaborate + sv2v + yosys)
#   ../verilator-bench/run_sweep.sh           (verilator build + bench runs)
#
# The worker for a single config is:
#   1. Try to pull a verilator build zip (generated-src + simulator binary)
#      from s3://ronitnag04-lynx/verilator_build_files/<cls>.zip.
#      On miss, run ``make CONFIG=<cls>`` under $VERILATOR_DIR to produce
#      BOTH the generated-src tree and the simulator binary.
#   2. Run zachjs-sv2v -DSYNTHESIS + yosys on the generated-src to produce
#      synth.log; parse with hw_cost_model.parse_yosys_stat and append one
#      row to --synth-output under flock.
#   3. For every (bench, op) pair that isn't already in --sim-output, run
#      the bench via ``make run-binary-fast`` and append the ACCEL_SUMMARY
#      row to --sim-output under flock. Ops default to ``des`` for
#      des-side configs and ``ser`` for ser-side configs (matching
#      run_sweep.sh's --op both default). Bench selection mirrors
#      run_sweep.sh (--benches / --random-bench / --iter-bench).
#   4. Optionally upload the build zip (verilator_build_files/<cls>.zip)
#      and sim output zip (simulation_files/<cls>.zip) to S3.
#   5. Unless --keep-artifacts, delete the generated-src tree + simulator
#      binary + local output dir once every downstream artifact is either
#      persisted to S3 or intentionally skipped.
#
# The two output CSVs use the SAME schemas the individual scripts produce
# (run_yosys_sweep.sh reads/writes yosys_sweep_results_<side>.csv;
# run_sweep.sh reads/writes a per-bench CSV keyed on config_name/bench/op),
# so downstream tooling (fit_from_yosys.py, ml_model training) works
# without changes.
#
# Sweep-csv format (identical to gen_sweep_configs.py's yosys_synth_sweep_configs.csv,
# and to what gen_pareto_validation_configs.py now emits):
#   config_name,side,sample_group,<des_*>,<ser_*>
# The ``sample_group`` column is optional -- if missing, it's treated as empty.
#
# Prereqs (in the chipyard conda env):
#   - yosys, zachjs-sv2v, python3, make, gnu parallel
#   - awscli + unzip + zip (optional; for S3 pull/push)
#   - hw_cost_model package importable (this script sets PYTHONPATH)
#
# Usage:
#   ./run_combined_sweep.sh \
#       --side des \
#       --sweep-csv ../sweep_configs/des_pareto_validation_sweep_configs.csv \
#       --synth-output ./yosys_validation_des.csv \
#       --sim-output   ../verilator-bench/results/validation_des.csv \
#       --workers 8 --jobs 8 \
#       [--benches bench0 bench1] [--random-bench] [--iter-bench] \
#       [--pull-s3-builds] [--push-s3-builds] [--skip-upload] [--keep-artifacts]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYNX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHIPYARD_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
VERILATOR_DIR="$CHIPYARD_ROOT/sims/verilator"
VERILATOR_GEN_ROOT="$VERILATOR_DIR/generated-src"
VERILATOR_OUTPUT_ROOT="$VERILATOR_DIR/output"
BENCH_BUILD_DIR="$CHIPYARD_ROOT/generators/protoacc/software/verilator-bench/build"
CLASSPATH_JAR="$CHIPYARD_ROOT/.classpath_cache/chipyard.jar"

# S3: share the same prefixes as run_sweep.sh / run_yosys_sweep.sh so a
# build produced here is a cache hit for either of those, and vice versa.
S3_BUCKET="ronitnag04-lynx"
S3_BUILD_KEY_PREFIX="validation_build_files"
S3_SIM_KEY_PREFIX="validation_sim_files"
S3_BUILD_PREFIX="s3://${S3_BUCKET}/${S3_BUILD_KEY_PREFIX}"
S3_SIM_PREFIX="s3://${S3_BUCKET}/${S3_SIM_KEY_PREFIX}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-lynx}"

usage() {
  cat <<EOF
Usage: $0 --side {des|ser} --sweep-csv PATH --synth-output PATH --sim-output PATH [flags]

Required:
  --side {des|ser}       ProtoAccel (des) or ProtoAccelSerializer (ser).
  --sweep-csv PATH       Input CSV (config_name,side,[sample_group,]<des_*>,<ser_*>).
                         Rows with side != --side and != 'joint' are skipped.
  --synth-output PATH    yosys_sweep_results-style CSV (appended under flock).
  --sim-output   PATH    verilator sweep-style CSV (appended under flock).

Optional:
  --workers N            Parallel per-config workers (default: nproc/2).
  --jobs N               make -j for each Verilator build (default: 1).
  --bench-timeout SEC    Per-bench wall cap (default: 3600).
  --bench-parallel N     Concurrent (bench,op) sims per config after build
                         completes (default: 1). Peak concurrent sims
                         = workers * bench-parallel; each can eat multiple
                         GB of RAM.
  --benches NAME [NAME...]
                         Bench list (default: bench0..bench5 or every
                         bench<N>_<op>.riscv discovered under BENCH_BUILD_DIR).
  --random-bench         Per config, run one bench picked at random from
                         the discovered pool.
  --iter-bench           Pair the i-th config (per side) with the i-th synth
                         bench (bench<N> with N >= --hpb-synth-start);
                         configs past the synth pool draw a random HPB bench.
  --hpb-synth-start N    Boundary bench-id between HPB and synth (default 6).
  --config-filter REGEX  Only process configs whose config_name matches.
  --limit-configs N      Cap post-filter row count (0 = no cap).
  --pull-s3-builds       Try to fetch verilator_build_files/<cls>.zip from S3;
                         fall back to local build on miss.
  --push-s3-builds       After a fresh local build, upload the build zip
                         to S3. Idempotent (head-object skip).
  --skip-upload          Don't upload the simulation output zip to S3.
  --keep-artifacts       Don't delete generated-src/sim/output after use.
  --skip-synth           Only run the verilator sim step (mirror --skip-yosys
                         if you want CSV rows for sim but not synth).
  --skip-sim             Only run the yosys synth step (mirror the old
                         run_yosys_sweep.sh behavior).
  --skip-build           Don't build missing sims; skip a config if it has
                         no cached simulator binary and no S3 hit.
  --dry-run              Print the plan and exit.
  -h, --help             Show this help.

Notes:
  * Each worker elaborates once (or reuses a pulled/on-disk build). The
    same generated-src tree feeds both yosys and the sim run, so synth
    results and sim results are guaranteed to reflect the same RTL.
  * Both output CSVs are resume-friendly: rows already present for a
    given config (synth) or (config,bench,op) (sim) are skipped.
EOF
}

SIDE=""
SWEEP_CSV=""
SYNTH_OUTPUT=""
SIM_OUTPUT=""
WORKERS="$(( $(nproc) / 2 ))"
(( WORKERS < 1 )) && WORKERS=1
JOBS=1
BENCH_TIMEOUT=3600
BENCH_PARALLEL=1
BENCHES=()
RANDOM_BENCH=0
ITER_BENCH=0
HPB_SYNTH_START=6
CONFIG_FILTER=""
LIMIT_CONFIGS=0
PULL_S3=0
PUSH_S3=0
SKIP_UPLOAD=0
KEEP_ARTIFACTS=0
SKIP_SYNTH=0
SKIP_SIM=0
SKIP_BUILD=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --side)             SIDE=$2; shift 2 ;;
    --sweep-csv)        SWEEP_CSV=$2; shift 2 ;;
    --synth-output)     SYNTH_OUTPUT=$2; shift 2 ;;
    --sim-output)       SIM_OUTPUT=$2; shift 2 ;;
    --workers)          WORKERS=$2; shift 2 ;;
    --jobs)             JOBS=$2; shift 2 ;;
    --bench-timeout)    BENCH_TIMEOUT=$2; shift 2 ;;
    --bench-parallel)   BENCH_PARALLEL=$2; shift 2 ;;
    --benches)          shift
                        while [[ $# -gt 0 && $1 != --* ]]; do BENCHES+=("$1"); shift; done ;;
    --random-bench)     RANDOM_BENCH=1; shift ;;
    --iter-bench)       ITER_BENCH=1; shift ;;
    --hpb-synth-start)  HPB_SYNTH_START=$2; shift 2 ;;
    --config-filter)    CONFIG_FILTER=$2; shift 2 ;;
    --limit-configs)    LIMIT_CONFIGS=$2; shift 2 ;;
    --pull-s3-builds)   PULL_S3=1; shift ;;
    --push-s3-builds)   PUSH_S3=1; shift ;;
    --skip-upload)      SKIP_UPLOAD=1; shift ;;
    --keep-artifacts)   KEEP_ARTIFACTS=1; shift ;;
    --skip-synth)       SKIP_SYNTH=1; shift ;;
    --skip-sim)         SKIP_SIM=1; shift ;;
    --skip-build)       SKIP_BUILD=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -z $SIDE || -z $SWEEP_CSV || -z $SYNTH_OUTPUT || -z $SIM_OUTPUT ]] && { usage; exit 2; }
[[ $SIDE == des || $SIDE == ser ]] || { echo "--side must be des or ser" >&2; exit 2; }
[[ -f $SWEEP_CSV ]] || { echo "sweep-csv not found: $SWEEP_CSV" >&2; exit 2; }
if ! [[ $LIMIT_CONFIGS =~ ^[0-9]+$ ]]; then
  echo "--limit-configs must be a non-negative integer (got '$LIMIT_CONFIGS')" >&2; exit 2
fi
if (( RANDOM_BENCH == 1 && ITER_BENCH == 1 )); then
  echo "--random-bench and --iter-bench are mutually exclusive" >&2; exit 2
fi
if ! [[ $BENCH_PARALLEL =~ ^[0-9]+$ ]] || (( BENCH_PARALLEL < 1 )); then
  echo "--bench-parallel must be a positive integer (got '$BENCH_PARALLEL')" >&2; exit 2
fi
if (( SKIP_SYNTH == 1 && SKIP_SIM == 1 )); then
  echo "--skip-synth and --skip-sim can't both be set (nothing to do)" >&2; exit 2
fi

# Resolve the yosys top module name from the side flag.
TOP="ProtoAccel"; [[ $SIDE == ser ]] && TOP="ProtoAccelSerializer"

# --- Pre-flight tool checks ----------------------------------------------
# Core tools (needed regardless of skip flags -- verilog elab is always required).
for tool in python3 make; do
  command -v "$tool" >/dev/null \
    || { echo "[preflight] required tool '$tool' not on PATH -- aborting." >&2; exit 1; }
done
if (( SKIP_SYNTH == 0 )); then
  for tool in yosys zachjs-sv2v; do
    command -v "$tool" >/dev/null \
      || { echo "[preflight] required tool '$tool' not on PATH -- aborting." >&2; exit 1; }
  done
fi
if ! command -v parallel >/dev/null; then
  echo "[preflight] GNU parallel not on PATH -- workers will run serially." >&2
fi

# Optional-feature tool checks -- disable the feature with a warning.
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
if (( SKIP_UPLOAD == 0 )); then
  if ! command -v zip >/dev/null; then
    echo "[preflight] 'zip' not on PATH; disabling sim-output upload." >&2
    SKIP_UPLOAD=1
  elif ! command -v aws >/dev/null; then
    echo "[preflight] 'aws' CLI not on PATH; disabling sim-output upload." >&2
    SKIP_UPLOAD=1
  fi
fi

# --- Bench pool discovery (mirrors run_sweep.sh) --------------------------
_discover_default_benches() {
  local -a benches=()
  if [[ -d "$BENCH_BUILD_DIR" ]]; then
    local f name
    while IFS= read -r -d '' f; do
      name="$(basename "$f")"
      name="${name%_ser.riscv}"
      [[ "$name" =~ ^bench[0-9]+$ ]] && benches+=("$name")
    done < <(find "$BENCH_BUILD_DIR" -maxdepth 1 -type f -name 'bench*_ser.riscv' -print0 2>/dev/null)
  fi
  if [[ ${#benches[@]} -eq 0 ]]; then
    benches=(bench0 bench1 bench2 bench3 bench4 bench5)
  fi
  printf '%s\n' "${benches[@]}" | sort -V
}
mapfile -t DEFAULT_BENCHES < <(_discover_default_benches)

if (( RANDOM_BENCH == 0 && ITER_BENCH == 0 )); then
  [[ ${#BENCHES[@]} -eq 0 ]] && BENCHES=("${DEFAULT_BENCHES[@]}")
fi

SYNTH_BENCHES=()
HPB_BENCHES_POOL=()
for _b in "${DEFAULT_BENCHES[@]}"; do
  _id=${_b#bench}
  [[ "$_id" =~ ^[0-9]+$ ]] || continue
  if (( _id >= HPB_SYNTH_START )); then
    SYNTH_BENCHES+=("$_b")
  else
    HPB_BENCHES_POOL+=("$_b")
  fi
done
unset _b _id

if (( ITER_BENCH == 1 && ${#SYNTH_BENCHES[@]} == 0 )); then
  echo "[iter-bench] no synth benches (bench<N>_ser.riscv with N >= $HPB_SYNTH_START) under $BENCH_BUILD_DIR." >&2
  exit 2
fi

# --- Interrupt handling (bottom-up TERM/KILL, matches run_sweep.sh) ------
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
          *yosys*|*zachjs-sv2v*|*firtool*|*verilator*|*cc1plus*|*simulator-chipyard*|*"chipyard.Generator"*|*sbt*)
            kill -"$sig" "$pid" 2>/dev/null || true
            ;;
        esac
        ;;
    esac
  done < <(pgrep -u "$u" -af . 2>/dev/null || true)
}
cleanup_tmp() {
  [[ -n "${SIM_LOCK:-}"  ]] && rm -f "$SIM_LOCK"  2>/dev/null || true
  [[ -n "${_ROWS_TMP:-}" ]] && rm -f "$_ROWS_TMP" 2>/dev/null || true
}
on_interrupt() {
  trap '' INT TERM
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

# --- CSV header setup ----------------------------------------------------
# Synth CSV: hw_cost_model.parse_yosys_stat writes its own header on first
# append, so we only need to mkdir the parent.
mkdir -p "$(dirname "$SYNTH_OUTPUT")"
mkdir -p "$(dirname "$SIM_OUTPUT")"

# Sim CSV header: matches run_sweep.sh's column order so any downstream
# tool that reads either script's output works unchanged. The param
# columns follow the des_*/ser_* order the input CSV uses.
CSV_HEADER_STR="$(head -1 "$SWEEP_CSV")"
IFS=',' read -r -a CSV_HEADER <<< "$CSV_HEADER_STR"
PARAM_KEYS=()
for col in "${CSV_HEADER[@]}"; do
  case "$col" in
    config_name|side|sample_group) ;;
    *) PARAM_KEYS+=("$col") ;;
  esac
done
SIM_HEADER="config_name,side,bench,op,iters,cycles,bytes,throughput_bytes_per_sec,wall_s,build_wall_s,build_was_cached"
for k in "${PARAM_KEYS[@]}"; do SIM_HEADER+=",$k"; done
if [[ ! -f "$SIM_OUTPUT" ]]; then echo "$SIM_HEADER" > "$SIM_OUTPUT"; fi

echo "[sweep] side=$SIDE top=$TOP workers=$WORKERS jobs=$JOBS bench_parallel=$BENCH_PARALLEL"
echo "[sweep] sweep_csv=$SWEEP_CSV"
echo "[sweep] synth_output=$SYNTH_OUTPUT"
echo "[sweep] sim_output=$SIM_OUTPUT"
echo "[sweep] pull_s3=$PULL_S3 push_s3=$PUSH_S3 skip_upload=$SKIP_UPLOAD"
echo "[sweep] skip_synth=$SKIP_SYNTH skip_sim=$SKIP_SIM skip_build=$SKIP_BUILD"
if (( SKIP_SIM == 0 )); then
  if (( RANDOM_BENCH )); then
    echo "[sweep] bench mode: random (from pool of ${#DEFAULT_BENCHES[@]})"
  elif (( ITER_BENCH )); then
    echo "[sweep] bench mode: iter (synth pool=${#SYNTH_BENCHES[@]}, hpb pool=${#HPB_BENCHES_POOL[@]})"
  else
    echo "[sweep] bench mode: fixed (${#BENCHES[@]} benches: ${BENCHES[*]})"
  fi
fi

# --- S3 helpers (shape matches run_sweep.sh / run_yosys_sweep.sh) --------
# All uploads head-object first so re-runs don't re-upload identical zips;
# all failures are non-fatal and only log to stderr.

# Try to fetch a verilator build zip from S3. On success, generated-src +
# simulator binary are restored under $VERILATOR_DIR. Returns 0 iff a
# usable build (simulator binary + gen-collateral) is present after.
_s3_pull_build() {
  local cls=$1
  local key="$S3_BUILD_KEY_PREFIX/$cls.zip"
  local gen_dir="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"
  local sim_bin="$VERILATOR_DIR/simulator-chipyard.harness-$cls"

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
  if ! (cd "$VERILATOR_DIR" && unzip -o -q "$zip_tmp") >"$pull_log" 2>&1; then
    echo "[pull-fail] $cls unzip: see tail" >&2
    tail -c 400 "$pull_log" >&2
    rm -f "$zip_tmp" "$pull_log"
    return 1
  fi
  rm -f "$zip_tmp" "$pull_log"

  [[ -f $sim_bin ]] && chmod +x "$sim_bin" 2>/dev/null || true
  [[ -f "${sim_bin}.debug" ]] && chmod +x "${sim_bin}.debug" 2>/dev/null || true

  # Synth needs gen-collateral. Sim needs the executable. Accept a pull
  # that has EITHER but warn if the caller will still have to build.
  # In practice a run_sweep.sh-produced zip has both; a synth-only zip
  # (from run_yosys_sweep.sh) has only gen-collateral and we fall through
  # to a local build below.
  if [[ ! -d "$gen_dir/gen-collateral" && ! -x "$sim_bin" ]]; then
    echo "[pull-fail] $cls: neither gen-collateral nor simulator after unzip" >&2
    return 1
  fi

  local sz
  sz=$(du -sh "$gen_dir" 2>/dev/null | cut -f1)
  echo "[pull-ok]  $S3_BUILD_PREFIX/$cls.zip -> $gen_dir (${sz})" >&2
  return 0
}

# Upload the (generated-src + simulator binary) tree to S3.
_s3_upload_build() {
  local cls=$1
  local gen_src="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"
  local sim_bin="$VERILATOR_DIR/simulator-chipyard.harness-$cls"
  local sim_bin_debug="$VERILATOR_DIR/simulator-chipyard.harness-$cls.debug"
  local key="$S3_BUILD_KEY_PREFIX/$cls.zip"

  if [[ ! -d $gen_src && ! -x $sim_bin && ! -f $sim_bin_debug ]]; then
    return 0
  fi
  if aws --profile "$AWS_PROFILE_NAME" s3api head-object \
      --bucket "$S3_BUCKET" --key "$key" >/dev/null 2>&1; then
    echo "[upload-skip] $S3_BUILD_PREFIX/$cls.zip already exists" >&2
    return 0
  fi

  local -a zip_inputs=()
  [[ -d $gen_src       ]] && zip_inputs+=("generated-src/chipyard.harness.TestHarness.$cls")
  [[ -x $sim_bin       ]] && zip_inputs+=("simulator-chipyard.harness-$cls")
  [[ -f $sim_bin_debug ]] && zip_inputs+=("simulator-chipyard.harness-$cls.debug")
  (( ${#zip_inputs[@]} == 0 )) && return 0

  local zip_tmp zip_log
  zip_tmp=$(mktemp -t "build_${cls}.XXXXXX.zip"); rm -f "$zip_tmp"
  zip_log=$(mktemp -t "build_${cls}.XXXXXX.ziplog")
  if ! (cd "$VERILATOR_DIR" && zip -r -q "$zip_tmp" "${zip_inputs[@]}") \
      >"$zip_log" 2>&1; then
    echo "[upload-fail] $cls zip (build): see tail" >&2
    tail -c 400 "$zip_log" >&2
    rm -f "$zip_tmp" "$zip_log"
    return 0
  fi
  if aws --profile "$AWS_PROFILE_NAME" s3 cp --only-show-errors \
      "$zip_tmp" "$S3_BUILD_PREFIX/$cls.zip" >"$zip_log" 2>&1; then
    local sz; sz=$(stat -c %s "$zip_tmp" 2>/dev/null || echo ?)
    echo "[upload-ok]  $S3_BUILD_PREFIX/$cls.zip (${sz}B)" >&2
  else
    echo "[upload-fail] $S3_BUILD_PREFIX/$cls.zip: see tail" >&2
    tail -c 400 "$zip_log" >&2
  fi
  rm -f "$zip_tmp" "$zip_log"
}

_s3_upload_sim() {
  local cls=$1
  local sim_out="$VERILATOR_OUTPUT_ROOT/chipyard.harness.TestHarness.$cls"
  local key="$S3_SIM_KEY_PREFIX/$cls.zip"

  [[ -d $sim_out ]] || return 0
  if aws --profile "$AWS_PROFILE_NAME" s3api head-object \
      --bucket "$S3_BUCKET" --key "$key" >/dev/null 2>&1; then
    echo "[upload-skip] $S3_SIM_PREFIX/$cls.zip already exists" >&2
    return 0
  fi

  local zip_tmp zip_log
  zip_tmp=$(mktemp -t "sim_${cls}.XXXXXX.zip"); rm -f "$zip_tmp"
  zip_log=$(mktemp -t "sim_${cls}.XXXXXX.ziplog")
  if ! (cd "$VERILATOR_OUTPUT_ROOT" && \
        zip -r -q "$zip_tmp" "chipyard.harness.TestHarness.$cls") \
      >"$zip_log" 2>&1; then
    echo "[upload-fail] $cls zip (sim): see tail" >&2
    tail -c 400 "$zip_log" >&2
    rm -f "$zip_tmp" "$zip_log"
    return 0
  fi
  if aws --profile "$AWS_PROFILE_NAME" s3 cp --only-show-errors \
      "$zip_tmp" "$S3_SIM_PREFIX/$cls.zip" >"$zip_log" 2>&1; then
    local sz; sz=$(stat -c %s "$zip_tmp" 2>/dev/null || echo ?)
    echo "[upload-ok]  $S3_SIM_PREFIX/$cls.zip (${sz}B)" >&2
  else
    echo "[upload-fail] $S3_SIM_PREFIX/$cls.zip: see tail" >&2
    tail -c 400 "$zip_log" >&2
  fi
  rm -f "$zip_tmp" "$zip_log"
}

# --- Per-config worker ---------------------------------------------------
# Always returns 0: per-config failures log and are counted in the CSV
# absence, so a single bad config doesn't halt the sweep.
worker_one_config() {
  local row_csv=$1
  local plan_side_idx=$2   # 1-based per-side index for --iter-bench
  local cls side_col
  cls=$(echo "$row_csv" | awk -F',' '{print $1}')
  side_col=$(echo "$row_csv" | awk -F',' '{print $2}')

  if [[ -n $CONFIG_FILTER && ! $cls =~ $CONFIG_FILTER ]]; then
    return 0
  fi
  if [[ $side_col != "$SIDE" && $side_col != "joint" ]]; then
    return 0
  fi

  local gen_dir="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"
  local sim_bin="$VERILATOR_DIR/simulator-chipyard.harness-$cls"

  # Split (config_name, side, sample_group, param_json) once for the
  # synth parser call. When --skip-synth is set we don't even need this.
  local extract=""
  if (( SKIP_SYNTH == 0 )); then
    extract=$(CSV_ROW="$row_csv" CSV_HEADER_STR="$CSV_HEADER_STR" python3 -c '
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
  fi

  # Decide up front whether synth work is still pending for this config.
  local synth_pending=1
  if (( SKIP_SYNTH == 1 )); then
    synth_pending=0
  elif [[ -f "$SYNTH_OUTPUT" ]] && grep -q "^${cls}," "$SYNTH_OUTPUT" 2>/dev/null; then
    synth_pending=0
    echo "[skip-synth-done] $cls" >&2
  fi

  # Which (bench, op) pairs are still pending in the sim CSV?
  local -a ops=()
  case $side_col in
    des)   ops=(des) ;;
    ser)   ops=(ser) ;;
    joint) ops=($SIDE) ;;
    *)     ops=($SIDE) ;;
  esac

  local -a pending_bench=() pending_op=()
  if (( SKIP_SIM == 0 )); then
    local bench op
    if (( RANDOM_BENCH )); then
      bench=${DEFAULT_BENCHES[$((RANDOM % ${#DEFAULT_BENCHES[@]}))]}
      for op in "${ops[@]}"; do
        if ! grep -q "^${cls},${side_col},${bench},${op}," "$SIM_OUTPUT" 2>/dev/null; then
          pending_bench+=("$bench"); pending_op+=("$op")
        fi
      done
    elif (( ITER_BENCH )); then
      local idx0=$((plan_side_idx - 1))
      if (( idx0 < ${#SYNTH_BENCHES[@]} )); then
        bench=${SYNTH_BENCHES[$idx0]}
      elif (( ${#HPB_BENCHES_POOL[@]} > 0 )); then
        bench=${HPB_BENCHES_POOL[$((RANDOM % ${#HPB_BENCHES_POOL[@]}))]}
      else
        echo "[iter-bench-skip] $cls: no synth or HPB pool" >&2
        bench=""
      fi
      if [[ -n $bench ]]; then
        for op in "${ops[@]}"; do
          if ! grep -q "^${cls},${side_col},${bench},${op}," "$SIM_OUTPUT" 2>/dev/null; then
            pending_bench+=("$bench"); pending_op+=("$op")
          fi
        done
      fi
    else
      for op in "${ops[@]}"; do
        for bench in "${BENCHES[@]}"; do
          if ! grep -q "^${cls},${side_col},${bench},${op}," "$SIM_OUTPUT" 2>/dev/null; then
            pending_bench+=("$bench"); pending_op+=("$op")
          fi
        done
      done
    fi
  fi

  # Nothing to do? Still try to upload cached artifacts (idempotent),
  # then bail. Matches the resume-friendly behavior of run_sweep.sh.
  if (( synth_pending == 0 )) && (( ${#pending_bench[@]} == 0 )); then
    echo "[skip-done] $cls (synth + sim both cached)" >&2
    if (( PUSH_S3 )); then _s3_upload_build "$cls" || true; fi
    if (( SKIP_UPLOAD == 0 )) && (( SKIP_SIM == 0 )); then _s3_upload_sim "$cls" || true; fi
    _cleanup_after "$cls"
    return 0
  fi

  # --- Build / pull the design once, use for both synth and sim -----
  local pulled_from_s3=0
  local build_wall=0
  local build_was_cached=0
  local need_gen_collateral=$(( synth_pending == 1 ? 1 : 0 ))
  local need_sim_bin=0
  (( ${#pending_bench[@]} > 0 )) && need_sim_bin=1

  # 1. Reuse anything already on disk.
  if [[ -x $sim_bin ]] && (( need_sim_bin )); then
    build_was_cached=1
  fi
  if [[ -d "$gen_dir/gen-collateral" ]] && (( need_gen_collateral )); then
    :  # already have gen-collateral; treat as cached for synth purposes
  fi

  # 2. Try an S3 pull if either artifact is missing.
  if (( PULL_S3 )); then
    local pull_needed=0
    (( need_sim_bin )) && [[ ! -x $sim_bin ]] && pull_needed=1
    (( need_gen_collateral )) && [[ ! -d "$gen_dir/gen-collateral" ]] && pull_needed=1
    if (( pull_needed )) && _s3_pull_build "$cls"; then
      pulled_from_s3=1
      [[ -x $sim_bin ]] && build_was_cached=1
    fi
  fi

  # 3. Local build if still missing anything we need. When sim is pending
  # we run the full ``make CONFIG=<cls>`` (produces both gen-collateral
  # and the simulator binary in one go). When only synth is pending we
  # run the cheaper ``make verilog CONFIG=<cls>`` which stops after the
  # firtool emission -- enough for zachjs-sv2v + yosys and shaves
  # verilator C++ build time off the hot path.
  local need_full_build=0 need_verilog_only=0
  if (( need_sim_bin )) && [[ ! -x $sim_bin ]]; then
    need_full_build=1
  elif (( need_gen_collateral )) && [[ ! -d "$gen_dir/gen-collateral" ]]; then
    need_verilog_only=1
  fi
  if (( need_full_build == 1 || need_verilog_only == 1 )); then
    if (( SKIP_BUILD == 1 )); then
      echo "[skip-build] $cls (build needed but --skip-build set)" >&2
      (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
      return 0
    fi
    local build_log; build_log=$(mktemp -t "build_${cls}.XXXXXX")
    local build_target_desc="full"
    (( need_verilog_only == 1 )) && build_target_desc="verilog-only"
    echo "[build]    $cls ($build_target_desc)" >&2
    local t0=$SECONDS
    local build_rc=0
    if (( need_full_build == 1 )); then
      (cd "$VERILATOR_DIR" && make CONFIG="$cls" -j"$JOBS") \
        >"$build_log" 2>&1 || build_rc=$?
    else
      (cd "$VERILATOR_DIR" && make verilog CONFIG="$cls" -j"$JOBS") \
        >"$build_log" 2>&1 || build_rc=$?
    fi
    if (( build_rc != 0 )); then
      build_wall=$((SECONDS - t0))
      echo "[build-fail] $cls wall=${build_wall}s tail:" >&2
      tail -c 600 "$build_log" >&2
      rm -f "$build_log"
      (( KEEP_ARTIFACTS == 0 )) && rm -rf -- "$gen_dir"
      return 0
    fi
    build_wall=$((SECONDS - t0))
    rm -f "$build_log"
    echo "[build-ok] $cls wall=${build_wall}s"
  fi

  # --- Synth step -----------------------------------------------------
  if (( synth_pending == 1 )); then
    if ! _run_synth "$cls" "$extract"; then
      echo "[synth-fail] $cls (see per-config yosys workdir)" >&2
      # keep going; sim rows may still be worth collecting
    fi
  fi

  # --- Sim step -------------------------------------------------------
  if (( ${#pending_bench[@]} > 0 )); then
    _run_sim_pairs "$cls" "$side_col" "$build_wall" "$build_was_cached" \
      "$row_csv" pending_bench[@] pending_op[@]
  fi

  # --- S3 upload + cleanup -------------------------------------------
  if (( PUSH_S3 )) && (( pulled_from_s3 == 0 )); then
    _s3_upload_build "$cls" || true
  fi
  if (( SKIP_UPLOAD == 0 )) && (( SKIP_SIM == 0 )); then
    _s3_upload_sim "$cls" || true
  fi
  _cleanup_after "$cls"
  return 0
}

# Run zachjs-sv2v + yosys on the already-elaborated gen-collateral and
# append one row to $SYNTH_OUTPUT via hw_cost_model.parse_yosys_stat.
# Args: cls extract_json
_run_synth() {
  local cls=$1
  local extract=$2
  local gen_dir="$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"

  if [[ ! -d "$gen_dir/gen-collateral" ]]; then
    echo "[synth-fail] $cls: gen-collateral missing" >&2
    return 1
  fi

  local work="$gen_dir/yosys"
  mkdir -p "$work"
  local flat="$work/${TOP}_flat.v"

  if ! (cd "$work" && zachjs-sv2v -DSYNTHESIS --write="$flat" "$gen_dir"/gen-collateral/*.sv) \
       >"$work/sv2v.log" 2>&1; then
    echo "[sv2v-fail] $cls" >&2
    tail -c 400 "$work/sv2v.log" >&2
    return 1
  fi
  cp -f "$gen_dir/gen-collateral/plusarg_reader.v" "$work/"
  cp -f "$gen_dir/gen-collateral/EICG_wrapper.v" "$work/"

  local synth_log="$work/synth.log"
  local t1=$SECONDS
  if ! (cd "$work" && yosys -Q -l "$synth_log" -p "
      read_verilog -DSYNTHESIS ${TOP}_flat.v plusarg_reader.v EICG_wrapper.v
      hierarchy -check -top $TOP
      synth -top $TOP
      stat
  ") >/dev/null 2>&1; then
    echo "[yosys-fail] $cls" >&2
    tail -c 400 "$synth_log" >&2
    return 1
  fi
  local synth_wall=$((SECONDS - t1))

  # Sanity: an all-zero row is worse than nothing -- bail on empty stat.
  if ! grep -q "^=== " "$synth_log"; then
    echo "[yosys-no-stat] $cls: log has no '=== <mod> ===' blocks" >&2
    return 1
  fi

  local params_json sample_group
  params_json=$(echo "$extract" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["params"]))')
  sample_group=$(echo "$extract" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sample_group"])')

  # parse_yosys_stat's --csv-append already does its own fcntl LOCK_EX,
  # so no outer flock is needed here.
  if ! python3 -m hw_cost_model.parse_yosys_stat \
      --synth-log "$synth_log" \
      --side "$SIDE" \
      --config-name "$cls" \
      --config-params-json "$params_json" \
      --sample-group "$sample_group" \
      --csv-append "$SYNTH_OUTPUT" 2>>"$work/parse.log"; then
    echo "[parse-fail] $cls (see $work/parse.log)" >&2
    return 1
  fi

  echo "[synth-ok] $cls  synth=${synth_wall}s"
  return 0
}

# Run every pending (bench, op) sim for one config and append to
# $SIM_OUTPUT via flock. Build cost is attributed to the first row only
# (build_was_cached=1 for the rest) so downstream readers don't
# double-count. Args: cls side build_wall build_was_cached row_csv bench_ref op_ref
_run_sim_pairs() {
  local cls=$1 side=$2
  local build_wall=$3 build_was_cached=$4 row_csv=$5
  local -a bench_list=("${!6}") op_list=("${!7}")

  # Precompute the trailing param-value suffix once (parameter columns
  # in the same order as the input CSV header, so the sim CSV suffix
  # matches the header we wrote up front).
  local csv_suffix=""
  local extract_json
  extract_json=$(CSV_ROW="$row_csv" CSV_HEADER_STR="$CSV_HEADER_STR" python3 -c '
import os
header = os.environ["CSV_HEADER_STR"].split(",")
row    = os.environ["CSV_ROW"].split(",")
d = dict(zip(header, row))
out = []
for h in header:
    if h in ("config_name", "side", "sample_group"):
        continue
    out.append(d.get(h, ""))
print(",".join(out))
')
  csv_suffix=",$extract_json"

  # Build attribution: first row gets the wall_s cost, the rest are 0/1.
  local -a bw_list=() cached_list=()
  local i
  for (( i=0; i<${#bench_list[@]}; i++ )); do
    if (( i == 0 )); then
      bw_list+=("$build_wall")
      cached_list+=("$build_was_cached")
    else
      bw_list+=(0)
      cached_list+=(1)
    fi
  done

  if (( BENCH_PARALLEL > 1 )) && (( ${#bench_list[@]} > 1 )); then
    echo "[run-p]    $cls x ${#bench_list[@]} benches (parallel=$BENCH_PARALLEL)" >&2
    local running=0 idx
    for (( idx=0; idx<${#bench_list[@]}; idx++ )); do
      while (( running >= BENCH_PARALLEL )); do
        wait -n 2>/dev/null || true
        running=$((running - 1))
      done
      _run_bench_pair "$cls" "$side" \
        "${bench_list[$idx]}" "${op_list[$idx]}" \
        "${bw_list[$idx]}" "${cached_list[$idx]}" \
        "$csv_suffix" &
      running=$((running + 1))
    done
    wait
  else
    local idx
    for (( idx=0; idx<${#bench_list[@]}; idx++ )); do
      _run_bench_pair "$cls" "$side" \
        "${bench_list[$idx]}" "${op_list[$idx]}" \
        "${bw_list[$idx]}" "${cached_list[$idx]}" \
        "$csv_suffix"
    done
  fi
}

# Run one bench and append its CSV row. See run_sweep.sh's original for
# the parallelism story on why this is concurrency-safe under one CONFIG.
# Args: cls side bench op build_wall cached_col csv_suffix
_run_bench_pair() {
  set -u
  local cls=$1 side=$2 bench=$3 op=$4
  local this_build_wall=$5 cached_col=$6 csv_suffix=$7

  local elf="$BENCH_BUILD_DIR/${bench}_${op}.riscv"
  if [[ ! -f $elf ]]; then
    echo "[elf-missing] $elf" >&2
    return 0
  fi

  echo "[run]      $cls x ${bench}_${op}" >&2
  local rt0=$SECONDS
  timeout "$BENCH_TIMEOUT" \
    make -C "$VERILATOR_DIR" \
      CONFIG="$cls" BREAK_SIM_PREREQ=1 LOADMEM=1 \
      run-binary-fast BINARY="$elf" >/dev/null 2>&1 || true
  local wall=$((SECONDS - rt0))

  local log="$VERILATOR_OUTPUT_ROOT/chipyard.harness.TestHarness.$cls/${bench}_${op}.log"
  if [[ ! -f $log ]]; then
    echo "[no-log]   $log" >&2
    return 0
  fi
  local summary
  summary=$(grep 'ACCEL_SUMMARY:' "$log" | tail -1 || true)
  if [[ -z $summary ]]; then
    echo "[no-summary] $log (wall=${wall}s)" >&2
    return 0
  fi

  local iters cycles bytes
  iters=$(grep -oP 'iters=\K[0-9]+'        <<<"$summary" || echo 0)
  cycles=$(grep -oP 'total_cycles=\K[0-9]+' <<<"$summary" || echo 0)
  bytes=$(grep -oP 'total_bytes=\K[0-9]+'   <<<"$summary" || echo 0)
  local tput=0
  if (( cycles > 0 )); then
    tput=$(awk -v b="$bytes" -v c="$cycles" 'BEGIN{printf "%.6f", b*1e9/c}')
  fi

  local row="$cls,$side,$bench,$op,$iters,$cycles,$bytes,$tput,$wall,$this_build_wall,$cached_col$csv_suffix"
  (
    flock -w 60 9
    echo "$row" >> "$SIM_OUTPUT"
  ) 9>"$SIM_LOCK"

  echo "[done]     $cls x ${bench}_${op}: ${wall}s, ${bytes}B / ${cycles}cyc" >&2
}

# Delete per-config artifacts once uploads have had their shot at them.
_cleanup_after() {
  (( KEEP_ARTIFACTS == 1 )) && return 0
  local cls=$1
  rm -rf -- "$VERILATOR_GEN_ROOT/chipyard.harness.TestHarness.$cls"
  rm -f  -- "$VERILATOR_DIR/simulator-chipyard.harness-$cls" \
            "$VERILATOR_DIR/simulator-chipyard.harness-$cls.debug"
  rm -rf -- "$VERILATOR_OUTPUT_ROOT/chipyard.harness.TestHarness.$cls"
  echo "[cleanup]  $cls" >&2
}

export -f worker_one_config _run_synth _run_sim_pairs _run_bench_pair \
  _cleanup_after _s3_pull_build _s3_upload_build _s3_upload_sim \
  _kill_descendants_bottom_up _pkill_chipyard_stragglers
export SIDE TOP VERILATOR_DIR VERILATOR_GEN_ROOT VERILATOR_OUTPUT_ROOT
export BENCH_BUILD_DIR SYNTH_OUTPUT SIM_OUTPUT
export CSV_HEADER_STR
export S3_BUCKET S3_BUILD_KEY_PREFIX S3_SIM_KEY_PREFIX
export S3_BUILD_PREFIX S3_SIM_PREFIX AWS_PROFILE_NAME
export PULL_S3 PUSH_S3 SKIP_UPLOAD KEEP_ARTIFACTS SKIP_SYNTH SKIP_SIM SKIP_BUILD
export CONFIG_FILTER JOBS BENCH_TIMEOUT BENCH_PARALLEL
export RANDOM_BENCH ITER_BENCH HPB_SYNTH_START
export DEFAULT_BENCHES_STR="${DEFAULT_BENCHES[*]}"
export SYNTH_BENCHES_STR="${SYNTH_BENCHES[*]}"
export HPB_BENCHES_POOL_STR="${HPB_BENCHES_POOL[*]}"
export BENCHES_STR="${BENCHES[*]}"
# Re-hydrate the bench arrays inside each worker shell (parallel spawns fresh).
export _REHYDRATE_BENCHES=1

# Ensure python inside workers can find hw_cost_model.
export PYTHONPATH="$LYNX_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Sim CSV append lock. Synth append doesn't need one here -- parse_yosys_stat
# already does its own fcntl LOCK_EX on the target CSV.
export SIM_LOCK="${SIM_OUTPUT}.lock"

# --- Build the filtered work list ---------------------------------------
# Each output row is: "row_csv<TAB>side_idx". side_idx (1-based, per-side)
# is what --iter-bench uses to pair config-i-of-side with bench-i. Done
# before the env-source + prebuild so --dry-run stays cheap.
_filtered_rows() {
  tail -n +2 "$SWEEP_CSV" | awk -F',' -v side="$SIDE" -v re="$CONFIG_FILTER" '
    {
      if ($2 != side && $2 != "joint") next
      if (re != "" && $1 !~ re)         next
      s = $2 == "joint" ? side : $2
      cnt[s]++
      # Reconstruct the CSV row verbatim (awk splits on comma, join back).
      row = $0
      printf "%s\t%d\n", row, cnt[s]
    }
  '
}

if (( LIMIT_CONFIGS > 0 )); then
  _ROWS_TMP=$(mktemp -t "rows.XXXXXX")
  _filtered_rows > "$_ROWS_TMP"
  FILTERED=$(head -n "$LIMIT_CONFIGS" "$_ROWS_TMP")
  rm -f "$_ROWS_TMP"; _ROWS_TMP=""
  echo "[sweep] limit-configs=$LIMIT_CONFIGS (kept $(printf '%s\n' "$FILTERED" | grep -c .) rows)"
else
  FILTERED=$(_filtered_rows)
fi

TOTAL=$(printf '%s\n' "$FILTERED" | grep -c . || true)
echo "[sweep] plan: $TOTAL configs after filter"

if (( DRY_RUN )); then
  printf '%s\n' "$FILTERED" | awk -F'\t' '{
    n = split($1, a, ",")
    print "[dry-run] " a[1] " [side=" a[2] " idx=" $2 "]"
  }'
  exit 0
fi

# Source Chipyard env once so every worker's ``make`` sees sbt/firtool/$RISCV.
set +u
# shellcheck disable=SC1091
source "$CHIPYARD_ROOT/env.sh"
set -u

# Prebuild the jar serially -- same sbt-race reason both source scripts do this.
echo "[prebuild] ensuring $CLASSPATH_JAR is up to date..." >&2
make -C "$VERILATOR_DIR" CONFIG=ProtoAccelRocketConfig "$CLASSPATH_JAR" >/dev/null

# Wrap the worker so we can pass both fields via parallel's colsep.
_dispatch() {
  local combined=$1
  local row_csv side_idx
  row_csv=${combined%%$'\t'*}
  side_idx=${combined##*$'\t'}
  # Re-hydrate BENCHES / DEFAULT_BENCHES / SYNTH_BENCHES / HPB_BENCHES_POOL
  # inside the fresh shell.
  local -a BENCHES=() DEFAULT_BENCHES=() SYNTH_BENCHES=() HPB_BENCHES_POOL=()
  read -ra BENCHES         <<< "$BENCHES_STR"
  read -ra DEFAULT_BENCHES <<< "$DEFAULT_BENCHES_STR"
  read -ra SYNTH_BENCHES   <<< "$SYNTH_BENCHES_STR"
  read -ra HPB_BENCHES_POOL <<< "$HPB_BENCHES_POOL_STR"
  worker_one_config "$row_csv" "$side_idx"
}
export -f _dispatch

if command -v parallel >/dev/null; then
  # --halt-on-error 0: workers already swallow their own errors; this
  # keeps parallel from bailing on a signal-driven exit.
  printf '%s\n' "$FILTERED" \
    | parallel --line-buffer -j "$WORKERS" \
        --halt-on-error 0 \
        --termseq INT,1000,TERM,2000,KILL,25 \
        _dispatch {}
else
  echo "[warn] GNU parallel not on PATH; falling back to serial." >&2
  printf '%s\n' "$FILTERED" | while IFS= read -r combined; do
    [[ -z $combined ]] && continue
    _dispatch "$combined" || true
  done
fi

echo "[sweep] done"
echo "[sweep]   synth_output=$SYNTH_OUTPUT"
echo "[sweep]   sim_output=$SIM_OUTPUT"
