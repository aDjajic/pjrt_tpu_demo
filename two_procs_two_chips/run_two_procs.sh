#!/usr/bin/env bash
# ===--------------------------------------------------------------------=== #
# run_two_procs.sh: launch two independent processes, each owning one TPU chip
# ===--------------------------------------------------------------------=== #
#
#   ./two_procs_two_chips/run_two_procs.sh <libtpu_so> [options]
#
# Options:
#   --chips A,B     which two chips to use (default 0,1)
#   --iters N       matmuls per process (default 4000)
#   --serial        run the two processes one after the other instead
#   --compare       run serial first, then parallel, and compare wall time
#   --bin PATH      worker binary (default: build/two-procs-two-chips)
#
# Each worker is a plain single-device PJRT program. It pins itself to one
# chip with TPU_VISIBLE_CHIPS (+ 1x1x1 process/chip bounds) before loading
# libtpu, so two of them can coexist on one host. See README.md here.

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

LIBTPU=""
CHIPS="0,1"
ITERS=4000
MODE="parallel"
BIN="$ROOT/build/two-procs-two-chips"
MLIR="$DIR/matmul_mosaic.mlir"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chips)   CHIPS="$2"; shift 2 ;;
        --iters)   ITERS="$2"; shift 2 ;;
        --serial)  MODE="serial"; shift ;;
        --compare) MODE="compare"; shift ;;
        --bin)     BIN="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *)
            if [[ -z "$LIBTPU" ]]; then LIBTPU="$1"; shift
            else echo "Unexpected argument: $1" >&2; exit 1; fi ;;
    esac
done

if [[ -z "$LIBTPU" ]]; then
    echo "Usage: $0 <libtpu_so> [--chips A,B] [--iters N] [--serial|--compare]" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "Worker binary not found: $BIN" >&2
    echo "Build it first: cmake --build build --target two-procs-two-chips" >&2
    exit 1
fi

CHIP_A="${CHIPS%%,*}"
CHIP_B="${CHIPS##*,}"
if [[ "$CHIP_A" == "$CHIP_B" ]]; then
    echo "Note: both processes target chip $CHIP_A. The second one is" >&2
    echo "expected to fail with 'Device or resource busy'." >&2
fi

# One sub-directory per run under two_procs_two_chips/logs/ (git-ignored).
LOG_DIR="$DIR/logs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$LOG_DIR"
# libtpu writes its own driver logs here; keep them next to ours so you can
# grep for "Driver opened" to see which /dev/vfio/<n> each pid grabbed.
export TPU_LOG_DIR="${TPU_LOG_DIR:-$LOG_DIR/tpu_logs}"
mkdir -p "$TPU_LOG_DIR"

now_ms() { date +%s%3N; }

run_worker() {  # <chip> <logfile>
    "$BIN" "$MLIR" "$LIBTPU" "$1" "$ITERS" >"$2" 2>&1
}

show_log() {  # <role> <chip> <logfile>
    sed "s/^/[$1: chip $2] /" "$3"
}

exec_window() {  # <logfile> -> "start_ms end_ms"
    awk '/^EXEC_WINDOW/ {
        for (i = 1; i <= NF; i++) {
            split($i, kv, "=");
            if (kv[1] == "start_ms") s = kv[2];
            if (kv[1] == "end_ms")   e = kv[2];
        }
        print s, e
    }' "$1"
}

run_parallel() {
    local la="$LOG_DIR/parallel_A_chip${CHIP_A}.log"
    local lb="$LOG_DIR/parallel_B_chip${CHIP_B}.log"
    echo "################ PARALLEL: chip $CHIP_A and chip $CHIP_B at the same time"
    # Start barrier: each worker touches ready.<role> once its chip is set up
    # and waits for "go". We release both at once so the execute windows
    # start together instead of whenever each process happened to finish
    # libtpu init + compile.
    local barrier="$LOG_DIR/barrier"
    mkdir -p "$barrier"
    export TWO_PROCS_BARRIER_DIR="$barrier"

    local t0; t0=$(now_ms)
    TWO_PROCS_BARRIER_TAG=A run_worker "$CHIP_A" "$la" & local pa=$!
    TWO_PROCS_BARRIER_TAG=B run_worker "$CHIP_B" "$lb" & local pb=$!

    # Wait until both are ready (or one of them has died), then release.
    while [[ ! -e "$barrier/ready.A" || ! -e "$barrier/ready.B" ]]; do
        if ! kill -0 "$pa" 2>/dev/null || ! kill -0 "$pb" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    touch "$barrier/go"
    unset TWO_PROCS_BARRIER_DIR

    local ra=0 rb=0
    wait "$pa" || ra=$?
    wait "$pb" || rb=$?
    local t1; t1=$(now_ms)
    PARALLEL_WALL_MS=$((t1 - t0))

    show_log A "$CHIP_A" "$la"
    echo
    show_log B "$CHIP_B" "$lb"
    echo
    echo "---- parallel summary"
    echo "exit codes: A (chip $CHIP_A) -> $ra, B (chip $CHIP_B) -> $rb"
    echo "total wall time (both processes, incl. libtpu init + compile): ${PARALLEL_WALL_MS} ms"

    read -r sa ea < <(exec_window "$la")
    read -r sb eb < <(exec_window "$lb")
    if [[ -n "${sa:-}" && -n "${sb:-}" ]]; then
        local base=$(( sa < sb ? sa : sb ))
        local ov_start=$(( sa > sb ? sa : sb ))
        local ov_end=$(( ea < eb ? ea : eb ))
        local overlap=$(( ov_end - ov_start ))
        (( overlap < 0 )) && overlap=0
        echo
        echo "execute windows (ms, relative to the earlier start):"
        printf "  A: chip %-3s  [%6d .. %6d]  duration %6d ms\n" "$CHIP_A" $((sa-base)) $((ea-base)) $((ea-sa))
        printf "  B: chip %-3s  [%6d .. %6d]  duration %6d ms\n" "$CHIP_B" $((sb-base)) $((eb-base)) $((eb-sb))
        printf "  overlap      %6d ms\n" "$overlap"
        if (( overlap > 0 )); then
            echo "  -> the two chips were executing at the same time"
        else
            echo "  -> no overlap (did one process fail or finish before the other started?)"
        fi
    fi
    echo
    echo "which physical device did each pid open (from libtpu driver logs):"
    grep -h -o -E "\[/dev/vfio/[0-9]+ [^]]*\] vf_id:[0-9]+ Driver opened" "$TPU_LOG_DIR"/*INFO* 2>/dev/null | sort -u | sed 's/^/  /' || true
    return $(( ra != 0 || rb != 0 ))
}

run_serial() {
    local la="$LOG_DIR/serial_A_chip${CHIP_A}.log"
    local lb="$LOG_DIR/serial_B_chip${CHIP_B}.log"
    echo "################ SERIAL: chip $CHIP_A, then chip $CHIP_B"
    local t0; t0=$(now_ms)
    local ra=0 rb=0
    run_worker "$CHIP_A" "$la" || ra=$?
    run_worker "$CHIP_B" "$lb" || rb=$?
    local t1; t1=$(now_ms)
    SERIAL_WALL_MS=$((t1 - t0))
    show_log A "$CHIP_A" "$la" | grep -E "\] \[(3|6|9)\]|EXEC_WINDOW|Max abs|done"
    show_log B "$CHIP_B" "$lb" | grep -E "\] \[(3|6|9)\]|EXEC_WINDOW|Max abs|done"
    echo "---- serial summary"
    echo "exit codes: A (chip $CHIP_A) -> $ra, B (chip $CHIP_B) -> $rb"
    echo "total wall time (one after the other): ${SERIAL_WALL_MS} ms"
    return $(( ra != 0 || rb != 0 ))
}

status=0
case "$MODE" in
    parallel) run_parallel || status=$? ;;
    serial)   run_serial   || status=$? ;;
    compare)
        run_serial || status=$?
        echo
        run_parallel || status=$?
        echo
        echo "################ COMPARE"
        echo "serial   : ${SERIAL_WALL_MS} ms"
        echo "parallel : ${PARALLEL_WALL_MS} ms"
        if (( PARALLEL_WALL_MS > 0 )); then
            awk -v s="$SERIAL_WALL_MS" -v p="$PARALLEL_WALL_MS" \
                'BEGIN { printf "speedup  : %.2fx\n", s / p }'
        fi
        ;;
esac

echo
echo "logs: $LOG_DIR"
exit $status
