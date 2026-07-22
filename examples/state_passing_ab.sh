#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs}
mkdir -p "$log_dir"
cd "$repo_root"

bench=(
    cargo run --release --no-default-features --features vulkan
    --example train_bench --
    --model tiny-turbo --micro-batch 4 --warmup 1 --steps 3
    --dtype f32 --ssd serial --checkpointing false --muon true
)

run_bench() {
    local label=$1
    local fused_state_passing=$2
    local fused_chunk_cumsum=$3
    local accum=$4
    local sampler_pid=""

    if command -v rocm-smi >/dev/null 2>&1; then
        (
            while true; do
                date -u +'%Y-%m-%dT%H:%M:%SZ'
                rocm-smi --showmeminfo vram
                sleep 1
            done
        ) >"$log_dir/vram-${label}.log" 2>&1 &
        sampler_pid=$!
    fi

    set +e
    BURN_MAMBA_FUSED_STATE_PASSING=$fused_state_passing \
        BURN_MAMBA_FUSED_CHUNK_CUMSUM=$fused_chunk_cumsum \
        "${bench[@]}" --accum "$accum" 2>&1 | tee "$log_dir/${label}.log"
    local benchmark_status=${PIPESTATUS[0]}
    set -e

    if [[ -n $sampler_pid ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
    return "$benchmark_status"
}

# 4 * 12 * 1024 is the issue's original 49,152-token optimizer step. Every
# stage uses the same binary, seed, model, data, warm-up, and token count. The
# middle run isolates K4 so the final run measures K1 incrementally.
run_bench reference-issue 0 0 12
run_bench cubecl-k4-issue 1 0 12
run_bench cubecl-k1-issue 1 1 12

# Keep the full 131,072-token preset under a real training-step and OOM check.
# Its three-step measured window is still shorter than one minute on the target.
run_bench cubecl-production 1 1 32

reference_throughput=$(sed -n 's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/reference-issue.log" | tail -n 1)
k4_throughput=$(sed -n 's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/cubecl-k4-issue.log" | tail -n 1)
k1_throughput=$(sed -n 's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/cubecl-k1-issue.log" | tail -n 1)
production_throughput=$(sed -n 's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/cubecl-production.log" | tail -n 1)

test -n "$reference_throughput"
test -n "$k4_throughput"
test -n "$k1_throughput"
test -n "$production_throughput"

awk '
    FNR == 1 { variant += 1 }
    /^measured / {
        for (field = 1; field <= NF; field += 1) {
            if ($field ~ /^loss=/) {
                value = $field
                sub(/^loss=/, "", value)
                losses[variant, ++count[variant]] = value
            }
        }
    }
    END {
        if (count[1] == 0 || count[1] != count[2] || count[1] != count[3]) {
            print "missing or incomplete measured loss sequence" > "/dev/stderr"
            exit 1
        }
        for (step = 1; step <= count[1]; step += 1) {
            for (variant = 2; variant <= 3; variant += 1) {
                difference = losses[1, step] - losses[variant, step]
                if (difference < 0) difference = -difference
                if (difference > 0.01) {
                    printf "loss mismatch at step %d: reference=%s candidate=%s\n", \
                        step, losses[1, step], losses[variant, step] > "/dev/stderr"
                    exit 1
                }
            }
        }
    }
' "$log_dir/reference-issue.log" "$log_dir/cubecl-k4-issue.log" \
    "$log_dir/cubecl-k1-issue.log"

awk -v reference="$reference_throughput" -v k4="$k4_throughput" \
    -v k1="$k1_throughput" -v production="$production_throughput" 'BEGIN {
    k4_change = 100 * (k4 / reference - 1)
    k1_change = 100 * (k1 / k4 - 1)
    total_change = 100 * (k1 / reference - 1)
    printf "stepwise A/B: reference=%d K4=%d (%+.1f%%) K4+K1=%d (%+.1f%% incremental, %+.1f%% total) production=%d tok/s\n", \
        reference, k4, k4_change, k1, k1_change, total_change, production
    if (k4 < reference * 0.95) {
        print "CubeCL K4 regressed by more than 5%" > "/dev/stderr"
        exit 1
    }
    if (k1 < k4 * 0.95) {
        print "CubeCL K1 regressed by more than 5% relative to K4" > "/dev/stderr"
        exit 1
    }
}'
