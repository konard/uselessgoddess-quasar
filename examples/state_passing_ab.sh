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
    local fused=$2
    local accum=$3
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
    BURN_MAMBA_FUSED_STATE_PASSING=$fused "${bench[@]}" --accum "$accum" \
        2>&1 | tee "$log_dir/${label}.log"
    local benchmark_status=${PIPESTATUS[0]}
    set -e

    if [[ -n $sampler_pid ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
    return "$benchmark_status"
}

# 4 * 12 * 1024 is the issue's original 49,152-token optimizer step. Both
# variants use the same binary, seed, model, data, warm-up, and token count.
run_bench reference-issue 0 12
run_bench cubecl-issue 1 12

# Keep the full 131,072-token preset under a real training-step and OOM check.
# Its three-step measured window is still shorter than one minute on the target.
run_bench cubecl-production 1 32

reference_throughput=$(sed -n 's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/reference-issue.log" | tail -n 1)
cubecl_throughput=$(sed -n 's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/cubecl-issue.log" | tail -n 1)
reference_loss=$(sed -n 's/^measured .* loss=\([^ ]*\) .*/\1/p' \
    "$log_dir/reference-issue.log" | tail -n 1)
cubecl_loss=$(sed -n 's/^measured .* loss=\([^ ]*\) .*/\1/p' \
    "$log_dir/cubecl-issue.log" | tail -n 1)

test -n "$reference_throughput"
test -n "$cubecl_throughput"
test -n "$reference_loss"
test -n "$cubecl_loss"

awk -v reference="$reference_loss" -v cubecl="$cubecl_loss" 'BEGIN {
    difference = reference - cubecl
    if (difference < 0) difference = -difference
    if (difference > 0.01) {
        printf "loss mismatch: reference=%s cubecl=%s\n", reference, cubecl > "/dev/stderr"
        exit 1
    }
}'

awk -v reference="$reference_throughput" -v cubecl="$cubecl_throughput" 'BEGIN {
    change = 100 * (cubecl / reference - 1)
    printf "state-passing A/B: reference=%d tok/s cubecl=%d tok/s change=%+.1f%%\n",
        reference, cubecl, change
    if (cubecl < reference * 0.95) {
        print "CubeCL state passing regressed by more than 5%" > "/dev/stderr"
        exit 1
    }
}'
