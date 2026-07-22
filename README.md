# quasar

A Mamba-3 language model family trained end to end on one consumer GPU, in Rust
on [burn](https://github.com/tracel-ai/burn) + wgpu.

Three presets, all hybrid stacks of Mamba-3 blocks with a sliding-window GQA
layer every fifth, sixth or seventh position:

| | params | fwd FLOPs/token | states fp32 | activations / micro-batch | micro-batches in 16 GiB |
| --- | --- | --- | --- | --- | --- |
| `tiny-turbo` | 77.7M | 161.5M | 1.16 GiB | 1.61 GiB | 9 |
| `tiny` | 162.5M | 360.8M | 2.42 GiB | 6.99 GiB | 2 |
| `base` | 1117.5M | 2306.2M | 16.65 GiB | 24.48 GiB | 0 |

`docs/DESIGN.md` justifies every number above, states the training-time budget
honestly, and explains why this is not a mixture of experts.
[`docs/MEMORY.md`](docs/MEMORY.md) takes apart the last two columns — where the
VRAM actually goes, which burn-mamba setting moves it, and what `tiny-turbo`
gives up to fit nine micro-batches where `tiny` fits two.

## Pipeline

```sh
# what a preset costs before committing a week to it
cargo run --release -- budget tiny

# a corpus: parquet, jsonl or txt, files or directories
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
    --include "sample/10BT/*" --local-dir data/fineweb-edu

cargo run --release -- tokenizer data/fineweb-edu --vocab-size 32768
cargo run --release -- prepare data/fineweb-edu --out data/shards
cargo run --release -- train tiny --data data/shards --out runs/tiny

cargo run --release -- eval runs/tiny --data data/shards
cargo run --release -- generate runs/tiny --prompt "The reason"
```

`train` resumes from the newest checkpoint under `--out`, so a run interrupted
at any point continues where it stopped. Overrides worth knowing:
`--steps`, `--micro-batch`, `--accum`, `--lr`, `--warmup`, `--decay`,
`--save-every`, `--eval-every`.

The default tiny recipe is 12,500 optimizer steps, or 3.2768B tokens with the
default `8 × 16 × 2048` effective batch. Changing either batch knob also changes
the total token budget unless `--steps` is adjusted. The startup summary prints
both quantities before training begins. A progress item in the dashboard is one
optimizer step, not one sequence or token.

When training is attached to a terminal it opens Burn's official TUI, with
live plots for training/validation loss, perplexity, bits-per-byte, learning
rate, throughput, tokens processed, ETA, and effective TFLOP/s. Press `q`, then
`s`, to stop cleanly; the loop writes a resumable checkpoint before exiting.
Redirected output and CI keep the line-oriented logs instead. See
[`docs/TRAINING_SPEED.md`](docs/TRAINING_SPEED.md) for interpreting these
numbers and the investigation behind the defaults.

Two knobs decide whether a preset fits the card, and both are on by default:
`--muon false` puts the hidden matrices back on AdamW (16 B/param of state
instead of 12, which is what stopped `base` fitting 16 GB), and
`--checkpointing false` stops recomputing activations in the backward, trading
memory back for about a third of the step time. See `docs/DESIGN.md` §3.

Validation reports negative log-likelihood, perplexity and **bits-per-byte** —
the last is the only figure comparable across tokenizers, and the one the design
targets are written in.

The first GPU step is not representative of training speed or peak live tensor
memory. With the GPU features, Burn compiles fused kernels and benchmarks
candidate implementations for the shapes it sees; utilization therefore comes
in bursts while VRAM grows before the steady loop begins. `budget` now prints an
`activations` breakdown and the largest micro-batch that fits 16 GiB alongside
the `states` figures, so the answer is available before a run allocates anything
— but it is an analytic estimate of the tensors autodiff must retain, not a
measurement, and it does not cover fusion/autotuning workspaces or the backend
allocator's cache. `--micro-batch`, `--seq-len`, `--state-rank`, `--mimo-rank`,
`--expand`, `--attn-window`, `--attn-period` and `--ssd-chunk` all work on
`budget`, so a shape can be swept without rebuilding a preset. Still start at
`--micro-batch 1` on new hardware and raise it once the first optimizer step has
completed; change `--accum` inversely if the effective token batch must remain
fixed.

## Backends

The default is a CPU backend, so `cargo test` needs no GPU. Training wants one:

```sh
# RDNA4 and anything else with a Vulkan driver
cargo run --release --no-default-features --features vulkan -- train tiny

# the same card through ROCm/HIP, which needs the ROCm toolchain installed
cargo run --release --no-default-features --features rocm -- train tiny
```

Available: `flex` (CPU, default), `ndarray` (CPU), `wgpu`, `vulkan`, `rocm`,
`cuda`. On AMD, `vulkan` is the same runtime as `wgpu` compiled to SPIR-V rather
than WGSL, which the drivers handle better; `rocm` goes through HIP instead, and
which of the two is faster on RDNA4 is a question for a measurement, not for a
README. It needs a ROCm installation whose `hipcc` knows the card's target
(`gfx1201` for RX 9070 XT); `rocminfo` says what it is.

All four GPU features go through `gpu`, which turns on fusion in burn *and* in
burn-mamba together — burn's GPU backends are `Fusion<CubeBackend<_>>`, and
burn-mamba only implements its SSD extension for `Fusion` when its own `fusion`
feature is on.

The normal `release` profile deliberately skips LTO so GPU experiments do not
pay a full-program link on every iteration. For an infrequent final local build,
opt in with `cargo build --profile release-lto`; that separate profile enables
thin LTO and a single codegen unit.

## Trying it without a GPU

```sh
examples/smoke.sh
```

Fits a tokenizer, shards a synthetic corpus, trains 50 steps, evaluates and
samples — the whole pipeline in under a minute on a CPU.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
