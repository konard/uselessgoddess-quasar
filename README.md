# quasar

A Mamba-3 language model family trained end to end on one consumer GPU, in Rust
on [burn](https://github.com/tracel-ai/burn) + wgpu.

Two presets, both hybrid stacks of Mamba-3 blocks with a sliding-window GQA
layer every sixth or seventh position:

| | params | fwd FLOPs/token | states fp32 |
| --- | --- | --- | --- |
| `tiny` | 162.5M | 360.8M | 2.42 GiB |
| `base` | 1117.5M | 2306.2M | 16.65 GiB |

`docs/DESIGN.md` justifies every number above, states the training-time budget
honestly, and explains why this is not a mixture of experts.

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

Validation reports negative log-likelihood, perplexity and **bits-per-byte** —
the last is the only figure comparable across tokenizers, and the one the design
targets are written in.

## Backends

The default is a CPU backend, so `cargo test` needs no GPU. Training wants one:

```sh
# RDNA4 and anything else with a Vulkan driver
cargo run --release --no-default-features --features vulkan -- train tiny
```

Available: `flex` (CPU, default), `ndarray` (CPU), `wgpu`, `vulkan`, `rocm`,
`cuda`. On AMD, `vulkan` is the same runtime as `wgpu` compiled to SPIR-V rather
than WGSL, which the drivers handle better.

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
