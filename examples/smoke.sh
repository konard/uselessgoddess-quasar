#!/usr/bin/env bash
# The whole pipeline on a toy corpus, in under a minute on a CPU.
#
# This is what a real run looks like with every number shrunk: fit a tokenizer,
# shard a corpus, train, score, sample. It exists so the harness can be checked
# end to end without a GPU or a download.
set -euo pipefail

work="${1:-$(mktemp -d)}"
mkdir -p "$work"
echo "workspace $work"

python3 - "$work/corpus.jsonl" <<'PY'
import json, random, sys

# Deterministic nonsense with real word statistics: enough structure that the
# loss moves, small enough that a toy model can hold it.
random.seed(7)
words = "the quick brown fox jumps over lazy dog while nine ravens watch".split()
with open(sys.argv[1], "w") as out:
    for _ in range(2000):
        text = " ".join(random.choice(words) for _ in range(random.randint(20, 60)))
        out.write(json.dumps({"text": text}) + "\n")
PY

cargo run --release -q -- tokenizer "$work/corpus.jsonl" \
    --out "$work/tokenizer.json" --vocab-size 512
cargo run --release -q -- prepare "$work/corpus.jsonl" \
    --tokenizer "$work/tokenizer.json" --out "$work/shards"
cargo run --release -q -- train toy \
    --data "$work/shards" --out "$work/run" \
    --steps 50 --micro-batch 8 --accum 1 --warmup 5 --decay 20 --save-every 25
cargo run --release -q -- eval "$work/run" --data "$work/shards" --batches 4 --batch 4
cargo run --release -q -- generate "$work/run" \
    --tokenizer "$work/tokenizer.json" --prompt "the quick" --tokens 16
