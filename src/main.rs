//! `quasar` — the command line around the crate.
//!
//! The pipeline is four commands in order: `budget` to see what a preset costs,
//! `tokenizer` to fit a vocabulary, `prepare` to turn a download into shards,
//! `train` to run. `eval` and `generate` inspect what came out.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Args, Parser, Subcommand, ValueEnum};

use quasar::data::{Batcher, Corpus, Shards, Tokenizer, prepare};
use quasar::model::Quasar;
use quasar::train::checkpoint;
use quasar::{config, eval, generate, train};

#[derive(Parser)]
#[command(name = "quasar", version, about = "A Mamba-3 language model on one GPU")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parameters, FLOPs and memory of a preset.
    Budget {
        #[arg(value_enum, default_value_t = Preset::Tiny)]
        preset: Preset,
    },
    /// Fit a byte-level BPE vocabulary on the corpus.
    Tokenizer {
        /// Files or directories of parquet/jsonl/txt documents.
        #[arg(required = true)]
        corpus: Vec<PathBuf>,
        #[arg(long, default_value = "data/tokenizer.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 32_768)]
        vocab_size: usize,
        /// Documents to fit on. The full corpus is not needed and would take
        /// hours; BPE merges converge long before it.
        #[arg(long, default_value_t = 2_000_000)]
        docs: usize,
        #[arg(long, default_value = "text")]
        field: String,
    },
    /// Tokenize the corpus into `train/` and `valid/` shards.
    Prepare {
        #[arg(required = true)]
        corpus: Vec<PathBuf>,
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "data/shards")]
        out: PathBuf,
        #[arg(long, default_value = "text")]
        field: String,
    },
    /// Train, resuming from the newest checkpoint under `--out`.
    Train {
        #[arg(value_enum, default_value_t = Preset::Tiny)]
        preset: Preset,
        #[arg(long, default_value = "data/shards")]
        data: PathBuf,
        #[arg(long, default_value = "runs/tiny")]
        out: PathBuf,
        #[command(flatten)]
        run: Overrides,
    },
    /// Score a checkpoint on the validation shards.
    Eval {
        run: PathBuf,
        #[arg(long, default_value = "data/shards")]
        data: PathBuf,
        #[arg(long, default_value_t = 100)]
        batches: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
    },
    /// Continue a prompt with a checkpoint.
    Generate {
        run: PathBuf,
        #[arg(long, default_value = "\n")]
        prompt: String,
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
    },
}

/// The run knobs worth changing from the command line; the rest live in
/// `run.json` next to the checkpoints and are read back on resume.
#[derive(Args)]
struct Overrides {
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    micro_batch: Option<usize>,
    #[arg(long)]
    accum: Option<usize>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    warmup: Option<usize>,
    #[arg(long)]
    decay: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    save_every: Option<usize>,
    #[arg(long)]
    eval_every: Option<usize>,
    /// Muon on the hidden matrices; `false` puts everything on AdamW.
    #[arg(long)]
    muon: Option<bool>,
    /// Recompute activations in the backward.
    #[arg(long)]
    checkpointing: Option<bool>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    Tiny,
    Base,
    Toy,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Budget { preset } => budget(preset.config()),
        Command::Tokenizer { corpus, out, vocab_size, docs, field } => {
            tokenizer(&corpus, &out, vocab_size, docs, &field)
        }
        Command::Prepare { corpus, tokenizer, out, field } => {
            let corpus = Corpus::open(&corpus, &field)?;
            let tokenizer = Tokenizer::load(&tokenizer)?;
            let prepared = prepare::run(&corpus, &tokenizer, &out)?;
            println!("train {} tokens", prepared.train.tokens);
            println!("valid {} tokens", prepared.valid.tokens);
            Ok(())
        }
        Command::Train { preset, data, out, run } => {
            // The tokenizer decides the vocabulary, not the preset: a corpus
            // whose BPE stopped short of the requested merges is common, and
            // silently building a wider embedding would be dead parameters.
            let mut cfg = preset.config();
            cfg.vocab_size = Shards::open(&data.join("train"))?.meta().vocab_size;
            let run = run.apply(train::Run::new());
            train::run(&cfg, &run, &data, &out, &Device::default())?;
            Ok(())
        }
        Command::Eval { run, data, batches, batch } => evaluate(&run, &data, batches, batch),
        Command::Generate { run, prompt, tokenizer, tokens, temperature, top_k, seed } => {
            let sampler = generate::Sampler { temperature, top_k, max_tokens: tokens, seed };
            sample(&run, &tokenizer, &prompt, &sampler)
        }
    }
}

/// What a preset costs, in the three currencies that decide whether it fits.
fn budget(cfg: config::Model) -> Result<()> {
    let budget = cfg.budget();
    let params = budget.total as f64;
    let gib = |bytes_per_param: f64| params * bytes_per_param / (1 << 30) as f64;
    let flops = cfg.flops_per_token();

    println!("{budget}\n");
    println!("seq_len          {}", cfg.seq_len);
    println!("fwd FLOPs/token  {:.1}M", flops / 1e6);
    println!("step FLOPs/token {:.1}M", 3.0 * flops / 1e6);
    // Weights, gradients and two Adam moments. Pure bf16 is 8 B/param; keeping
    // fp32 master weights and moments doubles it, which is what decides whether
    // a preset fits 16 GB before a single activation is allocated.
    println!("states bf16      {:.2} GiB", gib(8.0));
    println!("states fp32      {:.2} GiB", gib(16.0));
    // Muon keeps one momentum buffer where AdamW keeps two moments, and every
    // matrix in the stack is on Muon — only the vocabulary and the norms are not.
    println!("states muon      {:.2} GiB", gib(12.0));
    Ok(())
}

fn tokenizer(
    corpus: &[PathBuf],
    out: &Path,
    vocab_size: usize,
    docs: usize,
    field: &str,
) -> Result<()> {
    let corpus = Corpus::open(corpus, field)?;
    println!("fitting {vocab_size} tokens on up to {docs} documents");

    let stream = corpus.docs().filter_map(Result::ok).take(docs);
    let tokenizer = Tokenizer::train(stream, vocab_size)?;

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    tokenizer.save(out)?;
    println!("wrote {} tokens to {}", tokenizer.vocab_size(), out.display());
    Ok(())
}

fn evaluate(run: &Path, data: &Path, batches: usize, batch: usize) -> Result<()> {
    let (cfg, dir) = trained(run)?;
    let device = Device::default();
    let mut model = Quasar::new(&cfg, &device);
    checkpoint::weights(&dir, &mut model)?;

    let shards = Shards::open(&data.join("valid"))?;
    let valid = Batcher::new(shards, cfg.seq_len, batch, 0);
    println!("{}", eval::evaluate(&model, &valid, batches, &device));
    Ok(())
}

fn sample(run: &Path, tokenizer: &Path, prompt: &str, sampler: &generate::Sampler) -> Result<()> {
    let (cfg, dir) = trained(run)?;
    let device = Device::default();
    let mut model = Quasar::new(&cfg, &device);
    checkpoint::weights(&dir, &mut model)?;
    let tokenizer = Tokenizer::load(tokenizer)?;

    let text = generate::generate(&model, &tokenizer, prompt, cfg.seq_len, sampler, &device)?;
    println!("{prompt}{text}");
    Ok(())
}

/// The config and newest checkpoint of a run directory.
fn trained(run: &Path) -> Result<(config::Model, PathBuf)> {
    let cfg = config::Model::load(run.join("model.json"))
        .with_context(|| format!("no model.json in {}", run.display()))?;
    let dir =
        checkpoint::latest(run).with_context(|| format!("no checkpoint in {}", run.display()))?;
    println!("{}", dir.display());
    Ok((cfg, dir))
}

impl Preset {
    fn config(self) -> config::Model {
        match self {
            Self::Tiny => config::Model::tiny(),
            Self::Base => config::Model::base(),
            Self::Toy => config::Model::toy(),
        }
    }
}

impl Overrides {
    fn apply(&self, mut run: train::Run) -> train::Run {
        macro_rules! set {
            ($($field:ident),*) => {$(if let Some(value) = self.$field { run.$field = value; })*};
        }
        set!(
            steps,
            micro_batch,
            accum,
            lr,
            warmup,
            decay,
            seed,
            save_every,
            eval_every,
            muon,
            checkpointing
        );
        run
    }
}
