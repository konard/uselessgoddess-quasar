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
        /// Micro-batch the activation estimate is for.
        #[arg(long, default_value_t = 1)]
        micro_batch: usize,
        #[command(flatten)]
        shape: Shape,
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
        #[command(flatten)]
        shape: Shape,
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
    /// SSD algorithm: serial retains intermediates for speed; recalculated saves memory.
    #[arg(long, value_enum)]
    ssd: Option<Ssd>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Ssd {
    Minimal,
    Serial,
    Recalculated,
}

/// The model-shape knobs worth sweeping without editing a preset, because they
/// are the ones that decide whether a micro-batch fits. `quasar budget` answers
/// that before a run allocates anything.
#[derive(Args, Clone)]
struct Shape {
    #[arg(long)]
    seq_len: Option<usize>,
    #[arg(long)]
    state_rank: Option<usize>,
    #[arg(long)]
    mimo_rank: Option<usize>,
    #[arg(long)]
    expand: Option<usize>,
    /// Sliding-window radius; `0` means full causal attention.
    #[arg(long)]
    attn_window: Option<usize>,
    /// Attention every n-th layer; `0` means a pure-SSM stack.
    #[arg(long)]
    attn_period: Option<usize>,
    /// SSD scan chunk length; unset keeps the largest divisor of `seq_len`
    /// below burn-mamba's own rule of thumb.
    #[arg(long)]
    ssd_chunk: Option<usize>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    Tiny,
    /// `tiny`, cut down to what trains fastest inside 16 GB.
    TinyTurbo,
    Base,
    Toy,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Budget { preset, micro_batch, shape } => {
            budget(shape.apply(preset.config()), micro_batch)
        }
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
        Command::Train { preset, data, out, run, shape } => {
            // The tokenizer decides the vocabulary, not the preset: a corpus
            // whose BPE stopped short of the requested merges is common, and
            // silently building a wider embedding would be dead parameters.
            let mut cfg = shape.apply(preset.config());
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

/// What a preset costs, in the currencies that decide whether it fits.
fn budget(cfg: config::Model, micro_batch: usize) -> Result<()> {
    cfg.validate()?;
    let budget = cfg.budget();
    let params = budget.total as f64;
    let gib = |bytes_per_param: f64| params * bytes_per_param / (1 << 30) as f64;
    let flops = cfg.flops_per_token();

    println!("{budget}\n");
    println!("seq_len          {}", cfg.seq_len);
    println!("ssd chunk        {}", cfg.ssd_chunk_len());
    println!("attn keys/query  {}", cfg.attn_pairs() / cfg.seq_len);
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

    // States are the easy half: they are the same every step. The activations
    // are what an OOM at `micro_batch 2` is actually about, so they get their own
    // breakdown and the micro-batch the card has room for.
    let sixteen = (16u64 << 30) as f64;
    println!("\nactivations at micro_batch {micro_batch} (fp32, estimated)");
    println!("{}", cfg.activations(micro_batch));
    println!("\nmicro_batch in 16 GiB, muon states {}", cfg.micro_batch_within(sixteen, 12.0));
    println!("micro_batch in 16 GiB, fp32 adamw  {}", cfg.micro_batch_within(sixteen, 16.0));
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
            Self::TinyTurbo => config::Model::tiny_turbo(),
            Self::Base => config::Model::base(),
            Self::Toy => config::Model::toy(),
        }
    }
}

impl Shape {
    fn apply(&self, mut cfg: config::Model) -> config::Model {
        macro_rules! set {
            ($($field:ident),*) => {$(if let Some(value) = self.$field { cfg.$field = value; })*};
        }
        set!(seq_len, state_rank, mimo_rank, expand);
        // `0` is how a flag says "none" — clap has no `--attn-window=` spelling
        // for `Option<usize>` inside an `Option`.
        if let Some(window) = self.attn_window {
            cfg.attn_window = (window > 0).then_some(window);
        }
        if let Some(period) = self.attn_period {
            cfg.attn_period = (period > 0).then_some(period);
        }
        if let Some(chunk) = self.ssd_chunk {
            cfg.ssd_chunk = Some(chunk);
        }
        cfg
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
        if let Some(ssd) = self.ssd {
            run.ssd_mode = Some(match ssd {
                Ssd::Minimal => config::SsdMode::Minimal,
                Ssd::Serial => config::SsdMode::Serial,
                Ssd::Recalculated => config::SsdMode::Recalculated,
            });
        }
        run
    }
}
