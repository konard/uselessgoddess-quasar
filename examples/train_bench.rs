//! Short, data-independent GPU benchmark for the real tiny-turbo training step.
//!
//! It deliberately uses the full 32k vocabulary and the production optimizer;
//! only corpus I/O and checkpoint-file writes are omitted from the timed
//! region. The first complete optimizer step is warm-up so CubeCL fusion and
//! autotuning do not contaminate the measurement.

use std::time::Instant;

use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{DeviceConfig, FloatDType};
use clap::{Parser, ValueEnum};
use quasar::config::{Model, SsdMode};
use quasar::model::Quasar;
use quasar::train::{Optim, Run};

#[derive(Parser)]
#[command(about = "Measure a few synchronized tiny-turbo training steps")]
struct Args {
    #[arg(long, value_enum, default_value_t = BenchModel::TinyTurbo)]
    model: BenchModel,
    #[arg(long, default_value_t = 4)]
    micro_batch: usize,
    #[arg(long, default_value_t = 32)]
    accum: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    steps: usize,
    #[arg(long, value_enum, default_value_t = Dtype::F32)]
    dtype: Dtype,
    #[arg(long, value_enum, default_value_t = Ssd::Serial)]
    ssd: Ssd,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    checkpointing: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    muon: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Ssd {
    Minimal,
    Serial,
    Recalculated,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchModel {
    TinyTurbo,
    LegacyTurbo,
}

impl BenchModel {
    fn config(self) -> Model {
        let mut cfg = Model::tiny_turbo();
        if matches!(self, Self::LegacyTurbo) {
            // The pre-optimization shape remains available so the paired A/B
            // can be reproduced after the wide candidate became the preset.
            cfg.d_model = 512;
            cfg.n_layers = 20;
            cfg.attn_heads = 8;
        }
        cfg.validate().expect("benchmark model must be valid");
        cfg
    }
}

impl From<Dtype> for FloatDType {
    fn from(value: Dtype) -> Self {
        match value {
            Dtype::F32 => Self::F32,
            Dtype::F16 => Self::F16,
            Dtype::Bf16 => Self::BF16,
        }
    }
}

impl From<Ssd> for SsdMode {
    fn from(value: Ssd) -> Self {
        match value {
            Ssd::Minimal => Self::Minimal,
            Ssd::Serial => Self::Serial,
            Ssd::Recalculated => Self::Recalculated,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    assert!(args.micro_batch > 0, "micro-batch must be positive");
    assert!(args.accum > 0, "accum must be positive");
    assert!(args.steps > 0, "at least one measured step is required");

    let cfg = args.model.config();
    let mut base_device = Device::default();
    base_device.configure(DeviceConfig::default().float_dtype(FloatDType::from(args.dtype)))?;
    let device = base_device.clone().autodiff();
    let device = if args.checkpointing { device.gradient_checkpointing() } else { device };
    device.seed(1337);

    let ssd_mode = SsdMode::from(args.ssd);
    let mut model = Quasar::new_with_ssd(&cfg, ssd_mode.clone(), &device);
    let run = Run::new()
        .with_micro_batch(args.micro_batch)
        .with_accum(args.accum)
        .with_muon(args.muon)
        .with_checkpointing(args.checkpointing)
        .with_ssd_mode(Some(ssd_mode));
    let mut optim = Optim::new(&run, &model);
    let (input, target) = tokens(&cfg, args.micro_batch, &device);
    let tokens_per_step = args.micro_batch * args.accum * cfg.seq_len;

    println!(
        "bench device={base_device:?} model={:?} dtype={:?} micro_batch={} accum={} ssd={:?} checkpointing={} muon={} tokens/step={tokens_per_step}",
        args.model,
        args.dtype,
        args.micro_batch,
        args.accum,
        args.ssd,
        args.checkpointing,
        args.muon
    );

    for step in 0..args.warmup {
        let (next, loss) = optimizer_step(model, &mut optim, &input, &target, args.accum, &device)?;
        model = next;
        println!("warmup {}/{} loss={loss:.4}", step + 1, args.warmup);
    }

    let mut seconds = Vec::with_capacity(args.steps);
    for step in 0..args.steps {
        let started = Instant::now();
        let (next, loss) = optimizer_step(model, &mut optim, &input, &target, args.accum, &device)?;
        model = next;
        let elapsed = started.elapsed().as_secs_f64();
        let throughput = tokens_per_step as f64 / elapsed;
        println!(
            "measured {}/{} loss={loss:.4} seconds={elapsed:.3} throughput={throughput:.0} tok/s",
            step + 1,
            args.steps
        );
        seconds.push(elapsed);
    }

    seconds.sort_by(f64::total_cmp);
    let median = seconds[seconds.len() / 2];
    let throughput = tokens_per_step as f64 / median;
    let tflops = throughput * 3.0 * cfg.flops_per_token() / 1e12;
    println!(
        "result median_seconds={median:.3} throughput={throughput:.0} tok/s effective={tflops:.2} TFLOP/s"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape_trades_depth_for_width_without_changing_work() {
        let narrow = BenchModel::LegacyTurbo.config();
        let wide = BenchModel::TinyTurbo.config();

        let param_ratio = wide.budget().total as f64 / narrow.budget().total as f64;
        let flop_ratio = wide.flops_per_token() / narrow.flops_per_token();
        let activation_ratio = wide.activations(1).total / narrow.activations(1).total;

        assert!((param_ratio - 1.0).abs() < 0.01, "parameter ratio {param_ratio}");
        assert!((flop_ratio - 1.0).abs() < 0.01, "FLOP ratio {flop_ratio}");
        assert!(activation_ratio < 0.81, "activation ratio {activation_ratio}");
        assert_eq!((wide.d_model, wide.n_layers, wide.attn_heads), (640, 12, 10));
    }
}

fn tokens(cfg: &Model, batch: usize, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let len = batch * cfg.seq_len;
    let data: Vec<i32> = (0..=len).map(|i| (i % cfg.vocab_size) as i32).collect();
    let input = TensorData::new(data[..len].to_vec(), [batch, cfg.seq_len]);
    let target = TensorData::new(data[1..].to_vec(), [batch, cfg.seq_len]);
    (Tensor::from_data(input, device), Tensor::from_data(target, device))
}

fn optimizer_step(
    mut model: Quasar,
    optim: &mut Optim,
    input: &Tensor<2, Int>,
    target: &Tensor<2, Int>,
    accum: usize,
    device: &Device,
) -> Result<(Quasar, f32)> {
    let mut logged_loss = None;
    for _ in 0..accum {
        let loss = model.loss(input.clone(), target.clone());
        let nll = loss.nll.clone().detach();
        logged_loss = Some(match logged_loss.take() {
            Some(total) => total + nll,
            None => nll,
        });
        let grads = loss.total.div_scalar(accum as f64).backward();
        optim.accumulate(&model, grads);
    }
    model = optim.step(3e-3, model);
    device.sync()?;
    let loss = logged_loss.unwrap().div_scalar(accum as f64).into_scalar::<f32>();
    anyhow::ensure!(loss.is_finite(), "training produced a non-finite loss");
    Ok((model, loss))
}
