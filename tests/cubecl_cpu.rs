//! CI probe: does the real CubeCL CPU runtime (`cubecl-cpu`, LLVM-JIT) build
//! and run the whole model on a CPU-only runner?
//!
//! `flex`, the default backend, is pure Rust and never compiles a `#[cube]`
//! kernel, so a hand-written kernel cannot be exercised there. The CubeCL CPU
//! backend is the same runtime family as the GPU ones (`CubeBackend<R>` behind
//! `Fusion`), only targeting LLVM instead of a GPU — which makes it the one way
//! to check a custom kernel's forward *and* backward without a GPU. This file
//! confirms the backend is viable in CI before anything is built on top of it;
//! the fused-kernel equivalence checks live in `src/kernel`.
//!
//! Gated on `cpu`, so `cargo test --all-targets` (the default `flex` job)
//! compiles it to nothing and only `--features cpu` runs it.
#![cfg(feature = "cpu")]

use burn::optim::GradientsParams;
use burn::prelude::*;
use quasar::config;
use quasar::model::Quasar;

/// The floor: a single kernel dispatch through the CubeCL CPU runtime.
#[test]
fn a_tensor_op_runs_on_the_cubecl_cpu_backend() {
    let device = Device::default();
    let a = Tensor::<2>::ones([4, 4], &device);

    let sum = a.clone().matmul(a).into_data().to_vec::<f32>().unwrap();

    // Each entry of `ones([4,4]) @ ones([4,4])` is the row-column dot of four
    // ones. If the runtime miscompiled the matmul this is the first thing wrong.
    assert!(sum.iter().all(|&x| (x - 4.0).abs() < 1e-6), "{sum:?}");
}

/// The ceiling for this probe: the full toy model, burn-mamba's SSD included,
/// through a forward and a backward on the CubeCL CPU backend. If this passes,
/// a custom kernel wired into the model is verifiable here too.
#[test]
fn the_toy_model_runs_forward_and_backward_on_cubecl_cpu() {
    let device = Device::default().autodiff();
    let model = Quasar::new(&config::Model::toy(), &device);
    let tokens = Tensor::<2, Int>::zeros([1, 8], &device);

    let loss = model.loss(tokens.clone(), tokens).total;
    assert!(loss.clone().into_scalar::<f32>().is_finite(), "forward produced a non-finite loss");

    // `backward()` running to completion drives every backward kernel on this
    // runtime — burn-mamba's custom SSD backward included. Collecting the
    // per-parameter gradients and finding some is the signal that a custom
    // kernel could be checked here the same way.
    let mut grads = loss.backward();
    let params = GradientsParams::from_module(&mut grads, &model);
    assert!(!params.is_empty(), "backward produced no parameter gradients");
}
