//! Runs the hand-written CubeCL kernel on a real CubeCL backend without a GPU.
//!
//! `flex`, the default backend, is pure Rust and never compiles a `#[cube]`
//! kernel, so a hand-written kernel cannot be exercised there. The CubeCL CPU
//! runtime (`cubecl-cpu`, LLVM-JIT) is the same runtime family as the GPU
//! backends (`CubeBackend<R>`), only targeting LLVM — which makes it the one way
//! to compile *and run* a custom kernel on a CPU-only runner. This file:
//!
//!   1. confirms a single kernel dispatch works on the backend at all, then
//!   2. checks [`quasar::kernel::rms_norm`] for numeric equality against a
//!      plain-Rust RMSNorm reference.
//!
//! The kernel is launched directly on a raw `CubeBackend<CpuRuntime>` primitive
//! (bypassing the `Fusion`/dispatch layers, which would route to a default op
//! body rather than our `#[cube]` kernel). A model-level probe was tried and
//! deliberately dropped: `cubecl-cpu` aborts with SIGFPE while JITing
//! burn-mamba's full SSD backward — a maturity bug in the CPU runtime, unrelated
//! to this kernel, that would take down the whole test binary. Verifying a
//! small standalone kernel is what fits both the runtime's maturity and the CI
//! time budget; see `docs/KERNELS.md`.
//!
//! Gated on `cpu`, so `cargo test --all-targets` (the default `flex` job)
//! compiles it to nothing and only `--features cpu` runs it.
#![cfg(feature = "cpu")]

use burn::backend::ops::FloatTensorOps;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_cubecl::CubeBackend;
use burn_cubecl::ops::into_data_sync;
use cubecl::cpu::{CpuDevice, CpuRuntime};

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

/// RMSNorm on plain tensors: `y[r,i] = x[r,i] / sqrt(mean(x[r]²) + eps) * gamma[i]`.
/// The reference the kernel is checked against.
fn rms_norm_reference(x: &[f32], gamma: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let mut sum_sq = 0.0f32;
        for i in 0..dim {
            let v = x[r * dim + i];
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = x[r * dim + i] * inv * gamma[i];
        }
    }
    out
}

/// The custom `#[cube]` kernel must match that reference bit-for-(near-)bit on a
/// real CubeCL backend. Inputs are uploaded as raw `CubeBackend<CpuRuntime>`
/// primitives, the kernel is launched directly on them, and the result is read
/// back and compared — forward correctness of the fused kernel, checked without
/// a GPU and in well under the CI time budget.
#[test]
fn rms_norm_kernel_matches_reference_on_cubecl_cpu() {
    type Raw = CubeBackend<CpuRuntime>;
    let device = CpuDevice;

    let rows = 4usize;
    let dim = 10usize;
    let eps = 1e-5f32;

    // Deterministic pseudo-data spanning positive and negative values, so a
    // dropped `gamma` scale or a sign bug would show.
    let x: Vec<f32> = (0..rows * dim).map(|k| (k % 7) as f32 - 3.0).collect();
    let gamma: Vec<f32> = (0..dim).map(|i| 0.5 + 0.1 * i as f32).collect();

    let x_t = <Raw as FloatTensorOps<Raw>>::float_from_data(
        TensorData::new(x.clone(), [rows, dim]),
        &device,
    );
    let gamma_t = <Raw as FloatTensorOps<Raw>>::float_from_data(
        TensorData::new(gamma.clone(), [dim]),
        &device,
    );

    let out = quasar::kernel::rms_norm::<CpuRuntime>(x_t, gamma_t, eps);
    let got = into_data_sync::<CpuRuntime>(out).to_vec::<f32>().unwrap();

    let expected = rms_norm_reference(&x, &gamma, rows, dim, eps);

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "index {i}: kernel {a} vs reference {b}");
    }
}
