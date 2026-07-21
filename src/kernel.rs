//! A fused RMSNorm kernel written directly in CubeCL.
//!
//! RMSNorm is the most-called operation in the model — `norm_mix`/`norm_ffn` in
//! every one of the 24 blocks, the final `norm`, plus `q_norm`/`k_norm` in the
//! attention layers (~50 calls per `tiny` forward). Expressed as tensor ops it
//! is a chain of separate kernels — `x²`, `mean`, `rsqrt`, a broadcast `mul`,
//! then `mul` by `gamma` — and, being memory-bound, each one streams the whole
//! activation through global memory again. This module folds the whole thing
//! into a single `#[cube]` kernel: one pass reads `x`, one pass writes `y`.
//!
//! Why this and not the SSD mixer (the actual dominant cost): SSD lives in the
//! pinned `burn-mamba` (~4k lines, custom backward) and rewriting it correctly
//! without a GPU to check against is not realistic. RMSNorm belongs to quasar,
//! is small enough to verify against a plain-Rust reference in CI, and is a
//! faithful, runnable instance of the lever the issue points at.
//!
//! The kernel is checked for numeric equality against that reference on a real
//! CubeCL backend (the `cubecl-cpu` LLVM-JIT runtime) in `tests/cubecl_cpu.rs`.

use burn::backend::cubecl::dtype_to_storage_type;
use burn::tensor::Shape;
use burn_cubecl::CubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, cube};

/// One cube thread per row. Each thread reads its row of `x` once to accumulate
/// the sum of squares, computes the reciprocal RMS, then writes the normalised,
/// `gamma`-scaled row — a single load and a single store of the activation
/// instead of the several passes the tensor-op chain makes.
///
/// `x` and `output` are contiguous `[n_rows, dim]`; `gamma` is `[dim]`. `eps` is
/// a runtime scalar (passed by value, like `blank` in burn's `ctc_loss_kernel`);
/// `_dtype` makes the element type `F` runtime-defined via `#[define]`.
#[cube(launch)]
pub fn rms_norm_kernel<F: Float>(
    x: &Tensor<F>,
    gamma: &Tensor<F>,
    output: &mut Tensor<F>,
    eps: f32,
    #[define(F)] _dtype: StorageType,
) {
    let row = ABSOLUTE_POS_X as usize;

    let n_rows = output.shape(0);
    let dim = output.shape(1);

    if row >= n_rows {
        terminate!();
    }

    let base = row * dim;

    let mut sum_sq = F::new(0.0);
    for i in 0..dim {
        let v = x[base + i];
        sum_sq += v * v;
    }

    let mean = sum_sq * F::recip(F::cast_from(dim));
    let inv = F::recip((mean + F::cast_from(eps)).sqrt());

    for i in 0..dim {
        output[base + i] = x[base + i] * inv * gamma[i];
    }
}

/// Launch [`rms_norm_kernel`] on a raw `CubeBackend<R>` primitive, returning a
/// fresh contiguous `[n_rows, dim]` tensor. `x` is `[n_rows, dim]`, `gamma` is
/// `[dim]`. Modelled on burn's `custom-cubecl-kernel` example: make the inputs
/// contiguous, allocate the output buffer, and launch one cube per block of rows.
pub fn rms_norm<R: CubeRuntime>(x: CubeTensor<R>, gamma: CubeTensor<R>, eps: f32) -> CubeTensor<R> {
    let dtype = x.dtype;

    let x = into_contiguous(x);
    let gamma = into_contiguous(gamma);

    let ndims = x.meta.num_dims();
    let n_rows = x.meta.shape()[0];
    let dim = x.meta.shape()[ndims - 1];

    let shape_out = Shape::from(vec![n_rows, dim]);
    let buffer = x.client.empty(shape_out.num_elements() * dtype.size());
    let output =
        CubeTensor::new_contiguous(x.client.clone(), x.device.clone(), shape_out, buffer, dtype);

    let cube_dim = CubeDim { x: 64, y: 1, z: 1 };
    let cubes_needed = f32::ceil(n_rows as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    rms_norm_kernel::launch::<R>(
        &output.client,
        cube_count,
        cube_dim,
        x.into_tensor_arg(),
        gamma.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        eps,
        dtype_to_storage_type(dtype),
    );

    output
}
