#!/usr/bin/env python3

"""Benchmark a tiled Pallas TPU matmul kernel across several matrix sizes."""

import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_M = 1024
BLOCK_N = 1024
BLOCK_K = 2048
DTYPE = jnp.bfloat16
LOG_DIR = "./jax_profile_logs"  # xprof/TensorBoard trace output

# (N, iterations) - fewer repeats for the larger, slower sizes.
SIZES = (
    (512, 5),
    (4096, 5),
    (16384, 5),
)


def matmul_kernel(x_ref, y_ref, o_ref, acc_ref, *, k_tiles):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(pl.program_id(2) == k_tiles - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul_kernel_plain(x_ref, y_ref, o_ref):
    o_ref[...] = jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    ).astype(o_ref.dtype)


def make_matmul_plain(n, dtype=DTYPE):
    # No grid/BlockSpec: the whole (n, n) matmul runs as a single Mosaic block.
    return pl.pallas_call(
        matmul_kernel_plain,
        out_shape=jax.ShapeDtypeStruct((n, n), dtype),
    )


def make_matmul(n, block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, dtype=DTYPE):
    grid = (n // block_m, n // block_n, n // block_k)
    return pl.pallas_call(
        functools.partial(matmul_kernel, k_tiles=grid[2]),
        grid=grid,
        in_specs=[
            pl.BlockSpec((block_m, block_k), lambda i, j, k: (i, k)),
            pl.BlockSpec((block_k, block_n), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct((n, n), dtype),
        scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )


def benchmark(n, iters):
    kx, ky = jax.random.split(jax.random.PRNGKey(n))
    x = jax.random.normal(kx, (n, n), dtype=DTYPE)
    y = jax.random.normal(ky, (n, n), dtype=DTYPE)

    make_fn = make_matmul_plain if n == 512 else make_matmul
    fn = jax.jit(make_fn(n))

    out = fn(x, y)
    out.block_until_ready()  # compile + warm-up (Mosaic IR dump happens here), excluded from timing

    if n in (512, 4096):
        expected = jnp.dot(x, y, preferred_element_type=jnp.float32)
        ok = jnp.allclose(out.astype(jnp.float32), expected, atol=1e-1, rtol=1e-2)
        print(f"correctness check @ N={n}: {'OK' if ok else 'MISMATCH'}")

    jax.profiler.start_trace(LOG_DIR)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(x, y)
        out.block_until_ready()
    elapsed = time.perf_counter() - start
    jax.profiler.stop_trace()

    per_call = elapsed / iters
    return per_call


def main():
    print(f"{'N':>8} | {'time/call (ms)':>16}")
    print("-" * 42)
    for n, iters in SIZES:
        per_call = benchmark(n, iters)
        print(f"{n:>8} | {per_call * 1e3:>16.3f}")
    print(f"\nxprof traces written under {LOG_DIR}")
    print(f"  tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()
