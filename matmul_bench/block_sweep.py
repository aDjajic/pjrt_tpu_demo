#!/usr/bin/env python3

"""Sweep Pallas matmul block sizes to find the fastest one that fits VMEM.

Usage: python3 block_sweep.py [N]   (defaults to 4096)

pallas_example.py couples M/N/K into a single BLOCK cubed tile, so its VMEM
cost is dominated by the fp32 accumulator scratch (BLOCK_M * BLOCK_N * 4
bytes) plus double-buffered bf16 x/y blocks (~2 * 2 * BLOCK * BLOCK * 2
bytes). The accumulator scales with the output tile (M, N), not with K, so
this script decouples BLOCK_M/BLOCK_N from BLOCK_K and times every
combination that actually compiles, to find the fastest one that still fits.
"""

import functools
import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DTYPE = jnp.bfloat16
ITERS = 5


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


def make_matmul(n, block_m, block_n, block_k, dtype=DTYPE):
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


def time_config(n, block_m, block_n, block_k):
    kx, ky = jax.random.split(jax.random.PRNGKey(n))
    x = jax.random.normal(kx, (n, n), dtype=DTYPE)
    y = jax.random.normal(ky, (n, n), dtype=DTYPE)

    fn = jax.jit(make_matmul(n, block_m, block_n, block_k))
    out = fn(x, y)
    out.block_until_ready()  # compile + warm-up, excluded from timing

    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn(x, y)
        out.block_until_ready()
    elapsed = time.perf_counter() - start
    return elapsed / ITERS


def pow2_divisors_up_to(n, cap):
    block, out = 128, []
    while block <= cap and n % block == 0:
        out.append(block)
        block *= 2
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    candidates = pow2_divisors_up_to(n, n)

    configs = [(bmn, bmn, bk) for bmn in candidates for bk in candidates]

    print(f"N = {n}, trying {len(configs)} (BLOCK_M=BLOCK_N, BLOCK_K) combos\n")
    print(f"{'BLOCK_M=N':>10} | {'BLOCK_K':>8} | {'time/call (ms)':>16}")
    print("-" * 42)

    results = []
    for bm, bn, bk in configs:
        try:
            per_call = time_config(n, bm, bn, bk)
        except Exception as e:  # VMEM overflow / compile failure for this tile
            print(f"{bm:>10} | {bk:>8} | FAILED ({type(e).__name__})")
            continue
        results.append((per_call, bm, bk))
        print(f"{bm:>10} | {bk:>8} | {per_call * 1e3:>16.3f}")

    if results:
        results.sort()
        best_time, best_bmn, best_bk = results[0]
        print(
            f"\nBest: BLOCK_M=BLOCK_N={best_bmn}, BLOCK_K={best_bk} "
            f"-> {best_time * 1e3:.3f} ms/call"
        )


if __name__ == "__main__":
    main()
