#!/usr/bin/env python3

"""Benchmark plain jnp.dot (native XLA matmul) across several matrix sizes."""

import time

import jax
import jax.numpy as jnp

DTYPE = jnp.bfloat16
LOG_DIR = "./jax_profile_logs"  # xprof/TensorBoard trace output

# (N, iterations) - fewer repeats for the larger, slower sizes.
SIZES = (
    (512, 5),
    (4096, 5),
    (16384, 5),
)


@jax.jit
def matmul(x, y):
    return jnp.dot(x, y, preferred_element_type=jnp.float32).astype(DTYPE)


def benchmark(n, iters):
    kx, ky = jax.random.split(jax.random.PRNGKey(n))
    x = jax.random.normal(kx, (n, n), dtype=DTYPE)
    y = jax.random.normal(ky, (n, n), dtype=DTYPE)

    out = matmul(x, y)
    out.block_until_ready()  # compile + warm-up, excluded from timing

    jax.profiler.start_trace(LOG_DIR)
    start = time.perf_counter()
    for _ in range(iters):
        out = matmul(x, y)
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


if __name__ == "__main__":
    main()
