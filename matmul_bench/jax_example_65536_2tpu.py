#!/usr/bin/env python3

"""Benchmark a 65536x65536 jnp.dot matmul split across 2 TPU chips."""

import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

NUM_DEVICES = 2
N = 65536
ITERS = 3
DTYPE = jnp.bfloat16
LOG_DIR = "./jax_profile_logs"  # xprof/TensorBoard trace output


def local_matmul(x, y):
    # x: (N/NUM_DEVICES, N) - this device's shard, y: (N, N) - replicated
    return jnp.dot(x, y, preferred_element_type=DTYPE)


def make_x_shard():
    # device d's whole shard is filled with the constant (d + 1).
    idx = jax.lax.axis_index("dev")
    return jnp.full((N // NUM_DEVICES, N), (idx + 1).astype(DTYPE), dtype=DTYPE)


def make_y_shard():
    # Every device independently builds the same full replica locally.
    return jnp.ones((N, N), dtype=DTYPE)


def main():
    if jax.local_device_count() < NUM_DEVICES:
        raise SystemExit(
            f"Need {NUM_DEVICES} devices, "
            f"but only {jax.local_device_count()} available."
        )

    mesh = Mesh(jax.devices()[:NUM_DEVICES], axis_names=("dev",))

    make_x = jax.jit(
        jax.shard_map(
            make_x_shard, mesh=mesh, in_specs=(), out_specs=P("dev", None),
            check_vma=False,
        )
    )
    make_y = jax.jit(
        jax.shard_map(
            make_y_shard, mesh=mesh, in_specs=(), out_specs=P(None, None),
            check_vma=False,
        )
    )
    sharded_matmul = jax.jit(
        jax.shard_map(
            local_matmul, mesh=mesh,
            in_specs=(P("dev", None), P(None, None)),
            out_specs=P("dev", None),
            check_vma=False,
        )
    )

    x = make_x()
    y = make_y()

    print("=== X sharding across devices ===")
    try:
        jax.debug.visualize_array_sharding(x)  # requires the `rich` package
    except ValueError:
        print(x.sharding)

    out = sharded_matmul(x, y)
    out.block_until_ready()  # compile + warm-up, excluded from timing

    print("=== Correctness check (deterministic inputs) ===")
    print("out[0, :8]  =", out[0, :8])
    print("out[-1, :8] =", out[-1, :8])
    ok_first = bool(jnp.all(out[0] == N))
    ok_last = bool(jnp.all(out[-1] == 2 * N))
    print(f"row 0  (device 0 shard) == {N}:     {ok_first}")
    print(f"row -1 (device 1 shard) == {2 * N}: {ok_last}")

    jax.profiler.start_trace(LOG_DIR)
    start = time.perf_counter()
    for _ in range(ITERS):
        out = sharded_matmul(x, y)
        out.block_until_ready()
    elapsed = time.perf_counter() - start
    jax.profiler.stop_trace()

    per_call = elapsed / ITERS
    print(f"{'N':>8} | {'time/call (ms)':>16}")
    print("-" * 42)
    print(f"{N:>8} | {per_call * 1e3:>16.3f}")
    print(f"\nxprof traces written under {LOG_DIR}")
    print(f"  tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()
