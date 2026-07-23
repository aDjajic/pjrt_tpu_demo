import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

NUM_DEVICES = 4

M = 1024
K = 1024
N = 1024
DTYPE = jnp.float32

# Block size the Pallas grid processes in one iteration (per device)
BLOCK_M = 128
BLOCK_K = 256
BLOCK_N = 256


def matmul_kernel(x_ref, y_ref, o_ref):
    # The grid is (m, n, k); partial products are accumulated along k.
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += x_ref[...] @ y_ref[...]


def blocked_matmul(x, y):
    # x: (M/4, K) — the shard this device received, y: (K, N) — replicated
    m, _ = x.shape
    _, n = y.shape
    grid = (m // BLOCK_M, n // BLOCK_N, K // BLOCK_K)
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), DTYPE),
        in_specs=[
            pl.BlockSpec((BLOCK_M, BLOCK_K), lambda i, j, k: (i, k)),
            pl.BlockSpec((BLOCK_K, BLOCK_N), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j, k: (i, j)),
        grid=grid,
        debug=True
    )(x, y)


def main():
    if jax.local_device_count() < NUM_DEVICES:
        raise SystemExit(
            f"Need {NUM_DEVICES} devices, "
            f"but only {jax.local_device_count()} available."
        )

    mesh = Mesh(jax.devices()[:NUM_DEVICES], axis_names=("dev",))

    # X is sharded by rows across 4 devices, Y is replicated to all of them.
    sharded_matmul = jax.jit(
        jax.shard_map(
            blocked_matmul,
            mesh=mesh,
            in_specs=(P("dev", None), P(None, None)),
            out_specs=P("dev", None),
            check_vma=False,
        )
    )

    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.device_put(
        jax.random.normal(key_x, (M, K), dtype=DTYPE),
        NamedSharding(mesh, P("dev", None)),
    )
    y = jax.device_put(
        jax.random.normal(key_y, (K, N), dtype=DTYPE),
        NamedSharding(mesh, P(None, None)),
    )

    print("=== X sharding across devices ===")
    try:
        jax.debug.visualize_array_sharding(x)  # requires the `rich` package
    except ValueError:
        print(x.sharding)

    print("=== StableHLO ===")
    lowered = sharded_matmul.lower(x, y)
    print(lowered.as_text())

    # Warmup — compile outside the trace so the profile has no compile time.
    sharded_matmul(x, y).block_until_ready()

    log_dir = "./jax_profile_logs"
    jax.profiler.start_trace(log_dir)
    out = None
    for _ in range(100):
        out = sharded_matmul(x, y)
    out.block_until_ready()
    jax.profiler.stop_trace()

    print("=== Result (first 8 elements) ===")
    print(out[0, :8])

    expected = jnp.dot(x, y, preferred_element_type=DTYPE)
    print("=== Correctness check ===")
    print("max absolute error:", jnp.max(jnp.abs(out - expected)))


if __name__ == "__main__":
    main()
