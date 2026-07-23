import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

NUM_DEVICES = 4

M = 1024
N = 1024
DTYPE = jnp.float32

BLOCK_M = 128
BLOCK_N = 256


def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def blocked_add(x, y):
    # x, y: (M/4, N) — the shard this device received
    m, n = x.shape
    grid = (m // BLOCK_M, n // BLOCK_N)
    block_spec = pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j))
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), DTYPE),
        in_specs=[block_spec, block_spec],
        out_specs=block_spec,
        grid=grid,
        debug=True,
    )(x, y)


def main():
    if jax.local_device_count() < NUM_DEVICES:
        raise SystemExit(
            f"Need {NUM_DEVICES} devices, "
            f"but only {jax.local_device_count()} available."
        )

    mesh = Mesh(jax.devices()[:NUM_DEVICES], axis_names=("dev",))
    row_sharding = NamedSharding(mesh, P("dev", None))

    sharded_add = jax.jit(
        jax.shard_map(
            blocked_add,
            mesh=mesh,
            in_specs=(P("dev", None), P("dev", None)),
            out_specs=P("dev", None),
            check_vma=False,
        )
    )

    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.device_put(jax.random.normal(key_x, (M, N), dtype=DTYPE), row_sharding)
    y = jax.device_put(jax.random.normal(key_y, (M, N), dtype=DTYPE), row_sharding)

    print("=== X sharding across devices ===")
    try:
        jax.debug.visualize_array_sharding(x)  # requires the `rich` package
    except ValueError:
        print(x.sharding)

    print("=== StableHLO ===")
    print(sharded_add.lower(x, y).as_text())

    # Warmup — compile outside the trace so the profile has no compile time.
    sharded_add(x, y).block_until_ready()

    log_dir = "./jax_profile_logs"
    jax.profiler.start_trace(log_dir)
    out = None
    for _ in range(100):
        out = sharded_add(x, y)
    out.block_until_ready()
    jax.profiler.stop_trace()

    print("=== Result (first 8 elements) ===")
    print(out[0, :8])
    print("=== Correctness check ===")
    print("max absolute error:", jnp.max(jnp.abs(out - (x + y))))


if __name__ == "__main__":
    main()
