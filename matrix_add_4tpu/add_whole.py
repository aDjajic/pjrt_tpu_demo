import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Matrix addition with a single "whole" Pallas kernel: no grid and no
# BlockSpecs, so the entire matrices are DMA-ed from HBM into VMEM at once
# and the kernel adds them in a single invocation. Because of that all
# three matrices must fit in VMEM (~16 MB): 3 x 512x512xf32 = 3 MB.

M = 512
N = 512
DTYPE = jnp.float32


def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def add_whole(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), DTYPE),
        debug=True,
    )(x, y)


def main():
    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.random.normal(key_x, (M, N), dtype=DTYPE)
    y = jax.random.normal(key_y, (M, N), dtype=DTYPE)

    fn = jax.jit(add_whole)

    print("=== StableHLO ===")
    print(fn.lower(x, y).as_text())

    # Warmup — compile outside the trace so the profile has no compile time.
    fn(x, y).block_until_ready()

    log_dir = "./jax_profile_logs"
    jax.profiler.start_trace(log_dir)
    out = None
    for _ in range(100):
        out = fn(x, y)
    out.block_until_ready()
    jax.profiler.stop_trace()
    
    print("=== Result (first 8 elements) ===")
    print(out[0, :8])
    print("=== Correctness check ===")
    print("max absolute error:", jnp.max(jnp.abs(out - (x + y))))


if __name__ == "__main__":
    main()
