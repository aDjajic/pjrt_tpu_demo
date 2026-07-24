import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Concatenation with a single "whole" Pallas kernel: no grid and no
# BlockSpecs, so all four input matrices are DMA-ed into VMEM at once and
# the kernel copies each one into its slice of the output ref. Because of
# that all five matrices must fit in VMEM (~16 MB): 5 x 256x512xf32 = 2.5 MB.

M = 256
N = 512
NUM_INPUTS = 4
DTYPE = jnp.float32


def concat_kernel(x0_ref, x1_ref, x2_ref, x3_ref, o_ref):
    o_ref[0 * M:1 * M, :] = x0_ref[...]
    o_ref[1 * M:2 * M, :] = x1_ref[...]
    o_ref[2 * M:3 * M, :] = x2_ref[...]
    o_ref[3 * M:4 * M, :] = x3_ref[...]


def concat_whole(x0, x1, x2, x3):
    return pl.pallas_call(
        concat_kernel,
        out_shape=jax.ShapeDtypeStruct((NUM_INPUTS * M, N), DTYPE),
        debug=True,
    )(x0, x1, x2, x3)


def main():
    keys = jax.random.split(jax.random.key(0), NUM_INPUTS)
    xs = [jax.random.normal(k, (M, N), dtype=DTYPE) for k in keys]

    fn = jax.jit(concat_whole)

    print("=== StableHLO ===")
    print(fn.lower(*xs).as_text())

    out = fn(*xs).block_until_ready()
    print("=== Result shape ===")
    print(out.shape)
    print("=== Correctness check ===")
    expected = jnp.concatenate(xs, axis=0)
    print("max absolute error:", jnp.max(jnp.abs(out - expected)))


if __name__ == "__main__":
    main()
