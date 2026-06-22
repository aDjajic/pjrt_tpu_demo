import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

M = K = N = 256
DTYPE = jnp.float32


def matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]


def make_matmul():
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), DTYPE),
        debug=True,
    )

NATIVE_DIM = 1024

def native_matmul(x, y):
    z = x
    for _ in range(8):
        z = jnp.dot(z, y, preferred_element_type=DTYPE)
        z = jnp.tanh(z)
    return z


def main():
    x = jnp.arange(M * K, dtype=DTYPE).reshape(M, K)
    y = jnp.ones((K, N), dtype=DTYPE)

    fn = make_matmul()

    print("=== Mosaic ===")
    lowered = jax.jit(fn).lower(x, y)
    print("=== StableHLO ===")
    print(lowered.as_text())

    compiled = jax.jit(fn)

    log_dir = "./jax_profile_logs"
    jax.profiler.start_trace(log_dir)
    out = None
    for _ in range(100):
        out = compiled(x, y)
    out.block_until_ready()
    jax.profiler.stop_trace()

    print("=== Result (the first 8 elements) ===")
    print(out[0, :8])

    nx = jnp.ones((NATIVE_DIM, NATIVE_DIM), dtype=DTYPE)
    nw = jnp.ones((NATIVE_DIM, NATIVE_DIM), dtype=DTYPE) * 0.001
    native = jax.jit(native_matmul)
    print("=== StableHLO ===")
    print(lowered.as_text())

    jax.profiler.start_trace(log_dir)
    nout = None
    for _ in range(100):
        nout = native(nx, nw)
    nout.block_until_ready()
    jax.profiler.stop_trace()
    print("=== Native result (the first 8 elements) ===")
    print(nout[0, :8])


if __name__ == "__main__":
    main()
