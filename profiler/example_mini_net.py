import numpy as np
import jax
import jax.numpy as jnp

dim = 128
batch_size = 8  # M multiple of the 8-sublane MXU tile.
USE_SOFTMAX = (
    False  # Flip on to exercise softmax (expected v6e libtpu blocker).
)
log_dir = "./jax_profile_logs"
options = jax.profiler.ProfileOptions()
options.enable_hlo_proto = True
options.advanced_configuration = {"tpu_trace_mode" : "TRACE_COMPUTE_AND_SYNC"}

x = jnp.ones((batch_size, dim), dtype=jnp.float32)

w1_np = (np.random.randn(dim, dim) / np.sqrt(dim)).astype(np.float32)
w2_np = (np.random.randn(dim, dim) / np.sqrt(dim)).astype(np.float32)
b1_vec = (np.random.randn(dim) * 0.1).astype(np.float32)
b2_vec = (np.random.randn(dim) * 0.1).astype(np.float32)

b1_np = np.ascontiguousarray(np.broadcast_to(b1_vec, (batch_size, dim)))
b2_np = np.ascontiguousarray(np.broadcast_to(b2_vec, (batch_size, dim)))

@jax.jit
def mlp_forward(x):
    hidden = jax.nn.relu(jnp.matmul(x, w1_np) + b1_np)
    logits = jnp.matmul(hidden, w2_np) + b2_np
    return jax.nn.softmax(logits) if USE_SOFTMAX else logits

jax.profiler.start_trace(log_dir, profiler_options=options)
retval = mlp_forward(x)
retval.block_until_ready()
jax.profiler.stop_trace()
print(retval)
