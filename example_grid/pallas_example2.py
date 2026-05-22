#!/usr/bin/env python3

#import os
#
#DUMP_ROOT = "compiler_dump/"
#
#HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")
#
#LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")
#
#MOSAIC_DUMP_PATH = os.path.join(DUMP_ROOT, "mosaic")
#
#os.makedirs(HLO_DUMP_PATH, exist_ok=True)
#
#os.makedirs(LLO_DUMP_PATH, exist_ok=True)
#os.makedirs(MOSAIC_DUMP_PATH, exist_ok=True)
#
## XLA flags — dump HLO + Mosaic MLIR passes
#os.environ["XLA_FLAGS"] = (
#    f"--xla_dump_hlo_as_text "
#    f"--xla_dump_to={HLO_DUMP_PATH} "
#)
#
## libtpu flags — dump LLO
#os.environ["LIBTPU_INIT_ARGS"] = (
#    f"--xla_jf_dump_to={LLO_DUMP_PATH} "
#    f"--xla_jf_dump_hlo_text=true "
#    f"--xla_jf_dump_llo_text=true "
#    f"--xla_jf_emit_annotations=true "
#    #f"--xla_jf_debug_level=2 "
#    f"--xla_mosaic_dump_to={MOSAIC_DUMP_PATH} "
#)

# Import JAX AFTER setting env vars

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

x = jnp.arange(8 * 1024, dtype=jnp.float32).reshape(8, 1024)

y = jnp.ones((8, 1024), dtype=jnp.float32)

result = pl.pallas_call(
    add_kernel,
    out_shape=jax.ShapeDtypeStruct((8, 1024), jnp.float32),
    grid=(1, 8),
    in_specs=[
        pl.BlockSpec((8, 128), lambda i, j: (i, j)),
        pl.BlockSpec((8, 128), lambda i, j: (i, j))
    ],
    out_specs=pl.BlockSpec(
        (8, 128), lambda i, j: (i, j),
    ),
    debug=True,
)(x, y)

result.block_until_ready()

print("result[:5] =", result[:5])

print("result[-5:] =", result[-5:])

#print(f"\nDumps written to:")
#
#print(f"  HLO:    {HLO_DUMP_PATH}")
#
#print(f"  Mosaic: {MOSAIC_DUMP_PATH}")
#
#print(f"  LLO:    {LLO_DUMP_PATH}")
