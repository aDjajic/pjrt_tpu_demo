#!/usr/bin/env python3

import os

DUMP_ROOT = "compiler_dump/"

HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")

LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")

MOSAIC_DUMP_PATH = os.path.join(DUMP_ROOT, "mosaic")

os.makedirs(HLO_DUMP_PATH, exist_ok=True)

os.makedirs(LLO_DUMP_PATH, exist_ok=True)
os.makedirs(MOSAIC_DUMP_PATH, exist_ok=True)

# XLA flags — dump HLO + Mosaic MLIR passes
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DUMP_PATH} "
)

# libtpu flags — dump LLO
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_PATH} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_emit_annotations=true "
#    f"--xla_jf_debug_level=2 "
    f"--xla_mosaic_dump_to={MOSAIC_DUMP_PATH} "
    f"xla_jf_dump_tlo_text=true"
    f"--xla_jf_dump_tpu_isa_text=true"
)

# Import JAX AFTER setting env vars


import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import lax


def sum_kernel(n_ref, x_ref, o_ref):
    # n_ref: (1,) i32
    # x_ref: (N, D)
    # o_ref: (1, D)
    n = n_ref[0]

    def body(i, acc):
        return acc + x_ref[i]

    init = jnp.zeros((x_ref.shape[1],), dtype=x_ref.dtype)
    o_ref[0] = lax.fori_loop(0, n, body, init)


N, D = 8, 128
x = jnp.arange(N * D, dtype=jnp.float32).reshape(N, D)


def run(n_value: int):
    n = jnp.array([n_value], dtype=jnp.int32)
    result = pl.pallas_call(
        sum_kernel,
        out_shape=jax.ShapeDtypeStruct((1, D), x.dtype),
        debug=True,
    )(n, x)
    result.block_until_ready()
    return result


for n_value in (3, 5, 8):
    result = run(n_value)
    expected = x[:n_value].sum(axis=0)
    ok = jnp.allclose(result[0], expected)
    print(f"n={n_value}: result[0,:4]={result[0, :4]}, "
        f"result[0,-4:]={result[0, -4:]}, matches numpy sum: {ok}")

print(f"\nDumps written to:")

print(f"  HLO:    {HLO_DUMP_PATH}")

print(f"  Mosaic: {MOSAIC_DUMP_PATH}")

print(f"  LLO:    {LLO_DUMP_PATH}")
