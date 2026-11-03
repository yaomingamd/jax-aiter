# SPDX-License-Identifier: MIT
"""Integration tests for gfx950 A6W6 MXFP6 GEMM."""

import jax
import jax.numpy as jnp
import pytest

from jax_aiter.ops.gemm_mxfp6 import (
    gemm_mxfp6,
    mxfp6_gemm_pack_size,
    quantize_mxfp6_gemm,
)


def test_mxfp6_gemm_pack_sizes_include_abi_guard_tiles():
    # ceil(257 / 256) * (ceil(129 / 128) + 2) = 8 tiles.
    assert mxfp6_gemm_pack_size(257, 129) == (8 * 24576, 8 * 1024)


@pytest.mark.parametrize("m,n,k", [(256, 256, 128), (257, 259, 129)])
def test_mxfp6_gemm_runs_and_jit_is_stable(m, n, k):
    key_a, key_b = jax.random.split(jax.random.key(m + n + k))
    a = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16)

    packed, scales = quantize_mxfp6_gemm(a)
    expected_packed, expected_scales = mxfp6_gemm_pack_size(m, k)
    assert packed.shape == (expected_packed,)
    assert scales.shape == (expected_scales,)

    eager = gemm_mxfp6(a, b).block_until_ready()
    compiled = jax.jit(gemm_mxfp6)(a, b).block_until_ready()
    assert eager.shape == (m, n)
    assert eager.dtype == jnp.bfloat16
    assert jnp.array_equal(eager, compiled)


def test_mxfp6_gemm_rejects_invalid_operands():
    with pytest.raises(ValueError, match="bfloat16"):
        gemm_mxfp6(jnp.ones((16, 16), jnp.float32), jnp.ones((16, 16), jnp.float32))
    with pytest.raises(ValueError, match="K mismatch"):
        gemm_mxfp6(jnp.ones((16, 16), jnp.bfloat16), jnp.ones((16, 8), jnp.bfloat16))
