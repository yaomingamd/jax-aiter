# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 FMHA v4 producers and the dense ASM launch, checked against aiter."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_aiter.ops.mha_v4 import (
    mha_v4_mxfp4,
    mha_v4_mxfp4_quantize_k,
    mha_v4_mxfp4_quantize_q,
    mha_v4_mxfp4_quantize_v,
)

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "aiter"))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

_LOG2E = 1.4426950408889634


def _torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("aiter")
    return torch


def _pair(batch, sequence_q, sequence_k, heads_q, heads_k, seed):
    key = jax.random.PRNGKey(seed)
    q = jax.random.normal(key, (batch, sequence_q, heads_q, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, sequence_k, heads_k, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
    return q, k, v


def _to_torch(torch, array):
    host = np.asarray(array)
    return torch.from_numpy(host.view(np.int16).astype(np.uint16)).view(torch.bfloat16).cuda()


def _zero_unwritten_k(raw, batch, sequence, heads):
    """Clear K-tile bytes the producer never stores.

    The coalesced kernel writes one 64-byte row per real token and leaves the
    rest of the final 128-token tile untouched. Those bytes are allocator
    garbage, so a full-buffer compare is only stable when the sequence fills
    the tile.
    """
    out = np.array(raw, copy=True)
    tail = sequence % 128
    if tail == 0:
        return out
    tiles = (sequence + 127) // 128
    for index in range(batch * heads):
        base = (index * tiles + tiles - 1) * 8192
        for token in range(tail, 128):
            for chunk in range(4):
                offset = base + chunk * 2048 + token * 16
                out[offset : offset + 16] = 0
    return out


@pytest.mark.parametrize(
    ("batch", "sequence", "heads"),
    [(1, 256, 4), (1, 300, 2), (2, 128, 2)],
)
def test_mxfp4_producers_match_aiter(batch, sequence, heads):
    torch = _torch()
    from aiter.ops.mha_v4_quant import (
        quantize_mxfp4_k,
        quantize_mxfp4_q,
        quantize_v_mxfp4_fp6_p,
    )

    q, k, v = _pair(batch, sequence, sequence, heads, heads, 7)
    multiplier = (128 ** -0.5) * _LOG2E
    jq, jqs = mha_v4_mxfp4_quantize_q(q, multiplier)
    jk, jks = mha_v4_mxfp4_quantize_k(k)
    jv, jvs = mha_v4_mxfp4_quantize_v(v)
    tq, tk, tv = _to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v)
    pq, pqs = quantize_mxfp4_q(tq, multiplier)
    pk, pks = quantize_mxfp4_k(tk)
    pv, pvs = quantize_v_mxfp4_fp6_p(tv)
    torch.cuda.synchronize()

    assert np.array_equal(np.asarray(jq), pq.cpu().numpy())
    assert np.array_equal(np.asarray(jqs)[: pqs.numel()], pqs.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(
        _zero_unwritten_k(np.asarray(jk), batch, sequence, heads),
        _zero_unwritten_k(pk.cpu().numpy(), batch, sequence, heads),
    )
    assert np.array_equal(np.asarray(jks)[: pks.numel()], pks.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(np.asarray(jv), pv.cpu().numpy())
    assert np.array_equal(np.asarray(jvs)[: pvs.numel()], pvs.detach().cpu().numpy().reshape(-1))


@pytest.mark.parametrize(
    ("sequence_q", "sequence_k", "heads_q", "heads_k"),
    [(256, 256, 4, 4), (256, 384, 8, 2), (129, 200, 4, 4)],
)
def test_mxfp4_dense_matches_aiter(sequence_q, sequence_k, heads_q, heads_k):
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    q, k, v = _pair(1, sequence_q, sequence_k, heads_q, heads_k, 11)
    got = np.asarray(mha_v4_mxfp4(q, k, v), dtype=np.float32)
    fmt = AttentionFormat.MXFP4
    ref = mha_v4(_to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v), fmt, fmt, fmt)
    torch.cuda.synchronize()
    ref_np = ref.float().cpu().numpy()
    cosine = float(np.dot(got.ravel(), ref_np.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref_np) + 1e-12))
    assert cosine > 0.999
    assert np.allclose(got, ref_np, rtol=2e-2, atol=2e-2)


def _pair_hd(batch, sequence_q, sequence_k, heads_q, heads_k, head_dim, seed):
    key = jax.random.PRNGKey(seed)
    q = jax.random.normal(key, (batch, sequence_q, heads_q, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(
        jax.random.fold_in(key, 1), (batch, sequence_k, heads_k, head_dim), dtype=jnp.bfloat16
    )
    v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
    return q, k, v


@pytest.mark.parametrize(
    ("batch", "sequence", "heads"),
    [(1, 256, 1), (2, 512, 2)],
)
def test_mxfp4_hd256_producers_match_aiter(batch, sequence, heads):
    torch = _torch()
    from aiter.ops.mha_v4_quant import (
        quantize_mxfp4_k_hd256,
        quantize_mxfp4_q_hd256,
        quantize_v_mxfp4_fp6_p_hd256,
    )

    q, k, v = _pair_hd(batch, sequence, sequence, heads, heads, 256, 19)
    multiplier = (256 ** -0.5) * _LOG2E
    jq, jqs = mha_v4_mxfp4_quantize_q(q, multiplier)
    jk, jks = mha_v4_mxfp4_quantize_k(k)
    jv, jvs = mha_v4_mxfp4_quantize_v(v)
    tq, tk, tv = _to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v)
    pq, pqs = quantize_mxfp4_q_hd256(tq, multiplier)
    pk, pks = quantize_mxfp4_k_hd256(tk)
    pv, pvs = quantize_v_mxfp4_fp6_p_hd256(tv)
    torch.cuda.synchronize()

    assert np.array_equal(np.asarray(jq), pq.cpu().numpy())
    assert np.array_equal(np.asarray(jqs)[: pqs.numel()], pqs.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(np.asarray(jk), pk.cpu().numpy())
    assert np.array_equal(np.asarray(jks)[: pks.numel()], pks.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(np.asarray(jv), pv.cpu().numpy())
    assert np.array_equal(np.asarray(jvs)[: pvs.numel()], pvs.detach().cpu().numpy().reshape(-1))


@pytest.mark.parametrize(
    ("batch", "sequence_q", "sequence_k", "heads_q", "heads_k"),
    [(1, 256, 256, 1, 1), (2, 512, 512, 2, 2), (1, 768, 576, 2, 2), (1, 257, 320, 16, 4)],
)
def test_mxfp4_hd256_dense_matches_aiter(batch, sequence_q, sequence_k, heads_q, heads_k):
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    q, k, v = _pair_hd(batch, sequence_q, sequence_k, heads_q, heads_k, 256, 23)
    got = np.asarray(mha_v4_mxfp4(q, k, v), dtype=np.float32)
    fmt = AttentionFormat.MXFP4
    ref = mha_v4(_to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v), fmt, fmt, fmt)
    torch.cuda.synchronize()
    ref_np = ref.float().cpu().numpy()
    cosine = float(
        np.dot(got.ravel(), ref_np.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref_np) + 1e-12)
    )
    assert got.shape[-1] == 256
    assert cosine > 0.999
    assert np.allclose(got, ref_np, rtol=2e-2, atol=2e-2)
