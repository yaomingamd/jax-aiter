#!/usr/bin/env python3
"""Tune gfx950 A6W6 MXFP6 GEMM kernels for one or more ``M,N,K`` shapes.

Example::

    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa \
      python benchmarks/tune_gemm_mxfp6.py --shape 8192,3072,9216 \
      --output /tmp/mxfp6_gemm_tuned.csv

Set ``JA_MXFP6_GEMM_CONFIG`` to the resulting CSV when launching JAX.  The
tuner only admits manifest candidates whose exact-shape/swizzle contract
covers the padded launch, and retains a candidate only if it is bit-identical
to AITER's safe dmabig reference.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

# Permit ``python benchmarks/tune_gemm_mxfp6.py`` from a built checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "build" / "jax_aiter_build").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp

from jax_aiter.ops.gemm_mxfp6 import gemm_mxfp6_packed, quantize_mxfp6_gemm

_DEFAULT = "f6gemm_dmabig_kernel_func"
_MANIFEST = "f6gemm_bf16_per1x32Fp6.csv"


def _ceil(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _parse_shape(text: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected M,N,K") from error
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected positive M,N,K")
    return shape


def _manifest_path() -> Path:
    asm_root = os.environ.get("AITER_ASM_DIR", "").strip()
    if not asm_root:
        raise RuntimeError("set AITER_ASM_DIR to the gfx950 hsa directory")
    root = Path(asm_root)
    for directory in (root, root / "gfx950"):
        path = directory / "f6gemm" / _MANIFEST
        if path.is_file():
            return path
    raise FileNotFoundError(f"MXFP6 manifest not found below: {root}")


def _candidates(m: int, n: int, k: int) -> list[str]:
    padded = (_ceil(m, 256), _ceil(n, 256), _ceil(k, 128))
    valid: list[str] = []
    with _manifest_path().open(newline="") as file:
        for row in csv.DictReader(file):
            if int(row["splitK"]) != 0 or row["pack_layout"] != "mxfp6_c0c1_256_padk2":
                continue
            exact = tuple(int(row[key]) for key in ("exact_M", "exact_N", "exact_K"))
            if exact != (0, 0, 0):
                if exact != padded:
                    continue
            else:
                limits = tuple(
                    int(row[key]) for key in ("swizzle_max_M", "swizzle_max_N", "swizzle_max_K")
                )
                # Zero max-K means that this kernel has no shape restriction.
                if limits[2] > 0 and any(value > limit for value, limit in zip(padded, limits)):
                    continue
            valid.append(row["knl_name"])
    return list(dict.fromkeys(valid))


def _median_us(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn().block_until_ready()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def tune_shape(m: int, n: int, k: int, *, warmup: int, repeats: int, seed: int) -> tuple[str, float]:
    key_a, key_b = jax.random.split(jax.random.key(seed))
    a = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16)
    ap, ascale = quantize_mxfp6_gemm(a)
    bp, bscale = quantize_mxfp6_gemm(b)
    ap.block_until_ready()
    bp.block_until_ready()

    calls = {}

    def call(kernel: str):
        if kernel not in calls:
            calls[kernel] = jax.jit(
                lambda kernel=kernel: gemm_mxfp6_packed(
                    ap, bp, ascale, bscale, m=m, n=n, k=k, kernel_name=kernel
                )
            )
        return calls[kernel]()

    reference = call(_DEFAULT)
    reference.block_until_ready()
    best_kernel, best_us = _DEFAULT, _median_us(lambda: call(_DEFAULT), repeats)
    for kernel in _candidates(m, n, k):
        for _ in range(warmup):
            call(kernel).block_until_ready()
        output = call(kernel)
        if not bool(jnp.array_equal(reference, output)):
            print(f"skip {kernel}: output differs from {_DEFAULT}")
            continue
        elapsed = _median_us(lambda: call(kernel), repeats)
        print(f"{m:6d},{n:6d},{k:6d}  {kernel:<75} {elapsed:9.2f} us")
        if elapsed < best_us:
            best_kernel, best_us = kernel, elapsed
    return best_kernel, best_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_parse_shape, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")

    rows = []
    for index, (m, n, k) in enumerate(args.shape):
        kernel, elapsed = tune_shape(
            m, n, k, warmup=args.warmup, repeats=args.repeats, seed=args.seed + index
        )
        print(f"selected {m},{n},{k}: {kernel} ({elapsed:.2f} us)")
        rows.append({"M": m, "N": n, "K": k, "kernel_name": kernel})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("M", "N", "K", "kernel_name"))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
