# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path

import pytest


def get_gpu_arch():
    """Detect GPU architecture string (e.g., 'gfx942', 'gfx950')."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "gfx9" in line and "Name:" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return os.environ.get("GPU_ARCHS", "gfx950").split(";")[0]


_gpu_arch = None

def gpu_arch():
    global _gpu_arch
    if _gpu_arch is None:
        _gpu_arch = get_gpu_arch()
    return _gpu_arch


def _is_gfx942():
    return gpu_arch() == "gfx942"

def _is_gfx950():
    return gpu_arch() == "gfx950"

requires_gfx942 = pytest.mark.skipif(
    not _is_gfx942(),
    reason="Requires gfx942 (MI300) GPU"
)

requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(),
    reason="Requires gfx950 (MI350) GPU"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dev_build_ready(root: Path) -> bool:
    """True when a local `make` tree exists (editable / source checkout)."""
    return (root / "build" / "jax_aiter_build" / "libjax_aiter.so").is_file()


def _installed_wheel_ready() -> bool:
    """True when jax-aiter was pip-installed with bundled _lib/_hsa."""
    import sysconfig

    site = Path(sysconfig.get_paths()["purelib"])
    lib = site / "jax_aiter" / "_lib" / "jax_aiter_build" / "libjax_aiter.so"
    if lib.is_file():
        return True
    # Fallback if layout differs (editable / custom prefix).
    try:
        import jax_aiter
    except ImportError:
        return False
    pkg = Path(jax_aiter.__file__).resolve().parent
    return (pkg / "_lib" / "jax_aiter_build" / "libjax_aiter.so").is_file()


def _prefer_wheel_mode(root: Path) -> bool:
    """Use wheel/site-packages layout instead of JA_ROOT_DIR dev paths."""
    if os.environ.get("JA_FORCE_DEV") == "1":
        return False
    if os.environ.get("JA_USE_WHEEL") == "1":
        return True
    ja_root = os.environ.get("JA_ROOT_DIR", "")
    if ja_root and _dev_build_ready(Path(ja_root)):
        return False
    if _dev_build_ready(root):
        return False
    return _installed_wheel_ready()


def _strip_repo_package_shadow(root: Path) -> None:
    """Repo checkout on PYTHONPATH shadows the installed wheel (slim Docker)."""
    import sys

    root_s = str(root)
    sys.path[:] = [p for p in sys.path if p not in ("", root_s)]


def _overlay_wheel_runtime_patches(root: Path) -> None:
    """Patch site-packages wheel with repo runtime fixes (slim image hotfix)."""
    import shutil
    import sysconfig

    site_pkg = Path(sysconfig.get_paths()["purelib"]) / "jax_aiter"
    if not site_pkg.is_dir():
        return

    for rel in ("ja_compat/config.py",):
        src = root / "jax_aiter" / rel
        dst = site_pkg / rel
        if src.is_file() and dst.parent.is_dir():
            shutil.copy2(src, dst)


def pytest_configure(config):
    root = _repo_root()
    os.environ.setdefault("AITER_SYMBOL_VISIBLE", "1")
    os.environ.setdefault("GPU_ARCHS", "gfx950")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    if _prefer_wheel_mode(root):
        # Drop stale docker/dev env (e.g. JA_ROOT_DIR=/opt/jax-aiter on slim images).
        os.environ.pop("JA_ROOT_DIR", None)
        os.environ.pop("AITER_ASM_DIR", None)
        _strip_repo_package_shadow(root)
        _overlay_wheel_runtime_patches(root)

    # Full-suite GPU tests OOM/abort on 8-GPU nodes when every device accumulates
    # XLA allocations. Default to one visible device unless explicitly overridden.
    if (
        "HIP_VISIBLE_DEVICES" not in os.environ
        and os.environ.get("JA_TEST_ALL_GPUS") != "1"
        and not os.environ.get("PYTEST_XDIST_WORKER")
    ):
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
    else:
        os.environ.setdefault("JA_ROOT_DIR", str(root))
        os.environ.setdefault(
            "AITER_ASM_DIR", str(root / "third_party" / "aiter" / "hsa") + "/"
        )

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id:
        gpu_idx = int(worker_id.replace("gw", ""))
        visible = os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        os.environ["HIP_VISIBLE_DEVICES"] = visible[gpu_idx % len(visible)]


@pytest.fixture(autouse=True)
def _release_jax_memory_after_test():
    """Long GPU test runs can OOM/abort later tests without cache cleanup."""
    yield
    if os.environ.get("JA_SKIP_JAX_GC") == "1":
        return
    import gc

    try:
        import jax
        import jax.numpy as jnp

        gc.collect()
        jax.clear_caches()
        # Drop cached device allocations before the next test.
        try:
            buf = jnp.zeros(1)
            buf.block_until_ready()
            del buf
        except Exception:
            pass
        gc.collect()
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):
    total = int(os.environ.get("PYTEST_SHARD_TOTAL", "1"))
    idx = int(os.environ.get("PYTEST_SHARD_INDEX", "0"))
    if total > 1:
        items[:] = [item for i, item in enumerate(items) if i % total == idx]

    max_tests = int(os.environ.get("PYTEST_MAX_TESTS", "0"))
    if max_tests > 0:
        items[:] = items[:max_tests]
