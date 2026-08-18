import logging
import os
import sysconfig
from pathlib import Path
from typing import Optional

try:
    from importlib.resources import files as pkg_files
except ImportError:
    from importlib import resources

    pkg_files = resources.files

logger = logging.getLogger("JAX_AITER")


def _this_package_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _site_packages_jax_aiter() -> Optional[Path]:
    site_pkg = Path(sysconfig.get_paths()["purelib"]) / "jax_aiter"
    return site_pkg if site_pkg.is_dir() else None


def _wheel_lib_root() -> Optional[Path]:
    for pkg in (_this_package_dir(), _site_packages_jax_aiter() or Path()):
        if not pkg.is_dir():
            continue
        lib = pkg / "_lib"
        if (lib / "jax_aiter_build" / "libjax_aiter.so").is_file():
            return lib
    return None


def _wheel_hsa_root() -> Optional[Path]:
    for pkg in (_this_package_dir(), _site_packages_jax_aiter() or Path()):
        if not pkg.is_dir():
            continue
        hsa = pkg / "_hsa"
        if hsa.is_dir():
            return hsa
    return None


def _dev_lib_root(ja_root: str) -> Optional[Path]:
    build = Path(ja_root).resolve() / "build"
    if (build / "jax_aiter_build" / "libjax_aiter.so").is_file():
        return build
    return None


def _dev_hsa_root(ja_root: str) -> Optional[Path]:
    hsa = Path(ja_root).resolve() / "third_party" / "aiter" / "hsa"
    return hsa if hsa.is_dir() else None


def get_packaged_lib_dir():
    """Returns a Traversable to jax_aiter/_lib inside the installed package."""
    return pkg_files("jax_aiter")


def get_package_install_dir() -> Path:
    """Filesystem path to jax_aiter with bundled wheel artifacts (_lib/_hsa)."""
    local = _this_package_dir()
    if (local / "_lib" / "jax_aiter_build" / "libjax_aiter.so").is_file():
        return local
    wheel_pkg = _site_packages_jax_aiter()
    if wheel_pkg and (wheel_pkg / "_lib" / "jax_aiter_build" / "libjax_aiter.so").is_file():
        return wheel_pkg
    return local


def get_lib_root() -> Path:
    sanitize_runtime_env()
    ja_root = os.environ.get("JA_ROOT_DIR")
    if ja_root:
        dev = _dev_lib_root(ja_root)
        if dev is not None:
            return dev
        logger.warning(
            "JA_ROOT_DIR=%s has no build/jax_aiter_build/libjax_aiter.so; "
            "using installed wheel _lib",
            ja_root,
        )

    wheel = _wheel_lib_root()
    if wheel is not None:
        return wheel

    packaged = get_packaged_lib_dir() / "_lib"
    if packaged is not None:
        return Path(str(packaged))

    raise FileNotFoundError(
        "Can't find JAX Aiter library. Set JA_ROOT_DIR to a built tree or "
        "install the jax-aiter wheel."
    )


def get_umbrella_lib() -> Path:
    """Get the path to the umbrella shared library."""
    return get_lib_root() / "jax_aiter_build" / "libjax_aiter.so"


_env_sanitized = False


def sanitize_runtime_env() -> None:
    """Drop stale JA_ROOT_DIR/AITER_ASM_DIR (e.g. slim Docker /opt/jax-aiter)."""
    global _env_sanitized
    if _env_sanitized:
        return
    _env_sanitized = True

    ja_root = os.environ.get("JA_ROOT_DIR")
    if ja_root and _dev_lib_root(ja_root) is None:
        logger.warning(
            "JA_ROOT_DIR=%s has no build tree; ignoring for wheel runtime",
            ja_root,
        )
        os.environ.pop("JA_ROOT_DIR", None)

    asm_dir = os.environ.get("AITER_ASM_DIR")
    if asm_dir and not Path(asm_dir).is_dir():
        logger.warning(
            "AITER_ASM_DIR=%s missing; ignoring for wheel runtime",
            asm_dir,
        )
        os.environ.pop("AITER_ASM_DIR", None)


def set_aiter_asm_dir():
    """
    Set AITER_ASM_DIR environment variable to the base HSA directory.
    AITER_ASM_DIR should point to the base hsa/ directory (no arch suffix).
    AITER's get_asm_dir() function will append the architecture dynamically.

    The path is set to:
    - For development mode: <repo_root>/third_party/aiter/hsa/
    - For installed packages: <package_location>/jax_aiter/_hsa/
    """
    sanitize_runtime_env()
    existing = os.environ.get("AITER_ASM_DIR")
    if existing:
        if Path(existing).is_dir():
            logger.info("AITER_ASM_DIR already set to: %s", existing)
            return
        logger.warning(
            "AITER_ASM_DIR=%s missing; falling back to installed wheel _hsa",
            existing,
        )
        os.environ.pop("AITER_ASM_DIR", None)

    try:
        from .chip_info import get_gfx

        gfx_arch = get_gfx()

        ja_root = os.environ.get("JA_ROOT_DIR")
        hsa_base = _dev_hsa_root(ja_root) if ja_root else None
        if hsa_base is None:
            hsa_base = _wheel_hsa_root()

        if hsa_base is None:
            logger.warning(
                "HSA base directory not found. Assembly kernels will not be available."
            )
            return

        arch_dir = hsa_base / gfx_arch
        if not arch_dir.exists():
            logger.warning(
                "HSA arch directory not found: %s. "
                "Assembly kernels may not be available for %s.",
                arch_dir,
                gfx_arch,
            )

        os.environ["AITER_ASM_DIR"] = str(hsa_base) + "/"
        logger.info("Set AITER_ASM_DIR to: %s", os.environ["AITER_ASM_DIR"])
    except Exception as e:
        logger.warning("Failed to set AITER_ASM_DIR: %s", e)
