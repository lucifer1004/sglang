import ctypes
import glob
import importlib.util
import logging
import os
import shutil
import sysconfig
from pathlib import Path
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _get_compute_capability():
    """Get the compute capability of the current GPU."""
    if not torch.cuda.is_available():
        return None

    # Get the current device
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)

    # Return as integer (major * 10 + minor)
    return properties.major * 10 + properties.minor


def _filter_compiled_extensions(file_list):
    """Filter and prioritize compiled extensions over Python source files."""
    compiled_extensions = [".so", ".pyd", ".dll"]  # Common compiled extension suffixes
    compiled_files = []
    other_files = []

    for file_path in file_list:
        path = Path(file_path)
        # Check if it's a compiled extension (including complex names like .abi3.so, .cpython-312.so)
        if any(
            str(path).endswith(ext) or ext in str(path) for ext in compiled_extensions
        ):
            compiled_files.append(file_path)
        else:
            other_files.append(file_path)

    # Return compiled files first, then others
    return compiled_files + other_files


def _candidate_package_dirs(package_dir: Path) -> List[Path]:
    """Return package directories that may contain compiled extension modules.

    Editable installs import Python sources from the checkout, while compiled
    extensions are installed into the environment's site-packages directory.
    """
    candidate_dirs = [package_dir]
    sysconfig_paths = sysconfig.get_paths()

    for key in ("purelib", "platlib"):
        base_path = sysconfig_paths.get(key)
        if not base_path:
            continue

        candidate_dir = Path(base_path) / package_dir.name
        if candidate_dir not in candidate_dirs:
            candidate_dirs.append(candidate_dir)

    return candidate_dirs


def _find_compiled_module(
    package_dirs: List[Path], relative_pattern: str
) -> Tuple[Optional[Path], List[Tuple[str, List[str]]]]:
    attempts: List[Tuple[str, List[str]]] = []

    for package_dir in package_dirs:
        pattern = str(package_dir / relative_pattern)
        raw_matching_files = glob.glob(pattern)
        matching_files = _filter_compiled_extensions(raw_matching_files)
        attempts.append((pattern, matching_files))

        logger.debug(f"[sgl_kernel] Looking for library matching pattern: {pattern}")
        logger.debug(f"[sgl_kernel] Found files: {raw_matching_files}")
        logger.debug(f"[sgl_kernel] Prioritized files: {matching_files}")

        if matching_files:
            return Path(matching_files[0]), attempts

    return None, attempts


def _load_common_ops_from_path(ops_path: Path):
    spec = importlib.util.spec_from_file_location("common_ops", str(ops_path))
    if spec is None:
        raise ImportError(f"Could not create module spec for {ops_path}")

    common_ops = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {ops_path}")

    spec.loader.exec_module(common_ops)
    return common_ops


def _format_attempts(attempts: List[Tuple[str, List[str]]]) -> str:
    return "\n".join(
        f"- {pattern} - found files: {matching_files}"
        for pattern, matching_files in attempts
    )


def _load_architecture_specific_ops():
    """Load the appropriate common_ops library based on GPU architecture."""
    compute_capability = _get_compute_capability()
    logger.debug(
        f"[sgl_kernel] GPU Detection: compute_capability = {compute_capability}"
    )

    # Get the directory where sgl_kernel is installed
    sgl_kernel_dir = Path(__file__).parent
    logger.debug(f"[sgl_kernel] sgl_kernel directory: {sgl_kernel_dir}")
    sgl_kernel_dirs = _candidate_package_dirs(sgl_kernel_dir)
    logger.debug(f"[sgl_kernel] candidate sgl_kernel directories: {sgl_kernel_dirs}")

    # Determine which version to load based on GPU architecture
    if compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
    else:
        ops_subdir = "sm100"
        variant_name = "CPU/No GPU detected (using precise math)"

    logger.debug(f"[sgl_kernel] Attempting to load {variant_name}")
    ops_path, ops_attempts = _find_compiled_module(
        sgl_kernel_dirs, f"{ops_subdir}/common_ops.*"
    )

    previous_import_errors: List[Exception] = []

    # Try to load from the architecture-specific directory
    if ops_path is not None:
        logger.debug(f"[sgl_kernel] Found architecture-specific library: {ops_path}")
        try:
            logger.debug(f"[sgl_kernel] Loading module from {ops_path}...")
            common_ops = _load_common_ops_from_path(ops_path)
            logger.debug(f"[sgl_kernel] ✓ Successfully loaded {variant_name}")
            logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
            return common_ops

        except Exception as e:
            previous_import_errors.append(e)
            logger.debug(
                f"[sgl_kernel] ✗ Failed to load from {ops_path}: {type(e).__name__}: {e}"
            )
            # Continue to fallback
    else:
        logger.debug(
            "[sgl_kernel] ✗ Architecture-specific library not found in any "
            f"candidate directory: {ops_attempts}"
        )

    # Try alternative directory (in case installation structure differs)
    logger.debug("[sgl_kernel] Attempting fallback: looking for common_ops.*")
    alt_path, alt_attempts = _find_compiled_module(sgl_kernel_dirs, "common_ops.*")

    if alt_path is not None:
        logger.debug(f"[sgl_kernel] Found fallback library: {alt_path}")
        try:
            logger.debug(f"[sgl_kernel] Loading fallback module from {alt_path}...")
            common_ops = _load_common_ops_from_path(alt_path)
            logger.debug(f"[sgl_kernel] ✓ Successfully loaded fallback library")
            logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
            return common_ops

        except Exception as e:
            previous_import_errors.append(e)
            logger.debug(
                f"[sgl_kernel] ✗ Failed to load fallback from {alt_path}: {type(e).__name__}: {e}"
            )
    else:
        logger.debug(
            f"[sgl_kernel] ✗ Fallback library not found: {alt_attempts}"
        )

    # Final attempt: try standard Python import (for backward compatibility)
    logger.debug(
        f"[sgl_kernel] Final attempt: trying standard Python import 'common_ops'"
    )
    try:
        import common_ops

        logger.debug(f"[sgl_kernel] ✓ Successfully imported via standard Python import")
        logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
        return common_ops
    except ImportError as e:
        previous_import_errors.append(e)
        logger.debug(f"[sgl_kernel] ✗ Standard Python import failed: {e}")

    attempt_error_msg = "\n".join(
        f"- {type(err).__name__}: {err}" for err in previous_import_errors
    )

    # All attempts failed
    cuda_version = torch.version.cuda
    if cuda_version and cuda_version.startswith("12"):
        install_hint = (
            "pip install sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/"
        )
    else:
        install_hint = "pip install --upgrade sglang-kernel"

    error_msg = f"""
[sgl_kernel] CRITICAL: Could not load any common_ops library!

Attempted locations:
1. Architecture-specific patterns:
{_format_attempts(ops_attempts)}
2. Fallback patterns:
{_format_attempts(alt_attempts)}
3. Standard Python import: common_ops - failed

GPU Info:
- Compute capability: {compute_capability}
- Expected variant: {variant_name}
- CUDA version: {cuda_version}

Please ensure sgl_kernel is properly installed with:
{install_hint}

Error details from previous import attempts:
{attempt_error_msg}
"""
    logger.debug(error_msg)
    raise ImportError(error_msg)


# copy & modify from torch/utils/cpp_extension.py
def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            cuda_home = "/usr/local/cuda"
    return cuda_home


def _preload_cuda_library():
    """Preload the CUDA runtime library to help avoid 'libcudart.so not found' issues."""
    cuda_home = Path(_find_cuda_home())

    candidate_dirs = [
        cuda_home / "lib",
        cuda_home / "lib64",
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
    ]

    # Determine CUDA major version to try the matching library first.
    # On CUDA 13 systems (e.g., DGX Spark), only libcudart.so.13 exists.
    cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else "12"
    lib_versions = list(dict.fromkeys([cuda_major, "13", "12"]))

    for base in candidate_dirs:
        for lib_version in lib_versions:
            candidate = base / f"libcudart.so.{lib_version}"
            if candidate.exists():
                try:
                    cuda_runtime_lib = candidate.resolve()
                    ctypes.CDLL(str(cuda_runtime_lib), mode=ctypes.RTLD_GLOBAL)
                    logger.debug(f"Preloaded CUDA runtime under {cuda_runtime_lib}")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load {candidate}: {e}")
                    continue

    logger.debug("[sgl_kernel] Could not preload CUDA runtime library")
