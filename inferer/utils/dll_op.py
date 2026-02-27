import ctypes
import os
import sys
from pathlib import Path


_DLL_DIR_HANDLES = []


def _prepend_env_path(env_key, path_str):
    current = os.environ.get(env_key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if path_str in parts:
        return
    if current:
        os.environ[env_key] = path_str + os.pathsep + current
    else:
        os.environ[env_key] = path_str


def configure_dll_path(dll_path):
    """
    Configure process runtime library search paths and eagerly load shared libraries.

    Args:
        dll_path: directory containing CUDA/ORT runtime libraries.

    Returns:
        dict with load metadata.
    """
    if dll_path is None:
        return None

    lib_dir = Path(dll_path).expanduser().resolve()
    if not lib_dir.exists():
        raise FileNotFoundError(f"--dll_path does not exist: {lib_dir}")
    if not lib_dir.is_dir():
        raise NotADirectoryError(f"--dll_path is not a directory: {lib_dir}")

    if sys.platform.startswith("win"):
        patterns = ("*.dll",)
    else:
        patterns = ("*.so", "*.so.*", "*.dll")

    lib_files = []
    for pattern in patterns:
        lib_files.extend(sorted(lib_dir.glob(pattern)))
    if len(lib_files) == 0:
        raise FileNotFoundError(
            f"No runtime library files were found under --dll_path: {lib_dir}"
        )

    _prepend_env_path("PATH", str(lib_dir))
    if not sys.platform.startswith("win"):
        _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir))

    if hasattr(os, "add_dll_directory"):
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib_dir)))

    load_errors = []
    loaded_count = 0
    for lib in lib_files:
        ext = lib.suffix.lower()
        try:
            if ext == ".dll":
                if sys.platform.startswith("win"):
                    ctypes.WinDLL(str(lib))
                    loaded_count += 1
            else:
                ctypes.CDLL(str(lib))
                loaded_count += 1
        except OSError as exc:
            load_errors.append((lib.name, str(exc)))

    if load_errors:
        preview = "\n".join(
            [f"  - {name}: {err}" for name, err in load_errors[:10]]
        )
        raise RuntimeError(
            f"Failed to import {len(load_errors)} runtime libraries from {lib_dir}.\n"
            f"{preview}"
        )

    return {
        "library_dir": str(lib_dir),
        "library_count": len(lib_files),
        "loaded_count": loaded_count,
    }
