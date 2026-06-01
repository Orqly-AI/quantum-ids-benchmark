"""Environment snapshot for result provenance.

This repo is intentionally git-less, so instead of a commit hash we capture a
snapshot of the runtime: platform, Python, key library versions, CUDA/GPU info,
and an md5 of the relevant source files. Stored verbatim in every results JSON
so a result can always be traced back to the exact stack that produced it.
"""
from __future__ import annotations
import hashlib
import os
import platform
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
# Source files whose contents define the experiment; hashed for provenance.
_TRACKED_SOURCES = (
    "config.py", "runner.py", "model_factory.py", "data.py",
    "quantum_model.py", "seeding.py",
)


def _version(modname: str) -> str | None:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def _source_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for fn in _TRACKED_SOURCES:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[fn] = hashlib.md5(f.read()).hexdigest()
    return out


def _gpu_info() -> dict:
    info: dict = {"cuda_available": False}
    try:
        import torch
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info


def env_snapshot() -> dict:
    """Capture a JSON-serializable snapshot of the runtime environment."""
    versions = {m: _version(m) for m in
                ("numpy", "torch", "pennylane", "sklearn", "scipy",
                 "pandas", "xgboost", "yaml")}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "versions": {k: v for k, v in versions.items() if v is not None},
        "gpu": _gpu_info(),
        "source_md5": _source_hashes(),
    }
