"""
Tracemalloc hook for vLLM server memory profiling.

Lazy-start approach: tracemalloc is NOT started at import time.
Instead, it's started on-demand via the /debug/tracemalloc/start endpoint.
"""

import atexit
import os
import pickle
import sys
import threading
import tracemalloc
from pathlib import Path

SNAPSHOT_DIR = Path("/workspace/rlee-tools/kv-bug/snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

_snapshot_counter = 0
_snapshot_lock = threading.Lock()


def _get_gpu_memory_mb() -> dict:
    """Get GPU memory stats using torch.cuda if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }
    except ImportError:
        pass
    return {"gpu_allocated_mb": 0, "gpu_reserved_mb": 0, "gpu_max_allocated_mb": 0}


def take_snapshot(label: str = None) -> dict:
    """Take a tracemalloc snapshot and save it to disk."""
    global _snapshot_counter

    if not tracemalloc.is_tracing():
        return {"error": "tracemalloc not running - call /debug/tracemalloc/start first"}

    with _snapshot_lock:
        snapshot = tracemalloc.take_snapshot()

        if label is None:
            label = f"snapshot_{_snapshot_counter}"

        filename = f"{_snapshot_counter:03d}_{label}.pickle"
        filepath = SNAPSHOT_DIR / filename

        with open(filepath, 'wb') as f:
            pickle.dump(snapshot, f)

        current, peak = tracemalloc.get_traced_memory()
        gpu_stats = _get_gpu_memory_mb()

        result = {
            "snapshot_number": _snapshot_counter,
            "label": label,
            "file": str(filepath),
            "current_memory_mb": current / (1024 * 1024),
            "peak_memory_mb": peak / (1024 * 1024),
            **gpu_stats,
        }

        _snapshot_counter += 1

        print(f"[tracemalloc] Snapshot saved: {filename} "
              f"(CPU: {result['current_memory_mb']:.2f} MB, "
              f"GPU: {result['gpu_allocated_mb']:.2f} MB)")

        return result


def _patch_fastapi_app(app):
    """Add debug endpoints to a FastAPI app."""
    from fastapi import Query
    from fastapi.responses import JSONResponse

    @app.post("/debug/tracemalloc/start")
    async def debug_start(nframes: int = Query(default=10)):
        """Start tracemalloc tracing."""
        if tracemalloc.is_tracing():
            return JSONResponse(content={"status": "already running", "nframes": tracemalloc.get_traceback_limit()})
        tracemalloc.start(nframes)
        print(f"[tracemalloc] Started with {nframes} frames")
        return JSONResponse(content={"status": "started", "nframes": nframes})

    @app.post("/debug/tracemalloc/stop")
    async def debug_stop():
        """Stop tracemalloc tracing."""
        if not tracemalloc.is_tracing():
            return JSONResponse(content={"status": "not running"})
        tracemalloc.stop()
        print("[tracemalloc] Stopped")
        return JSONResponse(content={"status": "stopped"})

    @app.post("/debug/snapshot")
    async def debug_snapshot(label: str = Query(default=None)):
        """Take a tracemalloc snapshot."""
        result = take_snapshot(label)
        return JSONResponse(content=result)

    @app.get("/debug/tracemalloc/stats")
    async def debug_stats():
        """Get current tracemalloc statistics."""
        gpu_stats = _get_gpu_memory_mb()
        if not tracemalloc.is_tracing():
            return JSONResponse(content={"is_tracing": False, **gpu_stats})
        current, peak = tracemalloc.get_traced_memory()
        return JSONResponse(content={
            "is_tracing": True,
            "current_memory_mb": current / (1024 * 1024),
            "peak_memory_mb": peak / (1024 * 1024),
            "traceback_limit": tracemalloc.get_traceback_limit(),
            **gpu_stats,
        })

    print(f"[tracemalloc] Added debug endpoints (pid={os.getpid()})")


def _install_fastapi_hook():
    """Install an import hook to patch FastAPI apps when they're created."""

    class FastAPIAppPatcher:
        def __init__(self):
            self._patched_apps = set()
            self._original_init = None

        def patch(self):
            try:
                import fastapi
                self._original_init = fastapi.FastAPI.__init__
                patcher = self

                def patched_init(app_self, *args, **kwargs):
                    patcher._original_init(app_self, *args, **kwargs)
                    app_id = id(app_self)
                    if app_id not in patcher._patched_apps:
                        patcher._patched_apps.add(app_id)
                        _patch_fastapi_app(app_self)

                fastapi.FastAPI.__init__ = patched_init
            except ImportError:
                pass
            return True

    patcher = FastAPIAppPatcher()

    if 'fastapi' in sys.modules:
        patcher.patch()
    else:
        class FastAPIImportHook:
            def find_module(self, name, path=None):
                if name == 'fastapi' or name.startswith('fastapi.'):
                    return self
                return None

            def load_module(self, name):
                if self in sys.meta_path:
                    sys.meta_path.remove(self)
                import importlib
                module = importlib.import_module(name)
                if name == 'fastapi':
                    patcher.patch()
                return module

        sys.meta_path.insert(0, FastAPIImportHook())


# Install the hook (but don't start tracemalloc yet)
_install_fastapi_hook()
print(f"[tracemalloc] Hook installed - use /debug/tracemalloc/start to begin tracing")
