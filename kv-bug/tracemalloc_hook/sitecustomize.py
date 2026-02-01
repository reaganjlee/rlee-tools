"""
Tracemalloc hook for vLLM server memory profiling.

This file is auto-loaded by Python when the directory is in PYTHONPATH.
It starts tracemalloc and patches the vLLM server to add a /debug/snapshot endpoint.
"""

import atexit
import pickle
import sys
import threading
import tracemalloc
from pathlib import Path

# Start tracemalloc with 25 frames deep for detailed stack traces
tracemalloc.start(25)

SNAPSHOT_DIR = Path("/workspace/rlee-tools/kv-bug/snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

_snapshot_counter = 0
_snapshot_lock = threading.Lock()


def take_snapshot(label: str = None) -> dict:
    """Take a tracemalloc snapshot and save it to disk."""
    global _snapshot_counter

    with _snapshot_lock:
        snapshot = tracemalloc.take_snapshot()

        if label is None:
            label = f"snapshot_{_snapshot_counter}"

        filename = f"{_snapshot_counter:03d}_{label}.pickle"
        filepath = SNAPSHOT_DIR / filename

        with open(filepath, 'wb') as f:
            pickle.dump(snapshot, f)

        # Get current memory stats
        current, peak = tracemalloc.get_traced_memory()

        result = {
            "snapshot_number": _snapshot_counter,
            "label": label,
            "file": str(filepath),
            "current_memory_mb": current / (1024 * 1024),
            "peak_memory_mb": peak / (1024 * 1024),
        }

        _snapshot_counter += 1

        print(f"[tracemalloc] Snapshot saved: {filename} "
              f"(current: {result['current_memory_mb']:.2f} MB, "
              f"peak: {result['peak_memory_mb']:.2f} MB)")

        return result


def _patch_fastapi_app(app):
    """Add /debug/snapshot endpoint to a FastAPI app."""
    from fastapi import Query
    from fastapi.responses import JSONResponse

    @app.post("/debug/snapshot")
    async def debug_snapshot(label: str = Query(default=None)):
        """Take a tracemalloc snapshot."""
        result = take_snapshot(label)
        return JSONResponse(content=result)

    @app.get("/debug/tracemalloc/stats")
    async def debug_tracemalloc_stats():
        """Get current tracemalloc statistics."""
        current, peak = tracemalloc.get_traced_memory()
        return JSONResponse(content={
            "current_memory_mb": current / (1024 * 1024),
            "peak_memory_mb": peak / (1024 * 1024),
            "is_tracing": tracemalloc.is_tracing(),
            "traceback_limit": tracemalloc.get_traceback_limit(),
        })

    print("[tracemalloc] Added /debug/snapshot and /debug/tracemalloc/stats endpoints")


def _install_fastapi_hook():
    """Install an import hook to patch FastAPI apps when they're created."""

    class FastAPIAppPatcher:
        """Patches FastAPI app instances after creation."""

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
                print("[tracemalloc] Installed FastAPI hook")
            except ImportError:
                print("[tracemalloc] FastAPI not yet available, will retry on import")
                return False
            return True

    patcher = FastAPIAppPatcher()

    # Try to patch immediately if FastAPI is already imported
    if 'fastapi' in sys.modules:
        patcher.patch()
    else:
        # Install an import hook to patch when FastAPI is imported
        class FastAPIImportHook:
            def find_module(self, name, path=None):
                if name == 'fastapi' or name.startswith('fastapi.'):
                    return self
                return None

            def load_module(self, name):
                # Remove ourselves to avoid recursion
                if self in sys.meta_path:
                    sys.meta_path.remove(self)

                # Import the real module
                import importlib
                module = importlib.import_module(name)

                # Patch after import
                if name == 'fastapi':
                    patcher.patch()

                return module

        sys.meta_path.insert(0, FastAPIImportHook())
        print("[tracemalloc] Installed FastAPI import hook")


# Install the hook
_install_fastapi_hook()

# Log that tracemalloc is active
print(f"[tracemalloc] Started with {tracemalloc.get_traceback_limit()} frame limit")
print(f"[tracemalloc] Snapshots will be saved to: {SNAPSHOT_DIR}")


def _save_final_snapshot():
    """Save a final snapshot on exit."""
    try:
        take_snapshot("exit")
    except Exception as e:
        print(f"[tracemalloc] Failed to save exit snapshot: {e}")


# Register exit handler
atexit.register(_save_final_snapshot)
