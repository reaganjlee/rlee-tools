# Tracemalloc Profiling Implementation - Handoff Document

## Overview

This document describes the tracemalloc-based memory profiling system created to investigate memory leaks in the vLLM server. The implementation replaces the previous memray-based approach with Python's built-in `tracemalloc` module, enabling snapshot-based memory analysis.

## Files Created

### 1. `/workspace/rlee-tools/kv-bug/tracemalloc_hook/sitecustomize.py`

A Python hook that auto-loads when the directory is in `PYTHONPATH`. It:

- Starts `tracemalloc` with 25-frame deep stack traces
- Patches FastAPI to add debugging endpoints:
  - `POST /debug/snapshot?label=<name>` - Takes a memory snapshot
  - `GET /debug/tracemalloc/stats` - Returns current memory statistics
- Uses import hooks to intercept FastAPI app creation
- Registers an exit handler to save a final snapshot on shutdown

### 2. `/workspace/rlee-tools/kv-bug/tracemalloc_profile.py`

Main profiling script (based on `memray_profile.py`). It:

- Starts the vLLM server with `PYTHONPATH` set to load the tracemalloc hook
- Takes snapshots at strategic points during benchmarking
- Analyzes memory growth between consecutive snapshots
- Outputs top memory-growing allocations

**Snapshot Timeline:**
| # | Label | Timing |
|---|-------|--------|
| 0 | baseline | After server ready |
| 1 | before_bench_1 | Before benchmark 1 |
| 2 | after_bench_1 | After benchmark 1 |
| ... | ... | ... |
| 11 | before_bench_6 | Before benchmark 6 |
| 12 | after_bench_6 | After benchmark 6 |
| 13 | final | After 30s sleep |

### 3. `/workspace/rlee-tools/kv-bug/snapshots/`

Directory where `.pickle` snapshot files are saved.

## How It Works

1. The profiling script sets `PYTHONPATH=/workspace/rlee-tools/kv-bug/tracemalloc_hook`
2. When Python starts, it auto-loads `sitecustomize.py`
3. The hook starts tracemalloc and installs a FastAPI patcher
4. When vLLM creates its FastAPI app, the patcher adds `/debug/snapshot` endpoint
5. The profiling script calls this endpoint to trigger snapshots via HTTP
6. After benchmarks complete, the script loads all `.pickle` files and compares them

## Usage

```bash
cd /workspace/rlee-tools/kv-bug
python3 tracemalloc_profile.py
```

## Current Blocker

**Disk space is exhausted** - The server cannot start because there's no space to download/cache model weights:

```
$ df -h /
Filesystem      Size  Used Avail Use% Mounted on
overlay          32G   32G  784K 100% /
```

The error from vLLM:
```
RuntimeError: Data processing error: CAS service error : IO Error: No space left on device (os error 28)
```

## To Resume

1. **Free disk space** - Options:
   - Clear unused HuggingFace cache: `rm -rf /workspace/hf_cache/hub/models--*`
   - Delete old profiling artifacts
   - Use a smaller/already-cached model

2. **Run the profiler**:
   ```bash
   python3 /workspace/rlee-tools/kv-bug/tracemalloc_profile.py
   ```

3. **Check results**:
   - Snapshots saved to `/workspace/rlee-tools/kv-bug/snapshots/*.pickle`
   - Console output shows top memory-growing allocations
   - Compare baseline to final for overall memory growth

## Verification Checklist

- [ ] Disk has sufficient free space (need ~10GB for model + snapshots)
- [ ] Server starts and `/health` returns 200
- [ ] `/debug/snapshot` endpoint responds
- [ ] 14 snapshot files created in `snapshots/` directory
- [ ] Console shows memory growth analysis

## Technical Notes

- The hook uses Python's import system (`sys.meta_path`) to intercept FastAPI imports
- Snapshots are Python pickle files containing `tracemalloc.Snapshot` objects
- Stack traces are 25 frames deep for detailed attribution
- The analysis compares snapshots using `snapshot.compare_to()` by line number
