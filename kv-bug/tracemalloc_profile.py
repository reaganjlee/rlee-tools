#!/usr/bin/env python3
"""
Tracemalloc Profiling Script for vLLM Server

Takes memory snapshots before and after each benchmark run to track memory growth.

Usage:
    python tracemalloc_profile.py

Output:
    - snapshots/*.pickle: Raw tracemalloc snapshot files
    - Console: Top 10 memory-growing allocations between snapshots

Snapshot Timeline:
    0: baseline (after server ready)
    1: before_bench_1
    2: after_bench_1
    ...
    13: after_bench_6
    14: final (after 30s sleep)
"""

import os
import pickle
import signal
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import psutil
import requests

# Server configuration (same as memray_profile.py)
SERVER_ARGS = [
    "-m", "vllm.entrypoints.cli.main", "serve", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--limit-mm-per-prompt.video", "0",
    "--max-model-len", "25000",
    "--override-generation-config", '{"max_new_tokens": 1}'
]

BENCHMARK_CMD = [
    "vllm", "bench", "serve",
    "--backend", "openai-chat",
    "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--endpoint", "/v1/chat/completions",
    "--dataset-name", "hf",
    "--dataset-path", "lmarena-ai/VisionArena-Chat",
    "--hf-split", "train",
    "--num-prompts", "1000"
]

SERVER_HEALTH_URL = "http://localhost:8000/health"
SNAPSHOT_URL = "http://localhost:8000/debug/snapshot"
TRACEMALLOC_START_URL = "http://localhost:8000/debug/tracemalloc/start"
TRACEMALLOC_STATS_URL = "http://localhost:8000/debug/tracemalloc/stats"
SERVER_STARTUP_TIMEOUT = 180  # 3 minutes (no tracemalloc overhead now)
BENCHMARK_TIMEOUT = 180  # 3 minutes per benchmark run
NUM_BENCHMARK_RUNS = 6
POST_BENCHMARK_SLEEP = 30  # seconds to wait after final benchmark

# Output paths
SCRIPT_DIR = Path(__file__).parent
SNAPSHOT_DIR = SCRIPT_DIR / "snapshots"
TRACEMALLOC_HOOK_DIR = SCRIPT_DIR / "tracemalloc_hook"
VLLM_REPO_PATH = Path("/workspace/vllm")


def log(message: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_gpu_mem_mb() -> float:
    """Get GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip().split('\n')[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0.0


def start_server() -> subprocess.Popen:
    """Start vLLM server with tracemalloc hook."""
    # Build the command to run inside activated venv
    # Quote the JSON arg properly for shell
    server_args_str = ' '.join(
        f"'{arg}'" if '{' in arg else arg for arg in SERVER_ARGS
    )
    bash_cmd = f"source .venv/bin/activate && python {server_args_str}"
    cmd = ["bash", "-c", bash_cmd]

    log(f"Starting server with tracemalloc: bash -c '{bash_cmd}'")

    # Create a log file for server output
    server_log = SCRIPT_DIR / "server_output.log"
    log_handle = open(server_log, 'w')

    # Set up environment (no PYTHONPATH hook needed - endpoints are in vLLM source)
    env = os.environ.copy()
    env["HF_HOME"] = "/workspace/hf_cache"

    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group
        cwd=VLLM_REPO_PATH,
        env=env,
    )

    # Store log handle on process for cleanup
    process._log_handle = log_handle
    process._log_file = server_log

    return process


def wait_for_server_ready(server_process: subprocess.Popen,
                          timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll health endpoint until server is ready or timeout."""
    log(f"Waiting for server to be ready (timeout: {timeout}s)")
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if server process has exited
        exit_code = server_process.poll()
        if exit_code is not None:
            log(f"Server process exited with code {exit_code}")
            # Show server log output
            if hasattr(server_process, '_log_file'):
                server_process._log_handle.flush()
                try:
                    with open(server_process._log_file, 'r') as f:
                        content = f.read()
                        if content:
                            log("Server output:")
                            # Show last 50 lines
                            lines = content.strip().split('\n')
                            for line in lines[-50:]:
                                print(f"  {line}")
                except Exception as e:
                    log(f"Failed to read server log: {e}")
            return False

        try:
            response = requests.get(SERVER_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                log("Server is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    log(f"Server failed to start within {timeout} seconds")
    return False


def stop_server(process: subprocess.Popen) -> None:
    """Gracefully stop server with process group termination."""
    log("Stopping server...")

    try:
        # Send SIGTERM to process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=30)
            log("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            log("Server didn't stop gracefully, force killing")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
    except (ProcessLookupError, OSError) as e:
        log(f"Error stopping server: {e}")

    # Close log handle if present
    if hasattr(process, '_log_handle'):
        try:
            process._log_handle.close()
        except Exception:
            pass

    # Wait for GPU memory to be released
    log("Waiting for GPU memory to be released...")
    for _ in range(30):
        gpu_mem = get_gpu_mem_mb()
        if gpu_mem < 500:
            break
        time.sleep(1)


def start_tracemalloc(nframes: int = 10) -> bool:
    """Start tracemalloc on the server."""
    try:
        response = requests.post(f"{TRACEMALLOC_START_URL}?nframes={nframes}", timeout=10)
        if response.status_code == 200:
            result = response.json()
            log(f"Tracemalloc: {result.get('status')} (frames={result.get('nframes')})")
            return True
        else:
            log(f"Failed to start tracemalloc: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log(f"Failed to start tracemalloc: {e}")
        return False


def take_snapshot(label: str) -> dict:
    """Request a snapshot from the server."""
    try:
        response = requests.post(f"{SNAPSHOT_URL}?label={label}", timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                log(f"Snapshot '{label}' error: {result['error']}")
                return None
            gpu_alloc = result.get('gpu_allocated_mb', 0)
            gpu_reserved = result.get('gpu_reserved_mb', 0)
            log(f"Snapshot '{label}': CPU {result['current_memory_mb']:.2f} MB "
                f"(peak: {result['peak_memory_mb']:.2f} MB) | "
                f"GPU alloc: {gpu_alloc:.2f} MB, reserved: {gpu_reserved:.2f} MB")
            return result
        else:
            log(f"Failed to take snapshot '{label}': HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        log(f"Failed to take snapshot '{label}': {e}")
        return None


def run_benchmark(run_number: int, timeout: int = BENCHMARK_TIMEOUT) -> bool:
    """Run a single benchmark iteration."""
    log(f"Starting benchmark run {run_number}/{NUM_BENCHMARK_RUNS}")

    # Use same HF cache as server
    env = os.environ.copy()
    env["HF_HOME"] = "/workspace/hf_cache"

    try:
        process = subprocess.Popen(
            BENCHMARK_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
            cwd=VLLM_REPO_PATH,
            env=env,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            log(f"Benchmark run {run_number} timed out, killing process")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.communicate()
            return False

        if process.returncode != 0:
            log(f"Benchmark run {run_number} failed with return code {process.returncode}")
            log(f"Stderr: {stderr[:500] if stderr else 'none'}")
            return False

        log(f"Benchmark run {run_number} completed successfully")
        return True

    except Exception as e:
        log(f"Error running benchmark: {e}")
        return False


def analyze_snapshots() -> None:
    """Analyze memory growth between snapshots."""
    log("=" * 60)
    log("Analyzing memory snapshots...")
    log("=" * 60)

    # Find all snapshot files
    snapshot_files = sorted(SNAPSHOT_DIR.glob("*.pickle"))

    if len(snapshot_files) < 2:
        log(f"Not enough snapshots to analyze (found {len(snapshot_files)})")
        return

    log(f"Found {len(snapshot_files)} snapshots")

    # Load snapshots
    snapshots = []
    for filepath in snapshot_files:
        try:
            with open(filepath, 'rb') as f:
                snapshot = pickle.load(f)
                snapshots.append((filepath.name, snapshot))
        except Exception as e:
            log(f"Failed to load {filepath.name}: {e}")

    if len(snapshots) < 2:
        log("Not enough valid snapshots to compare")
        return

    # Compare consecutive snapshots
    print("\n" + "=" * 80)
    print("MEMORY GROWTH ANALYSIS")
    print("=" * 80)

    for i in range(len(snapshots) - 1):
        name1, snap1 = snapshots[i]
        name2, snap2 = snapshots[i + 1]

        print(f"\n--- {name1} -> {name2} ---")

        try:
            # Compare snapshots
            top_stats = snap2.compare_to(snap1, 'lineno')

            # Show top 10 memory increases
            print("Top 10 memory increases:")
            for stat in top_stats[:10]:
                if stat.size_diff > 0:
                    print(f"  {stat.size_diff / 1024:.1f} KB: {stat.traceback}")
        except Exception as e:
            print(f"  Error comparing: {e}")

    # Overall comparison: baseline to final
    print("\n" + "=" * 80)
    print("OVERALL: BASELINE -> FINAL")
    print("=" * 80)

    baseline_name, baseline_snap = snapshots[0]
    final_name, final_snap = snapshots[-1]

    try:
        top_stats = final_snap.compare_to(baseline_snap, 'lineno')

        print(f"\nComparing {baseline_name} to {final_name}")
        print("Top 20 memory increases:")
        for stat in top_stats[:20]:
            if stat.size_diff > 0:
                print(f"  {stat.size_diff / 1024:.1f} KB: {stat.traceback}")

        # Also show total memory change
        baseline_total = sum(stat.size for stat in baseline_snap.statistics('lineno'))
        final_total = sum(stat.size for stat in final_snap.statistics('lineno'))
        diff = final_total - baseline_total

        print(f"\nTotal traced memory:")
        print(f"  Baseline: {baseline_total / (1024*1024):.2f} MB")
        print(f"  Final:    {final_total / (1024*1024):.2f} MB")
        print(f"  Growth:   {diff / (1024*1024):.2f} MB ({diff / baseline_total * 100:.1f}%)")

    except Exception as e:
        print(f"Error in overall comparison: {e}")


def clean_snapshots() -> None:
    """Remove old snapshot files."""
    if SNAPSHOT_DIR.exists():
        for f in SNAPSHOT_DIR.glob("*.pickle"):
            f.unlink()
        log(f"Cleaned snapshot directory: {SNAPSHOT_DIR}")


def main():
    log("=" * 60)
    log("Tracemalloc Profiling Script for vLLM Server")
    log("=" * 60)

    # Clean up old snapshots
    clean_snapshots()
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    server_process = None
    snapshot_results = []

    try:
        # Start server with tracemalloc
        server_process = start_server()

        # Wait for server to be ready
        if not wait_for_server_ready(server_process):
            log("Failed to start server")
            sys.exit(1)

        # Give server a moment to stabilize
        time.sleep(5)

        # Start tracemalloc now (lazy start - avoids startup overhead)
        log("Starting tracemalloc on server...")
        if not start_tracemalloc(nframes=10):
            log("Warning: Failed to start tracemalloc")

        # Take baseline snapshot
        result = take_snapshot("baseline")
        if result:
            snapshot_results.append(result)

        log(f"Running {NUM_BENCHMARK_RUNS} benchmark iterations with snapshots...")

        # Run benchmarks with before/after snapshots
        successful_runs = 0
        for i in range(1, NUM_BENCHMARK_RUNS + 1):
            # Snapshot before benchmark
            result = take_snapshot(f"before_bench_{i}")
            if result:
                snapshot_results.append(result)

            # Run benchmark
            if run_benchmark(i):
                successful_runs += 1
            else:
                log(f"Warning: Benchmark run {i} failed")

            # Snapshot after benchmark
            result = take_snapshot(f"after_bench_{i}")
            if result:
                snapshot_results.append(result)

        log(f"Completed {successful_runs}/{NUM_BENCHMARK_RUNS} benchmark runs")

        # Wait and take final snapshot
        log(f"Waiting {POST_BENCHMARK_SLEEP}s before final snapshot...")
        time.sleep(POST_BENCHMARK_SLEEP)

        result = take_snapshot("final")
        if result:
            snapshot_results.append(result)

    finally:
        # Stop server
        if server_process:
            stop_server(server_process)

    # Analyze snapshots
    analyze_snapshots()

    # GPU memory summary from collected results
    if snapshot_results:
        log("=" * 60)
        log("GPU MEMORY TREND")
        log("=" * 60)
        for r in snapshot_results:
            label = r.get('label', 'unknown')
            gpu_alloc = r.get('gpu_allocated_mb', 0)
            gpu_reserved = r.get('gpu_reserved_mb', 0)
            cpu_mem = r.get('current_memory_mb', 0)
            log(f"  {label:20s}: CPU {cpu_mem:8.2f} MB | GPU alloc {gpu_alloc:8.2f} MB | reserved {gpu_reserved:8.2f} MB")

        # Show deltas
        if len(snapshot_results) >= 2:
            first = snapshot_results[0]
            last = snapshot_results[-1]
            cpu_delta = last.get('current_memory_mb', 0) - first.get('current_memory_mb', 0)
            gpu_delta = last.get('gpu_allocated_mb', 0) - first.get('gpu_allocated_mb', 0)
            log(f"\n  GROWTH (baseline -> final):")
            log(f"    CPU: {cpu_delta:+.2f} MB")
            log(f"    GPU: {gpu_delta:+.2f} MB")

    # Summary
    log("=" * 60)
    log("Profiling complete!")
    log(f"  Snapshots saved to: {SNAPSHOT_DIR}")
    snapshot_files = list(SNAPSHOT_DIR.glob("*.pickle"))
    log(f"  Total snapshots: {len(snapshot_files)}")
    for f in sorted(snapshot_files):
        size_kb = f.stat().st_size / 1024
        log(f"    - {f.name} ({size_kb:.1f} KB)")
    log("=" * 60)


if __name__ == "__main__":
    main()
