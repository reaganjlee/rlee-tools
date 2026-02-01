#!/usr/bin/env python3
"""
Memray Profiling Script for vLLM Server

Automates memory profiling of the vLLM server:
1. Starts the vLLM server under memray tracking
2. Runs benchmarks 6 times
3. Stops the server and generates a flame graph

Usage:
    python memray_profile.py

Output:
    - memray_output.bin: Raw memray profiling data
    - flamegraph_leaks.html: Flame graph visualization of memory leaks
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests

# Server configuration (same as bisect_verify.py)
SERVER_ARGS = [
    "-m", "vllm.entrypoints.cli.main", "serve", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--limit-mm-per-prompt.video", "0",
    "--gpu-memory-utilization", "0.35",
    "--max-model-len", "2048",
    "--override-generation-config", '{"max_new_tokens": 1}'
]

BENCHMARK_CMD = [
    "vllm", "bench", "serve",
    "--backend", "openai-chat",
    "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--endpoint", "/v1/chat/completions",
    "--dataset-name", "random-mm",
    "--num-prompts", "1000"
]

SERVER_HEALTH_URL = "http://localhost:8000/health"
SERVER_STARTUP_TIMEOUT = 180  # 3 minutes
BENCHMARK_TIMEOUT = 180  # 3 minutes per benchmark run
NUM_BENCHMARK_RUNS = 6

# Output paths
SCRIPT_DIR = Path(__file__).parent
MEMRAY_OUTPUT = SCRIPT_DIR / "memray_output.bin"
FLAMEGRAPH_OUTPUT = SCRIPT_DIR / "flamegraph_leaks.html"
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


def start_server_with_memray() -> subprocess.Popen:
    """Start vLLM server under memray tracking."""
    memray_cmd = [
        "memray", "run",
        "--output", str(MEMRAY_OUTPUT),
        "--force",  # Overwrite existing output file
        "--follow-fork",
    ] + SERVER_ARGS

    log(f"Starting server with memray: {' '.join(memray_cmd)}")

    # Create a log file for server output
    server_log = SCRIPT_DIR / "server_output.log"
    log_handle = open(server_log, 'w')

    # Set up environment with fresh HF cache location
    env = os.environ.copy()
    env["HF_HOME"] = "/workspace/hf_cache"
    # Use malloc allocator for accurate leak detection with memray --leaks
    # Python's default allocator doesn't always release memory to the system
    env["PYTHONMALLOC"] = "malloc"

    process = subprocess.Popen(
        memray_cmd,
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
            process.wait(timeout=30)  # Longer timeout for memray to write output
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


def generate_flamegraph() -> bool:
    """Generate flame graph from memray output."""
    if not MEMRAY_OUTPUT.exists():
        log(f"Memray output file not found: {MEMRAY_OUTPUT}")
        return False

    log(f"Generating flame graph: {FLAMEGRAPH_OUTPUT}")

    try:
        result = subprocess.run(
            ["memray", "flamegraph", "--leaks", "--force", str(MEMRAY_OUTPUT), "-o", str(FLAMEGRAPH_OUTPUT)],
            capture_output=True,
            text=True,
            check=True
        )
        log("Flame graph generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Failed to generate flame graph: {e.stderr}")
        return False


def main():
    log("=" * 60)
    log("Memray Profiling Script for vLLM Server")
    log("=" * 60)

    # Clean up any existing output files
    if MEMRAY_OUTPUT.exists():
        log(f"Removing existing memray output: {MEMRAY_OUTPUT}")
        MEMRAY_OUTPUT.unlink()

    server_process = None

    try:
        # Start server with memray
        server_process = start_server_with_memray()

        # Wait for server to be ready
        if not wait_for_server_ready(server_process):
            log("Failed to start server")
            sys.exit(1)

        log(f"Running {NUM_BENCHMARK_RUNS} benchmark iterations...")

        # Run benchmarks
        successful_runs = 0
        for i in range(1, NUM_BENCHMARK_RUNS + 1):
            if run_benchmark(i):
                successful_runs += 1
            else:
                log(f"Warning: Benchmark run {i} failed")

        log(f"Completed {successful_runs}/{NUM_BENCHMARK_RUNS} benchmark runs")

    finally:
        # Stop server
        if server_process:
            stop_server(server_process)

        # Wait a moment for memray to finalize output
        log("Waiting for memray to write output...")
        time.sleep(5)

    # Generate flame graph
    if MEMRAY_OUTPUT.exists():
        file_size = MEMRAY_OUTPUT.stat().st_size
        log(f"Memray output file size: {file_size / (1024*1024):.2f} MB")

        if generate_flamegraph():
            log("=" * 60)
            log("Profiling complete!")
            log(f"  Memray data: {MEMRAY_OUTPUT}")
            log(f"  Flame graph: {FLAMEGRAPH_OUTPUT}")
            log("=" * 60)
        else:
            log("Failed to generate flame graph")
            sys.exit(1)
    else:
        log("Memray output file was not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
