#!/usr/bin/env python3
"""
vLLM Git Bisect Verification Script

Detects memory leaks by comparing RAM growth rates: if the growth rate decreases
over time, memory is stabilizing (good). If growth rate stays high, there's a
leak (bad).

Algorithm:
1. Warmup phase (first N runs): Calculate initial growth rate
2. Test phase (subsequent runs): Track RAM and calculate recent growth rate
3. Comparison: If recent_growth_rate < initial_growth_rate * threshold -> GOOD
   If recent_growth_rate >= initial_growth_rate * threshold -> BAD (leak)

Usage:
    # Binary search through filtered commits (recommended)
    python bisect_verify.py --bisect

    # Linear scan through all filtered commits
    python bisect_verify.py --run-all

    # Manual single-commit test
    python bisect_verify.py --commit abc123

    # Test current HEAD
    python bisect_verify.py

    # Test with custom benchmark duration (10 minutes)
    python bisect_verify.py --benchmark-duration 600

    # Customize warmup runs and growth rate threshold
    python bisect_verify.py --warmup-runs 5 --growth-rate-threshold 0.3

    # Fail if RAM exceeds 32GB (optional hard limit)
    python bisect_verify.py --ram-threshold 32

    # Test with multimodal processor cache disabled
    python bisect_verify.py --bisect --disable-mm-cache

    # With native git bisect (skips commits not in filtered list)
    git bisect start v0.11.1 v0.11.0
    git bisect run python bisect_verify.py --skip-unlisted

Notes:
    - Uses target_commits.csv as the list of commits to test
    - Skips commits already in bisect_results.csv (uses cached results)
    - Results are saved to bisect_results.csv
    - Detailed logs are saved to logs/<short_hash>.log
    - Use --disable-mm-cache to test with --mm-processor-cache-gb 0
    - Pass/fail is determined by growth rate stabilization:
      * GOOD: Growth rate decreased (memory stabilizing) or initial rate < 10 MB/run
      * BAD: Growth rate stayed high (memory leak detected)
"""

import argparse
import csv
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import requests

# Constants
BASE_SERVER_CMD = [
    "vllm", "serve", "Qwen/Qwen2.5-VL-3B-Instruct",
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
    # "--dataset-name", "hf",
    # "--dataset-path", "lmarena-ai/VisionArena-Chat",
    # "--hf-split", "train",
    "--num-prompts", "1000"
]

# Global flags for mm processor cache (set by command line)
DISABLE_MM_CACHE = False
USE_DEPRECATED_MM_FLAG = False
MOCK_ENCODER = False  # Mock vision encoder for faster testing
SERVER_HEALTH_URL = "http://localhost:8000/health"
SERVER_STARTUP_TIMEOUT = 120  # 2 minutes for model download/load
BENCHMARK_TIMEOUT = 300
BENCHMARK_TIMEOUT_MM_CACHE_DISABLED = 600  # 10 minutes when mm cache is disabled
DEFAULT_REQUIRED_RUNS = 6
REQUIRED_SUCCESSFUL_RUNS = 6  # Can be overridden by --num-runs
RAM_THRESHOLD_MB = 32 * 1024  # 32GB - fail if RAM exceeds this

# Time-based benchmark settings
DEFAULT_BENCHMARK_DURATION_SEC = 300  # 5 minutes of benchmarking
BENCHMARK_DURATION_SEC = DEFAULT_BENCHMARK_DURATION_SEC
RAM_SETTLE_WAIT_SEC = 30  # Wait time after benchmarks to check if RAM decreases
RAM_DECREASE_THRESHOLD_PERCENT = 1  # Expect at least 1% decrease if memory is being freed

# Growth rate stabilization settings
DEFAULT_WARMUP_RUNS = 3  # Number of initial runs for baseline growth rate
WARMUP_RUNS = DEFAULT_WARMUP_RUNS
DEFAULT_GROWTH_RATE_THRESHOLD = 0.5  # Growth rate must decrease to 50% of initial
GROWTH_RATE_THRESHOLD = DEFAULT_GROWTH_RATE_THRESHOLD
MIN_GROWTH_RATE_MB = 10  # If initial growth < this MB/run, already stable (good)

# RAM threshold check (optional, disabled by default)
RAM_THRESHOLD_ENABLED = False
RAM_THRESHOLD_GB = 32  # Fail if RAM exceeds this (when enabled)
LOG_DIR = Path("logs")
RESULTS_FILE = Path("bisect_results.csv")
COMMITS_FILE = Path("filtered_commits.txt")

# Commit range to search (inclusive)
# Set to None to use all commits in the file
RANGE_START = None  # Use all commits from filtered_commits.txt
RANGE_END = None    # Use all commits from filtered_commits.txt

MEMORY_ERROR_PATTERNS = [
    "cuda out of memory",
    "outofmemoryerror",
    "torch.cuda.outofmemoryerror",
    "runtimeerror: cuda error",
    "memoryerror",
    "std::bad_alloc",
    "cudamalloc failed",
    "cuda error: out of memory",
    "cuda error: an illegal memory access",
    "cumemalloc failed",
]

# CSV columns
CSV_COLUMNS = [
    "commit_hash", "short_hash", "timestamp", "author", "message",
    "status", "error_type", "error_message", "successful_runs",
    "mm_cache_disabled",
    "ram_idle_mb", "ram_peak_mb", "ram_after_settle_mb",
    "ram_decreased", "ram_decrease_percent",
    "initial_growth_rate_mb", "final_growth_rate_mb", "growth_rate_decreased",
    "ram_run1_mb", "ram_run2_mb", "ram_run3_mb",
    "ram_run4_mb", "ram_run5_mb", "ram_run6_mb", "ram_run7_mb",
    "ram_run8_mb", "ram_run9_mb", "ram_run10_mb",
    "gpu_mem_idle_mb", "gpu_mem_run1_mb", "gpu_mem_run2_mb",
    "gpu_mem_run3_mb", "gpu_mem_run4_mb", "gpu_mem_run5_mb",
    "gpu_mem_run6_mb", "gpu_mem_run7_mb", "gpu_mem_run8_mb",
    "gpu_mem_run9_mb", "gpu_mem_run10_mb",
    "gpu_mem_peak_mb", "total_duration_sec", "log_file"
]


def load_commit_list(filepath: Path) -> list[str]:
    """Load commit hashes from the commits file (CSV or text)"""
    commits = []
    if not filepath.exists():
        return commits

    with open(filepath) as f:
        first_line = f.readline().strip()
        f.seek(0)

        # Check if it's a CSV with header
        if first_line.startswith('commit_hash'):
            reader = csv.DictReader(f)
            for row in reader:
                commit_hash = row.get('commit_hash', '').strip()
                if commit_hash:
                    commits.append(commit_hash)
        else:
            # Plain text format
            for line in f:
                line = line.strip()
                if line:
                    # Extract commit hash (first field before space or comma)
                    commit_hash = line.split()[0].split(',')[0]
                    commits.append(commit_hash)

    # Filter to specified range if set
    if RANGE_START or RANGE_END:
        start_idx = 0
        end_idx = len(commits)

        for i, commit in enumerate(commits):
            if RANGE_START and commit.startswith(RANGE_START):
                start_idx = i
            if RANGE_END and commit.startswith(RANGE_END):
                end_idx = i + 1  # Include the end commit

        commits = commits[start_idx:end_idx]

    return commits


def is_commit_in_list(commit_hash: str, commit_list: list[str]) -> bool:
    """Check if commit (full or short hash) matches any in the list"""
    short_hash = commit_hash[:9]  # Match the 9-char format in the file
    for c in commit_list:
        if c.startswith(short_hash) or short_hash.startswith(c):
            return True
    return False


def load_existing_results() -> dict[tuple[str, bool], str]:
    """Load existing results from CSV, returns dict of (short_hash, mm_cache_disabled) -> status"""
    results = {}
    if not RESULTS_FILE.exists():
        return results

    with open(RESULTS_FILE, 'r', newline='') as f:
        # Peek at first line to check if it's a header
        first_line = f.readline().strip()
        if not first_line:
            return results

        f.seek(0)  # Reset to beginning

        # Check if first line is header (starts with 'commit_hash') or data (starts with hex)
        has_header = first_line.startswith('commit_hash')

        if has_header:
            reader = csv.DictReader(f)
        else:
            # No header - provide fieldnames explicitly
            reader = csv.DictReader(f, fieldnames=CSV_COLUMNS)

        for row in reader:
            short_hash = row.get('short_hash', '')
            status = row.get('status', '')
            # Parse mm_cache_disabled (handle missing column for old results)
            mm_disabled_str = row.get('mm_cache_disabled', 'False')
            mm_disabled = mm_disabled_str.lower() == 'true'

            if short_hash and status:
                results[(short_hash, mm_disabled)] = status
    return results


def get_commit_status(commit: str, existing_results: dict[tuple[str, bool], str]) -> Optional[str]:
    """Check if commit has already been benchmarked with current mm_cache setting, return status or None"""
    short_hash = commit[:8]
    return existing_results.get((short_hash, DISABLE_MM_CACHE))


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and stdout"""
    logger = logging.getLogger("bisect_verify")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    return logger


def get_current_commit() -> str:
    """Get current HEAD commit hash"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def get_commit_info(commit_hash: str) -> dict:
    """Get commit metadata (timestamp, author, message)"""
    # Get timestamp
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ci", commit_hash],
        capture_output=True, text=True, check=True
    )
    timestamp = result.stdout.strip()

    # Get author
    result = subprocess.run(
        ["git", "show", "-s", "--format=%an", commit_hash],
        capture_output=True, text=True, check=True
    )
    author = result.stdout.strip()

    # Get message (first line)
    result = subprocess.run(
        ["git", "show", "-s", "--format=%s", commit_hash],
        capture_output=True, text=True, check=True
    )
    message = result.stdout.strip()

    return {
        "timestamp": timestamp,
        "author": author,
        "message": message
    }


def get_ram_mb(server_process: Optional[subprocess.Popen] = None) -> float:
    """Get system RAM usage in MB for server process tree"""
    if server_process is None:
        return psutil.virtual_memory().used / (1024 * 1024)

    try:
        parent = psutil.Process(server_process.pid)
        total_rss = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def get_gpu_mem_mb() -> float:
    """Get GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip().split('\n')[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0.0


def check_for_memory_error(output: str) -> tuple[bool, Optional[str]]:
    """
    Check if output contains memory errors.
    Returns (is_memory_error, error_type)
    """
    output_lower = output.lower()
    for pattern in MEMORY_ERROR_PATTERNS:
        if pattern in output_lower:
            return True, "OOM"
    return False, None


QWEN_MODEL_PATH = Path("vllm/model_executor/models/qwen2_5_vl.py")
MOCK_MARKER = "# VLLM_MOCK_VISION_ENCODER"

MOCK_CODE = '''
        # VLLM_MOCK_VISION_ENCODER - Auto-injected mock for faster testing
        import os as _mock_os
        if _mock_os.environ.get("VLLM_MOCK_VISION_ENCODER") == "1":
            mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
            if not mm_input_by_modality:
                return []
            import torch as _mock_torch
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            out_hidden_size = self.config.vision_config.out_hidden_size
            device = next(self.parameters()).device
            dtype = _mock_torch.bfloat16
            multimodal_embeddings = ()
            for modality in mm_input_by_modality:
                multimodal_input = mm_input_by_modality[modality]
                if modality == "image":
                    grid_thw = multimodal_input["image_grid_thw"]
                elif modality == "video":
                    grid_thw = multimodal_input["video_grid_thw"]
                else:
                    continue
                sizes = (grid_thw.prod(-1) // spatial_merge_size // spatial_merge_size).tolist()
                for num_tokens in sizes:
                    dummy_embed = _mock_torch.zeros((num_tokens, out_hidden_size), dtype=dtype, device=device)
                    multimodal_embeddings += (dummy_embed,)
            return multimodal_embeddings
        # END VLLM_MOCK_VISION_ENCODER
'''


def patch_source_for_mock(logger: logging.Logger) -> bool:
    """
    Patch qwen2_5_vl.py to add mock encoder support.
    Returns True if patch was applied, False if already patched or failed.
    """
    source_file = Path(__file__).parent / QWEN_MODEL_PATH
    if not source_file.exists():
        logger.warning(f"Source file not found: {source_file}")
        return False

    content = source_file.read_text()

    # Check if already patched
    if MOCK_MARKER in content:
        logger.info("Source already patched for mock encoder")
        return True

    # Find the embed_multimodal method and inject mock code
    target = "def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:"
    if target not in content:
        logger.warning(f"Could not find embed_multimodal method in {source_file}")
        return False

    # Insert mock code right after the function signature
    patched_content = content.replace(
        target,
        target + MOCK_CODE
    )

    source_file.write_text(patched_content)
    logger.info(f"Patched {source_file} with mock encoder code")
    return True


def get_server_cmd() -> list[str]:
    """Build server command with optional mm cache disable flag"""
    cmd = BASE_SERVER_CMD.copy()
    if DISABLE_MM_CACHE:
        if USE_DEPRECATED_MM_FLAG:
            # Older boolean flag (deprecated in v0.12+)
            cmd.append("--disable-mm-preprocessor-cache")
        else:
            # Newer flag that takes a value (recommended)
            cmd.extend(["--mm-processor-cache-gb", "0"])
    return cmd


def start_server(log_file: Path, logger: logging.Logger) -> subprocess.Popen:
    """Start vLLM server, return process handle"""
    # Patch source code for mock encoder if enabled
    if MOCK_ENCODER:
        patch_source_for_mock(logger)

    server_cmd = get_server_cmd()
    logger.info(f"Starting server with command: {' '.join(server_cmd)}")
    logger.info(f"MM processor cache disabled: {DISABLE_MM_CACHE}")
    logger.info(f"Vision encoder mock enabled: {MOCK_ENCODER}")

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"SERVER START: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(server_cmd)}\n")
        f.write(f"MM processor cache disabled: {DISABLE_MM_CACHE}\n")
        f.write(f"Vision encoder mock enabled: {MOCK_ENCODER}\n")
        f.write(f"{'='*60}\n\n")

    # Prepare environment with mock encoder if enabled
    env = os.environ.copy()
    if MOCK_ENCODER:
        env["VLLM_MOCK_VISION_ENCODER"] = "1"
        # Add mock script directory to PYTHONPATH
        mock_script_dir = str(Path(__file__).parent)
        env["PYTHONPATH"] = f"{mock_script_dir}:{env.get('PYTHONPATH', '')}"

    # Open log file for appending server output
    log_handle = open(log_file, 'a')

    process = subprocess.Popen(
        server_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group for clean termination
        env=env
    )

    return process


def wait_for_server_ready(logger: logging.Logger, server_process: subprocess.Popen,
                          log_file: Path, timeout: int = SERVER_STARTUP_TIMEOUT) -> tuple[bool, Optional[str]]:
    """
    Poll health endpoint until ready, process exits, or timeout.
    Returns (success, error_message).
    """
    logger.info(f"Waiting for server to be ready (timeout: {timeout}s)")
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if server process has exited (crashed)
        exit_code = server_process.poll()
        if exit_code is not None:
            # Process has exited - read log file for error details
            error_msg = f"Server process exited with code {exit_code}"
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                # Get last 50 lines for error context
                log_lines = log_content.strip().split('\n')
                last_lines = '\n'.join(log_lines[-50:])
                error_msg = f"{error_msg}\n\nLast 50 lines of log:\n{last_lines}"
            except Exception as e:
                error_msg = f"{error_msg} (failed to read log: {e})"

            logger.error(f"Server process crashed with exit code {exit_code}")
            return False, error_msg

        try:
            response = requests.get(SERVER_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True, None
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    logger.error(f"Server failed to start within {timeout} seconds")
    return False, f"Server failed to start within {timeout} seconds"


def stop_server(process: subprocess.Popen, logger: logging.Logger) -> None:
    """Gracefully stop server, force kill if needed"""
    logger.info("Stopping server...")

    try:
        # Try graceful termination first
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=10)
            logger.info("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            # Force kill
            logger.warning("Server didn't stop gracefully, force killing")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
    except (ProcessLookupError, OSError) as e:
        logger.warning(f"Error stopping server: {e}")

    # Wait for GPU memory to be released
    logger.info("Waiting for GPU memory to be released...")
    for _ in range(30):  # Wait up to 30 seconds
        gpu_mem = get_gpu_mem_mb()
        if gpu_mem < 500:  # Less than 500MB indicates memory released
            break
        time.sleep(1)


def get_benchmark_timeout() -> int:
    """Get benchmark timeout based on mm cache setting"""
    if DISABLE_MM_CACHE:
        return BENCHMARK_TIMEOUT_MM_CACHE_DISABLED
    return BENCHMARK_TIMEOUT


def run_benchmark(logger: logging.Logger, log_file: Path, run_number: int,
                  timeout: int = None) -> tuple[bool, str, Optional[str]]:
    if timeout is None:
        timeout = get_benchmark_timeout()
    """
    Run benchmark once.
    Returns (success, stdout+stderr, error_type if failed)
    """
    logger.info(f"Starting benchmark run {run_number}")

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"BENCHMARK RUN {run_number}: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(BENCHMARK_CMD)}\n")
        f.write(f"{'='*60}\n\n")

    process = None
    try:
        # Use Popen with process group so we can kill entire tree on timeout
        process = subprocess.Popen(
            BENCHMARK_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            logger.warning(f"Benchmark run {run_number} timed out, killing process group")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()  # Collect any remaining output

            output = f"Timeout after {timeout} seconds\nStdout: {stdout}\nStderr: {stderr}"

            with open(log_file, 'a') as f:
                f.write(f"\nTIMEOUT: {output}\n")

            logger.error(f"Benchmark run {run_number} timed out")
            return False, output, "timeout"

        output = stdout + "\n" + stderr

        # Write output to log
        with open(log_file, 'a') as f:
            f.write(output)
            f.write(f"\nReturn code: {process.returncode}\n")

        # Check for memory errors
        is_memory_error, error_type = check_for_memory_error(output)
        if is_memory_error:
            logger.error(f"Memory error detected in run {run_number}")
            return False, output, error_type

        if process.returncode != 0:
            logger.error(f"Benchmark run {run_number} failed with return code {process.returncode}")
            return False, output, "crash"

        logger.info(f"Benchmark run {run_number} completed successfully")
        return True, output, None

    except Exception as e:
        # Clean up process if something went wrong
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        raise


def calculate_growth_rate(ram_measurements: list[float], window_size: int = 3) -> float:
    """
    Calculate RAM growth rate in MB per run over the last window_size measurements.
    Returns the average growth rate (can be negative if RAM is decreasing).
    """
    if len(ram_measurements) < 2:
        return 0.0

    # Use the last window_size measurements (or all if fewer available)
    window = ram_measurements[-window_size:] if len(ram_measurements) >= window_size else ram_measurements

    if len(window) < 2:
        return 0.0

    # Calculate average growth rate: (last - first) / (num_intervals)
    growth = window[-1] - window[0]
    intervals = len(window) - 1
    return growth / intervals


def verify_commit(commit_hash: str) -> dict:
    """
    Main verification function for a single commit.

    1. Start server
    2. Wait for ready
    3. Record idle memory
    4. Warmup phase: Run initial benchmarks to calculate baseline growth rate
    5. Test phase: Continue running, track recent growth rate
    6. Compare: If growth rate decreased sufficiently, commit is "good"
    7. Stop server
    8. Return results dict matching CSV columns

    A commit is "good" if:
    - Initial growth rate is very small (< MIN_GROWTH_RATE_MB per run), OR
    - Recent growth rate is less than initial_rate * GROWTH_RATE_THRESHOLD
    A commit is "bad" if growth rate stays high (memory leak).
    """
    start_time = time.time()
    short_hash = commit_hash[:8]
    log_file = LOG_DIR / f"{short_hash}.log"

    logger = setup_logging(log_file)
    logger.info(f"Starting verification for commit {commit_hash}")

    # Get commit info
    try:
        commit_info = get_commit_info(commit_hash)
    except subprocess.CalledProcessError:
        commit_info = {"timestamp": "", "author": "", "message": ""}

    # Initialize results
    results = {
        "commit_hash": commit_hash,
        "short_hash": short_hash,
        "timestamp": commit_info["timestamp"],
        "author": commit_info["author"],
        "message": commit_info["message"],
        "status": "bad",
        "error_type": None,
        "error_message": None,
        "successful_runs": 0,
        "mm_cache_disabled": DISABLE_MM_CACHE,
        "ram_idle_mb": None,
        "ram_run1_mb": None,
        "ram_run2_mb": None,
        "ram_run3_mb": None,
        "ram_run4_mb": None,
        "ram_run5_mb": None,
        "ram_run6_mb": None,
        "ram_run7_mb": None,
        "ram_run8_mb": None,
        "ram_run9_mb": None,
        "ram_run10_mb": None,
        "gpu_mem_idle_mb": None,
        "gpu_mem_run1_mb": None,
        "gpu_mem_run2_mb": None,
        "gpu_mem_run3_mb": None,
        "gpu_mem_run4_mb": None,
        "gpu_mem_run5_mb": None,
        "gpu_mem_run6_mb": None,
        "gpu_mem_run7_mb": None,
        "gpu_mem_run8_mb": None,
        "gpu_mem_run9_mb": None,
        "gpu_mem_run10_mb": None,
        "gpu_mem_peak_mb": 0.0,
        "total_duration_sec": 0.0,
        "log_file": str(log_file),
        # New fields for time-based approach
        "ram_peak_mb": None,
        "ram_after_settle_mb": None,
        "ram_decreased": None,
        "ram_decrease_percent": None,
        # Growth rate stabilization fields
        "initial_growth_rate_mb": None,
        "final_growth_rate_mb": None,
        "growth_rate_decreased": None,
    }

    server_process = None

    try:
        # Start server
        server_process = start_server(log_file, logger)

        # Wait for server to be ready
        server_ready, startup_error = wait_for_server_ready(logger, server_process, log_file)
        if not server_ready:
            results["status"] = "bad"
            results["error_type"] = "server_startup"
            results["error_message"] = startup_error[:500] if startup_error else "Server failed to start"
            return results

        # Record idle memory
        time.sleep(5)  # Let server stabilize
        results["ram_idle_mb"] = get_ram_mb(server_process)
        results["gpu_mem_idle_mb"] = get_gpu_mem_mb()
        results["gpu_mem_peak_mb"] = results["gpu_mem_idle_mb"]
        results["ram_peak_mb"] = results["ram_idle_mb"]

        logger.info(f"Idle memory - RAM: {results['ram_idle_mb']:.1f}MB, GPU: {results['gpu_mem_idle_mb']:.1f}MB")
        logger.info(f"Running benchmarks for {BENCHMARK_DURATION_SEC} seconds...")
        logger.info(f"Warmup runs: {WARMUP_RUNS}, Growth rate threshold: {GROWTH_RATE_THRESHOLD}")

        # Run benchmarks for the specified duration
        benchmark_start_time = time.time()
        run_num = 0

        # Track RAM measurements for growth rate calculation
        ram_measurements = [results["ram_idle_mb"]]
        initial_growth_rate = None
        ram_ever_decreased = False
        max_decrease_percent = 0.0
        decrease_detected_at_run = None

        while time.time() - benchmark_start_time < BENCHMARK_DURATION_SEC:
            run_num += 1
            elapsed = time.time() - benchmark_start_time
            remaining = BENCHMARK_DURATION_SEC - elapsed

            logger.info(f"Starting run {run_num} (elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s)")

            success, output, error_type = run_benchmark(logger, log_file, run_num)

            # Record memory after run
            current_ram = get_ram_mb(server_process)
            current_gpu = get_gpu_mem_mb()

            # Check if RAM decreased compared to previous measurement
            prev_ram = ram_measurements[-1]
            if prev_ram > 0:
                decrease_percent = ((prev_ram - current_ram) / prev_ram) * 100
                if decrease_percent >= RAM_DECREASE_THRESHOLD_PERCENT:
                    if not ram_ever_decreased:
                        ram_ever_decreased = True
                        decrease_detected_at_run = run_num
                        logger.info(f"RAM DECREASE DETECTED at run {run_num}: {prev_ram:.1f}MB -> {current_ram:.1f}MB ({decrease_percent:.1f}%)")
                    max_decrease_percent = max(max_decrease_percent, decrease_percent)

            ram_measurements.append(current_ram)

            # Calculate initial growth rate after warmup phase
            if run_num == WARMUP_RUNS and initial_growth_rate is None:
                initial_growth_rate = calculate_growth_rate(ram_measurements, window_size=WARMUP_RUNS + 1)
                logger.info(f"WARMUP COMPLETE: Initial growth rate = {initial_growth_rate:.2f} MB/run")
                if initial_growth_rate < MIN_GROWTH_RATE_MB:
                    logger.info(f"Initial growth rate ({initial_growth_rate:.2f} MB/run) < {MIN_GROWTH_RATE_MB} MB/run - memory already stable")

            # Track peak RAM
            if current_ram > results["ram_peak_mb"]:
                results["ram_peak_mb"] = current_ram
            results["gpu_mem_peak_mb"] = max(results["gpu_mem_peak_mb"], current_gpu)

            # Store in run-specific fields if available
            if run_num <= 10:
                ram_key = f"ram_run{run_num}_mb"
                gpu_key = f"gpu_mem_run{run_num}_mb"
                results[ram_key] = current_ram
                results[gpu_key] = current_gpu

            # Log current growth rate after warmup
            if run_num > WARMUP_RUNS:
                current_growth_rate = calculate_growth_rate(ram_measurements, window_size=WARMUP_RUNS + 1)
                logger.info(f"After run {run_num} - RAM: {current_ram:.1f}MB (peak: {results['ram_peak_mb']:.1f}MB), GPU: {current_gpu:.1f}MB, Growth rate: {current_growth_rate:.2f} MB/run")
            else:
                logger.info(f"After run {run_num} - RAM: {current_ram:.1f}MB (peak: {results['ram_peak_mb']:.1f}MB), GPU: {current_gpu:.1f}MB (warmup phase)")

            # Check if RAM exceeds threshold (if enabled)
            if RAM_THRESHOLD_ENABLED:
                threshold_mb = RAM_THRESHOLD_GB * 1024
                if current_ram > threshold_mb:
                    results["status"] = "bad"
                    results["error_type"] = "OOM"
                    results["error_message"] = f"RAM usage ({current_ram:.1f}MB) exceeded threshold ({threshold_mb}MB / {RAM_THRESHOLD_GB}GB) after run {run_num}"
                    results["successful_runs"] = run_num - 1
                    logger.error(f"RAM threshold exceeded: {current_ram:.1f}MB > {threshold_mb}MB")
                    return results

            if not success:
                results["status"] = "bad"
                results["error_type"] = error_type
                results["error_message"] = output[:500] if output else None
                results["successful_runs"] = run_num - 1
                logger.error(f"Benchmark failed at run {run_num}")
                return results

            results["successful_runs"] = run_num

        logger.info(f"Benchmark phase complete. Ran {run_num} iterations.")
        logger.info(f"Peak RAM during benchmarking: {results['ram_peak_mb']:.1f}MB")

        # Also check after settle period for one more chance to detect decrease
        logger.info(f"Waiting {RAM_SETTLE_WAIT_SEC} seconds for final RAM check...")
        time.sleep(RAM_SETTLE_WAIT_SEC)

        results["ram_after_settle_mb"] = get_ram_mb(server_process)
        ram_measurements.append(results["ram_after_settle_mb"])

        # Check if RAM decreased during settle period
        prev_ram = ram_measurements[-2]  # Second to last (before settle)
        if prev_ram > 0:
            settle_decrease_percent = ((prev_ram - results["ram_after_settle_mb"]) / prev_ram) * 100
            if settle_decrease_percent >= RAM_DECREASE_THRESHOLD_PERCENT:
                if not ram_ever_decreased:
                    ram_ever_decreased = True
                    decrease_detected_at_run = "settle"
                    logger.info(f"RAM DECREASE DETECTED during settle: {prev_ram:.1f}MB -> {results['ram_after_settle_mb']:.1f}MB ({settle_decrease_percent:.1f}%)")
                max_decrease_percent = max(max_decrease_percent, settle_decrease_percent)

        results["ram_decrease_percent"] = max_decrease_percent
        results["ram_decreased"] = ram_ever_decreased

        # Calculate final growth rate (using last few measurements including settle)
        final_growth_rate = calculate_growth_rate(ram_measurements, window_size=WARMUP_RUNS + 1)

        # Handle case where we didn't have enough runs for warmup
        if initial_growth_rate is None:
            initial_growth_rate = calculate_growth_rate(ram_measurements, window_size=len(ram_measurements))
            logger.warning(f"Not enough runs for warmup phase, using all measurements for initial rate: {initial_growth_rate:.2f} MB/run")

        results["initial_growth_rate_mb"] = initial_growth_rate
        results["final_growth_rate_mb"] = final_growth_rate

        logger.info(f"RAM after settle: {results['ram_after_settle_mb']:.1f}MB")
        logger.info(f"RAM ever decreased (>={RAM_DECREASE_THRESHOLD_PERCENT}%): {ram_ever_decreased}")
        if ram_ever_decreased:
            logger.info(f"First decrease detected at: run {decrease_detected_at_run}, max decrease: {max_decrease_percent:.1f}%")

        logger.info(f"Growth rate analysis:")
        logger.info(f"  Initial growth rate: {initial_growth_rate:.2f} MB/run")
        logger.info(f"  Final growth rate: {final_growth_rate:.2f} MB/run")
        logger.info(f"  Threshold for pass: {initial_growth_rate * GROWTH_RATE_THRESHOLD:.2f} MB/run (initial * {GROWTH_RATE_THRESHOLD})")

        # Determine pass/fail based on growth rate stabilization
        growth_rate_decreased = False

        # Case 1: Initial growth rate is very small (already stable)
        if initial_growth_rate < MIN_GROWTH_RATE_MB:
            results["status"] = "good"
            results["growth_rate_decreased"] = True
            growth_rate_decreased = True
            logger.info(f"PASS: Initial growth rate ({initial_growth_rate:.2f} MB/run) < {MIN_GROWTH_RATE_MB} MB/run - memory already stable")

        # Case 2: Growth rate has decreased sufficiently
        elif final_growth_rate < initial_growth_rate * GROWTH_RATE_THRESHOLD:
            results["status"] = "good"
            results["growth_rate_decreased"] = True
            growth_rate_decreased = True
            logger.info(f"PASS: Growth rate decreased from {initial_growth_rate:.2f} to {final_growth_rate:.2f} MB/run - memory is stabilizing")

        # Case 3: Growth rate stayed high (memory leak)
        else:
            results["status"] = "bad"
            results["growth_rate_decreased"] = False
            results["error_type"] = "memory_leak"
            results["error_message"] = (
                f"Growth rate did not decrease sufficiently. "
                f"Initial: {initial_growth_rate:.2f} MB/run, Final: {final_growth_rate:.2f} MB/run, "
                f"Threshold: {initial_growth_rate * GROWTH_RATE_THRESHOLD:.2f} MB/run"
            )
            logger.error(f"FAIL: Growth rate stayed high ({final_growth_rate:.2f} >= {initial_growth_rate * GROWTH_RATE_THRESHOLD:.2f} MB/run) - memory leak detected")

    except Exception as e:
        logger.exception(f"Unexpected error during verification: {e}")
        results["status"] = "bad"
        results["error_type"] = "crash"
        results["error_message"] = str(e)[:500]

    finally:
        # Clean up
        if server_process:
            stop_server(server_process, logger)

        results["total_duration_sec"] = time.time() - start_time
        logger.info(f"Total verification time: {results['total_duration_sec']:.1f}s")

        # Write final status to log
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"FINAL STATUS: {results['status']}\n")
            f.write(f"Successful runs: {results['successful_runs']}\n")
            f.write(f"RAM idle: {results.get('ram_idle_mb', 'N/A')}MB\n")
            f.write(f"RAM peak: {results.get('ram_peak_mb', 'N/A')}MB\n")
            f.write(f"RAM after settle: {results.get('ram_after_settle_mb', 'N/A')}MB\n")
            f.write(f"RAM decrease: {results.get('ram_decrease_percent', 'N/A')}%\n")
            f.write(f"RAM decreased: {results.get('ram_decreased', 'N/A')}\n")
            f.write(f"Initial growth rate: {results.get('initial_growth_rate_mb', 'N/A')} MB/run\n")
            f.write(f"Final growth rate: {results.get('final_growth_rate_mb', 'N/A')} MB/run\n")
            f.write(f"Growth rate decreased: {results.get('growth_rate_decreased', 'N/A')}\n")
            if results['error_type']:
                f.write(f"Error type: {results['error_type']}\n")
            if results['error_message']:
                f.write(f"Error message: {results['error_message']}\n")
            f.write(f"Total duration: {results['total_duration_sec']:.1f}s\n")
            f.write(f"{'='*60}\n")

    return results


def append_results_to_csv(results: dict) -> None:
    """Append results to CSV, creating file with headers if needed"""
    # Check if file exists AND has content (not just empty file)
    needs_header = not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0

    # Also check if existing file is missing header
    if not needs_header:
        with open(RESULTS_FILE, 'r') as f:
            first_line = f.readline().strip()
            if first_line and not first_line.startswith('commit_hash'):
                needs_header = False  # Has data but no header - don't add header mid-file

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if needs_header:
            writer.writeheader()

        writer.writerow(results)


def test_commit(commit: str, existing_results: dict[tuple[str, bool], str]) -> str:
    """
    Test a single commit - returns 'good', 'bad', or 'skip'.
    Uses cached result if available, otherwise runs verification.
    """
    # Check if already benchmarked with current mm_cache setting
    cached_status = get_commit_status(commit, existing_results)
    if cached_status:
        print(f"  Using cached result (mm_cache_disabled={DISABLE_MM_CACHE}): {cached_status}")
        return cached_status

    # Checkout and test
    try:
        subprocess.run(["git", "checkout", commit], check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  Failed to checkout: {e.stderr}")
        return "skip"

    full_commit = get_current_commit()
    results = verify_commit(full_commit)
    append_results_to_csv(results)

    # Update our cache
    existing_results[(commit[:8], DISABLE_MM_CACHE)] = results["status"]

    print(f"  Result: status={results['status']}, successful_runs={results['successful_runs']}")
    return results["status"]


def run_all_commits():
    """Run verification on all commits in the commit list (linear)"""
    commit_list = load_commit_list(COMMITS_FILE)
    if not commit_list:
        print(f"No commits found in {COMMITS_FILE}")
        sys.exit(1)

    print(f"Found {len(commit_list)} commits to verify")
    LOG_DIR.mkdir(exist_ok=True)

    existing_results = load_existing_results()
    print(f"Loaded {len(existing_results)} existing results from {RESULTS_FILE}")

    original_commit = get_current_commit()

    for i, commit in enumerate(commit_list, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(commit_list)}] Verifying: {commit}")
        print(f"{'='*60}")

        status = test_commit(commit, existing_results)

        if status == "bad":
            print(f"\n*** Found bad commit: {commit} ***")

    # Return to original commit
    print(f"\nReturning to original commit: {original_commit[:9]}")
    subprocess.run(["git", "checkout", original_commit], capture_output=True)
    print(f"\nResults saved to {RESULTS_FILE}")


def run_bisect():
    """
    Run binary search to find the first bad commit.

    Assumes commit list is ordered from OLDEST to NEWEST (good -> bad).
    The goal is to find the first commit where the bug appears.
    """
    commit_list = load_commit_list(COMMITS_FILE)
    if not commit_list:
        print(f"No commits found in {COMMITS_FILE}")
        sys.exit(1)

    # Reverse if needed - we want oldest first (index 0 = oldest = likely good)
    # The file appears to have newest first, so reverse it
    commit_list = list(reversed(commit_list))

    print(f"Found {len(commit_list)} commits to bisect")
    print(f"  Oldest (likely good): {commit_list[0]}")
    print(f"  Newest (likely bad):  {commit_list[-1]}")

    LOG_DIR.mkdir(exist_ok=True)

    existing_results = load_existing_results()
    print(f"Loaded {len(existing_results)} existing results from {RESULTS_FILE}")

    original_commit = get_current_commit()

    # Binary search: find first bad commit
    # Invariant: all commits before 'left' are good, all commits at/after 'right' are bad
    left = 0
    right = len(commit_list)

    # First, verify boundary conditions if not cached
    print(f"\n{'='*60}")
    print("Verifying boundary: oldest commit (should be good)")
    print(f"{'='*60}")
    oldest_status = test_commit(commit_list[0], existing_results)
    if oldest_status == "bad":
        print(f"\n*** Oldest commit is already bad! Bug predates this range. ***")
        print(f"First bad commit: {commit_list[0]}")
        subprocess.run(["git", "checkout", original_commit], capture_output=True)
        return

    print(f"\n{'='*60}")
    print("Verifying boundary: newest commit (should be bad)")
    print(f"{'='*60}")
    newest_status = test_commit(commit_list[-1], existing_results)
    if newest_status == "good":
        print(f"\n*** Newest commit is good! Bug not in this range. ***")
        subprocess.run(["git", "checkout", original_commit], capture_output=True)
        return

    # Binary search
    left = 0  # Known good
    right = len(commit_list) - 1  # Known bad

    iteration = 0
    while left + 1 < right:
        iteration += 1
        mid = (left + right) // 2
        remaining = right - left - 1

        print(f"\n{'='*60}")
        print(f"Bisect iteration {iteration}: testing index {mid}/{len(commit_list)-1}")
        print(f"  Range: [{left}..{right}], {remaining} commits remaining to search")
        print(f"  Testing: {commit_list[mid]}")
        print(f"{'='*60}")

        status = test_commit(commit_list[mid], existing_results)

        if status == "good":
            left = mid
            print(f"  -> Commit is good, searching newer commits")
        elif status == "bad":
            right = mid
            print(f"  -> Commit is bad, searching older commits")
        else:  # skip
            # Can't determine, try adjacent commit
            print(f"  -> Commit skipped, trying next commit")
            # Move towards right (newer) as a heuristic
            found_testable = False
            for offset in range(1, right - mid):
                if mid + offset < right:
                    test_idx = mid + offset
                    print(f"  Trying offset +{offset}: {commit_list[test_idx]}")
                    alt_status = test_commit(commit_list[test_idx], existing_results)
                    if alt_status == "good":
                        left = test_idx
                        found_testable = True
                        break
                    elif alt_status == "bad":
                        right = test_idx
                        found_testable = True
                        break
            if not found_testable:
                print(f"  Could not find testable commit in range, narrowing from left")
                left = mid

    # Found the boundary
    first_bad = commit_list[right]
    last_good = commit_list[left]

    print(f"\n{'='*60}")
    print("BISECT COMPLETE")
    print(f"{'='*60}")
    print(f"Last good commit:  {last_good}")
    print(f"First bad commit:  {first_bad}")
    print(f"Iterations: {iteration}")
    print(f"\nThe bug was likely introduced in: {first_bad}")

    # Return to original commit
    print(f"\nReturning to original commit: {original_commit[:9]}")
    subprocess.run(["git", "checkout", original_commit], capture_output=True)
    print(f"\nResults saved to {RESULTS_FILE}")


def main():
    global DISABLE_MM_CACHE, USE_DEPRECATED_MM_FLAG, REQUIRED_SUCCESSFUL_RUNS
    global BENCHMARK_DURATION_SEC, RAM_SETTLE_WAIT_SEC, RAM_DECREASE_THRESHOLD_PERCENT
    global RAM_THRESHOLD_ENABLED, RAM_THRESHOLD_GB
    global WARMUP_RUNS, GROWTH_RATE_THRESHOLD
    global MOCK_ENCODER

    parser = argparse.ArgumentParser(description="vLLM bisect verification")
    parser.add_argument("--commit", help="Specific commit to test (default: HEAD)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run verification on all commits in target_commits.csv (linear)")
    parser.add_argument("--bisect", action="store_true",
                        help="Run binary search to find first bad commit in target_commits.csv")
    parser.add_argument("--skip-unlisted", action="store_true",
                        help="Skip (exit 125) if commit not in target_commits.csv (for git bisect)")
    parser.add_argument("--disable-mm-cache", action="store_true",
                        help="Disable multimodal processor cache (--mm-processor-cache-gb 0)")
    parser.add_argument("--disable-mm-preprocessor-cache", action="store_true",
                        help="Disable multimodal processor cache using deprecated flag (--disable-mm-preprocessor-cache)")
    parser.add_argument("--mock-encoder", action="store_true",
                        help="Mock vision encoder to speed up testing (returns dummy embeddings)")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_REQUIRED_RUNS,
                        help=f"Number of benchmark runs required (default: {DEFAULT_REQUIRED_RUNS})")
    parser.add_argument("--benchmark-duration", type=int, default=DEFAULT_BENCHMARK_DURATION_SEC,
                        help=f"Duration in seconds to run benchmarks (default: {DEFAULT_BENCHMARK_DURATION_SEC})")
    parser.add_argument("--settle-wait", type=int, default=30,
                        help="Seconds to wait after benchmarking before checking RAM decrease (default: 30)")
    parser.add_argument("--ram-decrease-threshold", type=float, default=1.0,
                        help="Minimum RAM decrease percentage to pass (default: 1.0)")
    parser.add_argument("--ram-threshold", type=float, default=None,
                        help="Fail if RAM exceeds this many GB (e.g., --ram-threshold 32). Disabled by default.")
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS,
                        help=f"Number of initial runs for baseline growth rate (default: {DEFAULT_WARMUP_RUNS})")
    parser.add_argument("--growth-rate-threshold", type=float, default=DEFAULT_GROWTH_RATE_THRESHOLD,
                        help=f"How much the growth rate must decrease to pass (default: {DEFAULT_GROWTH_RATE_THRESHOLD} = 50%%). "
                             "e.g., 0.5 means final rate must be < 50%% of initial rate")
    args = parser.parse_args()

    # Set global flags for mm cache
    DISABLE_MM_CACHE = args.disable_mm_cache or args.disable_mm_preprocessor_cache
    USE_DEPRECATED_MM_FLAG = args.disable_mm_preprocessor_cache
    MOCK_ENCODER = args.mock_encoder
    REQUIRED_SUCCESSFUL_RUNS = args.num_runs

    # Set time-based benchmark settings
    BENCHMARK_DURATION_SEC = args.benchmark_duration
    RAM_SETTLE_WAIT_SEC = args.settle_wait
    RAM_DECREASE_THRESHOLD_PERCENT = args.ram_decrease_threshold

    # Set RAM threshold settings
    RAM_THRESHOLD_ENABLED = args.ram_threshold is not None
    RAM_THRESHOLD_GB = args.ram_threshold if args.ram_threshold else 32

    # Set growth rate stabilization settings
    WARMUP_RUNS = args.warmup_runs
    GROWTH_RATE_THRESHOLD = args.growth_rate_threshold

    LOG_DIR.mkdir(exist_ok=True)

    # Binary search mode
    if args.bisect:
        run_bisect()
        return

    # Run all commits mode (linear)
    if args.run_all:
        run_all_commits()
        return

    commit = args.commit or get_current_commit()

    # Check if commit should be skipped (for git bisect with filtered list)
    if args.skip_unlisted:
        commit_list = load_commit_list(COMMITS_FILE)
        if commit_list and not is_commit_in_list(commit, commit_list):
            print(f"Skipping commit {commit[:9]} (not in filtered list)")
            sys.exit(125)

    print(f"Verifying commit: {commit}")

    results = verify_commit(commit)
    append_results_to_csv(results)

    print(f"Results: status={results['status']}, successful_runs={results['successful_runs']}")

    # Exit codes for git bisect
    if results["status"] == "good":
        sys.exit(0)
    elif results["status"] == "skip":
        sys.exit(125)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
