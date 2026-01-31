# Memray Profiling for vLLM Memory Leak Investigation

## Overview

Use memray to identify which code paths in vLLM are causing continuous RAM growth during inference.

## Setup

```bash
pip install memray
```

## Step 1: Start server under memray (Terminal 1)

```bash
cd /workspace/vllm
memray run -o vllm_memory.bin --trace-python-allocators \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --limit-mm-per-prompt video=0 \
    --gpu-memory-utilization 0.35 \
    --max-model-len 2048
```

Wait for the server to be ready (you'll see "Uvicorn running on http://0.0.0.0:8000").

## Step 2: Run benchmarks (Terminal 2)

```bash
cd /workspace/vllm

# Single benchmark run
vllm bench serve \
    --backend openai-chat \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --endpoint /v1/chat/completions \
    --dataset-name random-mm \
    --num-prompts 1000
```

Run multiple times to accumulate memory growth:

```bash
for i in {1..3}; do
    echo "=== Run $i ==="
    vllm bench serve \
        --backend openai-chat \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --endpoint /v1/chat/completions \
        --dataset-name random-mm \
        --num-prompts 1000
done
```

## Step 3: Stop server (Terminal 1)

Press `Ctrl+C` to stop the server. Memray will finalize and write `vllm_memory.bin`.

## Step 4: Generate reports

```bash
cd /workspace/vllm

# Flamegraph showing LEAKED memory (most useful)
memray flamegraph --leaks vllm_memory.bin -o flamegraph_leaks.html

# Flamegraph showing ALL allocations
memray flamegraph vllm_memory.bin -o flamegraph_all.html

# Text summary of top allocators
memray summary vllm_memory.bin

# Tree view of allocations
memray tree vllm_memory.bin
```

## Step 5: View results

```bash
# Open in browser
xdg-open flamegraph_leaks.html

# Or view summary in terminal
memray summary vllm_memory.bin | head -50
```

## Understanding the output

### Flamegraph (`flamegraph_leaks.html`)
- **Width** = bytes of memory leaked
- **Stack depth** = call hierarchy
- **Hover** over bars to see exact function and allocation size
- Look for wide bars at the bottom - these are the root causes

### Summary output
Shows top functions by memory allocated. Look for vLLM-specific code paths like:
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/encoder_cache_manager.py`
- `vllm/v1/request.py`

## Tips

- Run more benchmark iterations to make leaks more visible
- The `--leaks` flag filters to only show memory that was never freed
- Compare flamegraphs between good (v0.11.0) and bad (v0.11.1) commits to see what changed
