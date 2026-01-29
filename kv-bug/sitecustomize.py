"""
Site customization script that installs the mock encoder hook.
This file is automatically loaded by Python when starting any process.
"""
import os
import sys

if os.environ.get("VLLM_MOCK_VISION_ENCODER", "0") == "1":
    # Add workspace to path if not already there
    workspace = "/workspace/vllm"
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

    # Import and install the mock encoder hook
    try:
        import mock_encoder
        print("[sitecustomize] Mock encoder hook loaded", flush=True)
    except ImportError as e:
        print(f"[sitecustomize] Warning: Could not load mock_encoder: {e}", flush=True)
