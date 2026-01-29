#!/bin/bash
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VLLM_REPO="https://github.com/reaganjlee/vllm.git"
PYTHON_VERSION="3.12"
VLLM_DIR="$HOME/vllm"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    log_info "Starting vLLM development environment setup..."
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    
    if ! command_exists git; then
        log_error "git is not installed. Please install git first."
        exit 1
    fi
    
    if ! command_exists uv; then
        log_info "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Clone vLLM repository
    if [ -d "$VLLM_DIR" ]; then
        log_warn "vLLM directory already exists at $VLLM_DIR"
        read -p "Do you want to remove and re-clone? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing directory..."
            rm -rf "$VLLM_DIR"
        else
            log_info "Using existing directory..."
            cd "$VLLM_DIR"
            git pull
        fi
    fi
    
    if [ ! -d "$VLLM_DIR" ]; then
        log_info "Cloning vLLM repository..."
        git clone "$VLLM_REPO" "$VLLM_DIR"
        cd "$VLLM_DIR"
    fi
    
    # Create virtual environment
    log_info "Creating Python $PYTHON_VERSION virtual environment..."
    uv venv --python "$PYTHON_VERSION" --seed
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source .venv/bin/activate
    
    # Install build requirements
    log_info "Installing build requirements..."
    uv pip install -r requirements/build.txt
    
    # Set environment variable for precompiled binaries
    log_info "Setting VLLM_USE_PRECOMPILED=1..."
    export VLLM_USE_PRECOMPILED=1
    
    # Install vLLM in editable mode
    log_info "Installing vLLM in editable mode (this may take a while)..."
    uv pip install --no-build-isolation -e .
    
    # Install benchmarking tools
    log_info "Installing vLLM benchmarking tools..."
    uv pip install vllm[bench]
    
    # Install xformers (specific version for compatibility)
    log_info "Installing xformers..."
    uv pip install xformers==0.0.33.post2
    
    # Optional: Uncomment to install CUDA requirements
    # log_info "Installing CUDA requirements..."
    # uv pip install -r requirements/cuda.txt
    
    # Install Claude Code
    if ! command_exists claude; then
        log_info "Installing Claude Code..."
        curl -fsSL https://claude.ai/install.sh | bash
        
        # Add to PATH for current and future sessions
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        fi
        export PATH="$HOME/.local/bin:$PATH"
    else
        log_info "Claude Code already installed"
    fi
    
    # Add activation helper to bashrc
    if ! grep -q "alias activate-vllm=" ~/.bashrc; then
        log_info "Adding activation alias to ~/.bashrc..."
        cat >> ~/.bashrc << 'EOF'
# vLLM development environment
alias activate-vllm='source $HOME/vllm/.venv/bin/activate && export VLLM_USE_PRECOMPILED=1'
EOF
    fi
    
    log_info "${GREEN}✨ Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Reload your shell: source ~/.bashrc"
    echo "  2. Activate environment: activate-vllm"
    echo "  3. Test installation: python -c 'import vllm; print(vllm.__version__)'"
    echo ""
    echo "vLLM location: $VLLM_DIR"
}

# Run main function
main "$@"
