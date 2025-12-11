#!/bin/bash
# 간단한 wrapper 스크립트 - 가상환경을 자동으로 활성화하고 스크립트 실행

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo "Create it with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Check JAX
if ! python -c "import jax" 2>/dev/null; then
    echo "❌ JAX not installed."
    echo "Install with:"
    echo "  source $VENV_PATH/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Execute command
echo "✓ Virtual environment activated"
echo "  JAX version: $(python -c 'import jax; print(jax.__version__)')"
echo "  Running command..."
echo ""

# Execute user command
exec "$@"




