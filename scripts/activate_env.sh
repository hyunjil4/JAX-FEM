#!/bin/bash
# JAX 가상환경 활성화 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/jax_fem_env_39"

# 가상환경이 있는지 확인
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 가상환경을 찾을 수 없습니다: $VENV_PATH"
    echo ""
    echo "가상환경을 생성하려면 다음 명령어를 실행하세요:"
    echo "  python3 -m venv jax_fem_env_39"
    echo "  source jax_fem_env_39/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check JAX
if python -c "import jax" 2>/dev/null; then
    JAX_VERSION=$(python -c "import jax; print(jax.__version__)" 2>/dev/null)
    BACKEND=$(python -c "import jax; print(jax.default_backend())" 2>/dev/null)
    echo "✓ Virtual environment activated!"
    echo "  JAX version: $JAX_VERSION"
    echo "  Backend: $BACKEND"
    echo ""
    echo "You can now run Python scripts:"
    echo "  python src/fem_solver.py"
    echo ""
    echo "To deactivate: deactivate"
    echo ""
    
    # 현재 셸에서 계속 사용할 수 있도록 bash를 시작
    # 하지만 이 스크립트를 source로 실행하면 현재 셸에 활성화됨
    if [ "$0" != "${BASH_SOURCE[0]}" ]; then
        # source로 실행된 경우 (이미 활성화됨)
        return 0
    else
        # 직접 실행된 경우 bash를 시작
        exec bash
    fi
else
    echo "⚠️  Virtual environment activated but JAX not found."
    echo "Install JAX with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi




