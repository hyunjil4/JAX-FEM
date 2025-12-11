#!/usr/bin/env python3
"""
3D Heat Transfer FEM Solver - Legacy CLI Entrypoint

This file maintains backward compatibility with the original API.
For new code, use src.solver.run_simulation() instead.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_fem_explicit, main

# Export for backward compatibility
__all__ = ['run_fem_explicit', 'main']

if __name__ == "__main__":
    main()
