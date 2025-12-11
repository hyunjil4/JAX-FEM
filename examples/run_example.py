#!/usr/bin/env python3
"""
Example script demonstrating usage of the FEM heat solver
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_simulation

def main():
    print("="*60)
    print("3D Heat Equation FEM Solver - Example")
    print("="*60)
    
    # Run with different mesh sizes
    mesh_sizes = [
        (10, 10, 10),
        (20, 20, 20),
    ]
    
    for nx, ny, nz in mesh_sizes:
        print(f"\nRunning simulation with mesh {nx}×{ny}×{nz}...")
        T, history = run_simulation(
            nx=nx, ny=ny, nz=nz,
            dt=1e-6,
            steps=50,
            T_bottom=100.0,
            T_top=0.0,
            kappa=1.0,
            verbose=True
        )
        print(f"Final temperature field shape: {T.shape}")
        print(f"Total time: {history['timing']['total_ms']:.2f} ms")

if __name__ == "__main__":
    main()
