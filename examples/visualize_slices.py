#!/usr/bin/env python3
"""
Generate 2D slice visualizations of temperature field.

Creates heatmap slices for XY, XZ, and YZ mid-planes and saves
them as PNG images for documentation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_simulation


def visualize_slices(T, output_dir="docs", mesh_name="default"):
    """
    Generate 2D slice visualizations of 3D temperature field.
    
    Args:
        T: Temperature field of shape (Nx, Ny, Nz)
        output_dir: Directory to save images
        mesh_name: Name identifier for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    Nx, Ny, Nz = T.shape
    
    # Mid-plane indices
    mid_x = Nx // 2
    mid_y = Ny // 2
    mid_z = Nz // 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY slice (at mid Z)
    im1 = axes[0].imshow(T[:, :, mid_z].T, origin='lower', cmap='hot', aspect='auto')
    axes[0].set_title(f'XY Slice (z = {mid_z}/{Nz-1})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='Temperature (°C)')
    
    # XZ slice (at mid Y)
    im2 = axes[1].imshow(T[:, mid_y, :].T, origin='lower', cmap='hot', aspect='auto')
    axes[1].set_title(f'XZ Slice (y = {mid_y}/{Ny-1})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[1], label='Temperature (°C)')
    
    # YZ slice (at mid X)
    im3 = axes[2].imshow(T[mid_x, :, :].T, origin='lower', cmap='hot', aspect='auto')
    axes[2].set_title(f'YZ Slice (x = {mid_x}/{Nx-1})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    plt.colorbar(im3, ax=axes[2], label='Temperature (°C)')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f"temperature_slices_{mesh_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved slice visualization: {output_file}")
    
    plt.close()


def main():
    """Main function to run simulation and generate visualizations."""
    print("="*60)
    print("Temperature Field Visualization - 2D Slices")
    print("="*60)
    
    # Run simulation
    nx, ny, nz = 20, 20, 20
    if len(sys.argv) >= 4:
        nx, ny, nz = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    print(f"\nRunning simulation with mesh {nx}×{ny}×{nz}...")
    T, history = run_simulation(
        nx=nx, ny=ny, nz=nz,
        dt=1e-6,
        steps=100,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
        verbose=True
    )
    
    # Generate visualizations
    mesh_name = f"{nx}x{ny}x{nz}"
    visualize_slices(T, output_dir="docs", mesh_name=mesh_name)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Temperature field shape: {T.shape}")
    print(f"  Min temperature: {T.min():.2f}°C")
    print(f"  Max temperature: {T.max():.2f}°C")


if __name__ == "__main__":
    main()
