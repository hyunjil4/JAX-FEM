#!/usr/bin/env python3
"""
Generate 3D volume rendering of temperature field.

Creates 3D visualization using matplotlib's 3D plotting capabilities
and saves as PNG image.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_simulation


def visualize_3d_isosurface(T, output_dir="docs", mesh_name="default"):
    """
    Generate 3D isosurface visualization of temperature field.
    
    Args:
        T: Temperature field of shape (Nx, Ny, Nz)
        output_dir: Directory to save images
        mesh_name: Name identifier for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    Nx, Ny, Nz = T.shape
    
    # Create coordinate arrays
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Convert to numpy for plotting
    T_np = np.array(T)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot isosurfaces at different temperature levels
    levels = np.linspace(T_np.min(), T_np.max(), 5)
    
    # Use scatter plot with color mapping for 3D visualization
    # Sample points for visualization (reduce density for performance)
    step = max(1, min(Nx, Ny, Nz) // 10)
    ax.scatter(X[::step, ::step, ::step].flatten(),
               Y[::step, ::step, ::step].flatten(),
               Z[::step, ::step, ::step].flatten(),
               c=T_np[::step, ::step, ::step].flatten(),
               cmap='hot', s=20, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Temperature Field Visualization')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hot', 
                               norm=plt.Normalize(vmin=T_np.min(), vmax=T_np.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    
    # Save figure
    output_file = output_path / f"temperature_3d_{mesh_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 3D visualization: {output_file}")
    
    plt.close()


def visualize_3d_slices(T, output_dir="docs", mesh_name="default"):
    """
    Generate 3D visualization with multiple slice planes.
    
    Args:
        T: Temperature field of shape (Nx, Ny, Nz)
        output_dir: Directory to save images
        mesh_name: Name identifier for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    Nx, Ny, Nz = T.shape
    T_np = np.array(T)
    
    # Create coordinate arrays
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot slices at different positions
    mid_x, mid_y, mid_z = Nx//2, Ny//2, Nz//2
    
    # XY plane at z=0.5
    xx, yy = np.meshgrid(x, y, indexing='ij')
    ax.contourf(xx, yy, T_np[:, :, mid_z], zdir='z', offset=0.5, 
                cmap='hot', alpha=0.7, levels=20)
    
    # XZ plane at y=0.5
    xx, zz = np.meshgrid(x, z, indexing='ij')
    ax.contourf(xx, T_np[:, mid_y, :], zz, zdir='y', offset=0.5,
                cmap='hot', alpha=0.7, levels=20)
    
    # YZ plane at x=0.5
    yy, zz = np.meshgrid(y, z, indexing='ij')
    ax.contourf(T_np[mid_x, :, :], yy, zz, zdir='x', offset=0.5,
                cmap='hot', alpha=0.7, levels=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Temperature Field - Multi-Slice View')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Save figure
    output_file = output_path / f"temperature_3d_slices_{mesh_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 3D slice visualization: {output_file}")
    
    plt.close()


def main():
    """Main function."""
    print("="*60)
    print("Temperature Field Visualization - 3D")
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
    visualize_3d_isosurface(T, output_dir="docs", mesh_name=mesh_name)
    visualize_3d_slices(T, output_dir="docs", mesh_name=mesh_name)
    
    print(f"\n✓ 3D visualization complete!")


if __name__ == "__main__":
    main()
