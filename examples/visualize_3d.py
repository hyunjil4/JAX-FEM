#!/usr/bin/env python3
"""
Generate 3D volume rendering of temperature field.

Creates 3D isosurface visualization using PyVista (preferred) or matplotlib fallback.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Ensure we're in project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Check if temperature.npy exists
temp_file = "temperature.npy"
if not os.path.exists(temp_file):
    print(f"Error: {temp_file} not found in project root.")
    print("Please run the solver first to generate temperature.npy")
    sys.exit(1)

# Load temperature field
try:
    T = np.load(temp_file)  # Expected shape (Nx, Ny, Nz)
    print(f"Loaded temperature field: shape {T.shape}")
except Exception as e:
    print(f"Error loading {temp_file}: {e}")
    sys.exit(1)

# Create output directory
os.makedirs("docs", exist_ok=True)

# Calculate mesh size for filename
nx, ny, nz = T.shape[0]-1, T.shape[1]-1, T.shape[2]-1
output_file = f"docs/temperature_3d_{nx}x{ny}x{nz}.png"

# Try PyVista first (preferred method)
try:
    import pyvista as pv
    
    print("Using PyVista for 3D rendering...")
    
    # Create uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = T.shape
    grid.point_data["T"] = T.flatten(order="F")
    
    # Create plotter (off-screen for headless rendering)
    plotter = pv.Plotter(off_screen=True)
    
    # Generate isosurfaces
    contours = grid.contour(isosurfaces=5)
    plotter.add_mesh(contours, cmap="coolwarm")
    
    # Save screenshot
    plotter.show(screenshot=output_file)
    plotter.close()
    
    # Verify file was created
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"✓ Saved 3D isosurface to {output_file}")
        sys.exit(0)
    else:
        raise RuntimeError(f"PyVista screenshot failed: {output_file} not created")
        
except ImportError:
    print("PyVista not available, using matplotlib fallback...")
    # Fallback to matplotlib 3D visualization
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D scatter plot with color mapping
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points for visualization (reduce density for performance)
        step = max(1, min(T.shape) // 5)
        x_coords = np.arange(0, T.shape[0], step)
        y_coords = np.arange(0, T.shape[1], step)
        z_coords = np.arange(0, T.shape[2], step)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        T_sampled = T[::step, ::step, ::step]
        
        # Flatten for scatter plot
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        T_flat = T_sampled.flatten()
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=T_flat, cmap='coolwarm', 
                            s=20, alpha=0.6, vmin=T.min(), vmax=T.max())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Temperature Field Visualization')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Temperature (°C)')
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"✓ Saved 3D visualization (matplotlib) to {output_file}")
            sys.exit(0)
        else:
            raise RuntimeError(f"Matplotlib save failed: {output_file} not created")
            
    except Exception as e:
        print(f"Error with matplotlib fallback: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    print(f"Error generating 3D visualization with PyVista: {e}")
    print("Falling back to matplotlib...")
    
    # Fallback to matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points
        step = max(1, min(T.shape) // 5)
        x_coords = np.arange(0, T.shape[0], step)
        y_coords = np.arange(0, T.shape[1], step)
        z_coords = np.arange(0, T.shape[2], step)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        T_sampled = T[::step, ::step, ::step]
        
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        T_flat = T_sampled.flatten()
        
        scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=T_flat, cmap='coolwarm',
                            s=20, alpha=0.6, vmin=T.min(), vmax=T.max())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Temperature Field Visualization')
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Temperature (°C)')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"✓ Saved 3D visualization (matplotlib fallback) to {output_file}")
            sys.exit(0)
        else:
            raise RuntimeError(f"Failed to create {output_file}")
            
    except Exception as e2:
        print(f"Error with matplotlib fallback: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
