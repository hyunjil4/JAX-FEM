#!/usr/bin/env python3
"""
Generate GIF animation from time step history.

Saves temperature field at each time step and creates an animated GIF
showing heat diffusion over time.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_simulation


def make_animation(T_history, output_file="docs/animation/heat_diffusion.gif",
                   fps=10, mesh_name="default"):
    """
    Create animated GIF from temperature field history.
    
    Args:
        T_history: List of temperature fields, each of shape (Nx, Ny, Nz)
        output_file: Path to output GIF file
        fps: Frames per second for animation
        mesh_name: Name identifier
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not T_history or len(T_history) == 0:
        print("Error: No temperature history provided")
        return
    
    print(f"\nCreating animation from {len(T_history)} time steps...")
    
    # Get dimensions
    Nx, Ny, Nz = T_history[0].shape
    mid_z = Nz // 2
    
    # Find global min/max for consistent color scale
    T_min = min(T.min() for T in T_history)
    T_max = max(T.max() for T in T_history)
    
    frames = []
    for i, T in enumerate(T_history):
        T_np = np.array(T)
        
        # Create frame
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(T_np[:, :, mid_z].T, origin='lower', cmap='hot',
                       vmin=T_min, vmax=T_max, aspect='auto')
        ax.set_title(f'Heat Diffusion - Step {i}/{len(T_history)-1}\n'
                    f'XY Slice at z = {mid_z}/{Nz-1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        # Convert to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
        
        if (i + 1) % max(1, len(T_history) // 10) == 0:
            print(f"  Processed {i+1}/{len(T_history)} frames")
    
    # Save as GIF
    print(f"Saving animation to {output_path}...")
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"✓ Animation saved: {output_path}")


def main():
    """Main function."""
    print("="*60)
    print("Heat Diffusion Animation Generator")
    print("="*60)
    
    # Run simulation with history
    nx, ny, nz = 20, 20, 20
    steps = 50  # Reduced for faster animation generation
    
    if len(sys.argv) >= 4:
        nx, ny, nz = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    print(f"\nRunning simulation with mesh {nx}×{ny}×{nz}, {steps} steps...")
    T_final, history = run_simulation(
        nx=nx, ny=ny, nz=nz,
        dt=1e-6,
        steps=steps,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
        save_history=True,
        verbose=True
    )
    
    # Generate animation
    if 'T_history' in history and history['T_history']:
        mesh_name = f"{nx}x{ny}x{nz}"
        output_file = f"docs/animation/heat_diffusion_{mesh_name}.gif"
        make_animation(history['T_history'], output_file=output_file, 
                      fps=5, mesh_name=mesh_name)
        print(f"\n✓ Animation generation complete!")
    else:
        print("\n⚠️  No temperature history available. Run with save_history=True.")


if __name__ == "__main__":
    main()
