import os
import sys
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we're in project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Add src to path for imports
sys.path.insert(0, str(project_root))

from src.solver import run_simulation

os.makedirs("docs", exist_ok=True)

# Run simulation with history to generate animation
print("Running simulation with history for animation...")
T_final, history = run_simulation(
    nx=20, ny=20, nz=20,
    dt=1e-6,
    steps=50,  # Reduced for faster animation
    T_bottom=100.0,
    T_top=0.0,
    kappa=1.0,
    save_history=True,
    verbose=False
)

# Check if history is available
if 'T_history' not in history or not history['T_history']:
    print("Warning: No temperature history available. Creating animation from final state only.")
    T_history = [T_final]
else:
    T_history = history['T_history']

frames = []
T_min = min(T.min() for T in T_history)
T_max = max(T.max() for T in T_history)

for i, T in enumerate(T_history):
    mid = T.shape[2] // 2
    
    plt.figure(figsize=(5, 4))
    plt.imshow(T[:, :, mid], cmap="inferno", vmin=T_min, vmax=T_max, origin='lower')
    plt.title(f"Timestep {i}/{len(T_history)-1}")
    plt.colorbar(label="Temperature")
    plt.tight_layout()
    
    # Save to temporary file
    temp_file = "temp_frame.png"
    plt.savefig(temp_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    frames.append(imageio.imread(temp_file))
    os.remove(temp_file)  # Clean up temp file

# Save GIF with mesh size in filename
nx, ny, nz = T_final.shape[0]-1, T_final.shape[1]-1, T_final.shape[2]-1
output_file = f"docs/heat_diffusion_{nx}x{ny}x{nz}.gif"
imageio.mimsave(output_file, frames, fps=5)
print(f"Saved animation: {output_file}")
