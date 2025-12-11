import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we're in project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Load simulation result
T = np.load("temperature.npy")  # expected shape (Nx, Ny, Nz)

os.makedirs("docs", exist_ok=True)

mid_x = T.shape[0] // 2
mid_y = T.shape[1] // 2
mid_z = T.shape[2] // 2

# Create combined figure with all three slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# XY slice
im1 = axes[0].imshow(T[:, :, mid_z], cmap="inferno", origin='lower')
axes[0].set_title("XY Slice")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
plt.colorbar(im1, ax=axes[0])

# XZ slice
im2 = axes[1].imshow(T[:, mid_y, :], cmap="inferno", origin='lower')
axes[1].set_title("XZ Slice")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Z")
plt.colorbar(im2, ax=axes[1])

# YZ slice
im3 = axes[2].imshow(T[mid_x, :, :], cmap="inferno", origin='lower')
axes[2].set_title("YZ Slice")
axes[2].set_xlabel("Y")
axes[2].set_ylabel("Z")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()

# Save with mesh size in filename - ALWAYS save to docs/temperature_slices_20x20x20.png
nx, ny, nz = T.shape[0]-1, T.shape[1]-1, T.shape[2]-1
output_file = f"docs/temperature_slices_{nx}x{ny}x{nz}.png"
os.makedirs("docs", exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved slice images to {output_file}")
