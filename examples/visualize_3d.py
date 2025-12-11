import os
import numpy as np
import pyvista as pv
from pathlib import Path

# Ensure we're in project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

T = np.load("temperature.npy")  # (Nx,Ny,Nz)

os.makedirs("docs", exist_ok=True)

grid = pv.UniformGrid()
grid.dimensions = T.shape
grid.point_data["T"] = T.flatten(order="F")

plotter = pv.Plotter(off_screen=True)
contours = grid.contour(isosurfaces=5)
plotter.add_mesh(contours, cmap="coolwarm")

# Save with mesh size in filename
nx, ny, nz = T.shape[0]-1, T.shape[1]-1, T.shape[2]-1
output_file = f"docs/temperature_3d_{nx}x{ny}x{nz}.png"
plotter.show(screenshot=output_file)

print(f"Saved 3D isosurface to {output_file}")
