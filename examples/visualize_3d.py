import os
import numpy as np
import pyvista as pv

T = np.load("temperature.npy")  # (Nx,Ny,Nz)

os.makedirs("docs/images", exist_ok=True)

grid = pv.UniformGrid()
grid.dimensions = T.shape
grid.point_data["T"] = T.flatten(order="F")

plotter = pv.Plotter(off_screen=True)
contours = grid.contour(isosurfaces=5)
plotter.add_mesh(contours, cmap="coolwarm")
plotter.show(screenshot="docs/images/3d_isosurface.png")

print("Saved 3D isosurface to docs/images/")
