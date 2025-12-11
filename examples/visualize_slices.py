import os
import numpy as np
import matplotlib.pyplot as plt

# Load simulation result (user may replace with actual saved numpy file)
T = np.load("temperature.npy")  # expected shape (Nx, Ny, Nz)

os.makedirs("docs/images", exist_ok=True)

mid_x = T.shape[0] // 2
mid_y = T.shape[1] // 2
mid_z = T.shape[2] // 2

slices = {
    "slice_xy.png": T[:, :, mid_z],
    "slice_xz.png": T[:, mid_y, :],
    "slice_yz.png": T[mid_x, :, :],
}

for name, sl in slices.items():
    plt.figure(figsize=(5,4))
    plt.imshow(sl, cmap="inferno")
    plt.title(name.replace(".png",""))
    plt.colorbar()
    plt.savefig(f"docs/images/{name}", dpi=150)
    plt.close()

print("Saved slice images to docs/images/")
