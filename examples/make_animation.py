import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

os.makedirs("docs/animation", exist_ok=True)

frames = []

# Example assumes temperature fields are saved per timestep:
# temperature_0.npy, temperature_1.npy, ...
for step in range(100):
    fname = f"temperature_{step}.npy"
    if not os.path.exists(fname):
        continue
    T = np.load(fname)
    mid = T.shape[2] // 2

    plt.figure(figsize=(5,4))
    plt.imshow(T[:,:,mid], cmap="inferno")
    plt.title(f"timestep {step}")
    plt.colorbar()
    plt.savefig("temp.png")
    plt.close()

    frames.append(imageio.imread("temp.png"))

imageio.mimsave("docs/animation/heat.gif", frames, fps=10)
print("Saved animation: docs/animation/heat.gif")
