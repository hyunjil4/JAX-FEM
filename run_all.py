#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import datetime

# --------- CONFIG ---------
NX = NY = NZ = 20
DT = 1e-6
STEPS = 100
# ---------------------------

def run_simulation():
    print("\n=== 1) Running FEM Simulation ===")
    cmd = f"python src/fem_solver.py {NX} {NY} {NZ}"
    subprocess.run(cmd, shell=True, check=True)

    # Assert that temperature.npy was created
    assert os.path.exists("temperature.npy"), "temperature.npy was not produced."
    print("✓ Simulation complete. temperature.npy saved.")

def generate_slices():
    print("\n=== 2) Generating 2D Slices ===")
    os.makedirs("docs", exist_ok=True)
    os.system("python examples/visualize_slices.py")
    print("✓ 2D slices generated")

def generate_3d():
    print("\n=== 3) Generating 3D Volume Rendering ===")
    os.system("python examples/visualize_3d.py")
    print("✓ 3D visualization generated")

def generate_gif():
    print("\n=== 4) Generating Animation GIF ===")
    os.system("python examples/make_animation.py")
    print("✓ Animation GIF generated")

def update_readme():
    print("\n=== 5) Updating README ===")

    md = "README.md"
    if not os.path.exists(md):
        print("⚠️ README.md not found. Skipping update.")
        return

    # new auto-updated section
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    new_section = f"""
---

## 🔄 Auto-Generated Visualizations (Updated {stamp})

### 2D Temperature Slices
![Slices](docs/temperature_slices_{NX}x{NY}x{NZ}.png)

### 3D Isosurface Visualization
![3D](docs/temperature_3d_{NX}x{NY}x{NZ}.png)

### Animation (Heat Diffusion)
![GIF](docs/heat_diffusion_{NX}x{NY}x{NZ}.gif)

---
"""

    # append to README
    with open(md, "a") as f:
        f.write(new_section)

    print("✓ README updated with new auto-generated images.")

def main():
    run_simulation()
    generate_slices()
    generate_3d()
    generate_gif()
    update_readme()
    print("\n🎉 ALL DONE! Check docs/ for outputs.\n")

if __name__ == "__main__":
    main()
