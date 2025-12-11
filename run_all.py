#!/usr/bin/env python3
"""
Master script to run complete FEM pipeline:
1. Run solver to generate temperature.npy
2. Generate all visualizations
3. Update README with generated images
"""
import os
import subprocess
import datetime
from pathlib import Path

# Configuration
NX = NY = NZ = 20
DT = 1e-6
STEPS = 100

# Ensure we're in project root
project_root = Path(__file__).parent
os.chdir(project_root)


def run_simulation():
    """Run FEM solver and verify temperature.npy is created."""
    print("\n=== 1) Running FEM Simulation ===")
    cmd = f"python src/fem_solver.py {NX} {NY} {NZ}"
    result = subprocess.run(cmd, shell=True, check=True)
    if result.returncode != 0:
        raise RuntimeError("Simulation failed")
    
    # Verify temperature.npy was created
    assert os.path.exists("temperature.npy"), "temperature.npy was not produced."
    print("✓ Simulation complete. temperature.npy saved.")


def generate_slices():
    """Generate 2D slice visualizations."""
    print("\n=== 2) Generating 2D Slices ===")
    os.makedirs("docs", exist_ok=True)
    subprocess.run("python examples/visualize_slices.py", shell=True, check=True)
    print("✓ 2D slices generated")


def generate_3d():
    """Generate 3D volume rendering."""
    print("\n=== 3) Generating 3D Volume Rendering ===")
    os.makedirs("docs", exist_ok=True)
    subprocess.run("python examples/visualize_3d.py", shell=True, check=True)
    print("✓ 3D visualization generated")


def generate_gif():
    """Generate animation GIF."""
    print("\n=== 4) Generating Animation GIF ===")
    os.makedirs("docs/animation", exist_ok=True)
    subprocess.run("python examples/make_animation.py", shell=True, check=True)
    print("✓ Animation GIF generated")


def update_readme():
    """Update README with auto-generated visualization section."""
    print("\n=== 5) Updating README ===")
    
    md = "README.md"
    if not os.path.exists(md):
        print("⚠️ README.md not found. Skipping update.")
        return

    # Read existing README
    with open(md, "r") as f:
        content = f.read()
    
    # Remove old auto-generated section if it exists
    lines = content.split("\n")
    new_lines = []
    skip_section = False
    for line in lines:
        if "## 🔄 Auto-Generated Visualizations" in line:
            skip_section = True
        elif skip_section and line.strip() == "---":
            skip_section = False
            continue
        if not skip_section:
            new_lines.append(line)
    
    # Add new auto-generated section
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    new_section = f"""
---

## 🔄 Auto-Generated Visualizations (Updated {stamp})

### 2D Temperature Slices
![Temperature Slices](docs/temperature_slices_{NX}x{NY}x{NZ}.png)

### 3D Isosurface Visualization
![3D Temperature Field](docs/temperature_3d_{NX}x{NY}x{NZ}.png)

### Animation (Heat Diffusion)
![Heat Diffusion Animation](docs/animation/heat_diffusion_{NX}x{NY}x{NZ}.gif)

---
"""
    
    # Write updated content
    with open(md, "w") as f:
        f.write("\n".join(new_lines))
        f.write(new_section)
    
    print("✓ README updated with new auto-generated images.")


def main():
    """Run complete pipeline."""
    print("="*70)
    print("JAX-FEM Heat Solver - Complete Pipeline")
    print("="*70)
    
    try:
        run_simulation()
        generate_slices()
        generate_3d()
        generate_gif()
        update_readme()
        print("\n" + "="*70)
        print("🎉 ALL DONE! Check docs/ for outputs.")
        print("="*70 + "\n")
    except AssertionError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
