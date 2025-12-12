#!/usr/bin/env python3
"""
Master script to run complete FEM pipeline:
1. Run solver to generate temperature.npy
2. Generate all visualizations
3. Verify all output files exist
4. Update README with generated images
"""
import os
import subprocess
import datetime
import numpy as np
from pathlib import Path

# Configuration
NX = NY = NZ = 20
DT = 1e-4
STEPS = 500

# Expected output files
EXPECTED_FILES = {
    "temperature.npy": "Project root",
    "docs/temperature_slices_20x20x20.png": "2D slice visualization",
    "docs/temperature_3d_20x20x20.png": "3D volume rendering",
    "docs/animation/heat_diffusion_yz_20x20x20.gif": "Animation GIF (YZ slice)"
}

# Ensure we're in project root
project_root = Path(__file__).parent
os.chdir(project_root)


def verify_file(filepath, description):
    """Verify that a file exists and is not empty."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ {description} not found: {filepath}\n"
            f"   Expected file was not generated. Please check the previous step."
        )
    if os.path.getsize(filepath) == 0:
        raise ValueError(
            f"❌ {description} is empty: {filepath}\n"
            f"   File was created but contains no data."
        )
    print(f"✓ Verified: {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")


def run_simulation():
    """Step 1: Run FEM solver and ensure temperature.npy is saved."""
    print("\n" + "="*70)
    print("STEP 1: Running FEM Simulation")
    print("="*70)
    
    # Run simulation with save_history=True using Python API
    import sys
    sys.path.insert(0, str(project_root))
    from src.solver import run_simulation
    
    T_final, history = run_simulation(
        nx=NX, ny=NY, nz=NZ,
        dt=DT,
        steps=STEPS,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
        save_history=True,
        verbose=True
    )
    
    # Save temperature history for animation
    if 'T_history' in history and history['T_history']:
        T_history_array = np.array(history['T_history'])
        np.save("temperature_history.npy", T_history_array)
        print(f"✓ Saved temperature history: {len(history['T_history'])} time steps")
    
    # Verify temperature.npy was created
    verify_file("temperature.npy", "Temperature field")
    print("✓ Step 1 complete: temperature.npy and temperature_history.npy saved\n")


def generate_slices():
    """Step 2: Generate 2D slice visualizations."""
    print("="*70)
    print("STEP 2: Generating 2D Slice Visualizations")
    print("="*70)
    
    os.makedirs("docs", exist_ok=True)
    result = subprocess.run(
        "python examples/visualize_slices.py",
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Verify output file exists
    output_file = f"docs/temperature_slices_{NX}x{NY}x{NZ}.png"
    verify_file(output_file, "2D slice visualization")
    print(f"✓ Step 2 complete: {output_file} generated\n")


def generate_3d():
    """Step 3: Generate 3D volume rendering."""
    print("="*70)
    print("STEP 3: Generating 3D Volume Rendering")
    print("="*70)
    
    os.makedirs("docs", exist_ok=True)
    result = subprocess.run(
        "python examples/visualize_3d.py",
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Verify output file exists
    output_file = f"docs/temperature_3d_{NX}x{NY}x{NZ}.png"
    verify_file(output_file, "3D volume rendering")
    print(f"✓ Step 3 complete: {output_file} generated\n")


def generate_animation():
    """Step 4: Generate animation GIF."""
    print("="*70)
    print("STEP 4: Generating Animation GIF")
    print("="*70)
    
    os.makedirs("docs/animation", exist_ok=True)
    result = subprocess.run(
        "python examples/make_animation.py",
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Verify output file exists
    output_file = f"docs/animation/heat_diffusion_yz_{NX}x{NY}x{NZ}.gif"
    verify_file(output_file, "Animation GIF (YZ slice)")
    print(f"✓ Step 4 complete: {output_file} generated\n")


def verify_all_outputs():
    """Step 5: Verify all expected output files exist."""
    print("="*70)
    print("STEP 5: Verifying All Output Files")
    print("="*70)
    
    all_verified = True
    for filepath, description in EXPECTED_FILES.items():
        try:
            verify_file(filepath, description)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n{e}")
            all_verified = False
    
    if not all_verified:
        raise RuntimeError(
            "❌ Some output files are missing or invalid.\n"
            "   Please check the error messages above and re-run the pipeline."
        )
    
    print("\n✓ All output files verified successfully!")
    print("="*70 + "\n")


def update_readme():
    """Step 6: Update README with auto-generated visualization section."""
    print("="*70)
    print("STEP 6: Updating README")
    print("="*70)
    
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
![Heat Diffusion Animation](docs/animation/heat_diffusion_yz_{NX}x{NY}x{NZ}.gif)

---
"""
    
    # Write updated content
    with open(md, "w") as f:
        f.write("\n".join(new_lines))
        f.write(new_section)
    
    print("✓ README updated with new auto-generated images.")
    print("="*70 + "\n")


def main():
    """Run complete pipeline."""
    print("="*70)
    print("JAX-FEM Heat Solver - Complete Pipeline")
    print("="*70)
    print(f"Configuration: Mesh {NX}×{NY}×{NZ}, {STEPS} steps")
    print("="*70)
    
    try:
        # Step 1: Run solver
        run_simulation()
        
        # Step 2: Generate 2D slices
        generate_slices()
        
        # Step 3: Generate 3D visualization
        generate_3d()
        
        # Step 4: Generate animation
        generate_animation()
        
        # Step 5: Verify all outputs
        verify_all_outputs()
        
        # Step 6: Update README
        update_readme()
        
        # Final success message
        print("="*70)
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        for filepath, description in EXPECTED_FILES.items():
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                print(f"  ✓ {filepath} ({size:.1f} KB)")
        print("\n✓ All images are ready for GitHub README!")
        print("="*70 + "\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
