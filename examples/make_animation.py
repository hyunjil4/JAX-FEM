#!/usr/bin/env python3
"""
Generate GIF animation from temperature field history.

Creates an animated GIF showing heat diffusion over time.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Check for imageio (required for GIF creation)
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Ensure we're in project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Create output directory
os.makedirs("docs/animation", exist_ok=True)
print("Output directory: docs/animation/")

# Try to load temperature history or run simulation
T_history = None

# Option 1: Try to load pre-saved history
history_file = "temperature_history.npy"
if os.path.exists(history_file):
    print(f"Loading temperature history from {history_file}...")
    try:
        T_history = np.load(history_file, allow_pickle=True)
        if isinstance(T_history, np.ndarray) and T_history.ndim == 4:
            # Convert from (steps, Nx, Ny, Nz) to list of (Nx, Ny, Nz)
            T_history = [T_history[i] for i in range(T_history.shape[0])]
            print(f"✓ Loaded {len(T_history)} time steps from history file")
        else:
            print("⚠ History file format unexpected, running simulation instead")
            T_history = None
    except Exception as e:
        print(f"⚠ Error loading history file: {e}")
        T_history = None

# Option 2: Run simulation with history if no history file exists
if T_history is None:
    print("No temperature history found. Running simulation with history...")
    try:
        # Add src to path for imports
        sys.path.insert(0, str(project_root))
        from src.solver import run_simulation
        
        T_final, history = run_simulation(
            nx=20, ny=20, nz=20,
            dt=1e-4,
            steps=500,
            T_bottom=100.0,
            T_top=0.0,
            kappa=1.0,
            save_history=True,
            verbose=False
        )
        
        # Extract history from simulation result
        if 'T_history' in history and history['T_history']:
            T_history = history['T_history']
            print(f"✓ Generated {len(T_history)} time steps from simulation")
        else:
            # Fallback: use final state only
            print("⚠ No history available, using final state only")
            T_history = [T_final]
    except Exception as e:
        print(f"Error running simulation: {e}")
        # Option 3: Try to use single temperature.npy file
        temp_file = "temperature.npy"
        if os.path.exists(temp_file):
            print(f"Using single temperature field from {temp_file}...")
            try:
                T = np.load(temp_file)
                T_history = [T]
                print("✓ Loaded single temperature field")
            except Exception as e:
                print(f"Error loading {temp_file}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Neither {history_file} nor {temp_file} found.")
            print("Please run the solver first to generate temperature data.")
            sys.exit(1)

# Validate we have temperature data
if not T_history or len(T_history) == 0:
    print("Error: No temperature data available for animation")
    sys.exit(1)

print(f"\nCreating animation from {len(T_history)} time steps...")

# Find global min/max for consistent color scale
T_min = min(T.min() for T in T_history)
T_max = max(T.max() for T in T_history)
print(f"Temperature range: {T_min:.2f} to {T_max:.2f}")

# Generate frames
frames = []
for i, T in enumerate(T_history):
    # YZ slice at mid X index
    mid = T.shape[0] // 2
    slice2d = T[mid, :, :]  # YZ slice
    
    plt.figure(figsize=(8, 6))
    plt.imshow(slice2d, cmap="inferno", vmin=T_min, vmax=T_max, origin='lower')
    plt.title(f"YZ Slice – Step {i+1}/{len(T_history)}", fontsize=12)
    plt.xlabel("Y", fontsize=10)
    plt.ylabel("Z", fontsize=10)
    plt.colorbar(label="Temperature (°C)")
    plt.tight_layout()
    
    # Save to temporary file
    temp_file = "temp_frame.png"
    plt.savefig(temp_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Read frame and clean up
    try:
        # Use matplotlib's imread (works without imageio)
        frame = plt.imread(temp_file)
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
        os.remove(temp_file)
    except Exception as e:
        print(f"Warning: Error processing frame {i}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    if (i + 1) % max(1, len(T_history) // 10) == 0:
        print(f"  Processed {i+1}/{len(T_history)} frames...")

# Calculate mesh size for filename
nx, ny, nz = T_history[0].shape[0]-1, T_history[0].shape[1]-1, T_history[0].shape[2]-1
output_file = f"docs/animation/heat_diffusion_yz_{nx}x{ny}x{nz}.gif"

# Save GIF - check if imageio is available
if not IMAGEIO_AVAILABLE:
    print("\n❌ Error: imageio is required to create GIF animations.")
    print("Please install it with: pip install imageio imageio-ffmpeg")
    print("\nAttempting to install imageio automatically...")
    
    # Try to install imageio
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ imageio installed successfully")
            import imageio
            IMAGEIO_AVAILABLE = True
        else:
            print(f"⚠ Installation failed: {result.stderr}")
            print("\nPlease install manually: pip install imageio imageio-ffmpeg")
            sys.exit(1)
    except Exception as e:
        print(f"⚠ Could not install imageio automatically: {e}")
        print("Please install manually: pip install imageio imageio-ffmpeg")
        sys.exit(1)

print(f"\nSaving GIF to {output_file}...")
try:
    imageio.mimsave(
        output_file, 
        frames, 
        fps=5,
        loop=0  # Loop forever
    )
    
    # Verify file was created
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"✓ Saved animation: {output_file} ({file_size:.1f} KB)")
        print(f"✓ GIF is valid and ready for GitHub")
    else:
        print(f"⚠ Warning: {output_file} was not created or is empty")
        sys.exit(1)
        
except Exception as e:
    print(f"Error saving GIF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

