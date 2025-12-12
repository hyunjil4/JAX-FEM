import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fem_solver import run_fem_explicit

os.makedirs("docs/benchmark", exist_ok=True)

mesh_sizes = [20, 40, 60, 80, 100]
times = []

for n in mesh_sizes:
    print(f"Running benchmark for mesh {n}^3...")
    start = time.time()
    run_fem_explicit(nx=n, ny=n, nz=n, steps=10)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)

plt.figure(figsize=(6,4))
plt.plot(mesh_sizes, times, marker='o')
plt.xlabel("Mesh size (N)")
plt.ylabel("Total runtime (ms)")
plt.title("FEM-JAX GPU Benchmark")
plt.grid(True)
plt.savefig("docs/benchmark/performance.png", dpi=150)
plt.close()

print("Benchmark saved to docs/benchmark/performance.png")

