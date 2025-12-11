#!/usr/bin/env python3
"""
Performance Benchmarking Script

Measures solver performance across multiple mesh sizes and generates
a performance plot for documentation.
"""

import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_simulation


def run_benchmark(mesh_sizes=None, output_file="docs/benchmark/performance.png"):
    """
    Run performance benchmark across multiple mesh sizes.
    
    Args:
        mesh_sizes: List of (nx, ny, nz) tuples. If None, uses default sizes.
        output_file: Path to save performance plot
    """
    if mesh_sizes is None:
        mesh_sizes = [
            (10, 10, 10),
            (20, 20, 20),
            (30, 30, 30),
            (40, 40, 40),
            (50, 50, 50),
        ]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'mesh_sizes': [],
        'elements': [],
        'nodes': [],
        'assembly_ms': [],
        'solve_ms': [],
        'total_ms': []
    }
    
    print("="*70)
    print("Performance Benchmark")
    print("="*70)
    print(f"{'Mesh':<15} {'Elements':<12} {'Nodes':<12} {'Assembly':<12} {'Solve':<12} {'Total':<12}")
    print("-"*70)
    
    for nx, ny, nz in mesh_sizes:
        print(f"Running {nx}×{ny}×{nz}...", end=' ', flush=True)
        
        T, history = run_simulation(
            nx=nx, ny=ny, nz=nz,
            dt=1e-6,
            steps=50,  # Reduced for faster benchmarking
            T_bottom=100.0,
            T_top=0.0,
            kappa=1.0,
            verbose=False
        )
        
        timing = history['timing']
        results['mesh_sizes'].append(f"{nx}×{ny}×{nz}")
        results['elements'].append(timing['num_elements'])
        results['nodes'].append(timing['num_nodes'])
        results['assembly_ms'].append(timing['assembly_ms'])
        results['solve_ms'].append(timing['solve_ms'])
        results['total_ms'].append(timing['total_ms'])
        
        print(f"✓ {timing['total_ms']:.1f} ms")
    
    # Create performance plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Time vs Mesh Size
    ax1 = axes[0]
    mesh_labels = [f"{nx}³" for nx, ny, nz in mesh_sizes]
    x_pos = np.arange(len(mesh_sizes))
    
    width = 0.25
    ax1.bar(x_pos - width, results['assembly_ms'], width, label='Assembly', alpha=0.8)
    ax1.bar(x_pos, results['solve_ms'], width, label='Solve', alpha=0.8)
    ax1.bar(x_pos + width, results['total_ms'], width, label='Total', alpha=0.8)
    
    ax1.set_xlabel('Mesh Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance vs Mesh Size')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mesh_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scaling analysis
    ax2 = axes[1]
    elements = np.array(results['elements'])
    total_times = np.array(results['total_ms'])
    
    ax2.loglog(elements, total_times, 'o-', linewidth=2, markersize=8, label='Total Time')
    ax2.loglog(elements, results['assembly_ms'], 's-', linewidth=2, markersize=8, label='Assembly')
    ax2.loglog(elements, results['solve_ms'], '^-', linewidth=2, markersize=8, label='Solve')
    
    # Add reference lines for scaling
    ref_elements = elements[0]
    ref_time = total_times[0]
    linear_scale = ref_time * (elements / ref_elements)
    quadratic_scale = ref_time * (elements / ref_elements) ** 1.5
    
    ax2.loglog(elements, linear_scale, '--', alpha=0.5, label='O(N) reference')
    ax2.loglog(elements, quadratic_scale, '--', alpha=0.5, label='O(N^1.5) reference')
    
    ax2.set_xlabel('Number of Elements')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Scaling Analysis (Log-Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Performance plot saved: {output_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("Benchmark Summary")
    print("="*70)
    print(f"{'Mesh':<15} {'Elements':<12} {'Nodes':<12} {'Assembly':<12} {'Solve':<12} {'Total':<12}")
    print("-"*70)
    for i, mesh_str in enumerate(results['mesh_sizes']):
        print(f"{mesh_str:<15} {results['elements'][i]:<12} {results['nodes'][i]:<12} "
              f"{results['assembly_ms'][i]:<10.1f} {results['solve_ms'][i]:<10.1f} "
              f"{results['total_ms'][i]:<10.1f}")
    print("="*70)
    
    plt.close()


def main():
    """Main function."""
    print("="*70)
    print("JAX-FEM Performance Benchmark")
    print("="*70)
    
    # Run benchmark
    run_benchmark()
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
