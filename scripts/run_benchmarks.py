#!/usr/bin/env python3
"""
Automated GPU Benchmark Workflow for JAX-FEM Heat Solver

Runs benchmarks across multiple mesh sizes, saves results to CSV,
generates plots, and updates README.md automatically.
"""

import os
import sys
import csv
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.solver import run_simulation

# Benchmark configuration
MESH_SIZES = [10, 20, 40, 60, 80, 100, 200]
STEPS = 50
OUTPUT_CSV = "docs/benchmark_results.csv"
OUTPUT_PLOT = "docs/benchmark_plot.png"
README_PATH = "README.md"


def run_benchmark(nx, ny, nz, steps=50):
    """
    Run a single benchmark for given mesh size.
    
    Returns:
        dict with keys: mesh_size, elements, nodes, assembly_ms, solve_ms, total_ms
        or None if simulation failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"Benchmarking mesh size: {nx}×{ny}×{nz}")
        print(f"{'='*60}")
        
        # Run simulation with auto-stable dt
        T, history = run_simulation(
            nx=nx, ny=ny, nz=nz,
            dt=None,  # Auto-computed for stability
            steps=steps,
            T_bottom=100.0,
            T_top=0.0,
            kappa=1.0,
            save_history=False,
            verbose=True
        )
        
        # Check for NaN/Inf
        if np.isnan(T).any() or np.isinf(T).any():
            print(f"⚠️  ERROR: NaN/Inf detected in results for {nx}×{ny}×{nz}")
            return None
        
        # Extract timing information
        timing = history['timing']
        mesh_size = nx  # Assuming cubic mesh
        
        result = {
            'mesh_size': mesh_size,
            'elements': timing['num_elements'],
            'nodes': timing['num_nodes'],
            'assembly_ms': timing['assembly_ms'],
            'solve_ms': timing['solve_ms'],
            'total_ms': timing['total_ms']
        }
        
        print(f"✓ Completed: {result['total_ms']:.2f} ms total")
        return result
        
    except RuntimeError as e:
        if "diverged" in str(e).lower() or "nan" in str(e).lower():
            print(f"⚠️  ERROR: Simulation diverged for {nx}×{ny}×{nz}: {e}")
            return None
        else:
            raise
    except Exception as e:
        print(f"⚠️  ERROR: Failed to benchmark {nx}×{ny}×{nz}: {e}")
        return None


def save_results_to_csv(results, csv_path):
    """Save benchmark results to CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mesh Size', 'Elements', 'Nodes', 'Assembly (ms)', 'Solve (ms)', 'Total (ms)'])
        
        for r in results:
            if r is not None:
                writer.writerow([
                    f"{r['mesh_size']}×{r['mesh_size']}×{r['mesh_size']}",
                    r['elements'],
                    r['nodes'],
                    f"{r['assembly_ms']:.2f}",
                    f"{r['solve_ms']:.2f}",
                    f"{r['total_ms']:.2f}"
                ])
    
    print(f"\n✓ Saved results to {csv_path}")


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Mesh Size':<15} {'Elements':<12} {'Nodes':<12} {'Assembly (ms)':<15} {'Solve (ms)':<15} {'Total (ms)':<15}")
    print("-"*80)
    
    for r in results:
        if r is not None:
            print(f"{r['mesh_size']}×{r['mesh_size']}×{r['mesh_size']:<8} "
                  f"{r['elements']:<12,} "
                  f"{r['nodes']:<12,} "
                  f"{r['assembly_ms']:<15.2f} "
                  f"{r['solve_ms']:<15.2f} "
                  f"{r['total_ms']:<15.2f}")
    
    print("="*80)


def generate_plot(results, plot_path):
    """Generate benchmark plot."""
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("⚠️  No valid results to plot")
        return
    
    mesh_sizes = [r['mesh_size'] for r in valid_results]
    total_times = [r['total_ms'] for r in valid_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(mesh_sizes, total_times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Mesh Size N (N×N×N)', fontsize=12)
    plt.ylabel('Total Runtime (ms)', fontsize=12)
    plt.title('FEM-JAX GPU Benchmark', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated plot: {plot_path}")


def update_readme_table(results, readme_path):
    """Update README.md with new benchmark table from results."""
    if not os.path.exists(readme_path):
        print(f"⚠️  README not found: {readme_path}")
        return
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Generate new table from results
    table_lines = [
        "| Mesh Size | Elements | Nodes | Assembly | Solve | Total |",
        "|-----------|----------|-------|----------|-------|-------|"
    ]
    
    for r in results:
        if r is not None:
            table_lines.append(
                f"| {r['mesh_size']}×{r['mesh_size']}×{r['mesh_size']} | "
                f"{r['elements']:,} | "
                f"{r['nodes']:,} | "
                f"{r['assembly_ms']:.2f} ms | "
                f"{r['solve_ms']:.2f} ms | "
                f"{r['total_ms']:.2f} ms |"
            )
    
    new_table = "\n".join(table_lines) + "\n"
    
    # Find and replace/insert table in Benchmark Results section
    # Pattern: ## Benchmark Results\n\n...![Performance Benchmark]...\n\n
    # We want to insert table after the image line
    
    # First, update plot path
    content = re.sub(
        r'!\[Performance Benchmark\]\(docs/benchmark/performance\.png\)',
        '![Performance Benchmark](docs/benchmark_plot.png)',
        content
    )
    
    # Try to find existing table and replace it
    table_pattern = r'(\| Mesh Size \|.*?\n\|-+\|.*?\n)((?:\|.*?\|.*?\n)+)'
    if re.search(table_pattern, content):
        # Replace existing table
        content = re.sub(
            table_pattern,
            lambda m: m.group(1) + '\n'.join(table_lines[2:]) + '\n',
            content,
            flags=re.MULTILINE | re.DOTALL
        )
    else:
        # Insert new table after the image line in Benchmark Results section
        # Pattern: ![Performance Benchmark](docs/benchmark_plot.png)\n\n*Left:...
        insert_pattern = r'(!\[Performance Benchmark\]\(docs/benchmark_plot\.png\)\n\n)'
        replacement = r'\1' + new_table
        if not re.search(insert_pattern, content):
            # Try alternative: after "Performance scaling analysis..."
            insert_pattern = r'(Performance scaling analysis across different mesh sizes:\n\n!\[Performance Benchmark\]\(docs/benchmark_plot\.png\)\n\n)'
            replacement = r'\1' + new_table
        content = re.sub(insert_pattern, replacement, content)
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {readme_path} with new benchmark table")


def main():
    """Main benchmark workflow."""
    print("="*80)
    print("JAX-FEM GPU BENCHMARK WORKFLOW")
    print("="*80)
    print(f"Mesh sizes to benchmark: {MESH_SIZES}")
    print(f"Time steps per run: {STEPS}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output plot: {OUTPUT_PLOT}")
    print("="*80)
    
    results = []
    
    # Run benchmarks
    for mesh_size in MESH_SIZES:
        result = run_benchmark(mesh_size, mesh_size, mesh_size, steps=STEPS)
        results.append(result)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("\n❌ ERROR: No successful benchmark runs!")
        sys.exit(1)
    
    # Print results table
    print_results_table(valid_results)
    
    # Save to CSV
    save_results_to_csv(valid_results, OUTPUT_CSV)
    
    # Generate plot
    generate_plot(valid_results, OUTPUT_PLOT)
    
    # Update README
    update_readme_table(valid_results, README_PATH)
    
    print("\n" + "="*80)
    print("✓ BENCHMARK WORKFLOW COMPLETED")
    print("="*80)
    print(f"✓ Results saved to: {OUTPUT_CSV}")
    print(f"✓ Plot saved to: {OUTPUT_PLOT}")
    print(f"✓ README updated: {README_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()
