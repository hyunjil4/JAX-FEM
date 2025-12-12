#!/usr/bin/env python3
"""
3D Heat Transfer FEM Solver - Main Solver Module

Refactored solver with reusable functions for mesh generation,
assembly, and time stepping. Includes logging and history tracking.
"""

import sys
import time
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from .fem_utils import (
    generate_mesh,
    compute_element_matrices,
    assemble_lumped_mass,
    apply_K_global,
    apply_boundary_conditions
)

jax.config.update("jax_enable_x64", False)


# ============================================================
# Time Stepping Functions
# ============================================================
@jit
def explicit_step(T, elem_dofs, Ke, M_lump, dt, kappa, dir_nodes, T_bc_vals):
    """
    Single explicit time step.
    
    Implements: T^{n+1} = T^n - Δt · κ · (K·T^n) / M_lump
    
    Args:
        T: Current temperature field of shape (Nnodes,)
        elem_dofs: Element connectivity of shape (Ne, 8)
        Ke: Element stiffness matrix of shape (8, 8)
        M_lump: Lumped mass matrix of shape (Nnodes,)
        dt: Time step size
        kappa: Thermal conductivity
        dir_nodes: Indices of Dirichlet nodes
        T_bc_vals: Boundary condition values
    
    Returns:
        T_new: Updated temperature field of shape (Nnodes,)
    """
    KT = apply_K_global(T, elem_dofs, Ke)
    T_new = T - dt * kappa * (KT / M_lump)
    T_new = T_new.at[dir_nodes].set(T_bc_vals)
    return T_new


# ============================================================
# Main Simulation Function
# ============================================================
def run_simulation(nx=20, ny=20, nz=20,
                   dt=None, steps=500,
                   T_bottom=100.0, T_top=0.0,
                   kappa=1.0,
                   Lx=1.0, Ly=1.0, Lz=1.0,
                   save_history=False,
                   log_file=None,
                   verbose=True):
    """
    Run complete FEM simulation with optional logging and history tracking.
    
    Solves the transient heat conduction equation:
        ρc_p ∂T/∂t = ∇·(κ∇T)
    
    using explicit time integration:
        T^{n+1} = T^n - Δt · κ · (K·T^n) / M_lump
    
    The time step dt is automatically computed using the CFL stability condition
    to ensure numerical stability. If dt is provided, it will be overridden.
    
    Args:
        nx, ny, nz: Number of elements in each direction
        dt: Time step size (ignored - computed automatically for stability)
        steps: Number of time steps
        T_bottom: Temperature at bottom face (z=0)
        T_top: Temperature at top face (z=Lz)
        kappa: Thermal conductivity
        Lx, Ly, Lz: Domain dimensions
        save_history: If True, save temperature field at each time step
        log_file: Path to CSV file for logging (min/max T per step)
        verbose: Print progress information
    
    Returns:
        T_final: Final temperature field of shape (Nx, Ny, Nz)
        history: Dictionary with timing and temperature history
    """
    t_total_start = time.time()
    history = {
        'T_min': [],
        'T_max': [],
        'timing': {}
    }
    T_history = [] if save_history else None

    # Device selection
    gpus = jax.devices("gpu")
    device = gpus[0] if gpus else jax.devices("cpu")[0]
    
    if verbose:
        print(f"Device: {device}")

    with jax.default_device(device):
        # ----------------- Assembly -----------------
        t_asm_start = time.time()

        # Generate mesh
        coords_global, elem_dofs, Nx, Ny, Nz, Ne, Nnodes = generate_mesh(
            nx, ny, nz, Lx, Ly, Lz
        )

        # Reference element coordinates (for uniform mesh, all elements are identical)
        hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
        coords0 = jnp.array([
            [0.0, 0.0, 0.0],
            [hx,  0.0, 0.0],
            [hx,  hy,  0.0],
            [0.0, hy,  0.0],
            [0.0, 0.0, hz],
            [hx,  0.0, hz],
            [hx,  hy,  hz],
            [0.0, hy,  hz],
        ], dtype=jnp.float32)

        # Compute element matrices (only once for uniform mesh)
        Ke, Me = compute_element_matrices(coords0)

        # Assemble lumped mass matrix
        M_lump = assemble_lumped_mass(Me, elem_dofs, Nnodes)

        # Compute stable time step using CFL condition for 3D heat conduction
        hx = Lx / nx
        hy = Ly / ny
        hz = Lz / nz
        h_min = min(hx, hy, hz)
        
        # Stable time step for explicit scheme: dt < h_min^2 / (6 * kappa)
        dt_stable = (h_min ** 2) / (6 * kappa)
        
        # Use a safety factor (0.5) for guaranteed stability
        dt = 0.5 * dt_stable
        
        if verbose:
            print(f"Computed stable time step: dt = {dt:.6e} (h_min = {h_min:.6f}, dt_stable = {dt_stable:.6e})")

        # Initial condition: hot sphere at center
        # Start with T_top everywhere
        T = jnp.full(Nnodes, T_top, dtype=jnp.float32)
        
        # Sphere center
        cx, cy, cz = Lx / 2.0, Ly / 2.0, Lz / 2.0
        # Sphere radius = domain/6 (using average of domain dimensions)
        R = min(Lx, Ly, Lz) / 6.0
        
        # Calculate distance from each node to sphere center
        X = coords_global[:, 0]
        Y = coords_global[:, 1]
        Z = coords_global[:, 2]
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        dist_sq = dx * dx + dy * dy + dz * dz
        R_sq = R * R
        
        # Set T_bottom inside sphere
        inside_sphere = dist_sq < R_sq
        T = jnp.where(inside_sphere, T_bottom, T)

        # Apply boundary conditions (re-applies T_bottom at z=0 and T_top at z=Lz)
        T, dir_nodes, T_bc_vals = apply_boundary_conditions(
            T, coords_global, T_bottom, T_top, Lz
        )

        # Ensure JAX has compiled things
        jax.block_until_ready(T)
        assembly_ms = (time.time() - t_asm_start) * 1000.0

        if verbose:
            print(f"\nMesh: {nx}×{ny}×{nz} elements, {Nnodes} nodes")
            print(f"Assembly time: {assembly_ms:.2f} ms")

        # ----------------- Solve (time stepping) -----------------
        t_solve_start = time.time()

        # Warm up (JIT compilation)
        T = explicit_step(T, elem_dofs, Ke, M_lump, dt, kappa, dir_nodes, T_bc_vals)
        jax.block_until_ready(T)

        # Initialize logging
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'time', 'T_min', 'T_max'])

        # Time stepping loop
        for step in range(steps):
            T = explicit_step(T, elem_dofs, Ke, M_lump, dt, kappa, dir_nodes, T_bc_vals)
            
            # Check for numerical instability
            if jnp.isnan(T).any() or jnp.isinf(T).any():
                raise RuntimeError(f"Simulation diverged at step {step}: dt too large. Try reducing mesh size or increasing steps.")
            
            # Log temperature statistics
            Tmin = float(T.min())
            Tmax = float(T.max())
            history['T_min'].append(Tmin)
            history['T_max'].append(Tmax)
            
            if log_file:
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, step * dt, Tmin, Tmax])
            
            if save_history:
                T_history.append(T.copy())
            
            if verbose and (step + 1) % max(1, steps // 10) == 0:
                print(f"  Step {step+1}/{steps}: T_min={Tmin:.2f}, T_max={Tmax:.2f}")

        jax.block_until_ready(T)
        solve_ms = (time.time() - t_solve_start) * 1000.0
        total_ms = (time.time() - t_total_start) * 1000.0

        history['timing'] = {
            'assembly_ms': assembly_ms,
            'solve_ms': solve_ms,
            'total_ms': total_ms,
            'mesh_size': (nx, ny, nz),
            'num_elements': Ne,
            'num_nodes': Nnodes
        }

        Tmin = float(T.min())
        Tmax = float(T.max())

        if verbose:
            print("\n" + "="*60)
            print("[FEM EXPLICIT (Pure JAX)]")
            print(f"Mesh     : {nx} x {ny} x {nz}")
            print(f"Elements : {Ne:6d}")
            print(f"Nodes    : {Nnodes:6d}")
            print(f"Assembly : {assembly_ms:8.2f} ms")
            print(f"Solve    : {solve_ms:8.2f} ms")
            print(f"Total    : {total_ms:8.2f} ms")
            print("--------------------------------")
            print(f"Tmin = {Tmin:.4f}, Tmax = {Tmax:.4f}")
            print("="*60 + "\n")

        T_final = T.reshape(Nx, Ny, Nz)
        
        if save_history:
            history['T_history'] = [T.reshape(Nx, Ny, Nz) for T in T_history]

        # Save temperature.npy for visualization (always save to project root)
        np.save("temperature.npy", np.array(T_final))
        print("Saved temperature.npy")

        return T_final, history


# ============================================================
# Backward Compatibility
# ============================================================
def run_fem_explicit(nx=20, ny=20, nz=20,
                     dt=None, steps=500,
                     T_bottom=100.0, T_top=0.0,
                     kappa=1.0):
    """
    Backward compatibility wrapper for run_simulation.
    
    This function maintains the original API while using the new
    modular implementation. The dt parameter is ignored - time step
    is computed automatically for stability.
    """
    T, _ = run_simulation(
        nx=nx, ny=ny, nz=nz,
        dt=None,  # Ignore provided dt, compute automatically
        steps=steps,
        T_bottom=T_bottom, T_top=T_top,
        kappa=kappa,
        save_history=False,
        verbose=True
    )
    
    return T





# ============================================================
# CLI Entrypoint
# ============================================================
def main():
    """Main entry point for command-line usage."""
    nx = ny = nz = 20

    if len(sys.argv) >= 4:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        nz = int(sys.argv[3])
        print(f"\nUsing mesh size from CLI: {nx} {ny} {nz}")
    else:
        print("\nUsing default mesh size: 20 20 20")
        print("Usage: python -m src.solver [nx] [ny] [nz]")

    # Run solver (dt computed automatically for stability)
    T, history = run_simulation(
        nx=nx, ny=ny, nz=nz,
        dt=None,  # Computed automatically
        steps=500,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
        verbose=True
    )


if __name__ == "__main__":
    main()
