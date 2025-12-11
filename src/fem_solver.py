#!/usr/bin/env python3
"""
3D Heat Transfer FEM Solver - Legacy CLI Entrypoint

This file maintains backward compatibility with the original API.
For new code, use src.solver.run_simulation() instead.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import run_fem_explicit, main

# Export for backward compatibility
__all__ = ['run_fem_explicit', 'main']

if __name__ == "__main__":
    main()


# ============================================================
# 1. Gauss Points (2×2×2 for HEX8)
# ============================================================
_ga = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)], dtype=jnp.float32)
_gp = jnp.stack(jnp.meshgrid(_ga, _ga, _ga, indexing="ij"), axis=-1).reshape(-1, 3)


# ============================================================
# 2. Shape Functions & Gradients (HEX8)
# ============================================================
def shape_functions_hex(xi, eta, zeta):
    """
    Compute shape functions for 8-node hexahedral element.
    
    Args:
        xi, eta, zeta: Natural coordinates in [-1, 1]
    
    Returns:
        Array of shape (8,) containing shape function values
    """
    return 0.125 * jnp.array([
        (1-xi)*(1-eta)*(1-zeta),
        (1+xi)*(1-eta)*(1-zeta),
        (1+xi)*(1+eta)*(1-zeta),
        (1-xi)*(1+eta)*(1-zeta),
        (1-xi)*(1-eta)*(1+zeta),
        (1+xi)*(1-eta)*(1+zeta),
        (1+xi)*(1+eta)*(1+zeta),
        (1-xi)*(1+eta)*(1+zeta),
    ], dtype=jnp.float32)


def shape_gradients_hex(xi, eta, zeta):
    """
    Compute shape function gradients in natural coordinates.
    
    Args:
        xi, eta, zeta: Natural coordinates in [-1, 1]
    
    Returns:
        Array of shape (3, 8) containing gradients [dN/dxi, dN/deta, dN/dzeta]
    """
    dN_dxi = 0.125 * jnp.array([
        -(1-eta)*(1-zeta),
         (1-eta)*(1-zeta),
         (1+eta)*(1-zeta),
        -(1+eta)*(1-zeta),
        -(1-eta)*(1+zeta),
         (1-eta)*(1+zeta),
         (1+eta)*(1+zeta),
        -(1+eta)*(1+zeta),
    ], dtype=jnp.float32)

    dN_deta = 0.125 * jnp.array([
        -(1-xi)*(1-zeta),
        -(1+xi)*(1-zeta),
         (1+xi)*(1-zeta),
         (1-xi)*(1-zeta),
        -(1-xi)*(1+zeta),
        -(1+xi)*(1+zeta),
         (1+xi)*(1+zeta),
         (1-xi)*(1+zeta),
    ], dtype=jnp.float32)

    dN_dzeta = 0.125 * jnp.array([
        -(1-xi)*(1-eta),
        -(1+xi)*(1-eta),
        -(1+xi)*(1+eta),
        -(1-xi)*(1+eta),
         (1-xi)*(1-eta),
         (1+xi)*(1-eta),
         (1+xi)*(1+eta),
         (1-xi)*(1+eta),
    ], dtype=jnp.float32)

    return jnp.stack([dN_dxi, dN_deta, dN_dzeta], axis=0)  # (3,8)


# ============================================================
# 3. Single Element Matrices (Ke, Me) via Gauss Integration
# ============================================================
def element_matrices(coords):
    """
    Compute element stiffness (Ke) and mass (Me) matrices via Gauss integration.
    
    Uses 2×2×2 Gauss quadrature (8 integration points) to compute:
    - Stiffness matrix: K_e^{ij} = κ ∫ ∇N_i · ∇N_j dΩ_e
    - Mass matrix: M_e^{ij} = ∫ N_i N_j dΩ_e
    
    Args:
        coords: Array of shape (8, 3) containing physical node coordinates
    
    Returns:
        Ke: Element stiffness matrix of shape (8, 8)
        Me: Element mass matrix of shape (8, 8)
    """
    Ke = jnp.zeros((8, 8), dtype=jnp.float32)
    Me = jnp.zeros((8, 8), dtype=jnp.float32)

    for xi, eta, zeta in _gp:
        N = shape_functions_hex(xi, eta, zeta)      # (8,)
        dN = shape_gradients_hex(xi, eta, zeta)     # (3,8)

        # Jacobian matrix
        J = dN @ coords                             # (3,3)
        detJ = jnp.linalg.det(J)
        invJ = jnp.linalg.inv(J)
        
        # B matrix: gradient operator in physical coordinates
        B = invJ @ dN                               # (3,8)

        # Stiffness matrix contribution: K_e = ∫ B^T B dV
        Ke += (B.T @ B) * detJ
        
        # Mass matrix contribution: M_e = ∫ N^T N dV
        Me += jnp.outer(N, N) * detJ

    return Ke, Me


# ============================================================
# 4. Matrix-Free Global K·T Application (Element-wise)
# ============================================================
@jax.jit
def apply_K_global(T, elem_dofs, Ke):
    """
    Apply global stiffness matrix to temperature field (matrix-free).
    
    This function computes K·T without assembling the full global matrix,
    making it memory-efficient and GPU-friendly.
    
    Args:
        T: Temperature field of shape (Nnodes,)
        elem_dofs: Element connectivity array of shape (Ne, 8)
        Ke: Element stiffness matrix of shape (8, 8)
    
    Returns:
        Result of K·T operation, shape (Nnodes,)
    """
    T_e = T[elem_dofs]                 # (Ne,8)
    KeTe = T_e @ Ke.T                  # (Ne,8)
    out = jnp.zeros_like(T)
    out = out.at[elem_dofs].add(KeTe)  # scatter-add
    return out


# ============================================================
# 5. Time Stepping (Explicit FEM)
# ============================================================
def run_fem_explicit(nx=20, ny=20, nz=20,
                     dt=1e-6, steps=100,
                     T_bottom=100.0, T_top=0.0,
                     kappa=1.0):
    """
    Run explicit time-stepping FEM solver for 3D heat equation.
    
    Solves the transient heat conduction equation:
        ρc_p ∂T/∂t = ∇·(κ∇T)
    
    using explicit time integration:
        T^{n+1} = T^n - Δt · κ · (K·T^n) / M_lump
    
    Args:
        nx, ny, nz: Number of elements in each direction
        dt: Time step size
        steps: Number of time steps
        T_bottom: Temperature at bottom face (z=0)
        T_top: Temperature at top face (z=Lz)
        kappa: Thermal conductivity
    
    Returns:
        T: Final temperature field of shape (Nx, Ny, Nz)
    """
    t_total_start = time.time()

    # ----------------- Assembly -----------------
    t_asm_start = time.time()

    Nx, Ny, Nz = nx + 1, ny + 1, nz + 1
    Lx = Ly = Lz = 1.0
    hx = Lx / nx
    hy = Ly / ny
    hz = Lz / nz

    # Global node coordinates
    x = jnp.linspace(0.0, Lx, Nx, dtype=jnp.float32)
    y = jnp.linspace(0.0, Ly, Ny, dtype=jnp.float32)
    z = jnp.linspace(0.0, Lz, Nz, dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    coords_global = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # (Nnodes,3)

    # Element DOF connectivity for structured grid
    I, J, K = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij")
    I = I.reshape(-1)
    J = J.reshape(-1)
    K = K.reshape(-1)
    Ne = I.shape[0]  # Number of elements

    # Compute number of nodes
    Nnodes = Nx * Ny * Nz

    base = (I * Ny * Nz + J * Nz + K).reshape(-1, 1)
    offsets = jnp.array([0,
                         Ny*Nz,
                         Ny*Nz + Nz,
                         Nz,
                         1,
                         Ny*Nz + 1,
                         Ny*Nz + Nz + 1,
                         Nz + 1], dtype=jnp.int32)
    elem_dofs = base + offsets            # (Ne,8), int32
    elem_dofs = elem_dofs.astype(jnp.int32)

    # One reference element coords (first cell)
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

    # Element matrices (only once, reused for all elements)
    Ke, Me = element_matrices(coords0)

    # Lumped mass: global diagonal from Me
    m_e = Me.sum(axis=1)                   # (8,)
    M_lump = jnp.zeros(Nx * Ny * Nz, dtype=jnp.float32)
    M_lump = M_lump.at[elem_dofs].add(m_e)  # scatter-add

    # Initial condition: linear from bottom to top
    T = T_bottom + (T_top - T_bottom) * (Z.reshape(-1) / Lz)

    # Dirichlet BC at bottom & top
    zz = coords_global[:, 2]
    eps = 1e-6
    bottom = jnp.where(jnp.abs(zz - 0.0) < eps)[0]
    top    = jnp.where(jnp.abs(zz - Lz) < eps)[0]
    dir_nodes = jnp.concatenate([bottom, top], axis=0)

    T_bc_vals = jnp.concatenate([
        jnp.full(bottom.shape[0], T_bottom, dtype=jnp.float32),
        jnp.full(top.shape[0],    T_top,    dtype=jnp.float32),
    ])

    # Apply initial BC
    T = T.at[bottom].set(T_bottom)
    T = T.at[top].set(T_top)

    # Ensure JAX has compiled things
    jax.block_until_ready(T)
    assembly_ms = (time.time() - t_asm_start) * 1000.0

    # ----------------- Solve (time stepping) -----------------
    t_solve_start = time.time()

    @jax.jit
    def time_step(T):
        """
        Single explicit time step.
        
        Implements: T^{n+1} = T^n - Δt · κ · (K·T^n) / M_lump
        
        Args:
            T: Current temperature field of shape (Nnodes,)
        
        Returns:
            T_new: Updated temperature field of shape (Nnodes,)
        """
        # Internal nodes update
        KT = apply_K_global(T, elem_dofs, Ke)
        T_new = T - dt * kappa * (KT / M_lump)

        # Re-apply Dirichlet BC
        T_new = T_new.at[dir_nodes].set(T_bc_vals)
        return T_new

    # Warm up (JIT compilation)
    T = time_step(T)
    jax.block_until_ready(T)

    # Actual time steps
    for _ in range(steps):
        T = time_step(T)

    jax.block_until_ready(T)
    solve_ms = (time.time() - t_solve_start) * 1000.0
    total_ms = (time.time() - t_total_start) * 1000.0

    Tmin = float(T.min())
    Tmax = float(T.max())

    print("\n[FEM EXPLICIT (Pure JAX)]")
    print(f"Mesh     : {nx} x {ny} x {nz}")
    print(f"Elements : {Ne:6d}")
    print(f"Nodes    : {Nnodes:6d}")
    print(f"Assembly : {assembly_ms:8.2f} ms")
    print(f"Solve    : {solve_ms:8.2f} ms")
    print(f"Total    : {total_ms:8.2f} ms")
    print("--------------------------------")
    print(f"Tmin = {Tmin:.4f}, Tmax = {Tmax:.4f}")
    print("--------------------------------\n")

    return T.reshape(Nx, Ny, Nz)


# ============================================================
# 6. CLI Entrypoint
# ============================================================
def main():
    """Main entry point for command-line usage."""
    # Default mesh
    nx = ny = nz = 20

    if len(sys.argv) == 4:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        nz = int(sys.argv[3])
        print(f"\nUsing mesh size from CLI: {nx} {ny} {nz}")
    else:
        print("\nUsing default mesh size: 20 20 20")
        print("Usage: python fem_solver.py [nx] [ny] [nz]")

    # Run solver
    # Note: dt should be chosen based on h^2 for stability
    T = run_fem_explicit(
        nx=nx, ny=ny, nz=nz,
        dt=1e-6,
        steps=100,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
    )


if __name__ == "__main__":
    main()
