#!/usr/bin/env python3
"""
FEM Utility Functions

Core finite element operations: shape functions, element matrices,
mesh generation, and matrix operations.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple

jax.config.update("jax_enable_x64", False)


# ============================================================
# Gauss Points (2×2×2 for HEX8)
# ============================================================
_ga = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)], dtype=jnp.float32)
_gp = jnp.stack(jnp.meshgrid(_ga, _ga, _ga, indexing="ij"), axis=-1).reshape(-1, 3)


# ============================================================
# Shape Functions & Gradients (HEX8)
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
# Element Matrices
# ============================================================
def compute_element_matrices(coords):
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
        detJ = jnp.abs(jnp.linalg.det(J))
        invJ = jnp.linalg.inv(J)
        
        # B matrix: gradient operator in physical coordinates
        B = invJ @ dN                               # (3,8)

        # Stiffness matrix contribution: K_e = ∫ B^T B dV
        Ke += (B.T @ B) * detJ
        
        # Mass matrix contribution: M_e = ∫ N^T N dV
        Me += jnp.outer(N, N) * detJ

    return Ke, Me


# ============================================================
# Matrix-Free Operations
# ============================================================
@jit
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
# Mesh Generation
# ============================================================
def generate_mesh(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=1.0):
    """
    Generate structured 3D hexahedral mesh.
    
    Args:
        nx, ny, nz: Number of elements in each direction
        Lx, Ly, Lz: Domain dimensions
    
    Returns:
        coords_global: Node coordinates of shape (Nnodes, 3)
        elem_dofs: Element connectivity of shape (Ne, 8)
        Nx, Ny, Nz: Number of nodes in each direction
        Ne: Number of elements
        Nnodes: Total number of nodes
    """
    Nx, Ny, Nz = nx + 1, ny + 1, nz + 1
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz

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
    Ne = I.shape[0]

    base = (I * Ny * Nz + J * Nz + K).reshape(-1, 1)
    offsets = jnp.array([0,
                         Ny*Nz,
                         Ny*Nz + Nz,
                         Nz,
                         1,
                         Ny*Nz + 1,
                         Ny*Nz + Nz + 1,
                         Nz + 1], dtype=jnp.int32)
    elem_dofs = base + offsets.astype(jnp.int32)

    Nnodes = Nx * Ny * Nz

    return coords_global, elem_dofs, Nx, Ny, Nz, Ne, Nnodes


# ============================================================
# Boundary Conditions
# ============================================================
def apply_boundary_conditions(T, coords_global, T_bottom, T_top, Lz=1.0, eps=1e-6):
    """
    Apply Dirichlet boundary conditions at bottom and top faces.
    Side faces (x=0, x=Lx, y=0, y=Ly) have Neumann (zero flux) boundary conditions.
    
    Args:
        T: Temperature field of shape (Nnodes,)
        coords_global: Node coordinates of shape (Nnodes, 3)
        T_bottom: Temperature at bottom face (z=0)
        T_top: Temperature at top face (z=Lz)
        Lz: Domain height
        eps: Tolerance for boundary detection
    
    Returns:
        T: Temperature field with BCs applied
        dir_nodes: Indices of Dirichlet nodes
        T_bc_vals: Boundary condition values
    """
    zz = coords_global[:, 2]
    bottom = jnp.where(jnp.abs(zz - 0.0) < eps)[0]
    top = jnp.where(jnp.abs(zz - Lz) < eps)[0]
    dir_nodes = jnp.concatenate([bottom, top], axis=0)

    T_bc_vals = jnp.concatenate([
        jnp.full(bottom.shape[0], T_bottom, dtype=jnp.float32),
        jnp.full(top.shape[0], T_top, dtype=jnp.float32),
    ])

    T = T.at[bottom].set(T_bottom)
    T = T.at[top].set(T_top)

    return T, dir_nodes, T_bc_vals


# ============================================================
# Lumped Mass Matrix Assembly
# ============================================================
def assemble_lumped_mass(Me, elem_dofs, Nnodes):
    """
    Assemble global lumped mass matrix from element mass matrices.
    
    Args:
        Me: Element mass matrix of shape (8, 8)
        elem_dofs: Element connectivity of shape (Ne, 8)
        Nnodes: Total number of nodes
    
    Returns:
        M_lump: Lumped mass matrix (diagonal) of shape (Nnodes,)
    """
    m_e = Me.sum(axis=1)  # (8,)
    M_lump = jnp.zeros(Nnodes, dtype=jnp.float32)
    M_lump = M_lump.at[elem_dofs].add(m_e)  # scatter-add
    return M_lump

