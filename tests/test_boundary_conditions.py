#!/usr/bin/env python3
"""
Unit tests for boundary conditions.
"""

import pytest
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fem_utils import apply_boundary_conditions, generate_mesh
from src.solver import explicit_step, run_simulation


def test_boundary_conditions_preserved():
    """Test that Dirichlet boundary conditions are preserved during time stepping."""
    # Small mesh for fast testing
    nx, ny, nz = 5, 5, 5
    T_bottom, T_top = 100.0, 0.0
    
    # Generate mesh
    coords_global, elem_dofs, Nx, Ny, Nz, Ne, Nnodes = generate_mesh(nx, ny, nz)
    
    # Create initial temperature field
    Z = coords_global[:, 2]
    T = T_bottom + (T_top - T_bottom) * Z
    
    # Apply boundary conditions
    T, dir_nodes, T_bc_vals = apply_boundary_conditions(
        T, coords_global, T_bottom, T_top, Lz=1.0
    )
    
    # Verify BCs are applied
    bottom_nodes = jnp.where(jnp.abs(coords_global[:, 2] - 0.0) < 1e-6)[0]
    top_nodes = jnp.where(jnp.abs(coords_global[:, 2] - 1.0) < 1e-6)[0]
    
    T_bottom_actual = float(T[bottom_nodes[0]])
    T_top_actual = float(T[top_nodes[0]])
    
    assert abs(T_bottom_actual - T_bottom) < 1e-5, \
        f"Bottom BC not applied correctly: expected {T_bottom}, got {T_bottom_actual}"
    assert abs(T_top_actual - T_top) < 1e-5, \
        f"Top BC not applied correctly: expected {T_top}, got {T_top_actual}"


def test_boundary_conditions_after_time_step():
    """Test that BCs are preserved after a time step."""
    # Run a short simulation
    T_final, history = run_simulation(
        nx=5, ny=5, nz=5,
        dt=1e-6,
        steps=10,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
        verbose=False
    )
    
    # Check that BCs are still satisfied
    T_flat = T_final.flatten()
    Nx, Ny, Nz = T_final.shape
    
    # Bottom face (z=0)
    bottom_slice = T_final[:, :, 0]
    bottom_values = np.array(bottom_slice)
    
    # Top face (z=1)
    top_slice = T_final[:, :, -1]
    top_values = np.array(top_slice)
    
    # Check that bottom is approximately 100.0 and top is approximately 0.0
    # (allowing for some numerical error)
    assert np.allclose(bottom_values, 100.0, atol=1e-3), \
        "Bottom boundary condition not preserved"
    assert np.allclose(top_values, 0.0, atol=1e-3), \
        "Top boundary condition not preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

