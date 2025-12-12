#!/usr/bin/env python3
"""
Unit tests for element matrices.
"""

import pytest
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fem_utils import compute_element_matrices


def test_element_matrices_symmetry():
    """Test that element matrices are symmetric."""
    # Create a simple cubic element
    coords = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=jnp.float32)
    
    Ke, Me = compute_element_matrices(coords)
    
    # Check symmetry
    Ke_diff = jnp.abs(Ke - Ke.T)
    Me_diff = jnp.abs(Me - Me.T)
    
    assert jnp.max(Ke_diff) < 1e-5, f"Ke is not symmetric: max diff = {jnp.max(Ke_diff)}"
    assert jnp.max(Me_diff) < 1e-5, f"Me is not symmetric: max diff = {jnp.max(Me_diff)}"


def test_element_matrices_positive_definite():
    """Test that mass matrix is positive definite."""
    coords = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=jnp.float32)
    
    Ke, Me = compute_element_matrices(coords)
    
    # Check that all eigenvalues of Me are positive
    eigenvals_Me = jnp.linalg.eigvals(Me)
    assert jnp.all(eigenvals_Me > 0), f"Me should be positive definite, but has negative eigenvalues: {eigenvals_Me}"


def test_element_matrices_positive_semi_definite_stiffness():
    """Test that stiffness matrix is positive semi-definite."""
    coords = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=jnp.float32)
    
    Ke, Me = compute_element_matrices(coords)
    
    # Check that all eigenvalues of Ke are non-negative
    eigenvals_Ke = jnp.linalg.eigvals(Ke)
    assert jnp.all(eigenvals_Ke >= -1e-6), f"Ke should be positive semi-definite, but has negative eigenvalues: {eigenvals_Ke}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

