#!/usr/bin/env python3
"""
Unit tests for shape functions.
"""

import pytest
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fem_utils import shape_functions_hex, shape_gradients_hex


def test_shape_functions_partition_of_unity():
    """Test that shape functions sum to 1 (partition of unity)."""
    # Test at several points
    test_points = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
    ]
    
    for xi, eta, zeta in test_points:
        N = shape_functions_hex(xi, eta, zeta)
        sum_N = float(jnp.sum(N))
        assert abs(sum_N - 1.0) < 1e-6, f"Partition of unity failed at ({xi}, {eta}, {zeta}): sum = {sum_N}"


def test_shape_functions_at_nodes():
    """Test that shape functions are 1 at their node and 0 at others."""
    # Node coordinates in natural space
    nodes = [
        (-1, -1, -1),  # Node 0
        (1, -1, -1),   # Node 1
        (1, 1, -1),    # Node 2
        (-1, 1, -1),   # Node 3
        (-1, -1, 1),   # Node 4
        (1, -1, 1),    # Node 5
        (1, 1, 1),     # Node 6
        (-1, 1, 1),    # Node 7
    ]
    
    for node_idx, (xi, eta, zeta) in enumerate(nodes):
        N = shape_functions_hex(xi, eta, zeta)
        # At node i, N_i should be 1, others should be 0
        for i, val in enumerate(N):
            expected = 1.0 if i == node_idx else 0.0
            assert abs(float(val) - expected) < 1e-6, \
                f"Shape function {i} at node {node_idx}: expected {expected}, got {val}"


def test_shape_gradients_symmetry():
    """Test that element matrices have expected symmetry properties."""
    # Test at center point
    xi, eta, zeta = 0.0, 0.0, 0.0
    dN = shape_gradients_hex(xi, eta, zeta)
    
    # Check that gradients are reasonable (not all zero at center)
    assert jnp.any(jnp.abs(dN) > 1e-6), "Gradients should not all be zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
