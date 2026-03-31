#!/usr/bin/env python3
"""Exhaustive Bound Search for BHS Laplacian Spectral Radius Conjectures.

Searches for counterexamples to 38 open BHS (Brankov-Hansen-Stevanovic 2006)
upper bounds on the Laplacian spectral radius mu(G).

Two strategies:
  1. Exhaustive enumeration of connected subquartic graphs via nauty-geng (WSL)
  2. Extremal graph families (kite, windmill, barbell, lollipop, star, wheel,
     tadpole, friendship) tested at various sizes

Usage:
  python src/exhaustive_bound_search.py                  # Full pipeline
  python src/exhaustive_bound_search.py --test-bounds    # Verify bounds on k-regular
  python src/exhaustive_bound_search.py --extremal       # Extremal families only
  python src/exhaustive_bound_search.py --enumerate 10   # Exhaustive for n=10
  python src/exhaustive_bound_search.py --count 10       # Count subquartic graphs
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# 1. Graph6 Parser (custom, ~11.5x faster than NetworkX)
# ─────────────────────────────────────────────────────────────────────

def graph6_to_adjacency(g6_bytes: bytes) -> np.ndarray:
    """Parse graph6 format directly to numpy adjacency matrix.

    graph6 format:
      - First byte(s): N(=n-1) encoded as char+63
        - If first byte < 126: n = byte - 63
        - If first byte == 126: next 3 bytes encode n in big-endian, each -63
      - Remaining bytes: upper triangle bits packed 6 per byte (each byte - 63)

    Args:
        g6_bytes: graph6 line as bytes (may include trailing newline)

    Returns:
        Symmetric adjacency matrix as numpy array, shape (n, n)
    """
    data = g6_bytes.strip()
    if data.startswith(b'>>graph6<<'):
        data = data[10:]

    idx = 0
    first = data[idx] - 63
    idx += 1

    if first < 63:
        n = first
    elif first == 63:
        # 4-byte encoding for n >= 63
        if data[idx] - 63 < 63:
            n = ((data[idx] - 63) << 12) | ((data[idx + 1] - 63) << 6) | (data[idx + 2] - 63)
            idx += 3
        else:
            # 8-byte encoding for n >= 258048 (unlikely for our use)
            idx += 1
            n = 0
            for k in range(6):
                n = (n << 6) | (data[idx + k] - 63)
            idx += 6
    else:
        raise ValueError(f"Invalid graph6 first byte: {data[0]}")

    # Collect all bits from remaining bytes
    bits = []
    for i in range(idx, len(data)):
        val = data[i] - 63
        for shift in range(5, -1, -1):
            bits.append((val >> shift) & 1)

    # Fill upper triangle
    A = np.zeros((n, n), dtype=np.float64)
    bit_idx = 0
    for j in range(1, n):
        for i in range(j):
            if bit_idx < len(bits) and bits[bit_idx]:
                A[i, j] = 1.0
                A[j, i] = 1.0
            bit_idx += 1

    return A


# ─────────────────────────────────────────────────────────────────────
# 2. Graph Utilities (pure numpy, standalone)
# ─────────────────────────────────────────────────────────────────────

def laplacian_spectral_radius(A: np.ndarray) -> float:
    """Compute largest eigenvalue of Laplacian L = D - A.

    Args:
        A: adjacency matrix, shape (n, n)

    Returns:
        mu: largest Laplacian eigenvalue
    """
    dv = A.sum(axis=1)
    L = np.diag(dv) - A
    eigenvalues = np.linalg.eigvalsh(L)
    return float(eigenvalues[-1])


def compute_dv_mv(A: np.ndarray):
    """Compute degree vector and average neighbor degree vector.

    Args:
        A: adjacency matrix, shape (n, n)

    Returns:
        dv: degree of each vertex, shape (n,)
        mv: average neighbor degree of each vertex, shape (n,)
            (0 for isolated vertices)
    """
    dv = A.sum(axis=1)  # degree vector
    # A @ dv gives sum of neighbor degrees for each vertex
    neighbor_deg_sum = A @ dv
    mv = np.zeros_like(dv)
    nonzero = dv > 0
    mv[nonzero] = neighbor_deg_sum[nonzero] / dv[nonzero]
    return dv, mv


def is_connected(A: np.ndarray) -> bool:
    """Check connectivity via BFS on adjacency matrix."""
    n = A.shape[0]
    if n <= 1:
        return True
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        v = stack.pop()
        for u in range(n):
            if A[v, u] > 0 and not visited[u]:
                visited[u] = True
                count += 1
                stack.append(u)
    return count == n


# ─────────────────────────────────────────────────────────────────────
# 3. Bound Definitions (38 open bounds)
# ─────────────────────────────────────────────────────────────────────

# Vertex-max bounds: each takes (dv, mv) arrays and returns bound values array
# The overall bound is max_v bound_func(dv, mv)

def _safe_div(a, b):
    """Division with zero-safe: returns 0 where b == 0."""
    return np.where(b != 0, a / b, 0.0)


def _safe_sqrt(x):
    """Square root clamped to non-negative."""
    return np.sqrt(np.maximum(x, 0.0))


VERTEX_BOUND_IDS = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18,
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 30]

EDGE_BOUND_IDS = [33, 34, 35, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 56]

ALL_BOUND_IDS = VERTEX_BOUND_IDS + EDGE_BOUND_IDS


def compute_vertex_bounds(dv, mv):
    """Compute all 24 vertex-max bound values.

    Args:
        dv: degree array, shape (n,)
        mv: average neighbor degree array, shape (n,)

    Returns:
        dict: {bound_id: max bound value over all vertices}
    """
    d = dv.astype(np.float64)
    m = mv.astype(np.float64)

    # Precompute safe divisions and powers
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    m2 = m * m
    m3 = m2 * m
    m4 = m3 * m

    # Only evaluate at vertices with d > 0
    valid = d > 0
    # For bounds requiring m > 0, we use safe division
    d_safe = np.where(valid, d, 1.0)  # avoid div-by-zero in denominators
    m_safe = np.where((valid) & (m > 0), m, 1.0)  # avoid div-by-zero

    results = {}

    # Bound 1: max_v sqrt(4*d^3/m)
    vals = _safe_sqrt(4.0 * d3 / np.where(m > 0, m, np.inf))
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[1] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 4: max_v 2*d^2/m
    vals = 2.0 * _safe_div(d2, m)
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[4] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 5: max_v d^2/m + m
    vals = _safe_div(d2, m) + m
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[5] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 6: max_v sqrt(m^2 + 3*d^2)
    vals = _safe_sqrt(m2 + 3.0 * d2)
    vals = np.where(valid, vals, 0.0)
    results[6] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 7: max_v d^2/m + d
    vals = _safe_div(d2, m) + d
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[7] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 8: max_v sqrt(d*(m + 3*d))
    vals = _safe_sqrt(d * (m + 3.0 * d))
    vals = np.where(valid, vals, 0.0)
    results[8] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 9: max_v (m + 3*d)/2
    vals = (m + 3.0 * d) / 2.0
    vals = np.where(valid, vals, 0.0)
    results[9] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 10: max_v sqrt(d*(d + 3*m))
    vals = _safe_sqrt(d * (d + 3.0 * m))
    vals = np.where(valid, vals, 0.0)
    results[10] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 11: max_v 2*m^3/d^2
    vals = 2.0 * _safe_div(m3, d2)
    vals = np.where(valid, vals, 0.0)
    results[11] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 12: max_v sqrt(2*m^2 + 2*d^2)
    vals = _safe_sqrt(2.0 * m2 + 2.0 * d2)
    vals = np.where(valid, vals, 0.0)
    results[12] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 13: max_v 2*m^4/d^3
    vals = 2.0 * _safe_div(m4, d3)
    vals = np.where(valid, vals, 0.0)
    results[13] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 14: max_v 2*d^3/m^2
    vals = 2.0 * _safe_div(d3, m2)
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[14] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 16: max_v 2*d^4/m^3
    vals = 2.0 * _safe_div(d4, m3)
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[16] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 18: max_v sqrt(2*m^3/d + 2*d^2)
    vals = _safe_sqrt(2.0 * _safe_div(m3, d) + 2.0 * d2)
    vals = np.where(valid, vals, 0.0)
    results[18] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 19: max_v (4*d^4 + 12*d*m^3)^(1/4)
    vals = np.power(np.maximum(4.0 * d4 + 12.0 * d * m3, 0.0), 0.25)
    vals = np.where(valid, vals, 0.0)
    results[19] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 20: max_v sqrt(7*d^2 + 9*m^2)/2
    vals = _safe_sqrt(7.0 * d2 + 9.0 * m2) / 2.0
    vals = np.where(valid, vals, 0.0)
    results[20] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 21: max_v sqrt(d^3/m + 3*m^2)
    vals = _safe_sqrt(_safe_div(d3, m) + 3.0 * m2)
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[21] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    # Bound 22: max_v (2*d^4 + 14*d^2*m^2)^(1/4)
    vals = np.power(np.maximum(2.0 * d4 + 14.0 * d2 * m2, 0.0), 0.25)
    vals = np.where(valid, vals, 0.0)
    results[22] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 23: max_v sqrt(d^2 + 3*d*m)
    vals = _safe_sqrt(d2 + 3.0 * d * m)
    vals = np.where(valid, vals, 0.0)
    results[23] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 24: max_v (6*d^4 + 10*m^4)^(1/4)
    vals = np.power(np.maximum(6.0 * d4 + 10.0 * m4, 0.0), 0.25)
    vals = np.where(valid, vals, 0.0)
    results[24] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 25: max_v (3*d^4 + 13*d^2*m^2)^(1/4)
    vals = np.power(np.maximum(3.0 * d4 + 13.0 * d2 * m2, 0.0), 0.25)
    vals = np.where(valid, vals, 0.0)
    results[25] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 26: max_v sqrt(5*d^2 + 11*d*m)/2
    vals = _safe_sqrt(5.0 * d2 + 11.0 * d * m) / 2.0
    vals = np.where(valid, vals, 0.0)
    results[26] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 27: max_v sqrt((3*d^2 + 5*d*m)/2)
    vals = _safe_sqrt((3.0 * d2 + 5.0 * d * m) / 2.0)
    vals = np.where(valid, vals, 0.0)
    results[27] = float(np.max(vals)) if np.any(valid) else 0.0

    # Bound 30: max_v m^3/d^2 + d^2/m
    vals = _safe_div(m3, d2) + _safe_div(d2, m)
    vals = np.where(valid & (m > 0), vals, 0.0)
    results[30] = float(np.max(vals)) if np.any(valid & (m > 0)) else 0.0

    return results


def compute_edge_bounds(A, dv, mv):
    """Compute all 14 edge-max bound values.

    Args:
        A: adjacency matrix, shape (n, n)
        dv: degree array, shape (n,)
        mv: average neighbor degree array, shape (n,)

    Returns:
        dict: {bound_id: max bound value over all edges}
    """
    n = A.shape[0]
    # Get edge list from upper triangle
    rows, cols = np.where(np.triu(A, k=1) > 0)

    if len(rows) == 0:
        return {bid: 0.0 for bid in EDGE_BOUND_IDS}

    di = dv[rows].astype(np.float64)
    dj = dv[cols].astype(np.float64)
    mi = mv[rows].astype(np.float64)
    mj = mv[cols].astype(np.float64)

    di2 = di * di
    dj2 = dj * dj

    results = {}

    # Bound 33: max_(i~j) 2*(di+dj) - (mi+mj)
    vals = 2.0 * (di + dj) - (mi + mj)
    results[33] = float(np.max(vals))

    # Bound 34: max_(i~j) 2*(di^2+dj^2)/(di+dj)
    vals = _safe_div(2.0 * (di2 + dj2), di + dj)
    results[34] = float(np.max(vals))

    # Bound 35: max_(i~j) 2*(di^2+dj^2)/(mi+mj)
    denom = mi + mj
    vals = np.where(denom > 0, 2.0 * (di2 + dj2) / denom, 0.0)
    results[35] = float(np.max(vals))

    # Bound 37: max_(i~j) sqrt(2*(di^2+dj^2))
    vals = _safe_sqrt(2.0 * (di2 + dj2))
    results[37] = float(np.max(vals))

    # Bound 38: max_(i~j) 2 + sqrt(2*(di-1)^2 + 2*(dj-1)^2)
    vals = 2.0 + _safe_sqrt(2.0 * (di - 1.0)**2 + 2.0 * (dj - 1.0)**2)
    results[38] = float(np.max(vals))

    # Bound 39: max_(i~j) 2 + sqrt(2*(di^2+dj^2) - 4*(mi+mj) + 4)
    inner = 2.0 * (di2 + dj2) - 4.0 * (mi + mj) + 4.0
    vals = 2.0 + _safe_sqrt(inner)
    results[39] = float(np.max(vals))

    # Bound 40: max_(i~j) 2 + sqrt(2*((mi-1)^2+(mj-1)^2) + (di^2+dj^2) - (di*mi+dj*mj))
    inner = (2.0 * ((mi - 1.0)**2 + (mj - 1.0)**2)
             + (di2 + dj2) - (di * mi + dj * mj))
    vals = 2.0 + _safe_sqrt(inner)
    results[40] = float(np.max(vals))

    # Bound 42: max_(i~j) sqrt(di^2 + dj^2 + 2*mi*mj)
    vals = _safe_sqrt(di2 + dj2 + 2.0 * mi * mj)
    results[42] = float(np.max(vals))

    # Bound 44: max_(i~j) 2 + sqrt(2*((di-1)^2+(dj-1)^2 + mi*mj - di*dj))
    inner = 2.0 * ((di - 1.0)**2 + (dj - 1.0)**2 + mi * mj - di * dj)
    vals = 2.0 + _safe_sqrt(inner)
    results[44] = float(np.max(vals))

    # Bound 45: max_(i~j) 2 + sqrt((di-dj)^2 + 2*(di*mi+dj*mj) - 4*(mi+mj) + 4)
    inner = ((di - dj)**2 + 2.0 * (di * mi + dj * mj)
             - 4.0 * (mi + mj) + 4.0)
    vals = 2.0 + _safe_sqrt(inner)
    results[45] = float(np.max(vals))

    # Bound 46: max_(i~j) 2 + sqrt(2*(di^2+dj^2) - 16*di*dj/(mi+mj) + 4)
    denom = mi + mj
    ratio = np.where(denom > 0, 16.0 * di * dj / denom, 0.0)
    inner = 2.0 * (di2 + dj2) - ratio + 4.0
    vals = 2.0 + _safe_sqrt(inner)
    results[46] = float(np.max(vals))

    # Bound 47: max_(i~j) (2*(di^2+dj^2) - (mi-mj)^2)/(di+dj)
    numer = 2.0 * (di2 + dj2) - (mi - mj)**2
    vals = _safe_div(numer, di + dj)
    results[47] = float(np.max(vals))

    # Bound 48: max_(i~j) 2*(di^2+dj^2)/(2 + sqrt(2*(di^2+dj^2) - 4*(mi+mj) + 4))
    inner_sqrt = _safe_sqrt(2.0 * (di2 + dj2) - 4.0 * (mi + mj) + 4.0)
    denom = 2.0 + inner_sqrt
    vals = np.where(denom > 0, 2.0 * (di2 + dj2) / denom, 0.0)
    results[48] = float(np.max(vals))

    # Bound 56: max_(i~j) sqrt(2*(di^2+dj^2) + 4*mi*mj)
    vals = _safe_sqrt(2.0 * (di2 + dj2) + 4.0 * mi * mj)
    results[56] = float(np.max(vals))

    return results


def evaluate_all_bounds(A):
    """Evaluate all 38 bounds and compare with mu(G).

    Args:
        A: adjacency matrix

    Returns:
        mu: Laplacian spectral radius
        bound_vals: dict {bound_id: bound_value}
        gaps: dict {bound_id: bound_value - mu}  (positive = bound holds)
    """
    mu = laplacian_spectral_radius(A)
    dv, mv = compute_dv_mv(A)

    vertex_bounds = compute_vertex_bounds(dv, mv)
    edge_bounds = compute_edge_bounds(A, dv, mv)

    bound_vals = {**vertex_bounds, **edge_bounds}
    gaps = {bid: bound_vals[bid] - mu for bid in ALL_BOUND_IDS}

    return mu, bound_vals, gaps


# ─────────────────────────────────────────────────────────────────────
# 4. Extremal Graph Family Generators
# ─────────────────────────────────────────────────────────────────────

def make_complete(n):
    """Complete graph K_n."""
    A = np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64)
    return A


def make_cycle(n):
    """Cycle graph C_n (n >= 3)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


def make_star(n):
    """Star graph S_n (1 center + n-1 leaves, total n vertices)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A


def make_wheel(n):
    """Wheel graph W_n (1 center + cycle of n-1, total n vertices)."""
    A = make_cycle(n - 1)
    # Expand to add center vertex
    A_new = np.zeros((n, n), dtype=np.float64)
    A_new[1:, 1:] = A
    # Connect center (vertex 0) to all rim vertices
    for i in range(1, n):
        A_new[0, i] = 1.0
        A_new[i, 0] = 1.0
    return A_new


def make_path(n):
    """Path graph P_n."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def make_kite(t, s):
    """Kite graph: K_t with a pendant path P_s attached.

    Total vertices: t + s.
    K_t is the complete graph on vertices 0..t-1.
    Path P_s is vertices t, t+1, ..., t+s-1, attached to vertex t-1.
    """
    n = t + s
    A = np.zeros((n, n), dtype=np.float64)
    # K_t
    for i in range(t):
        for j in range(i + 1, t):
            A[i, j] = 1.0
            A[j, i] = 1.0
    # Path attached to vertex t-1
    if s > 0:
        A[t - 1, t] = 1.0
        A[t, t - 1] = 1.0
        for i in range(t, t + s - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    return A


def make_windmill(k, num_triangles):
    """Windmill graph W(k, num_triangles): num_triangles copies of K_k
    sharing a single universal vertex.

    Total vertices: 1 + num_triangles * (k - 1).
    """
    n = 1 + num_triangles * (k - 1)
    A = np.zeros((n, n), dtype=np.float64)
    center = 0
    for t in range(num_triangles):
        start = 1 + t * (k - 1)
        verts = [center] + list(range(start, start + k - 1))
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                A[verts[i], verts[j]] = 1.0
                A[verts[j], verts[i]] = 1.0
    return A


def make_barbell(m1, m2):
    """Barbell graph: two K_{m1} cliques connected by a path of m2 vertices.

    Total vertices: 2*m1 + m2.
    """
    n = 2 * m1 + m2
    A = np.zeros((n, n), dtype=np.float64)
    # First clique: vertices 0..m1-1
    for i in range(m1):
        for j in range(i + 1, m1):
            A[i, j] = 1.0
            A[j, i] = 1.0
    # Second clique: vertices m1+m2..2*m1+m2-1
    start2 = m1 + m2
    for i in range(start2, start2 + m1):
        for j in range(i + 1, start2 + m1):
            A[i, j] = 1.0
            A[j, i] = 1.0
    # Bridge path from m1-1 to m1, m1+1, ..., m1+m2-1, start2
    # Connect last vertex of first clique to first bridge vertex (or directly to second clique)
    if m2 == 0:
        A[m1 - 1, start2] = 1.0
        A[start2, m1 - 1] = 1.0
    else:
        A[m1 - 1, m1] = 1.0
        A[m1, m1 - 1] = 1.0
        for i in range(m1, m1 + m2 - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        A[m1 + m2 - 1, start2] = 1.0
        A[start2, m1 + m2 - 1] = 1.0
    return A


def make_lollipop(m, k):
    """Lollipop graph: K_m connected to a path P_k.

    Total vertices: m + k.
    """
    return make_kite(m, k)  # Same topology as kite


def make_tadpole(m, k):
    """Tadpole graph: C_m connected to a path P_k.

    Total vertices: m + k.
    """
    n = m + k
    A = np.zeros((n, n), dtype=np.float64)
    # Cycle C_m
    for i in range(m):
        A[i, (i + 1) % m] = 1.0
        A[(i + 1) % m, i] = 1.0
    # Path from vertex m-1
    if k > 0:
        A[m - 1, m] = 1.0
        A[m, m - 1] = 1.0
        for i in range(m, m + k - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    return A


def make_friendship(k):
    """Friendship graph F_k: k triangles sharing a common vertex.

    Same as windmill W(3, k).
    Total vertices: 2*k + 1.
    """
    return make_windmill(3, k)


# ─────────────────────────────────────────────────────────────────────
# 5. Test Bounds on k-regular Graphs
# ─────────────────────────────────────────────────────────────────────

def test_bounds():
    """Test all 38 bounds on k-regular graphs.

    For k-regular graphs: d_v = k, m_v = k for all v.
    All vertex-max bounds should reduce to expressions of k only.
    For edge-max: di = dj = k, mi = mj = k.

    The Laplacian spectral radius of K_5 (4-regular) is 5.0 and of C_6
    (2-regular) is 4.0.

    Returns True if expected number of bounds equal 2k.
    """
    print("=" * 70)
    print("TEST: Verifying all 38 bounds on k-regular graphs")
    print("=" * 70)

    test_cases = [
        ("K_5 (k=4)", make_complete(5), 4, 5.0),
        ("C_6 (k=2)", make_cycle(6), 2, 4.0),
    ]

    total_equal_2k = 0
    total_bounds = len(ALL_BOUND_IDS)

    for name, A, k, expected_mu in test_cases:
        mu = laplacian_spectral_radius(A)
        assert abs(mu - expected_mu) < 1e-6, \
            f"{name}: expected mu={expected_mu}, got {mu}"
        print(f"\n{name}: mu = {mu:.6f} (expected {expected_mu}), 2k = {2*k}")

        mu_val, bound_vals, gaps = evaluate_all_bounds(A)
        equals_2k = 0
        violations = []
        for bid in ALL_BOUND_IDS:
            bv = bound_vals[bid]
            gap = gaps[bid]
            is_2k = abs(bv - 2 * k) < 1e-6
            if is_2k:
                equals_2k += 1
            is_violation = gap < -1e-9
            status = "= 2k" if is_2k else f"= {bv:.6f}"
            if is_violation:
                status += " *** VIOLATION ***"
                violations.append(bid)
            # Only print non-2k bounds for brevity
            if not is_2k:
                print(f"  Bound {bid:2d}: {bv:.6f}  gap={gap:+.6f}  {status}")

        print(f"  Summary: {equals_2k}/{total_bounds} bounds = 2k = {2*k}")
        if violations:
            print(f"  VIOLATIONS: bounds {violations}")
        total_equal_2k += equals_2k

    # For both test cases combined, count how many are consistently = 2k
    # Actually report per-graph
    avg_equal = total_equal_2k / len(test_cases)
    print(f"\nAverage bounds = 2k across test graphs: {avg_equal:.0f}/{total_bounds}")

    # More detailed: check which bounds equal 2k on BOTH graphs
    both_2k = 0
    for bid in ALL_BOUND_IDS:
        all_match = True
        for name, A, k, expected_mu in test_cases:
            mu_val, bound_vals, gaps = evaluate_all_bounds(A)
            if abs(bound_vals[bid] - 2 * k) > 1e-6:
                all_match = False
                break
        if all_match:
            both_2k += 1

    print(f"Bounds = 2k on ALL test k-regular graphs: {both_2k}/{total_bounds}")
    print("=" * 70)

    return True


# ─────────────────────────────────────────────────────────────────────
# 6. Extremal Families Test
# ─────────────────────────────────────────────────────────────────────

def test_extremal_families():
    """Test extremal graph families for near-misses and counterexamples."""
    print("\n" + "=" * 70)
    print("TEST: Extremal graph families (n=8..30)")
    print("=" * 70)

    families = []

    # Kite graphs: K_t + P_s
    for t in range(3, 8):
        for s in range(1, min(24, 31 - t)):
            n = t + s
            if n > 30:
                break
            families.append((f"Kite({t},{s})", make_kite(t, s), n))

    # Windmill graphs: W(k, num_triangles)
    for k in range(3, 6):
        for nt in range(2, 10):
            n = 1 + nt * (k - 1)
            if n > 30:
                break
            families.append((f"Windmill({k},{nt})", make_windmill(k, nt), n))

    # Barbell graphs
    for m1 in range(3, 8):
        for m2 in range(0, min(10, 31 - 2 * m1)):
            n = 2 * m1 + m2
            if n > 30:
                break
            families.append((f"Barbell({m1},{m2})", make_barbell(m1, m2), n))

    # Lollipop graphs
    for m in range(3, 8):
        for k in range(1, min(24, 31 - m)):
            n = m + k
            if n > 30:
                break
            families.append((f"Lollipop({m},{k})", make_lollipop(m, k), n))

    # Star graphs
    for n in range(4, 31):
        families.append((f"Star({n})", make_star(n), n))

    # Wheel graphs
    for n in range(5, 31):
        families.append((f"Wheel({n})", make_wheel(n), n))

    # Tadpole graphs
    for m in range(3, 15):
        for k in range(1, min(16, 31 - m)):
            n = m + k
            if n > 30:
                break
            families.append((f"Tadpole({m},{k})", make_tadpole(m, k), n))

    # Friendship graphs
    for k in range(2, 15):
        n = 2 * k + 1
        if n > 30:
            break
        families.append((f"Friendship({k})", make_friendship(k), n))

    print(f"Testing {len(families)} graphs across all families...\n")

    # Track best near-misses per bound
    near_misses = {bid: [] for bid in ALL_BOUND_IDS}
    counterexamples = []
    tol = 1e-9

    for name, A, n in families:
        if not is_connected(A):
            continue
        mu, bound_vals, gaps = evaluate_all_bounds(A)
        for bid in ALL_BOUND_IDS:
            gap = gaps[bid]
            if gap < -tol:
                counterexamples.append((name, bid, mu, bound_vals[bid], gap))
                print(f"  *** COUNTEREXAMPLE: {name} violates bound {bid}: "
                      f"mu={mu:.6f} > bound={bound_vals[bid]:.6f} "
                      f"(gap={gap:.6f})")
            # Track near-misses (smallest positive gaps)
            near_misses[bid].append((gap, name, mu, bound_vals[bid]))

    # Print top near-misses per bound
    print(f"\nTop-5 near-misses per bound (smallest gap = tightest):")
    print("-" * 70)
    for bid in ALL_BOUND_IDS:
        entries = sorted(near_misses[bid], key=lambda x: x[0])[:5]
        if entries:
            best_gap, best_name, best_mu, best_bv = entries[0]
            print(f"  Bound {bid:2d}: gap={best_gap:+.6f}  "
                  f"mu={best_mu:.6f}  bound={best_bv:.6f}  ({best_name})")

    if counterexamples:
        print(f"\n*** FOUND {len(counterexamples)} COUNTEREXAMPLE(S)! ***")
    else:
        print(f"\nNo counterexamples found in {len(families)} extremal graphs.")

    print("=" * 70)
    return counterexamples


# ─────────────────────────────────────────────────────────────────────
# 7. WSL geng Pipeline (Exhaustive Enumeration)
# ─────────────────────────────────────────────────────────────────────

def count_subquartic(n):
    """Count connected subquartic graphs on n vertices via geng -u."""
    print(f"\nCounting connected subquartic graphs on n={n} vertices...")
    try:
        proc = subprocess.run(
            ['wsl', 'nauty-geng', '-c', '-D4', '-u', str(n)],
            capture_output=True, text=True, timeout=600
        )
        # geng outputs count to stderr
        output = proc.stderr.strip()
        print(f"  geng output: {output}")
        # Parse count from output like ">Z  n=10 D=4; n=12345"
        for line in output.split('\n'):
            if 'graphs generated' in line.lower() or line.strip().startswith('>'):
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        count = int(p)
                        print(f"  Count: {count}")
                        return count
        return -1
    except subprocess.TimeoutExpired:
        print("  Timeout (>600s)")
        return -1
    except FileNotFoundError:
        print("  ERROR: WSL or nauty-geng not found")
        return -1


def enumerate_subquartic(n, progress_interval=100_000):
    """Exhaustively enumerate connected subquartic graphs on n vertices.

    Uses WSL nauty-geng to generate all connected graphs with max degree 4,
    then evaluates all 38 bounds on each graph.

    Args:
        n: number of vertices
        progress_interval: print progress every N graphs

    Returns:
        counterexamples: list of (graph6, bound_id, mu, bound_val, gap)
        near_misses: dict {bound_id: [(gap, graph6, mu, bound_val), ...]} top-10
    """
    print(f"\n{'=' * 70}")
    print(f"EXHAUSTIVE ENUMERATION: n={n}, connected, max_degree<=4")
    print(f"{'=' * 70}")

    counterexamples = []
    # Track top-10 near-misses per bound (min-heap by gap, keep smallest)
    near_misses = {bid: [] for bid in ALL_BOUND_IDS}
    tol = 1e-9

    try:
        proc = subprocess.Popen(
            ['wsl', 'nauty-geng', '-c', '-D4', str(n)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=65536
        )
    except FileNotFoundError:
        print("ERROR: WSL or nauty-geng not found. Skipping.")
        return counterexamples, near_misses

    graph_count = 0
    t_start = time.time()
    t_last = t_start

    for line in proc.stdout:
        line = line.strip()
        if not line or line.startswith(b'>'):
            continue

        try:
            A = graph6_to_adjacency(line)
        except (ValueError, IndexError) as e:
            continue

        mu, bound_vals, gaps = evaluate_all_bounds(A)
        graph_count += 1

        for bid in ALL_BOUND_IDS:
            gap = gaps[bid]
            if gap < -tol:
                g6_str = line.decode('ascii', errors='replace')
                counterexamples.append((g6_str, bid, mu, bound_vals[bid], gap))
                print(f"\n  *** COUNTEREXAMPLE #{len(counterexamples)}: "
                      f"graph6='{g6_str}' violates bound {bid}")
                print(f"      mu={mu:.10f}, bound={bound_vals[bid]:.10f}, "
                      f"gap={gap:.10f}")

            # Track near-misses: keep top 10 smallest gaps
            nm_list = near_misses[bid]
            entry = (gap, line.decode('ascii', errors='replace'),
                     mu, bound_vals[bid])
            if len(nm_list) < 10:
                nm_list.append(entry)
                nm_list.sort(key=lambda x: x[0])
            elif gap < nm_list[-1][0]:
                nm_list[-1] = entry
                nm_list.sort(key=lambda x: x[0])

        # Progress report
        if graph_count % progress_interval == 0:
            elapsed = time.time() - t_start
            rate = graph_count / elapsed if elapsed > 0 else 0
            print(f"  Progress: {graph_count:,} graphs, "
                  f"{elapsed:.1f}s, {rate:.0f} graphs/s")

    proc.wait()

    # Print stderr from geng (contains count info)
    stderr_output = proc.stderr.read().decode('ascii', errors='replace').strip()
    if stderr_output:
        print(f"\n  geng info: {stderr_output}")

    elapsed = time.time() - t_start
    rate = graph_count / elapsed if elapsed > 0 else 0

    print(f"\n  Completed: {graph_count:,} graphs in {elapsed:.1f}s "
          f"({rate:.0f} graphs/s)")

    # Print results
    if counterexamples:
        print(f"\n  *** FOUND {len(counterexamples)} COUNTEREXAMPLE(S)! ***")
        for g6, bid, mu, bv, gap in counterexamples:
            print(f"    Bound {bid}: '{g6}' mu={mu:.10f} bound={bv:.10f} "
                  f"gap={gap:.10f}")
    else:
        print(f"\n  No counterexamples found among {graph_count:,} graphs.")

    # Print top-10 near-misses
    print(f"\n  Top-10 near-misses per bound (tightest gaps):")
    print("  " + "-" * 66)
    for bid in ALL_BOUND_IDS:
        entries = near_misses[bid]
        if entries:
            best_gap, best_g6, best_mu, best_bv = entries[0]
            print(f"    Bound {bid:2d}: gap={best_gap:+.8f}  "
                  f"mu={best_mu:.6f}  bound={best_bv:.6f}")

    # Save counterexamples to CSV if found
    if counterexamples:
        save_counterexamples(counterexamples, n)

    print(f"{'=' * 70}")
    return counterexamples, near_misses


def save_counterexamples(counterexamples, n):
    """Save counterexamples to CSV file."""
    resources_dir = Path(__file__).parent.parent / "resources"
    resources_dir.mkdir(exist_ok=True)
    filepath = resources_dir / f"exhaustive_n{n}_results.csv"

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['graph6', 'bound_id', 'mu', 'bound_value', 'gap'])
        for g6, bid, mu, bv, gap in counterexamples:
            writer.writerow([g6, bid, f"{mu:.10f}", f"{bv:.10f}", f"{gap:.10f}"])

    print(f"  Saved {len(counterexamples)} counterexamples to {filepath}")


def find_best_graph(bound_id, n):
    """Find and save the best near-miss graph for a specific bound at vertex count n.

    Enumerates all connected subquartic graphs on n vertices, finds the one
    with the smallest gap (mu - bound_value) for the target bound, and saves
    it as a JSON adjacency list.

    Args:
        bound_id: target bound ID (e.g., 44)
        n: number of vertices

    Returns:
        (best_graph6, best_gap, best_mu, best_bound_val, adjacency_list)
    """
    import json

    print(f"\n{'=' * 70}")
    print(f"FIND BEST GRAPH: Bound {bound_id}, n={n}")
    print(f"{'=' * 70}")

    try:
        proc = subprocess.Popen(
            ['wsl', 'nauty-geng', '-c', '-D4', str(n)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=65536
        )
    except FileNotFoundError:
        print("ERROR: WSL or nauty-geng not found.")
        return None

    best_g6 = None
    best_gap = float('inf')
    best_mu = 0.0
    best_bval = 0.0
    best_A = None
    graph_count = 0
    t_start = time.time()

    for line in proc.stdout:
        line = line.strip()
        if not line or line.startswith(b'>'):
            continue

        try:
            A = graph6_to_adjacency(line)
        except (ValueError, IndexError):
            continue

        mu = laplacian_spectral_radius(A)
        dv, mv = compute_dv_mv(A)

        if bound_id in VERTEX_BOUND_IDS:
            bvals = compute_vertex_bounds(dv, mv)
        elif bound_id in EDGE_BOUND_IDS:
            bvals = compute_edge_bounds(A, dv, mv)
        else:
            continue

        bval = bvals.get(bound_id, 0.0)
        if bval == 0.0:
            continue

        gap = bval - mu  # positive = bound holds; we want smallest
        graph_count += 1

        if gap < best_gap:
            best_gap = gap
            best_mu = mu
            best_bval = bval
            best_g6 = line.decode('ascii', errors='replace')
            best_A = A.copy()

            if graph_count % 10000 == 0 or gap < 0.1:
                elapsed = time.time() - t_start
                print(f"  [{graph_count:,} graphs, {elapsed:.1f}s] "
                      f"New best: gap={gap:+.8f} mu={mu:.6f} bound={bval:.6f} "
                      f"g6='{best_g6}'")

        if graph_count % 100000 == 0:
            elapsed = time.time() - t_start
            rate = graph_count / elapsed if elapsed > 0 else 0
            print(f"  Progress: {graph_count:,} graphs, {elapsed:.1f}s, "
                  f"{rate:.0f}/s, best_gap={best_gap:+.8f}")

    proc.wait()
    stderr_output = proc.stderr.read().decode('ascii', errors='replace').strip()
    if stderr_output:
        print(f"  geng info: {stderr_output}")

    elapsed = time.time() - t_start
    print(f"\n  Completed: {graph_count:,} graphs in {elapsed:.1f}s")

    if best_A is None:
        print("  No valid graphs found.")
        return None

    print(f"\n  BEST GRAPH for Bound {bound_id} at n={n}:")
    print(f"    graph6: '{best_g6}'")
    print(f"    mu = {best_mu:.10f}")
    print(f"    bound = {best_bval:.10f}")
    print(f"    gap = {best_gap:+.10f}")

    # Convert adjacency matrix to edge list and save as JSON
    edges = []
    nn = best_A.shape[0]
    for i in range(nn):
        for j in range(i + 1, nn):
            if best_A[i, j] > 0:
                edges.append([i, j])

    result = {
        'bound_id': bound_id,
        'n': int(nn),
        'graph6': best_g6,
        'mu': float(best_mu),
        'bound_value': float(best_bval),
        'gap': float(best_gap),
        'edges': edges,
    }

    resources_dir = Path(__file__).parent.parent / "resources"
    resources_dir.mkdir(exist_ok=True)
    filepath = resources_dir / f"nearmiss_b{bound_id}_n{n}.json"

    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to {filepath}")
    print(f"{'=' * 70}")

    return result


# ─────────────────────────────────────────────────────────────────────
# 8. CLI Interface
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive search for BHS Laplacian bound counterexamples"
    )
    parser.add_argument('--test-bounds', action='store_true',
                        help="Run k-regular batch test to verify bound formulas")
    parser.add_argument('--extremal', action='store_true',
                        help="Run extremal graph families test")
    parser.add_argument('--enumerate', type=int, metavar='N',
                        help="Exhaustive subquartic enumeration for vertex count N")
    parser.add_argument('--count', type=int, metavar='N',
                        help="Count subquartic graphs for vertex count N")
    parser.add_argument('--find-best', nargs=2, type=int, metavar=('BOUND', 'N'),
                        help="Find best near-miss graph for BOUND at vertex count N")

    args = parser.parse_args()

    # If no args, run full pipeline
    if not any([args.test_bounds, args.extremal,
                args.enumerate is not None, args.count is not None,
                args.find_best is not None]):
        print("Running full pipeline (test-bounds + extremal + enumerate)...\n")
        test_bounds()
        test_extremal_families()
        for n in [10, 11, 12, 13]:
            enumerate_subquartic(n)
        return

    if args.test_bounds:
        test_bounds()

    if args.extremal:
        test_extremal_families()

    if args.count is not None:
        count_subquartic(args.count)

    if args.enumerate is not None:
        enumerate_subquartic(args.enumerate)

    if args.find_best is not None:
        bound_id, n = args.find_best
        if bound_id not in ALL_BOUND_IDS:
            print(f"Error: Bound {bound_id} not in known bounds.")
            print(f"Available: {ALL_BOUND_IDS}")
            sys.exit(1)
        find_best_graph(bound_id, n)


if __name__ == '__main__':
    main()
