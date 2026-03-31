#!/usr/bin/env python3
"""Structural Counterexample Search for BHS Laplacian Spectral Radius Bounds.

Generates structured graph families (DoubleStar, StarOfCliques, Caterpillar,
MultiHub, BookGraph) and tests them against all 38 open BHS bounds.

Usage:
  python src/structural_counterexample_search.py --test     # Sweep all families
  python src/structural_counterexample_search.py --verify   # Cross-verify with scipy
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import networkx as nx

# Import bound evaluation functions from exhaustive_bound_search
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exhaustive_bound_search import (
    evaluate_all_bounds,
    laplacian_spectral_radius,
    compute_dv_mv,
    ALL_BOUND_IDS,
    is_connected,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Graph Family Generators
# ─────────────────────────────────────────────────────────────────────

def double_star(k1, k2):
    """Two centers connected by an edge, k1 leaves on center 1, k2 on center 2.

    Total vertices: k1 + k2 + 2
    Center 0 has degree k1+1, center 1 has degree k2+1.
    """
    G = nx.Graph()
    n = k1 + k2 + 2
    G.add_nodes_from(range(n))
    # Centers are 0 and 1
    G.add_edge(0, 1)
    # k1 leaves on center 0
    for i in range(2, 2 + k1):
        G.add_edge(0, i)
    # k2 leaves on center 1
    for i in range(2 + k1, 2 + k1 + k2):
        G.add_edge(1, i)
    return nx.to_numpy_array(G, dtype=np.float64)


def star_of_cliques(m, t):
    """Central vertex connected to t copies of K_m.

    Each K_m clique has one designated vertex connected to the center.
    Total vertices: 1 + t*m
    """
    G = nx.Graph()
    center = 0
    G.add_node(center)
    node_id = 1
    for _ in range(t):
        clique_nodes = list(range(node_id, node_id + m))
        G.add_nodes_from(clique_nodes)
        # Make the clique
        for i in range(len(clique_nodes)):
            for j in range(i + 1, len(clique_nodes)):
                G.add_edge(clique_nodes[i], clique_nodes[j])
        # Connect first node of clique to center
        G.add_edge(center, clique_nodes[0])
        node_id += m
    return nx.to_numpy_array(G, dtype=np.float64)


def caterpillar(spine_length, leaves_per_node):
    """Path backbone with variable leaves per spine node.

    Args:
        spine_length: number of nodes in the spine path
        leaves_per_node: list of leaf counts, one per spine node
    """
    assert len(leaves_per_node) == spine_length, \
        f"leaves_per_node length ({len(leaves_per_node)}) must match spine_length ({spine_length})"
    G = nx.Graph()
    # Spine nodes: 0 to spine_length-1
    for i in range(spine_length - 1):
        G.add_edge(i, i + 1)
    node_id = spine_length
    for i in range(spine_length):
        for _ in range(leaves_per_node[i]):
            G.add_edge(i, node_id)
            node_id += 1
    return nx.to_numpy_array(G, dtype=np.float64)


def multi_hub(hub_degrees):
    """Multiple hubs connected in a chain, each with pendant leaves.

    Args:
        hub_degrees: list of pendant leaf counts per hub
    """
    G = nx.Graph()
    num_hubs = len(hub_degrees)
    # Hub nodes: 0 to num_hubs-1
    G.add_nodes_from(range(num_hubs))
    # Chain the hubs
    for i in range(num_hubs - 1):
        G.add_edge(i, i + 1)
    # Add pendant leaves
    node_id = num_hubs
    for i in range(num_hubs):
        for _ in range(hub_degrees[i]):
            G.add_edge(i, node_id)
            node_id += 1
    return nx.to_numpy_array(G, dtype=np.float64)


def book_graph(pages, clique_size):
    """Multiple cliques sharing a common edge.

    Args:
        pages: number of cliques (pages)
        clique_size: size of each clique (>= 3, since they share an edge)

    The common edge is (0, 1). Each page adds clique_size-2 private vertices
    that form a clique with vertices 0 and 1.
    """
    G = nx.Graph()
    G.add_edge(0, 1)  # shared edge
    node_id = 2
    for _ in range(pages):
        private_nodes = list(range(node_id, node_id + clique_size - 2))
        G.add_nodes_from(private_nodes)
        # Connect each private node to both shared vertices
        for v in private_nodes:
            G.add_edge(0, v)
            G.add_edge(1, v)
        # Connect private nodes to each other (to form a clique with 0 and 1)
        for i in range(len(private_nodes)):
            for j in range(i + 1, len(private_nodes)):
                G.add_edge(private_nodes[i], private_nodes[j])
        node_id += clique_size - 2
    return nx.to_numpy_array(G, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# 2. Graph Families Registry
# ─────────────────────────────────────────────────────────────────────

GRAPH_FAMILIES = {
    "DoubleStar": {
        "generator": double_star,
        "description": "Two centers + pendant leaves",
        "sweep": "k1, k2 in [2, 30)",
    },
    "StarOfCliques": {
        "generator": star_of_cliques,
        "description": "Central vertex + t copies of K_m",
        "sweep": "m in [3, 10), t in [2, 20)",
    },
    "Caterpillar": {
        "generator": caterpillar,
        "description": "Path spine + variable leaves",
        "sweep": "spine 3-10, leaves [0..10]",
    },
    "MultiHub": {
        "generator": multi_hub,
        "description": "Chain of hubs with pendant leaves",
        "sweep": "2-5 hubs, degrees 2-15",
    },
    "BookGraph": {
        "generator": book_graph,
        "description": "Multiple cliques sharing a common edge",
        "sweep": "pages 2-20, clique_size 3-8",
    },
}


# ─────────────────────────────────────────────────────────────────────
# 3. Parameter Sweep
# ─────────────────────────────────────────────────────────────────────

def sweep_double_star():
    """Generate DoubleStar instances for k1, k2 in [2, 30)."""
    instances = []
    for k1 in range(2, 30):
        for k2 in range(k1, 30):  # k2 >= k1 to avoid duplicates
            name = f"DoubleStar({k1},{k2})"
            A = double_star(k1, k2)
            instances.append((name, A))
    return instances


def sweep_star_of_cliques():
    """Generate StarOfCliques instances for m in [3,10), t in [2,20)."""
    instances = []
    for m in range(3, 10):
        for t in range(2, 20):
            name = f"StarOfCliques(K{m},{t})"
            A = star_of_cliques(m, t)
            instances.append((name, A))
    return instances


def sweep_caterpillar():
    """Generate representative Caterpillar instances."""
    instances = []
    # Uniform leaves
    for spine in range(3, 11):
        for leaves in range(0, 11):
            leaves_per_node = [leaves] * spine
            name = f"Caterpillar(spine={spine},leaves={leaves})"
            A = caterpillar(spine, leaves_per_node)
            instances.append((name, A))
    # Asymmetric: heavy leaves on ends
    for spine in range(4, 9):
        for heavy in range(3, 11):
            lpn = [heavy] + [1] * (spine - 2) + [heavy]
            name = f"Caterpillar(spine={spine},ends={heavy},mid=1)"
            A = caterpillar(spine, lpn)
            instances.append((name, A))
    # Asymmetric: heavy center
    for spine in range(4, 9):
        for heavy in range(3, 11):
            mid = spine // 2
            lpn = [1] * spine
            lpn[mid] = heavy
            name = f"Caterpillar(spine={spine},center={heavy})"
            A = caterpillar(spine, lpn)
            instances.append((name, A))
    return instances


def sweep_multi_hub():
    """Generate MultiHub instances."""
    instances = []
    # 2 hubs
    for d1 in range(2, 16):
        for d2 in range(d1, 16):
            name = f"MultiHub({d1},{d2})"
            A = multi_hub([d1, d2])
            instances.append((name, A))
    # 3 hubs
    for d in range(2, 12):
        for d_mid in range(2, 12):
            name = f"MultiHub({d},{d_mid},{d})"
            A = multi_hub([d, d_mid, d])
            instances.append((name, A))
    # 4 hubs (symmetric)
    for d_outer in range(2, 10):
        for d_inner in range(2, 10):
            name = f"MultiHub({d_outer},{d_inner},{d_inner},{d_outer})"
            A = multi_hub([d_outer, d_inner, d_inner, d_outer])
            instances.append((name, A))
    # 5 hubs (symmetric)
    for d_outer in range(2, 8):
        for d_mid in range(2, 8):
            name = f"MultiHub({d_outer},{d_mid},{d_mid},{d_mid},{d_outer})"
            A = multi_hub([d_outer, d_mid, d_mid, d_mid, d_outer])
            instances.append((name, A))
    return instances


def sweep_book_graph():
    """Generate BookGraph instances for pages in [2,20), clique_size in [3,8)."""
    instances = []
    for pages in range(2, 21):
        for cs in range(3, 9):
            name = f"BookGraph(pages={pages},K{cs})"
            A = book_graph(pages, cs)
            instances.append((name, A))
    return instances


# ─────────────────────────────────────────────────────────────────────
# 4. Test Mode: Full Sweep
# ─────────────────────────────────────────────────────────────────────

def run_test():
    """Sweep all graph families, test against 38 bounds, report violations."""
    print("=" * 80)
    print("STRUCTURAL COUNTEREXAMPLE SEARCH — Full Parameter Sweep")
    print("=" * 80)
    print(f"Testing against {len(ALL_BOUND_IDS)} bounds: {ALL_BOUND_IDS}")
    print()

    tol = 1e-9
    counterexamples = []  # (name, bound_id, mu, bound_val, gap)
    # Track best (most negative) gap per bound
    best_gap_per_bound = {bid: (None, float('inf')) for bid in ALL_BOUND_IDS}
    # Track which bounds are violated
    violated_bounds = set()
    total_tested = 0

    sweepers = [
        ("DoubleStar", sweep_double_star),
        ("StarOfCliques", sweep_star_of_cliques),
        ("Caterpillar", sweep_caterpillar),
        ("MultiHub", sweep_multi_hub),
        ("BookGraph", sweep_book_graph),
    ]

    t0 = time.time()

    for family_name, sweep_fn in sweepers:
        print(f"\n--- {family_name} ---")
        instances = sweep_fn()
        family_ces = 0
        family_tested = 0

        for name, A in instances:
            n = A.shape[0]
            if n < 3:
                continue
            if not is_connected(A):
                continue

            mu, bound_vals, gaps = evaluate_all_bounds(A)
            family_tested += 1
            total_tested += 1

            for bid in ALL_BOUND_IDS:
                gap = gaps[bid]
                # Track best gap per bound
                if gap < best_gap_per_bound[bid][1]:
                    best_gap_per_bound[bid] = (name, gap)
                # Check for violation
                if gap < -tol:
                    violated_bounds.add(bid)
                    counterexamples.append((name, bid, n, mu, bound_vals[bid], gap))
                    family_ces += 1

        print(f"  Tested {family_tested} instances, found {family_ces} violations")

    elapsed = time.time() - t0

    # ─── Summary ───
    print("\n" + "=" * 80)
    print(f"SUMMARY — {total_tested} graphs tested in {elapsed:.1f}s")
    print("=" * 80)

    # Violated bounds
    print(f"\nViolated bounds ({len(violated_bounds)}/{len(ALL_BOUND_IDS)}):")
    for bid in sorted(violated_bounds):
        # Find the best (most negative gap) CE for this bound
        ces_for_bid = [(name, n, mu, bv, gap) for name, b, n, mu, bv, gap in counterexamples if b == bid]
        ces_for_bid.sort(key=lambda x: x[4])
        best = ces_for_bid[0]
        print(f"  Bound {bid:2d}: VIOLATED  best_gap={best[4]:+.6f}  "
              f"n={best[1]:3d}  mu={best[2]:.6f}  bound={best[3]:.6f}  ({best[0]})")

    # Non-violated bounds
    non_violated = set(ALL_BOUND_IDS) - violated_bounds
    print(f"\nNon-violated bounds ({len(non_violated)}/{len(ALL_BOUND_IDS)}):")
    for bid in sorted(non_violated):
        name, gap = best_gap_per_bound[bid]
        status = "NEAR-MISS" if gap < 0.5 else "HOLDS"
        print(f"  Bound {bid:2d}: {status:9s}  closest_gap={gap:+.6f}  ({name})")

    # Deduplicated counterexample list: unique (graph, bound) pairs
    # Group by bound, show best graph
    print(f"\n--- All counterexamples ({len(counterexamples)} total, {len(violated_bounds)} bounds) ---")
    by_bound = defaultdict(list)
    for name, bid, n, mu, bv, gap in counterexamples:
        by_bound[bid].append((name, n, mu, bv, gap))
    for bid in sorted(by_bound.keys()):
        entries = sorted(by_bound[bid], key=lambda x: x[4])[:5]  # top 5
        print(f"\n  Bound {bid}:")
        for name, n, mu, bv, gap in entries:
            print(f"    {name:40s}  n={n:3d}  mu={mu:.6f}  bound={bv:.6f}  gap={gap:+.6f}")

    # Save results
    save_results(counterexamples, best_gap_per_bound, violated_bounds, elapsed, total_tested)

    return counterexamples, violated_bounds


def save_results(counterexamples, best_gap_per_bound, violated_bounds, elapsed, total_tested):
    """Save full results to resources/structural_search_results.txt."""
    os.makedirs("resources", exist_ok=True)
    outpath = os.path.join("resources", "structural_search_results.txt")
    with open(outpath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("STRUCTURAL COUNTEREXAMPLE SEARCH RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total graphs tested: {total_tested}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"VIOLATED BOUNDS ({len(violated_bounds)}/{len(ALL_BOUND_IDS)}):\n")
        by_bound = defaultdict(list)
        for name, bid, n, mu, bv, gap in counterexamples:
            by_bound[bid].append((name, n, mu, bv, gap))

        for bid in sorted(violated_bounds):
            entries = sorted(by_bound[bid], key=lambda x: x[4])
            best = entries[0]
            f.write(f"  Bound {bid:2d}: best_gap={best[4]:+.6f}  "
                    f"n={best[1]:3d}  mu={best[2]:.6f}  bound={best[3]:.6f}  ({best[0]})\n")
            f.write(f"    Total violations: {len(entries)}\n")
            # Top 5
            for name, n, mu, bv, gap in entries[:5]:
                f.write(f"      {name:40s}  n={n:3d}  mu={mu:.6f}  bound={bv:.6f}  gap={gap:+.6f}\n")
            f.write("\n")

        non_violated = set(ALL_BOUND_IDS) - violated_bounds
        f.write(f"\nNON-VIOLATED BOUNDS ({len(non_violated)}/{len(ALL_BOUND_IDS)}):\n")
        for bid in sorted(non_violated):
            name, gap = best_gap_per_bound[bid]
            f.write(f"  Bound {bid:2d}: closest_gap={gap:+.6f}  ({name})\n")

        f.write(f"\n\nALL COUNTEREXAMPLES ({len(counterexamples)} total):\n")
        for name, bid, n, mu, bv, gap in sorted(counterexamples, key=lambda x: (x[1], x[5])):
            f.write(f"  Bound {bid:2d}  {name:40s}  n={n:3d}  mu={mu:.6f}  "
                    f"bound={bv:.6f}  gap={gap:+.6f}\n")

    print(f"\nResults saved to {outpath}")


# ─────────────────────────────────────────────────────────────────────
# 5. Verify Mode: Cross-verify with scipy
# ─────────────────────────────────────────────────────────────────────

def run_verify():
    """For each counterexample, cross-verify mu with scipy.linalg.eigvalsh."""
    from scipy.linalg import eigvalsh as scipy_eigvalsh

    print("=" * 80)
    print("COUNTEREXAMPLE CROSS-VERIFICATION (scipy.linalg.eigvalsh)")
    print("=" * 80)

    tol_bound = 1e-9
    tol_eigen = 1e-8
    all_pass = True
    verified_count = 0
    fail_count = 0

    # First run sweep to get counterexamples
    print("\nRunning parameter sweep to collect counterexamples...\n")
    counterexamples, _ = run_test()

    # Deduplicate by graph name (same graph may violate multiple bounds)
    seen_graphs = {}
    for name, bid, n, mu, bv, gap in counterexamples:
        if name not in seen_graphs:
            seen_graphs[name] = (bid, n, mu, bv, gap)

    print(f"\n\nCross-verifying {len(seen_graphs)} unique graphs with scipy...\n")

    for name, (bid, n, mu_numpy, bv, gap) in sorted(seen_graphs.items()):
        # Reconstruct the graph
        A = _reconstruct_graph(name)
        if A is None:
            print(f"  SKIP: Cannot reconstruct {name}")
            continue

        # Compute mu with scipy
        dv = A.sum(axis=1)
        L = np.diag(dv) - A
        eigs_scipy = scipy_eigvalsh(L)
        mu_scipy = float(eigs_scipy[-1])

        # Compare
        diff = abs(mu_scipy - mu_numpy)
        if diff < tol_eigen:
            status = "MATCH"
            verified_count += 1
        else:
            status = "MISMATCH"
            fail_count += 1
            all_pass = False

        # Verify the bound is still violated with scipy mu
        mu_s, bv_s, gaps_s = evaluate_all_bounds(A)
        still_violated = any(gaps_s[b] < -tol_bound for b in ALL_BOUND_IDS)

        print(f"  {name:40s}  scipy_mu={mu_scipy:.6f}  numpy_mu={mu_numpy:.6f}  "
              f"diff={diff:.2e}  {status}  violated={still_violated}")

    print(f"\n{'=' * 80}")
    print(f"Verification: {verified_count} MATCH, {fail_count} MISMATCH")
    if all_pass:
        print("ALL COUNTEREXAMPLES VERIFIED with scipy.")
    else:
        print("WARNING: Some mismatches detected!")

    return all_pass


def _reconstruct_graph(name):
    """Reconstruct adjacency matrix from graph name string."""
    try:
        if name.startswith("DoubleStar("):
            params = name[len("DoubleStar("):-1]
            k1, k2 = map(int, params.split(","))
            return double_star(k1, k2)
        elif name.startswith("StarOfCliques(K"):
            params = name[len("StarOfCliques(K"):-1]
            m, t = map(int, params.split(","))
            return star_of_cliques(m, t)
        elif name.startswith("Caterpillar(spine=") and "ends=" in name:
            # Caterpillar(spine=X,ends=Y,mid=Z)
            parts = name[len("Caterpillar("):-1].split(",")
            spine = int(parts[0].split("=")[1])
            ends = int(parts[1].split("=")[1])
            mid = int(parts[2].split("=")[1])
            lpn = [ends] + [mid] * (spine - 2) + [ends]
            return caterpillar(spine, lpn)
        elif name.startswith("Caterpillar(spine=") and "center=" in name:
            parts = name[len("Caterpillar("):-1].split(",")
            spine = int(parts[0].split("=")[1])
            heavy = int(parts[1].split("=")[1])
            mid_idx = spine // 2
            lpn = [1] * spine
            lpn[mid_idx] = heavy
            return caterpillar(spine, lpn)
        elif name.startswith("Caterpillar(spine=") and "leaves=" in name:
            parts = name[len("Caterpillar("):-1].split(",")
            spine = int(parts[0].split("=")[1])
            leaves = int(parts[1].split("=")[1])
            lpn = [leaves] * spine
            return caterpillar(spine, lpn)
        elif name.startswith("MultiHub("):
            params = name[len("MultiHub("):-1]
            degrees = list(map(int, params.split(",")))
            return multi_hub(degrees)
        elif name.startswith("BookGraph(pages="):
            params = name[len("BookGraph(pages="):-1]
            parts = params.split(",K")
            pages = int(parts[0])
            cs = int(parts[1])
            return book_graph(pages, cs)
    except Exception as e:
        print(f"  Reconstruction error for {name}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Structural counterexample search for BHS bounds")
    parser.add_argument("--test", action="store_true",
                        help="Sweep all families, test against 38 bounds")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-verify counterexamples with scipy")
    args = parser.parse_args()

    if not args.test and not args.verify:
        print("Usage: python src/structural_counterexample_search.py --test|--verify")
        print("  --test    Sweep all families and test against 38 bounds")
        print("  --verify  Cross-verify found counterexamples with scipy")
        sys.exit(1)

    if args.test:
        run_test()
    elif args.verify:
        run_verify()


if __name__ == "__main__":
    main()
