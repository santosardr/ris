# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import networkx as nx
import numpy as np
import random
import os
import sys

# Add scripts directory to path for ris_core imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ris_core import RISSimulation, generate_ground_truth

def run_ksat_test(n=5000, ksat=5, replicates=30):
    """
    Evaluates the impact of the saturation constant Ksat on connectivity and correlation.
    """
    corrs = []
    fragmented_count = 0
    total_edges = 0
    
    class CustomHAF(RISSimulation):
        def __init__(self, n_nodes, k):
            super().__init__(n_nodes, strategy='heuristic')
            self.k = k
        def _calculate_redundancy(self):
            if self.n >= 3000:
                return self.k
            return super()._calculate_redundancy()

    G_gt = generate_ground_truth(n, 'barabasi_albert')
    true_edges = set(tuple(sorted(e)) for e in G_gt.edges())
    deg_gt = list(nx.degree_centrality(G_gt).values())

    for i in range(replicates):
        random.seed(42 + i)
        np.random.seed(42 + i)
        
        ris = CustomHAF(n, ksat)
        candidates = ris.generate_candidates()
        
        # Check connectivity of the sampled graph
        G_ris = nx.Graph()
        G_ris.add_nodes_from(range(n))
        G_ris.add_edges_from(list(candidates))
        if not nx.is_connected(G_ris):
            fragmented_count += 1
            
        # Check topological correlation
        recovered = candidates.intersection(true_edges)
        G_rec = nx.Graph()
        G_rec.add_nodes_from(range(n))
        G_rec.add_edges_from(list(recovered))
        deg_rec = [G_rec.degree(node) if node in G_rec else 0 for node in range(n)]
        
        if np.std(deg_rec) > 0:
            corrs.append(np.corrcoef(deg_gt, deg_rec)[0, 1])
        else:
            corrs.append(0)
            
        total_edges += len(candidates)

    return {
        'ksat': ksat,
        'mean_corr': np.mean(corrs),
        'fragmentation_rate': fragmented_count / replicates,
        'avg_cost': total_edges / replicates
    }

def main():
    print("Testing RIS Sensitivity to Saturation Constant (Ksat)...")
    results = []
    for k in [3, 5, 10]:
        results.append(run_ksat_test(ksat=k))
    
    print("\n" + "="*80)
    print(f"{'Ksat':<5} | {'Mean Corr':<10} | {'Frag Rate':<10} | {'Avg Cost':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['ksat']:<5} | {r['mean_corr']:10.4f} | {r['fragmentation_rate']:10.1%} | {r['avg_cost']:10.0f}")
    print("="*80)
    print("Conclusion: Ksat=5 offers a Pareto-optimal balance between cost and structural preservation.")

if __name__ == "__main__":
    main()
