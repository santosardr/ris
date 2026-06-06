# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import numpy as np
import networkx as nx
import pandas as pd
from ris_core import RISSimulation, generate_ground_truth

class CustomRISSimulation(RISSimulation):
    def _calculate_redundancy(self):
        """Extended strategy logic for validation experiments."""
        if self.strategy == 'constant_1pct':
            return int(self.n * 0.01)
        elif self.strategy.startswith('constant_k'):
            # Format: constant_k250 -> returns 250
            try:
                return int(self.strategy.replace('constant_k', ''))
            except ValueError:
                return 5
        else:
            return super()._calculate_redundancy()

def calculate_segmented_metrics(n, topology, strategy='heuristic'):
    # 1. Generate Ground Truth
    G_gt = generate_ground_truth(n, topology)
    true_degrees = dict(nx.degree(G_gt))
    
    # 2. Run RIS with Custom Simulation
    ris = CustomRISSimulation(n, strategy)
    candidate_edges = ris.generate_candidates()
    
    # 3. Build Recovered Graph
    G_rec = nx.Graph()
    G_rec.add_nodes_from(range(n))
    G_rec.add_edges_from(list(candidate_edges.intersection(set(tuple(sorted(e)) for e in G_gt.edges()))))
    rec_degrees = dict(nx.degree(G_rec))
    
    # 4. Prepare Data for Analysis
    df = pd.DataFrame({
        'node': list(range(n)),
        'deg_gt': [true_degrees.get(i, 0) for i in range(n)],
        'deg_rec': [rec_degrees.get(i, 0) for i in range(n)]
    })
    
    # Sort by Ground Truth Degree (Descending)
    df = df.sort_values('deg_gt', ascending=False).reset_index(drop=True)
    
    # --- METRIC 1: Degree Ratio per Class ---
    # Top 1% (Hubs)
    top_1_pct_idx = int(n * 0.01)
    ratio_hubs = df.iloc[:top_1_pct_idx]['deg_rec'].mean() / df.iloc[:top_1_pct_idx]['deg_gt'].mean()
    
    # Mid-Tier (Top 50% excluding top 1%)
    top_50_pct_idx = int(n * 0.50)
    ratio_mid = df.iloc[top_1_pct_idx:top_50_pct_idx]['deg_rec'].mean() / df.iloc[top_1_pct_idx:top_50_pct_idx]['deg_gt'].mean()
    
    # Low-Degree (Bottom 50%)
    ratio_low = df.iloc[top_50_pct_idx:]['deg_rec'].mean() / df.iloc[top_50_pct_idx:]['deg_gt'].mean()
    
    # --- METRIC 2: Rank Correlation Segments ---
    corr_total = df['deg_gt'].corr(df['deg_rec'], method='pearson')
    corr_top100 = df.iloc[:100]['deg_gt'].corr(df.iloc[:100]['deg_rec'], method='pearson')
    corr_top1000 = df.iloc[:1000]['deg_gt'].corr(df.iloc[:1000]['deg_rec'], method='pearson')
    
    # --- METRIC 3: Top-K Overlap (Set Intersection) ---
    top100_gt = set(df.iloc[:100]['node'])
    
    # Get Top-100 from Recovered
    df_rec_sorted = df.sort_values('deg_rec', ascending=False)
    top100_rec = set(df_rec_sorted.iloc[:100]['node'])
    
    overlap_top100 = len(top100_gt.intersection(top100_rec)) / 100.0

    return {
        'topology': topology,
        'ratio_hubs': ratio_hubs,
        'ratio_mid': ratio_mid,
        'ratio_low': ratio_low,
        'corr_total': corr_total,
        'overlap_top100': overlap_top100
    }

if __name__ == "__main__":
    N = 10000
    print(f"Running Segmented Analysis for N={N}...")
    
    results = []
    # Test on Scale-Free (Target) and Random (Control)
    # Strategy: Constant K=250 (2.5%) and K=500 (5%)
    for k_val in [250, 500]:
        strat_name = f'constant_k{k_val}'
        for topo in ['barabasi_albert', 'erdos_renyi']:
            print(f"  Processing {topo} ({strat_name})...")
            # Creating a custom runner for constant k
            metrics = calculate_segmented_metrics(N, topo, strat_name)
            metrics['k_val'] = k_val
            results.append(metrics)
        
    print("\n--- RESULTS ---")
    print(pd.DataFrame(results).to_csv(index=False))
