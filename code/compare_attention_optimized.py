# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import networkx as nx
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from collections import Counter
from ris_core import RISSimulation
from attention_patterns import AttentionPatterns

# Subclass to strictly separate testing logic from core library
class HighDensityRIS(RISSimulation):
    def __init__(self, n_nodes, target_density=0.1):
        super().__init__(n_nodes, strategy='heuristic')
        # CORRECTION: 
        # Density D = 2E / N^2 (approx)
        # We generate N * R edges (assuming low collision).
        # So D = 2(NR) / N^2 = 2R / N
        # Thus R = D * N / 2
        self.custom_r = max(1, int((target_density * n_nodes) / 2))
        
    def _calculate_redundancy(self):
        return self.custom_r

def load_livejournal(path):
    print(f"Loading LiveJournal from {path}...")
    G = nx.Graph()
    max_node_id = 50000 
    
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if not parts: continue
                u, v = int(parts[0]), int(parts[1])
                if u < max_node_id and v < max_node_id:
                    G.add_edge(u, v)
    except Exception as e:
        print(f"Error reading file: {e}")
        return nx.barabasi_albert_graph(2000, 5)

    print(f"Loaded Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # SHUFFLE LIVEJOURNAL
    print("Shuffling LiveJournal indices...")
    nodes = list(G.nodes())
    random.shuffle(nodes)
    mapping = dict(zip(nodes, range(len(nodes))))
    G = nx.relabel_nodes(G, mapping)
    
    return G

def compare_topologies(G_gt, name="Graph", target_density=0.1):
    n = G_gt.number_of_nodes()
    
    print(f"Running RIS on {name} with Target Density={target_density:.1%}...")
    ris = HighDensityRIS(n, target_density=target_density)
    
    r_val = ris._calculate_redundancy()
    print(f"  -> Calculated Redundancy R={r_val} neighbors per node")
    
    ris_candidates = ris.generate_candidates()
    
    budget = len(ris_candidates)
    actual_density = budget / (n*(n-1)/2) if n > 1 else 0
    print(f"  -> Generated {budget} edges (Actual Density: {actual_density:.5f})")
    
    print("Generating Longformer Pattern...")
    lf_edges = AttentionPatterns.longformer_simulation(n, budget)
    
    print("Generating BigBird Pattern...")
    bb_edges = AttentionPatterns.bigbird_simulation(n, budget)
    
    print("Evaluating Metrics (Memory Optimized)...")
    true_edges = set(tuple(sorted(e)) for e in G_gt.edges())
    
    # Pre-calculate Ground Truth Degrees
    deg_gt_dict = dict(nx.degree(G_gt))
    deg_gt = [deg_gt_dict.get(i, 0) for i in range(n)]
    
    # Identify Hubs (Top 1%)
    k = max(1, int(n * 0.01))
    top_k_gt = sorted(deg_gt_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    top_k_ids = set([x[0] for x in top_k_gt])
    
    def evaluate(candidates, label):
        # 1. Edge Recall
        recovered = candidates.intersection(true_edges)
        recall = len(recovered) / len(true_edges) if len(true_edges) > 0 else 0
        
        # 2. Centrality Correlation (Using Counter, avoiding nx.Graph)
        deg_counter = Counter()
        for u, v in candidates:
            deg_counter[u] += 1
            deg_counter[v] += 1
            
        deg_sp = [deg_counter.get(i, 0) for i in range(n)]
        
        if np.std(deg_gt) == 0 or np.std(deg_sp) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(deg_gt, deg_sp)[0, 1]
        
        # 3. Hub Recall
        top_k_sp = sorted(deg_counter.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_sp_ids = set([x[0] for x in top_k_sp])
        
        hub_overlap = len(top_k_ids.intersection(top_k_sp_ids)) / k if k > 0 else 0
        
        return {'Label': label, 'Recall': recall, 'Corr': corr, 'Hub_Recall': hub_overlap}

    results = []
    results.append(evaluate(ris_candidates, f'RIS (Dens={target_density})'))
    results.append(evaluate(lf_edges, 'Longformer (Sim)'))
    results.append(evaluate(bb_edges, 'BigBird (Sim)'))
    
    return results

def main():
    TARGET_DENSITY = 0.10
    
    # Use relative path for portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(base_dir, 'real_world_data', 'com-lj.ungraph.txt')
    
    results_all = []
    
    if os.path.exists(real_path):
        G_real = load_livejournal(real_path)
        res_real = compare_topologies(G_real, "LiveJournal (Shuffled)", target_density=TARGET_DENSITY)
        results_all.extend(res_real)
    else:
        print(f"Skipping Real Data (File not found at {real_path})")
        
    print("\nGenerating Synthetic BA Graph (N=5000)...")
    G_ba_ordered = nx.barabasi_albert_graph(5000, 5)
    
    nodes = list(G_ba_ordered.nodes())
    random.shuffle(nodes)
    mapping = dict(zip(nodes, range(5000)))
    G_ba_shuffled = nx.relabel_nodes(G_ba_ordered, mapping)
    
    res_ba = compare_topologies(G_ba_shuffled, "Synthetic BA (Shuffled)", target_density=TARGET_DENSITY)
    results_all.extend(res_ba)

    print("\n" + "="*80)
    print(f"{'Dataset':<25} | {'Method':<20} | {'Corr':<8} | {'Hub Rec':<8}")
    print("-" * 80)
    
    datasets = ["LiveJournal"] * 3 + ["Barabasi-Albert"] * 3
    for i, res in enumerate(results_all):
        if i < len(datasets):
            dset = datasets[i] 
        else:
            dset = "Synthetic"
        print(f"{dset:<25} | {res['Label']:<20} | {res['Corr']:.4f}   | {res['Hub_Recall']:.4f}")
    print("="*80)
    
    if len(results_all) > 0:
        labels = [r['Label'] for r in results_all[:3]]
        corrs = [r['Corr'] for r in results_all[:3]]
        x = np.arange(len(labels))
        plt.bar(x, corrs)
        plt.xticks(x, labels)
        plt.title(f'Comparison at {TARGET_DENSITY:.0%} Density')
        plt.savefig('comparison_plot_optimized.pdf')
        print("Plot saved.")

if __name__ == "__main__":
    main()
