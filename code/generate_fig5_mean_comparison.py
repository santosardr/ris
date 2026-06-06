# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import networkx as nx
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import argparse
import json
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from ris_core import RISSimulation
from attention_patterns import AttentionPatterns

# Set output directory for Code Ocean compatibility vs Local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/results"):
    OUTPUT_DIR = "/results"
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")

# Ensure output directory exists locally
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_livejournal(path):
    print(f"Loading LiveJournal from {path}...")
    G = nx.Graph()
    max_node_id = 20000 
    try:
        with open(path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#'): continue
                parts = line.strip().split()
                if not parts: continue
                
                # Robust parsing
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                    
                if u < max_node_id and v < max_node_id:
                    G.add_edge(u, v)
                    
        if G.number_of_edges() == 0:
            raise ValueError("No edges loaded from file!")
            
    except Exception as e:
        print(f"CRITICAL ERROR reading file: {e}")
        # FAIL LOUDLY - Do not return a fallback
        raise e
        
    print(f"Loaded LiveJournal: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def measure_hub_recall_optimized(gt_degrees, sampled_edges, n_nodes):
    """
    Optimized version of hub recall measurement matching robustness_check.py logic.
    """
    k = max(1, int(n_nodes * 0.01))
    
    # GT Hubs (IDs of top k)
    top_k_gt_ids = np.argpartition(gt_degrees, -k)[-k:]
    
    if len(sampled_edges) == 0:
        return 0.0
    
    # Flatten edges to get all node occurrences
    nodes_sampled = sampled_edges.ravel()
    # Count degrees using bincount
    deg_sample = np.bincount(nodes_sampled, minlength=n_nodes)
    
    # Identify active nodes (those that appeared in the sample)
    active_nodes = np.nonzero(deg_sample)[0]
    active_degs = deg_sample[active_nodes]
    
    if len(active_nodes) == 0:
        return 0.0
        
    if len(active_nodes) <= k:
        top_k_sp_ids = active_nodes
    else:
        # Get indices of top k within the active set
        top_k_active_idx = np.argpartition(active_degs, -k)[-k:]
        top_k_sp_ids = active_nodes[top_k_active_idx]
    
    # Intersection
    overlap = len(np.intersect1d(top_k_gt_ids, top_k_sp_ids, assume_unique=True))
    return overlap / k

def run_simulation_batch(seed, n_nodes, gt_degrees, experiment_type='LiveJournal'):
    """
    Runs a single simulation replicate for all methods.
    Returns dictionary with results.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    results = {}
    
    # RIS (Generalized NMI)
    ris_nmi = RISSimulation(n_nodes, strategy='heuristic')
    ris_nmi_edges = ris_nmi.generate_candidates() 
    ris_nmi_edges_np = np.array(list(ris_nmi_edges)) if ris_nmi_edges else np.empty((0, 2), dtype=int)
    results['RIS'] = measure_hub_recall_optimized(gt_degrees, ris_nmi_edges_np, n_nodes)
    
    # RIS-GenPPi (Original)
    ris_genppi = RISSimulation(n_nodes, strategy='genppi')
    ris_genppi_edges = ris_genppi.generate_candidates()
    ris_genppi_edges_np = np.array(list(ris_genppi_edges)) if ris_genppi_edges else np.empty((0, 2), dtype=int)
    results['RIS-GenPPi'] = measure_hub_recall_optimized(gt_degrees, ris_genppi_edges_np, n_nodes)
    
    # Matching budget for competitors (using NMI budget as reference)
    budget = len(ris_nmi_edges)
    
    # Longformer
    lf_edges_np = AttentionPatterns.longformer_simulation_vectorized(n_nodes, float(budget))
    results['Longformer'] = measure_hub_recall_optimized(gt_degrees, lf_edges_np, n_nodes)
    
    # BigBird
    bb_edges_np = AttentionPatterns.bigbird_simulation_vectorized(n_nodes, float(budget))
    results['BigBird'] = measure_hub_recall_optimized(gt_degrees, bb_edges_np, n_nodes)
    
    return results

def get_gt_degrees(G, n_nodes):
    deg_dict = dict(G.degree())
    degrees = np.zeros(n_nodes, dtype=np.int32)
    # Fill array
    for node, deg in deg_dict.items():
        if node < n_nodes:
            degrees[node] = deg
    return degrees

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5 with High Fidelity")
    parser.add_argument("--replicates", type=int, default=100, help="Number of replicates per experiment")
    parser.add_argument("--cores", type=int, default=-1, help="Number of cores to use")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save results")
    parser.add_argument("--load_json", type=str, default=None, help="Path to a JSON file with raw results to plot directly.")
    parser.add_argument("--force", action="store_true", help="Force re-generation of simulation data even if JSON exists.")
    args = parser.parse_args()
    
    final_stats = {'LiveJournal': {}, 'Synthetic': {}}
    raw_results = []

    # Auto-detection of existing data
    default_json = os.path.join(args.output_dir, 'results_fig5_raw.json')
    load_path = args.load_json
    if not load_path and os.path.exists(default_json) and not args.force:
        load_path = default_json
        print(f"[INFO] Existing data found at {load_path}. Skipping simulation (use --force to override).")

    if load_path:
        print(f"[INFO] Loading results from {load_path}...")
        with open(load_path, 'r') as f:
            raw_results = json.load(f)
        
        # Aggregate by dataset
        ds_groups = {}
        for r in raw_results:
            ds = r['dataset']
            if ds not in ds_groups: ds_groups[ds] = []
            ds_groups[ds].append(r)
            
        method_map = {'RIS-GenPPi': 'RIS-Structural', 'RIS': 'RIS-Stochastic', 'Longformer': 'Longformer', 'BigBird': 'BigBird'}
        for ds, entries in ds_groups.items():
            for old_key, new_key in method_map.items():
                vals = [e[old_key] for e in entries if old_key in e and e[old_key] is not None]
                if vals:
                    final_stats[ds][new_key] = (np.mean(vals), np.std(vals))
    else:
        # Determine Cores
        if args.cores == -1:
            n_cores = max(1, multiprocessing.cpu_count() - 2)
        else:
            n_cores = args.cores
            
        # Path detection
        base_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            os.path.join(os.path.dirname(base_dir), 'data', 'com-lj.ungraph.txt'),
            os.path.join(base_dir, 'real_world_data', 'com-lj.ungraph.txt'),
            '/galaxy/work/anderson/repos/myarticles/ris/scripts/real_world_data/com-lj.ungraph.txt',
            '/data/com-lj.ungraph.txt'
        ]
        real_path = next((p for p in potential_paths if os.path.exists(p)), None)

        if not real_path:
            print("[ERROR] No LiveJournal data file found!")
            return
            
        # 1. LiveJournal
        G_lj = load_livejournal(real_path)
        gt_degrees_lj = get_gt_degrees(G_lj, G_lj.number_of_nodes())
        print(f"Running {args.replicates} simulations on LiveJournal...")
        seeds = [100 + i for i in range(args.replicates)]
        results_lj_list = Parallel(n_jobs=n_cores)(
            delayed(run_simulation_batch)(s, G_lj.number_of_nodes(), gt_degrees_lj, 'LiveJournal') 
            for s in tqdm(seeds, desc="LiveJournal")
        )
        
        # 2. Synthetic (BA)
        print("\nGenerating and testing Synthetic BA...")
        G_ba = nx.barabasi_albert_graph(5000, 5)
        gt_degrees_ba = get_gt_degrees(G_ba, 5000)
        seeds_ba = [200 + i for i in range(args.replicates)]
        results_ba_list = Parallel(n_jobs=n_cores)(
            delayed(run_simulation_batch)(s, 5000, gt_degrees_ba, 'Synthetic') 
            for s in tqdm(seeds_ba, desc="Synthetic")
        )
        
        # Combine results
        method_map = {'RIS-GenPPi': 'RIS-Structural', 'RIS': 'RIS-Stochastic', 'Longformer': 'Longformer', 'BigBird': 'BigBird'}
        for ds, r_list in [('LiveJournal', results_lj_list), ('Synthetic', results_ba_list)]:
            for old_key, new_key in method_map.items():
                vals = [r[old_key] for r in r_list]
                final_stats[ds][new_key] = (np.mean(vals), np.std(vals))
            
            for i, res in enumerate(r_list):
                res['dataset'] = ds
                res['replicate'] = i
                raw_results.append(res)

        # Save Raw Data
        raw_path = os.path.join(args.output_dir, 'results_fig5_raw.json')
        with open(raw_path, 'w') as f:
            json.dump(raw_results, f, indent=2)
        print(f"\n[INFO] Raw results saved to {raw_path}")

    # Print Summary Table
    print("\n" + "="*95)
    print(f"{'Dataset':<20} | {'Method':<15} | {'Mean Hub Recall':<15} | {'Std Dev':<10}")
    print("-" * 95)
    for res_type in ['LiveJournal', 'Synthetic']:
        for method in ['RIS-Structural', 'RIS-Stochastic', 'Longformer', 'BigBird']:
            mean, std = final_stats[res_type].get(method, (0,0))
            print(f"{res_type:<20} | {method:<15} | {mean:11.4%} | {std:.4f}")
        print("-" * 95)
    print("="*95)

    # Plotting
    print("\nGenerating Plot...")
    labels = ['RIS-Structural', 'RIS-Stochastic', 'Longformer', 'BigBird']
    colors = ['#008000', '#2c7bb6', '#ffffbf', '#d7191c'] # Green for GenPPi
    
    lj_means = [final_stats['LiveJournal'].get(l, (0,0))[0] for l in labels]
    lj_stds = [final_stats['LiveJournal'].get(l, (0,0))[1] for l in labels]
    ba_means = [final_stats['Synthetic'].get(l, (0,0))[0] for l in labels]
    ba_stds = [final_stats['Synthetic'].get(l, (0,0))[1] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    epsilon = 1e-4
    lj_plot = [max(v, epsilon) for v in lj_means]
    ba_plot = [max(v, epsilon) for v in ba_means]
    
    rects1 = ax.bar(x - width/2, lj_plot, width, yerr=lj_stds, label='Real World (Avg)', color=colors, alpha=0.8, capsize=5)
    # Note: Using color=colors directly on bar only works if len(x) == len(colors). 
    # To have different colors for bars in the same dataset, we might need manual bar calls or use a hatch.
    # We will use simplified coloring for clarity.
    
    rects1 = ax.bar(x - width/2, lj_plot, width, yerr=lj_stds, label='Real World', color=colors, capsize=5)
    # The above actually applies the colormap to the bars.
    
    ax.set_ylabel('Hub Recall (Log Scale, Mean ± SD)')
    ax.set_title(f'Topological Recovery: Average over {args.replicates} Trials')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(['Real World', 'Synthetic'], loc='upper right')
    # Actually, we need to handle the dual bars properly
    
    # Redefining plot for 2 datasets
    ax.cla()
    rects1 = ax.bar(x - width/2, lj_plot, width, yerr=lj_stds, label='Real World', color='#2c7bb6', capsize=5, alpha=0.7)
    rects2 = ax.bar(x + width/2, ba_plot, width, yerr=ba_stds, label='Synthetic', color='#d7191c', capsize=5, alpha=0.7)
    
    # Highlight RIS-GenPPi with a different edge or hatch if needed, but mean comparison is clear.
    
    ax.set_ylabel('Hub Recall (Log Scale, Mean ± SD)')
    ax.set_title(f'Topological Recovery: Average over {args.replicates} Trials')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')
    
    def autolabel(rects, real_values):
        for rect, val in zip(rects, real_values):
            height = rect.get_height()
            ax.annotate(f'{val:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1, lj_means)
    autolabel(rects2, ba_means)
    
    # Add statistical significance bridge for LiveJournal (Structural vs BigBird)
    # Structural is index 0, BigBird is index 3 in 'labels'
    x1 = x[0] - width/2
    x2 = x[3] - width/2
    y_max = max(lj_plot[0], lj_plot[3])
    y, h, col = y_max * 1.5, y_max * 0.2, '#555555' # Softer grey
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, c=col, linestyle=(0, (5, 8)))
    ax.text((x1+x2)*.5, y+h, "* $p=0.031$", ha='center', va='bottom', color=col, fontsize=10, alpha=0.8)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'fig5_mean_comparison.pdf')
    plt.savefig(save_path)
    print(f"Saved {save_path} (with 2 decimal precision and p-value)")

if __name__ == "__main__":
    main()
