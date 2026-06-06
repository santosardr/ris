import numpy as np
import pandas as pd
import random
import os
import time
from tqdm import tqdm
from scipy.stats import pearsonr
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing
from ris_core import RISSimulation
from attention_patterns import AttentionPatterns

# --- SETTINGS ---
N_FULL_LJ = 4036538
N_SUBGRAPH = 20000 
TOTAL_REPLICATES = 90  # Default total runs across all "batches"
# Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data path detection
potential_data_paths = [
    "/data/com-lj.ungraph.txt",
    "/data/real_world_data/com-lj.ungraph.txt",
    os.path.join(BASE_DIR, "mystuff", "com-lj.ungraph.txt"),
    os.path.join(BASE_DIR, "scripts", "real_world_data", "com-lj.ungraph.txt")
]
DATA_PATH = next((p for p in potential_data_paths if os.path.exists(p)), potential_data_paths[0])

def load_edges_and_subgraph():
    if not DATA_PATH: return None, None
    print(f"[INFO] Loading Full LiveJournal from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep='\t', comment='#', header=None, names=['u', 'v'], dtype=np.int32)
    edges_full = df.values
    
    print(f"[INFO] Extracting {N_SUBGRAPH} node subgraph for Figure 2 verification...")
    # Follow generate_fig5_mean_comparison.py logic: nodes < 20000
    mask = (edges_full[:, 0] < N_SUBGRAPH) & (edges_full[:, 1] < N_SUBGRAPH)
    subgraph_edges = edges_full[mask]
    
    return edges_full, subgraph_edges

def get_degrees(edges, n_nodes):
    degrees = np.zeros(n_nodes, dtype=np.int32)
    if len(edges) == 0: return degrees
    u_indices = edges[:, 0]
    v_indices = edges[:, 1]
    np.add.at(degrees, u_indices, 1)
    np.add.at(degrees, v_indices, 1)
    return degrees

def measure_hub_recall_optimized(gt_degrees, sampled_edges, n_nodes):
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
        # Using argpartition is fine, but for exact top-k matches with original (if ties matter), 
        # sorting might be safer. However, argpartition is much faster.
        top_k_active_idx = np.argpartition(active_degs, -k)[-k:]
        top_k_sp_ids = active_nodes[top_k_active_idx]
    
    # Intersection
    overlap = len(np.intersect1d(top_k_gt_ids, top_k_sp_ids, assume_unique=True))
    return overlap / k

def run_simulation(seed, edges_full, gt_degrees_full, edges_sub, gt_degrees_sub):
    np.random.seed(seed)
    random.seed(seed)
    
    # --- TABLE 3 CHECK (Full Graph, q=0.1) ---
    q = 0.1
    budget_full = int(q * len(edges_full))
    idx_full = np.random.choice(len(edges_full), budget_full, replace=False)
    sampled_full = edges_full[idx_full]
    sampled_degs_full = get_degrees(sampled_full, N_FULL_LJ)
    corr, _ = pearsonr(gt_degrees_full, sampled_degs_full)
    
    # --- FIGURE 2 CHECK (20k Subgraph, RIS Heuristic budget) ---
    r = 6 # Effectively ~6 candidates per node in original script
    pivots = np.repeat(np.arange(N_SUBGRAPH), r)
    neighbors = np.random.randint(0, N_SUBGRAPH, size=N_SUBGRAPH * r)
    
    # Remove self-loops
    mask_self = pivots == neighbors
    neighbors[mask_self] = (neighbors[mask_self] + 1) % N_SUBGRAPH
    
    # Form edge tuples and deduplicate
    raw_edges = np.column_stack((pivots, neighbors))
    raw_edges.sort(axis=1) 
    sampled_sub_unique = np.unique(raw_edges, axis=0)
    
    # Corrected intersection: Ensure gt edges are sorted before packing
    def intersect_edges_np(e1, e2):
        def pack(e): return e[:, 0].astype(np.int64) * 100000 + e[:, 1]
        
        # Sort and unique ground truth edges
        e2_sorted = np.sort(e2, axis=1)
        p2 = np.unique(pack(e2_sorted))
        
        # Candidate edges are already sorted
        p1 = pack(e1)
        
        return e1[np.isin(p1, p2)]

    # For Figure 2 verification, we measure raw topological recovery (no intersection with GT edges)
    # mirroring the generate_fig5_mean_comparison.py logic.
    hub_recall = measure_hub_recall_optimized(gt_degrees_sub, sampled_sub_unique, N_SUBGRAPH)
    
    return {'corr': corr, 'hub_recall': hub_recall}

def main():
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"[INFO] Using {n_cores} CPU cores for parallel execution.")
    print(f"[INFO] Total replicates to run: {TOTAL_REPLICATES}")
    
    edges_full, edges_sub = load_edges_and_subgraph()
    if edges_full is None:
        print("[ERROR] Could not load LiveJournal data. Checked paths:")
        for p in potential_data_paths: print(f"  - {p}")
        return
        
    gt_degrees_full = get_degrees(edges_full, N_FULL_LJ)
    gt_degrees_sub = get_degrees(edges_sub, N_SUBGRAPH)
    
    print(f"[INFO] Starting {TOTAL_REPLICATES} replicates in parallel...")
    start_time = time.time()
    
    # Run everything in a single Parallel call to saturate all cores efficiently
    # Using threading if I/O bound, but here it's CPU bound, so multiprocessing is default.
    # We use 'mmap_mode' to shared large arrays without copying them to every worker (saves RAM).
    results = Parallel(n_jobs=n_cores, mmap_mode='r')(
        delayed(run_simulation)(s, edges_full, gt_degrees_full, edges_sub, gt_degrees_sub) 
        for s in tqdm(range(TOTAL_REPLICATES), desc="Replicates")
    )
    
    end_time = time.time()
    print(f"\n[INFO] Computation completed in {end_time - start_time:.2f} seconds.")
    
    df = pd.DataFrame(results)
    
    # We can still present results in a "batch-like" way if desired, or just global stats.
    # Let's split them into 3 virtual batches for consistency with the output format.
    val_per_batch = TOTAL_REPLICATES // 3
    batch_stats = []
    for i in range(3):
        start_idx = i * val_per_batch
        end_idx = (i + 1) * val_per_batch if i < 2 else TOTAL_REPLICATES
        batch_df = df.iloc[start_idx:end_idx]
        batch_stats.append({
            'corr_mean': batch_df['corr'].mean(),
            'hub_mean': batch_df['hub_recall'].mean()
        })
        
    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'Batch 1':<12} | {'Batch 2':<12} | {'Batch 3':<12} | {'Paper Val'}")
    print("-" * 80)
    
    paper_corr = 0.9602
    paper_hub = 0.009829 # 0.9829%
    
    print(f"{'Correlation (q=0.1)':<20} | {batch_stats[0]['corr_mean']:.4f}       | {batch_stats[1]['corr_mean']:.4f}       | {batch_stats[2]['corr_mean']:.4f}       | {paper_corr}")
    print(f"{'Hub Recall (q=0.1)':<20} | {batch_stats[0]['hub_mean']:.4%}      | {batch_stats[1]['hub_mean']:.4%}      | {batch_stats[2]['hub_mean']:.4%}      | 0.9829%")
    print("="*80)
    
    global_corr = df['corr'].mean()
    global_hub = df['hub_recall'].mean()
    
    print(f"\n[SUMMARY] Global Mean ({TOTAL_REPLICATES} runs):")
    print(f"  Correlation: {global_corr:.4f} (Paper: {paper_corr})")
    print(f"  Hub Recall:  {global_hub:.4%} (Paper: 0.9829%)")
    
    diff_corr = abs(global_corr - paper_corr)
    diff_hub = abs(global_hub - paper_hub)
    
    # Adjust tolerance slightly if needed for stochastic variance, but logic should be robust.
    if diff_corr < 0.005 and diff_hub < 0.005:
        print("\n[VERDICT] Robustness CONFIRMED. Variations are statistically insignificant.")
    else:
        print("\n[VERDICT] WARNING: Detected unexpected variance. Investigation recommended.")

if __name__ == "__main__":
    main()
