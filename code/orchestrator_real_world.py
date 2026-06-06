# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import random
from tqdm import tqdm
import os
import sys
import argparse
import time
from joblib import Parallel, delayed
import multiprocessing

# Handle paths for Code Ocean vs. Local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data path detection
potential_data_paths = [
    "/data/com-lj.ungraph.txt",
    os.path.join(os.path.dirname(BASE_DIR), "data", "com-lj.ungraph.txt"),
    os.path.join(BASE_DIR, "real_world_data", "com-lj.ungraph.txt")
]
FILE = next((p for p in potential_data_paths if os.path.exists(p)), potential_data_paths[0])

if os.path.exists("/results"):
    OUTPUT_DIR = "/results"
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "real_world_benchmark.csv")
RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_real_world_benchmark.csv")

N_NODES = 4036538
N_NODES = 4036538
REPLICATES_DEFAULT = 30
CPU_CORES_DEFAULT = max(1, multiprocessing.cpu_count() - 2)

def load_edge_list():
    print(f"[INFO] Loading edge list from {FILE}...")
    df = pd.read_csv(FILE, sep='\t', comment='#', header=None, names=['u', 'v'])
    return df.values

def get_degrees(edges, n_nodes):
    degrees = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(degrees, edges[:, 0], 1)
    np.add.at(degrees, edges[:, 1], 1)
    return degrees

def run_single_replicate(rep_id, edges, gt_degrees, ratios, n_edges):
    """Executa um único conjunto de testes (todos os q) para uma semente."""
    seed_value = hash(f"real-world-lj-{rep_id}") % (2**32)
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    rep_results = []
    for q in ratios:
        budget = int(q * n_edges)
        indices = np.random.choice(n_edges, budget, replace=False)
        sampled_edges = edges[indices]
        sampled_degrees = get_degrees(sampled_edges, N_NODES)
        
        corr, _ = pearsonr(gt_degrees, sampled_degrees)
        rep_results.append({
            'rep': rep_id,
            'q': q,
            'corr': corr,
            'budget': budget,
            'method': 'RIS (Uniform)'
        })
    return rep_results

def run_benchmark():
    parser = argparse.ArgumentParser(description="RIS Real-World Benchmark (High-Performance Parallel)")
    parser.add_argument("-randomize", action="store_true", help="Randomize seeds for each replication")
    parser.add_argument("-replicates", type=int, default=REPLICATES_DEFAULT, help="Number of replicates to run")
    parser.add_argument("-cores", type=int, default=CPU_CORES_DEFAULT, help="Number of cores to use")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    edges = load_edge_list()
    n_edges = len(edges)
    print(f"[INFO] Loaded {n_edges} edges.")

    print("[INFO] Computing ground truth degrees...")
    gt_degrees = get_degrees(edges, N_NODES)
    
    if args.randomize:
        random_offset = random.randint(0, 1000000)
        replicate_ids = [random_offset + r for r in range(args.replicates)]
        print(f"[INFO] Randomization enabled. Using offset: {random_offset}")
    else:
        replicate_ids = list(range(args.replicates))
        print(f"[INFO] Using fixed seeds (0 to {args.replicates-1}).")

    ratios = [0.001, 0.01, 0.05, 0.1]
    
    print(f"[INFO] Starting {args.replicates} replicates on {args.cores} cores...")
    start_time = time.time()
    
    # Parallel execution using Joblib
    results_nested = Parallel(n_jobs=args.cores)(
        delayed(run_single_replicate)(rep, edges, gt_degrees, ratios, n_edges) 
        for rep in tqdm(replicate_ids, desc="Simulating Replicates")
    )
    
    # Flatten results
    all_results = [item for sublist in results_nested for item in sublist]
    
    duration = time.time() - start_time
    print(f"[INFO] Computation completed in {duration:.2f} seconds.")

    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(RAW_OUTPUT_FILE, index=False)
    
    print(f"\n[INFO] Averaging results across {args.replicates} replicates...")
    df_avg = df_raw.groupby(['q', 'method']).agg({
        'corr': ['mean', 'std', 'sem'],
        'budget': 'first'
    }).reset_index()
    
    # Flatten columns
    df_avg.columns = ['q', 'method', 'corr_mean', 'corr_std', 'corr_sem', 'budget']
    
    df_avg.to_csv(OUTPUT_FILE, index=False)
    
    print("\n[SUMMARY] Results (Mean Correlation):")
    for _, row in df_avg.iterrows():
        print(f"  q={row['q']:.3f}: {row['corr_mean']:.6f} (±{row['corr_std']:.6f}, SEM={row['corr_sem']:.6f})")

    print(f"\n[SUCCESS] Benchmark results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()
