# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
import time
from ris_core import run_experiment
import random
import numpy as np
import argparse
import multiprocessing
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

# Handle paths for Code Ocean (/results) vs. Local development
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/results"):
    OUTPUT_DIR = "/results"
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
DATA_FILE = os.path.join(OUTPUT_DIR, "simulation_data.csv")
RAW_DATA_FILE = os.path.join(OUTPUT_DIR, "raw_simulation_data.csv")

# N-VALUES Refinados para ver o comportamento entre 0 e 2000
N_VALUES = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 5000, 10000]
N_VALUES_STRESS = [100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000]

TOPOLOGIES = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'stochastic_block', 'powerlaw_cluster']
STRATEGIES = ['heuristic', 'logarithmic']
REPLICATES_DEFAULT = 30 
CPU_CORES_DEFAULT = max(1, multiprocessing.cpu_count() - 2)

def plot_results(df):
    sns.set_theme(style="whitegrid")
    
    # Fig 1: Centralidade com destaque para N < 2000 + inset zoom
    fig, ax_main = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=df, x='n', y='centrality_corr', hue='topology', style='strategy', markers=True, ax=ax_main)
    ax_main.axvspan(0, 2000, color='gray', alpha=0.1, label='Stabilization Zone')
    ax_main.set_title("Topology Preservation Analysis (Pearson Correlation)", fontsize=14)
    ax_main.set_xlabel("Number of Entities (N)", fontsize=12)
    ax_main.set_ylabel("Centrality Correlation (Recovered vs. Ground Truth)", fontsize=12)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Inset zoom: N < 10,000
    ax_inset = fig.add_axes([0.38, 0.45, 0.32, 0.35])  # [left, bottom, width, height] in figure coords
    df_zoom = df[df['n'] <= 10000]
    sns.lineplot(data=df_zoom, x='n', y='centrality_corr', hue='topology', style='strategy', markers=True, ax=ax_inset, legend=False)
    ax_inset.axvspan(0, 2000, color='gray', alpha=0.08)
    ax_inset.set_title("Zoom: N ≤ 10,000", fontsize=10, fontweight='bold')
    ax_inset.set_xlabel("N", fontsize=9)
    ax_inset.set_ylabel("ρ", fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.patch.set_alpha(0.95)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1.2)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_centrality.pdf"), bbox_inches='tight')

    # Fig 2: Redução de Esforço
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='reduction', hue='strategy', markers=True)
    plt.title("Computational Effort Reduction Ratio (The Quadratic Wall Break)", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_reduction.pdf"), bbox_inches='tight')
    
    # Fig 3: Recall (Eficiência de Amostragem) por Topologia
    g = sns.FacetGrid(df, col="topology", hue="strategy", col_wrap=3, height=4)
    g.map(sns.lineplot, "n", "recall", marker="o")
    g.add_legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_recall.pdf"), bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="RIS Orchestrator (High-Performance)")
    parser.add_argument("-randomize", action="store_true", help="Randomize seeds for each replication")
    parser.add_argument("-stress", action="store_true", help="Run scalability stress test (up to N=100k)")
    parser.add_argument("-replicates", type=int, default=REPLICATES_DEFAULT, help="Number of replicates to run")
    parser.add_argument("-cores", type=int, default=CPU_CORES_DEFAULT, help="Number of cores to use")
    parser.add_argument("-plot_only", action="store_true", help="Skip simulation and only generate plots from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        if os.path.exists(DATA_FILE):
            print(f"[INFO] Plot-only mode. Loading existing data from {DATA_FILE}")
            df = pd.read_csv(DATA_FILE)
            plot_results(df)
            print(f"[SUCCESS] PDFs generated in {OUTPUT_DIR}")
            return
        else:
            print(f"[ERROR] {DATA_FILE} not found. Cannot run plot-only mode.")
            sys.exit(1)

    # Proteção OOM: Se no modo estresse (N=100k), use menos núcleos para economizar RAM
    if args.stress and args.cores > 8:
        print(f"[WARNING] Stress mode detected. Reducing cores from {args.cores} to 8 to prevent OOM.")
        args.cores = 8

    n_list = N_VALUES_STRESS if args.stress else N_VALUES
    
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    # Define seeds/replicate IDs
    if args.randomize:
        random_offset = random.randint(0, 1000000)
        replicate_ids = [random_offset + r for r in range(args.replicates)]
        print(f"[INFO] Randomization enabled. Using offset: {random_offset}")
    else:
        replicate_ids = list(range(args.replicates))
        print(f"[INFO] Using fixed seeds (0 to {args.replicates-1}).")

    configs = [{'n': n, 'topology': t, 'strategy': s, 'rep': r} 
               for n in n_list for t in TOPOLOGIES for s in STRATEGIES for r in replicate_ids]
    
    print(f"[INFO] Running {len(configs)} simulations on {args.cores} cores...")
    start = time.time()
    results = Parallel(n_jobs=args.cores)(delayed(run_experiment)(c) for c in tqdm(configs, desc="Simulating"))
    print(f"[INFO] Finished in {time.time() - start:.2f}s")
    
    df_raw = pd.DataFrame(results)
    df_raw.to_csv(RAW_DATA_FILE, index=False)
    
    # Average results 
    print(f"[INFO] Averaging results across {args.replicates} replicates...")
    df = df_raw.groupby(['n', 'topology', 'strategy']).mean().reset_index()
    df.to_csv(DATA_FILE, index=False)

    plot_results(df)
    print(f"[SUCCESS] Averaged data saved to {DATA_FILE}")
    print(f"[SUCCESS] PDFs generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
