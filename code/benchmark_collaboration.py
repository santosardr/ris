
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gzip
import requests
from ris_core import RISSimulation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data path: prioritize /data (Code Ocean) else check local
if os.path.exists('/data'):
    DATA_DIR = '/data'
else:
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')

# Results path: prioritize /results (Code Ocean) else use local
if os.path.exists('/results'):
    RESULTS_DIR = '/results'
else:
    RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'results')
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

RESULTS_FILE = os.path.join(RESULTS_DIR, 'collaboration_results.csv')
PLOT_FILE = os.path.join(RESULTS_DIR, 'collaboration_plot.png')

def load_ca_grqc():
    print("Loading ca-GrQc...")
    # SNAP format: from_node to_node (undirected)
    # Skip comments with #
    gz_path = os.path.join(DATA_DIR, 'ca-GrQc.txt.gz')
    txt_path = os.path.join(DATA_DIR, 'ca-GrQc.txt')
    # SNAP URL
    url = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
    
    # List of possible locations for uncompressed or compressed file
    possible_txt_paths = [
        os.path.join(DATA_DIR, 'real_world_data', 'ca-GrQc.txt'),
        os.path.join(DATA_DIR, 'ca-GrQc.txt'),
        txt_path
    ]
    
    target_file = None
    is_gz = False
    
    # 1. Search for uncompressed file
    for path in possible_txt_paths:
        if os.path.exists(path):
            target_file = path
            is_gz = False
            break
            
    # 2. Search for compressed file or download
    if not target_file:
        if os.path.exists(gz_path):
            target_file = gz_path
            is_gz = True
        else:
            print(f"File not found in {DATA_DIR}. Attempting download fallback...")
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    with open(gz_path, 'wb') as f:
                        f.write(r.content)
                    target_file = gz_path
                    is_gz = True
                else:
                    print(f"Failed to download: {r.status_code}")
                    return None
            except Exception as e:
                print(f"Error downloading: {e}")
                return None

    try:
        if is_gz:
            with gzip.open(target_file, 'rt') as f:
                G = nx.read_edgelist(f, comments='#', nodetype=int)
        else:
            G = nx.read_edgelist(target_file, comments='#', nodetype=int)
            
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    except Exception as e:
        print(f"Error loading ca-GrQc: {e}")
        return None

def run_analysis(G, name):
    results = []
    n = G.number_of_nodes()
    
    # Ground Truth Centrality
    print(f"  Computing GT centrality for {name}...")
    deg_gt = list(nx.degree_centrality(G).values())
    
    # RIS Strategies
    # Let's remap nodes to integers 0..N-1 for RIS compatibility
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    true_edges = set(tuple(sorted(e)) for e in G.edges())
    
    # Define scenarios
    scenarios = [
        {'strategy': 'heuristic', 'label': 'RIS (HAF)'},
        {'strategy': 'logarithmic', 'label': 'RIS (Log)'},
        {'strategy': 'high_fidelity', 'label': 'RIS (Hi-Fi 5%)'}
    ]
    
    for scen in scenarios:
        strat = scen['strategy']
        label = scen['label']
        print(f"  Running {label}...")
        
        # Run RIS
        if strat == 'high_fidelity':
            target_r = int(0.025 * n)
            target_r = max(target_r, 20) # Minimum safety
            class HiFiRIS(RISSimulation):
                def _calculate_redundancy(self):
                    return target_r
            ris = HiFiRIS(n, strategy='custom')
        else:
            ris = RISSimulation(n, strategy=strat)
            
        # Force re-seed for stability
        import random
        random.seed(42)
        candidate_edges = ris.generate_candidates()
        
        # Intersection
        recovered = candidate_edges.intersection(true_edges)
        
        # Build Sparse Graph
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(range(n))
        G_sparse.add_edges_from(list(recovered))
        
        # Calc Metrics
        deg_sparse = list(nx.degree_centrality(G_sparse).values())
        corr = np.corrcoef(deg_gt, deg_sparse)[0, 1]
        
        # Edge Reduction
        total_possible = n * (n - 1) / 2
        density = len(recovered) / len(true_edges) if len(true_edges) > 0 else 0
        reduction = 1 - (len(candidate_edges) / total_possible)
        
        print(f"    Corr: {corr:.4f}, Density: {density:.4f}")
        results.append({
            'Dataset': name,
            'Strategy': label,
            'Correlation': corr,
            'Recovered_Density': density,
            'Reduction': reduction
        })
        
    return results

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    G = load_ca_grqc()
    if G:
        results = run_analysis(G, 'ca-GrQc')
        df = pd.DataFrame(results)
        print("\nFinal Results:")
        print(df)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
