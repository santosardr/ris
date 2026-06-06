
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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

RESULTS_FILE = os.path.join(RESULTS_DIR, 'power_grid_results.csv')
PLOT_FILE = os.path.join(RESULTS_DIR, 'power_grid_plot.png')

def load_power_grid():
    print("Loading US Power Grid (opsahl-powergrid)...")
    import tarfile
    
    # Mirror link (fallback only)
    url = "https://raw.githubusercontent.com/skardhamar/network-analysis/master/power.edit"
    
    target_file = None
    
    # List of possible locations for the edgelist file
    possible_paths = [
        os.path.join(DATA_DIR, 'real_world_data', 'out.opsahl-powergrid'),
        os.path.join(DATA_DIR, 'out.opsahl-powergrid'),
        os.path.join(DATA_DIR, 'real_world_data', 'opsahl-powergrid', 'out.opsahl-powergrid'),
        os.path.join(DATA_DIR, 'opsahl-powergrid', 'out.opsahl-powergrid'),
        os.path.join(DATA_DIR, 'power.edit')
    ]
    
    print("[DEBUG] Searching for Power Grid data in:")
    for path in possible_paths:
        exists = "EXISTS" if os.path.exists(path) else "MISSING"
        print(f"  - {path} [{exists}]")
        if os.path.exists(path):
            target_file = path
            break
            
    # 2. If not found, try to extract from tarball if it exists
    if not target_file:
        tar_file = os.path.join(DATA_DIR, 'real_world_data', 'opsahl-powergrid.tar.bz2')
        if not os.path.exists(tar_file):
            tar_file = os.path.join(DATA_DIR, 'opsahl-powergrid.tar.bz2')
            
        if os.path.exists(tar_file):
            print(f"Extracting {tar_file}...")
            try:
                with tarfile.open(tar_file, 'r:bz2') as tar:
                    tar.extractall(path=DATA_DIR)
                # Check again after extraction
                for path in possible_paths:
                    if os.path.exists(path):
                        target_file = path
                        break
            except Exception as e:
                print(f"Extraction failed: {e}")
            
    # 3. Download if still missing (Code Ocean fallback)
    if not target_file:
        print(f"File not found in {DATA_DIR}. Attempting download fallback...")
        target_file = possible_paths[0] # Default download location
        try:
            r = requests.get(url)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                with open(target_file, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Failed to download mirror: {r.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

    try:
        # Opsahl format uses % as comment. Some mirrors use * or #
        # We try to detect the comment character or use the most common one
        G = nx.read_edgelist(target_file, comments='%', nodetype=int)
        if G.number_of_nodes() == 0: # Fallback for different comment char
            G = nx.read_edgelist(target_file, comments='*', nodetype=int)
            
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    except Exception as e:
        print(f"Error loading Power Grid: {e}")
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
        
    G = load_power_grid()
    if G:
        results = run_analysis(G, 'US Power Grid')
        df = pd.DataFrame(results)
        print("\nFinal Results:")
        print(df)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
