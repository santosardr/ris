
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gzip
import requests
import io
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

RESULTS_FILE = os.path.join(RESULTS_DIR, 'financial_results.csv')
URL = "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
FILENAME = "soc-sign-bitcoin-alpha.csv.gz"

def load_bitcoin_alpha():
    print("Loading Bitcoin Alpha Trust Network...")
    file_path = os.path.join(DATA_DIR, FILENAME)
    
    # Download if not exists
    if not os.path.exists(file_path):
        print(f"Downloading {URL}...")
        try:
            r = requests.get(URL)
            if r.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Failed to download: {r.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

    try:
        # Format: SOURCE, TARGET, RATING, TIME
        # We ignore rating/time, treat as undirected trust topology
        df = pd.read_csv(file_path, compression='gzip', header=None, names=['source', 'target', 'rating', 'time'])
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        
        # Make undirected and remove self-loops
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    except Exception as e:
        print(f"Error loading Bitcoin Alpha: {e}")
        return None

def run_analysis(G, name):
    results = []
    n = G.number_of_nodes()
    
    # Ground Truth Centrality
    print(f"  Computing GT centrality for {name}...")
    deg_gt = list(nx.degree_centrality(G).values())
    
    # RIS Strategies
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    true_edges = set(tuple(sorted(e)) for e in G.edges())
    
    scenarios = [
        {'strategy': 'heuristic', 'label': 'RIS (HAF)'},
        {'strategy': 'logarithmic', 'label': 'RIS (Log)'},
        {'strategy': 'high_fidelity', 'label': 'RIS (Hi-Fi 5%)'}
    ]
    
    for scen in scenarios:
        strat = scen['strategy']
        label = scen['label']
        print(f"  Running {label}...")
        
        if strat == 'high_fidelity':
            target_r = int(0.025 * n)
            target_r = max(target_r, 20)
            class HiFiRIS(RISSimulation):
                def _calculate_redundancy(self):
                    return target_r
            ris = HiFiRIS(n, strategy='custom')
        else:
            ris = RISSimulation(n, strategy=strat)
            
        import random
        random.seed(42)
        candidate_edges = ris.generate_candidates()
        
        recovered = candidate_edges.intersection(true_edges)
        
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(range(n))
        G_sparse.add_edges_from(list(recovered))
        
        deg_sparse = list(nx.degree_centrality(G_sparse).values())
        corr = np.corrcoef(deg_gt, deg_sparse)[0, 1]
        
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
        
    G = load_bitcoin_alpha()
    if G:
        results = run_analysis(G, 'Bitcoin Alpha (Finance)')
        df = pd.DataFrame(results)
        print("\nFinal Results:")
        print(df)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
