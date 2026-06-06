# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_graphical_abstract():
    # Set seeds for absolute consistency
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Nature dimensions optimized for landscape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    
    # --- BASE NETWORK CONSTRUCTION ---
    # Increased nodes by 100% (from 85 to 170) for maximum visual impact
    N = 170
    # Scaling initial edges slightly for complexity
    G_truth = nx.barabasi_albert_graph(N, 2, seed=random_seed)
    
    # Unified layout - Increased k and iterations to prevent overlapping with 170 nodes
    pos = nx.spring_layout(G_truth, seed=random_seed, k=0.15, iterations=100)
    
    degrees = dict(G_truth.degree())
    # Define Hubs clearly
    hub_threshold = 7 
    hubs = [n for n in G_truth.nodes() if degrees[n] > hub_threshold]
    
    # Color scheme: Professional Blues
    color_hub = '#3182bd' # Strong Blue
    color_node = '#9ecae1'  # Light Blue
    color_edges_dense = '#444444'
    color_edges_ris = '#444444' 
    
    # --- PANEL 1: THE DENSE QUADRATIC WALL ---
    # Complete Graph background represents O(N^2) complexity
    G_complete = nx.complete_graph(N)
    nx.draw_networkx_edges(G_complete, pos, ax=ax1, alpha=0.04, edge_color=color_edges_dense, width=0.2)
    
    # Node sizes for the "Dense" view (Slightly larger base size to be visible)
    node_sizes_dense = [(degrees[n] * 30 + 40) * 0.25 if n in hubs else 15 for n in G_truth.nodes()]
    node_colors_dense = [color_hub if n in hubs else color_node for n in G_truth.nodes()]
    
    nx.draw_networkx_nodes(G_truth, pos, ax=ax1, node_size=node_sizes_dense, 
                           node_color=node_colors_dense, linewidths=0.3, edgecolors='white')
    
    ax1.set_title("Standard Relationship Inference\n$O(N^2)$ All-to-All Complexity", 
                 fontsize=12, fontweight='bold', color='#1a1a1a', pad=15)
    ax1.text(0.5, -0.1, f"Redundancy in {N}-node system\nHigh Computational Barrier", 
             horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, 
             fontsize=9, color='#666666', style='italic')
    ax1.axis('off')

    # --- PANEL 2: THE RIS FRAMEWORK (SPARSIFIED) ---
    # RIS logic: Keep ONLY edges connected to hubs. 
    ris_edges = [e for e in G_truth.edges() if e[0] in hubs or e[1] in hubs]
    
    # Hub sizes in RIS refined, and increased base size for visibility
    node_sizes_ris = [ (degrees[n] * 30 + 40) * 0.36 * 0.25 if n in hubs else 12 for n in G_truth.nodes()]
    
    # Same blue color scheme
    node_colors_ris = [color_hub if n in hubs else color_node for n in G_truth.nodes()]
    
    # Draw cleaned network - showing disconnected nodes clearly
    nx.draw_networkx_edges(G_truth, pos, ax=ax2, edgelist=ris_edges, 
                           alpha=0.8, edge_color=color_edges_ris, width=0.2)
    nx.draw_networkx_nodes(G_truth, pos, ax=ax2, node_size=node_sizes_ris, 
                           node_color=node_colors_ris, linewidths=0.5, edgecolors='white')
    
    ax2.set_title("RIS Framework\n$O(N)$ Adaptive Sparsification", 
                 fontsize=12, fontweight='bold', color='#1a1a1a', pad=15)
    ax2.text(0.5, -0.1, "Topological Preservation\n99% Gain in Efficiency", 
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, 
             fontsize=9, color=color_hub, fontweight='bold')
    ax2.axis('off')
    
    # Central Transformation Arrow
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch((0.485, 0.5), (0.515, 0.5), mutation_scale=20, 
                            color='#333333', arrowstyle='-|>')
    fig.add_artist(arrow)

    plt.tight_layout()
    
    # Save to /results if on Code Ocean, otherwise local
    if os.path.exists("/results"):
        out_dir = "/results"
    else:
        out_dir = os.path.join(os.path.dirname(BASE_DIR), "results")
        
    svg_path = os.path.join(out_dir, 'graphical_abstract.svg')
    pdf_path = os.path.join(out_dir, 'graphical_abstract.pdf')
    
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    
    print(f"Final compared graphical abstract (N={N}, Blue) saved to {svg_path} and {pdf_path}")

if __name__ == "__main__":
    generate_graphical_abstract()
