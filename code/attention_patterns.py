# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import numpy as np
import random

class AttentionPatterns:
    """
    Simulates the edge-generation logic of common Efficient Transformer patterns
    to allow for topological comparison with RIS.
    """
    
    @staticmethod
    def sliding_window(n_nodes, window_size, periodic=False):
        """
        Simulates Local Attention (e.g., Longformer local radius).
        Connects node i to [i - w/2, ..., i + w/2].
        """
        edges = set()
        half_window = window_size // 2
        
        for i in range(n_nodes):
            start = i - half_window
            end = i + half_window + 1
            
            for j in range(start, end):
                if j == i: continue
                
                target = j
                if periodic:
                    target = j % n_nodes
                elif target < 0 or target >= n_nodes:
                    continue
                
                # Undirected graph storage
                edge = tuple(sorted((i, target)))
                edges.add(edge)
                
        return edges

    @staticmethod
    def pure_random(n_nodes, num_edges):
        """
        Simulates standard Random Edge (RE) or 'Random' attention blocks.
        """
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, n_nodes - 1)
            v = random.randint(0, n_nodes - 1)
            if u != v:
                edges.add(tuple(sorted((u, v))))
        return edges

    @staticmethod
    def bigbird_simulation(n_nodes, density_budget):
        """
        Simulates BigBird's three components:
        1. Window attention (Local)
        2. Random attention (Stochastic)
        3. Global attention (ITC - Internal Transformer Construction, global tokens)
        
        Approximation:
        - 50% of budget to Window
        - 20% of budget to Global Tokens (first/last few nodes as CLS/SEP proxies)
        - 30% of budget to Random
        """
        total_edges = int(n_nodes * (n_nodes - 1) // 2 * density_budget) if density_budget < 1.0 else int(density_budget)
        
        # 1. Window (Global Local)
        # Solve k * N approx 0.5 * Budget
        # Each node connects to w neighbors. Total edges approx N * w / 2 (undirected)
        # w = Budget * 0.5 * 2 / N = Budget / N
        effective_w = max(2, int((total_edges * 0.5 * 2) / n_nodes))
        window_edges = AttentionPatterns.sliding_window(n_nodes, effective_w)
        
        # 2. Global Tokens (g). Connect g nodes to ALL other nodes.
        # Budget 20%. Edges = g * N.
        # g = Budget * 0.2 / N
        num_global = max(2, int((total_edges * 0.2) / n_nodes))
        global_edges = set()
        # Make first 'num_global' nodes the global tokens (like CLS)
        for i in range(num_global):
            for j in range(n_nodes):
                if i != j:
                    global_edges.add(tuple(sorted((i, j))))
                    
        # 3. Random
        current_count = len(window_edges.union(global_edges))
        remaining = max(0, total_edges - current_count)
        random_edges = AttentionPatterns.pure_random(n_nodes, remaining)
        
        return window_edges.union(global_edges).union(random_edges)

    @staticmethod
    def longformer_simulation(n_nodes, density_budget):
        """
        Simulates Longformer:
        Dominantly Window + minimal Global (dilated not implemented for simplicity as it maps to window at scale).
        
        Approximation:
        - 90% Window
        - 10% Global (Task specific tokens)
        """
        total_edges = int(n_nodes * (n_nodes - 1) // 2 * density_budget) if density_budget < 1.0 else int(density_budget)
        
        # w = Budget * 0.9 * 2 / N
        effective_w = max(2, int((total_edges * 0.9 * 2) / n_nodes))
        window_edges = AttentionPatterns.sliding_window(n_nodes, effective_w)
        
        num_global = max(1, int((total_edges * 0.1) / n_nodes))
        global_edges = set()
        for i in range(num_global):
             for j in range(n_nodes):
                if i != j:
                    global_edges.add(tuple(sorted((i, j))))
                    
        return window_edges.union(global_edges)

    @staticmethod
    def longformer_simulation_vectorized(n_nodes, density_budget):
        """
        Vectorized simulation of Longformer patterns using NumPy.
        Returns a NumPy array of edges (M, 2).
        """
        total_edges = int(n_nodes * (n_nodes - 1) // 2 * density_budget) if density_budget < 1.0 else int(density_budget)
        
        # 1. Window Attention (90% budget)
        effective_w = max(2, int((total_edges * 0.9 * 2) / n_nodes))
        half_w = effective_w // 2
        
        # Create window edges using broadcasting
        # For each node i, connect to i+1 ... i+half_w
        # We only generate i < j edges to avoid duplicates and self-loops naturally
        
        # Nodes: [0, 1, ..., N-1]
        nodes = np.arange(n_nodes)
        
        # Offsets: [1, 2, ..., half_w]
        offsets = np.arange(1, half_w + 1)
        
        # Broadcast: nodes[:, None] + offsets[None, :] -> (N, half_w)
        # We need to handle boundary conditions (periodic or clipped). 
        # Longformer is usually not periodic, so we clip or mask. 
        # But simple addition works if we filter out >= n_nodes later.
        u_window = np.repeat(nodes, half_w)
        v_window = (nodes[:, None] + offsets[None, :]).ravel()
        
        # Filter valid edges (v < n_nodes)
        mask = v_window < n_nodes
        u_window = u_window[mask]
        v_window = v_window[mask]
        
        window_edges_np = np.column_stack((u_window, v_window))
        
        # 2. Global Attention (10% budget)
        # Connect first 'num_global' nodes to all others
        num_global = max(1, int((total_edges * 0.1) / n_nodes))
        
        # Global sources: [0, 1, ..., num_global-1]
        g_sources = np.arange(num_global)
        # Global targets: [0, ..., N-1]
        g_targets = np.arange(n_nodes)
        
        # Cartesian product
        u_global = np.repeat(g_sources, n_nodes)
        v_global = np.tile(g_targets, num_global)
        
        # Remove self-loops and duplicates (since global tokens are also in window)
        # But strictly speaking, set union handles this. For performance, we can just concat and unique later.
        # Filter self-loops
        mask_self = u_global != v_global
        u_global = u_global[mask_self]
        v_global = v_global[mask_self]
        
        # Ensure u < v for global edges to match undirected format
        global_edges_np = np.column_stack((u_global, v_global))
        # Sort each row
        global_edges_np.sort(axis=1)
        
        # unique global edges
        global_edges_np = np.unique(global_edges_np, axis=0)
        
        # Combine
        all_edges = np.vstack((window_edges_np, global_edges_np))
        all_edges = np.unique(all_edges, axis=0) # Remove duplicates between sets
        
        return all_edges

    @staticmethod
    def bigbird_simulation_vectorized(n_nodes, density_budget):
        """
        Vectorized simulation of BigBird patterns.
        Returns a NumPy array of edges (M, 2).
        """
        total_edges = int(n_nodes * (n_nodes - 1) // 2 * density_budget) if density_budget < 1.0 else int(density_budget)
        
        # 1. Window (50% budget)
        effective_w = max(2, int((total_edges * 0.5 * 2) / n_nodes))
        half_w = effective_w // 2
        
        nodes = np.arange(n_nodes)
        offsets = np.arange(1, half_w + 1)
        
        u_window = np.repeat(nodes, half_w)
        v_window = (nodes[:, None] + offsets[None, :]).ravel()
        
        mask = v_window < n_nodes
        u_window = u_window[mask]
        v_window = v_window[mask]
        window_edges_np = np.column_stack((u_window, v_window))
        
        # 2. Global (20% budget)
        num_global = max(2, int((total_edges * 0.2) / n_nodes))
        
        g_sources = np.arange(num_global)
        g_targets = np.arange(n_nodes)
        
        u_global = np.repeat(g_sources, n_nodes)
        v_global = np.tile(g_targets, num_global)
        
        mask_self = u_global != v_global
        u_global = u_global[mask_self]
        v_global = v_global[mask_self]
        
        global_edges_np = np.column_stack((u_global, v_global))
        global_edges_np.sort(axis=1)
        
        # 3. Random (30% budget + remainder)
        # Estimate current edges to calculate remainder accurately
        # (This is an approximation since window/global overlap)
        est_current = len(window_edges_np) + len(global_edges_np) 
        # Overlap is small for low density, so we assume disjoint for speed or just aim for target count
        
        # Let's just create random edges based on 30% budget
        n_random = int(total_edges * 0.3)
        
        if n_random > 0:
            u_rand = np.random.randint(0, n_nodes, size=n_random)
            v_rand = np.random.randint(0, n_nodes, size=n_random)
            
            # Remove self-loops
            mask_r = u_rand != v_rand
            u_rand = u_rand[mask_r]
            v_rand = v_rand[mask_r]
            
            rand_edges_np = np.column_stack((u_rand, v_rand))
            rand_edges_np.sort(axis=1) # Sort for undirected consistency
        else:
            rand_edges_np = np.empty((0, 2), dtype=int)

        # 4. Combine all
        all_edges = np.vstack((window_edges_np, global_edges_np, rand_edges_np))
        all_edges = np.unique(all_edges, axis=0)
        
        return all_edges
