# Copyright (C) 2024-2026 Anderson R. Santos
# Faculty of Computing, Federal University of Uberlândia (UFU), Brazil
# Distributed under the Apache License 2.0
# For commercial licensing inquiries, contact: santosardr@ufu.br

import numpy as np
import networkx as nx
import math
import random

class RISSimulation:
    def __init__(self, n_nodes, strategy='heuristic'):
        self.n = n_nodes
        self.strategy = strategy
        self.nodes = list(range(n_nodes))

    def _calculate_redundancy(self):
        """Implementação da Família de Funções RIS."""
        # Use common logic for HAF unless specifically logarithmic
        if self.strategy in ['heuristic', 'genppi']:
            # Implementação fiel ao código Lisp (HAF - Heuristic Adaptive Function)
            if self.n < 250:
                return max(1, int(round(self.n * 0.5)))
            elif self.n < 3000:
                val = int(round(167.49 * (0.999**self.n)))
                return max(5, val)
            else:
                return 5
        elif self.strategy == 'logarithmic':
            if self.n <= 1: return 1
            return int(math.ceil(3 * math.log2(self.n)))
        else:
            return 5

    def generate_candidates(self):
        candidates = set()
        r = self._calculate_redundancy()
        shuffled_nodes = self.nodes.copy()
        random.shuffle(shuffled_nodes)
        
        if self.strategy == 'genppi':
            # Original GenPPi Algorithm: Shuffle -> Partition (Sublists) -> Redundancy
            dividend = 10
            block_size = max(1, self.n // dividend)
            
            node_to_block = {}
            for b in range(dividend):
                start_id = b * block_size
                end_id = min((b + 1) * block_size, self.n) if b < dividend - 1 else self.n
                block_nodes = shuffled_indices = shuffled_nodes[start_id:end_id]
                
                for node in block_nodes:
                    node_to_block[node] = b
                
                # Intra-sublist dense connections (Local Cliques)
                for i in range(len(block_nodes)):
                    for j in range(i + 1, len(block_nodes)):
                        candidates.add(tuple(sorted((block_nodes[i], block_nodes[j]))))
            
            # Phase 3: Controlled Redundancy (Inter-sublist connectivity)
            for pivot in shuffled_nodes:
                pivot_block = node_to_block[pivot]
                # Sample 'r' distal neighbors
                attempts = 0
                count = 0
                while count < r and attempts < r * 3:
                    target = random.choice(self.nodes)
                    if target != pivot and node_to_block[target] != pivot_block:
                        candidates.add(tuple(sorted((pivot, target))))
                        count += 1
                    attempts += 1
                    
        else:
            # Generalized RIS (NMI Version) - Simple stochastic sampling
            for pivot in shuffled_nodes:
                # Amostragem global redundante
                potential_neighbors = random.sample(self.nodes, min(len(self.nodes), r + 1))
                for neighbor in potential_neighbors:
                    if pivot != neighbor:
                        edge = tuple(sorted((pivot, neighbor)))
                        candidates.add(edge)
        return candidates

def generate_ground_truth(n_nodes, topology):
    """Gera 5 tipos de estruturas de 'Verdade Oculta'."""
    if topology == 'erdos_renyi':
        p = (2 * math.log(n_nodes)) / n_nodes
        return nx.erdos_renyi_graph(n_nodes, p)
    
    elif topology == 'barabasi_albert':
        return nx.barabasi_albert_graph(n_nodes, 5)
    
    elif topology == 'watts_strogatz':
        return nx.watts_strogatz_graph(n_nodes, 10, 0.1)
    
    elif topology == 'stochastic_block':
        # Simula comunidades (4 blocos de tamanhos iguais)
        sizes = [n_nodes // 4] * 4
        # Escala as probabilidades com N para evitar explosão de memória (OOM) no teste de estresse
        # Em N=1000, p_in ~ 0.1. Em N=100k, p_in ~ 0.001
        p_in = min(0.1, (100 * math.log(n_nodes)) / n_nodes)
        p_out = min(0.001, (10 * math.log(n_nodes)) / n_nodes)
        
        probs = [[p_in, p_out, p_out, p_out], 
                 [p_out, p_in, p_out, p_out], 
                 [p_out, p_out, p_in, p_out], 
                 [p_out, p_out, p_out, p_in]]
        return nx.stochastic_block_model(sizes, probs)

    elif topology == 'powerlaw_cluster':
        # BA com alta clusterização (p=0.1 de formar triângulos)
        return nx.powerlaw_cluster_graph(n_nodes, 5, 0.1)
    
    else:
        raise ValueError("Unknown topology")

def run_experiment(config):
    n, topo, strat, rep = config['n'], config['topology'], config['strategy'], config['rep']

    seed_value = hash(f"{n}-{topo}-{strat}-{rep}") % (2**32)
    random.seed(seed_value)
    np.random.seed(seed_value)
    G_gt = generate_ground_truth(n, topo)
    true_edges = set(tuple(sorted(e)) for e in G_gt.edges())
    
    ris = RISSimulation(n, strat)
    candidate_edges = ris.generate_candidates()
    
    recovered_edges = candidate_edges.intersection(true_edges)
    
    G_recovered = nx.Graph()
    G_recovered.add_nodes_from(range(n))
    G_recovered.add_edges_from(list(recovered_edges))
    
    # Métricas
    recall = len(recovered_edges) / len(true_edges) if len(true_edges) > 0 else 0
    total_possible = (n * (n - 1)) / 2
    reduction = 1 - (len(candidate_edges) / total_possible)
    
    try:
        deg_gt = list(nx.degree_centrality(G_gt).values())
        deg_rec = list(nx.degree_centrality(G_recovered).values())
        centrality_corr = np.corrcoef(deg_gt, deg_rec)[0, 1]
    except:
        centrality_corr = 0
        
    return {
        'n': n, 'topology': topo, 'strategy': strat,
        'recall': recall, 'reduction': reduction,
        'centrality_corr': centrality_corr,
        'edges_gt': len(true_edges), 'edges_ris': len(candidate_edges)
    }
