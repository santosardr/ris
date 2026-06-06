import os
import json
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from ris_core import RISSimulation
from attention_patterns import AttentionPatterns

# ---------------------------------------------------------
# Environment detection for Code Ocean vs Local (ibteci)
# ---------------------------------------------------------
# Na ibteci, as pastas /data e /code não existem como raízes globais.
# No Ocean Code elas são o padrão.
IS_CODE_OCEAN = os.path.exists("/data") and os.path.exists("/code")

if IS_CODE_OCEAN:
    RES_DIR = "/results"
    SEQ_LENGTHS = [512, 1024] # Optimized for Code Ocean (CPU/Memory constraints)
    print("[INFO] Code Ocean environment detected. Using optimized sequence lengths (Max 1k).")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RES_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
    # Na ibteci rodamos o benchmark completo, do 512 até o 16k
    SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384] 
    print("[INFO] Local/Server environment detected (ibteci). Using full evaluation range (Max 16k).")

os.makedirs(RES_DIR, exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_TRIALS = 30

def get_attention_matrix(model_name, seq_len):
    print(f"Loading model {model_name} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map="cpu")
    
    realistic_text = """
    Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Recently, generative artificial intelligence networks have surpassed many previous benchmarks. 
    The Transformer architecture, introduced in 2017 by Vaswani et al., relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. This allows for significantly more parallelization during training, enabling the training of large language models (LLMs) on unprecedented amounts of data. 
    However, the self-attention mechanism computes a relatedness score for every pair of tokens in the input sequence, leading to a computational and memory complexity of O(N^2), where N is the sequence length. This quadratic bottleneck fundamentally limits the context window size of these models.
    To mitigate this, various sparse attention mechanisms have been proposed, such as Longformer, BigBird, and Reformer, which use local windows, random connections, or hashing to approximate the full attention matrix in O(N log N) or O(N) time.
    Despite these advances, representing global semantic structure without computing all N^2 pairs remains an open challenge. Network science offers a different perspective: complex systems often exhibit scale-free properties, where a few high-degree hubs mediate most of the information flow. 
    By modeling the attention matrix as a complex network, we hypothesize that stochastically sampling edges to preserve these hub structures—rather than relying on fixed sliding windows—can better approximate the global attention pattern while maintaining linear or quasi-linear complexity.
    """ * (seq_len // 200 + 1)
    
    inputs = tokenizer(realistic_text, return_tensors="pt", max_length=seq_len, truncation=True)
    actual_seq_len = inputs.input_ids.shape[1]
    
    print(f"Running forward pass for sequence length {actual_seq_len}...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    attentions = outputs.attentions
    
    avg_attention = torch.zeros((actual_seq_len, actual_seq_len))
    for layer_attn in attentions:
        avg_attention += layer_attn[0].mean(dim=0).cpu()
        
    avg_attention /= len(attentions)
    sym_attention = (avg_attention + avg_attention.T) / 2.0
    
    return sym_attention.numpy(), actual_seq_len


def evaluate_sparsity_pattern(attention_matrix, edges):
    mass = 0.0
    for u, v in edges:
        mass += attention_matrix[u, v]
    return mass

def run_evaluation():
    for seq_len in SEQ_LENGTHS:
        print(f"\n=========================================")
        print(f"Starting evaluation for sequence length: {seq_len}")
        print(f"=========================================")
        
        output_file = os.path.join(RES_DIR, f"llm_evaluation_results_{seq_len}.json")
        attention_matrix, n_nodes = get_attention_matrix(MODEL_NAME, seq_len)
        top_k_hubs = max(5, int(n_nodes * 0.01))
        
        results = {
            "model": MODEL_NAME,
            "sequence_length": n_nodes,
            "top_k_hubs": top_k_hubs,
            "trials": []
        }
        
        total_possible_edges = (n_nodes * (n_nodes - 1)) / 2
        
        for trial in range(NUM_TRIALS):
            print(f"--- Trial {trial + 1}/{NUM_TRIALS} ---")
            # Set fixed seed for deterministic evaluation matching paper
            np.random.seed(2026 + trial)
            random.seed(2026 + trial)
            
            ris = RISSimulation(n_nodes, strategy='heuristic') 
            ris_edges = ris.generate_candidates()
            
            budget = len(ris_edges)
            density = budget / total_possible_edges
            
            longformer_edges_np = AttentionPatterns.longformer_simulation_vectorized(n_nodes, budget)
            longformer_edges = set([tuple(sorted(e)) for e in longformer_edges_np])
            
            bigbird_edges_np = AttentionPatterns.bigbird_simulation_vectorized(n_nodes, budget)
            bigbird_edges = set([tuple(sorted(e)) for e in bigbird_edges_np])
            
            ris_mass = evaluate_sparsity_pattern(attention_matrix, ris_edges)
            lf_mass = evaluate_sparsity_pattern(attention_matrix, longformer_edges)
            bb_mass = evaluate_sparsity_pattern(attention_matrix, bigbird_edges)
            
            def calc_centrality_corr(edges):
                degrees = np.zeros(n_nodes)
                for u, v in edges:
                    degrees[u] += 1
                    degrees[v] += 1
                true_degrees = np.sum(attention_matrix, axis=1)
                corr = np.corrcoef(true_degrees, degrees)[0, 1]
                return corr
                
            def hub_recall_top_k(edges, top_k):
                degrees = np.zeros(n_nodes)
                for u, v in edges:
                    degrees[u] += 1
                    degrees[v] += 1
                sparse_hubs = set(np.argsort(degrees)[-top_k:])
                true_hubs = set(np.argsort(np.sum(attention_matrix, axis=1))[-top_k:])
                recovered = true_hubs.intersection(sparse_hubs)
                return len(recovered) / top_k, recovered
                
            def calc_mean_edge_distance(edges, recovered_hubs):
                hub_edges = [e for e in edges if e[0] in recovered_hubs or e[1] in recovered_hubs]
                if not hub_edges:
                    return 0.0
                distances = [abs(e[0] - e[1]) for e in hub_edges]
                return np.mean(distances)
                
            ris_corr = calc_centrality_corr(ris_edges)
            lf_corr = calc_centrality_corr(longformer_edges)
            bb_corr = calc_centrality_corr(bigbird_edges)
            
            ris_hub_recall, ris_recovered = hub_recall_top_k(ris_edges, top_k_hubs)
            lf_hub_recall, lf_recovered = hub_recall_top_k(longformer_edges, top_k_hubs)
            bb_hub_recall, bb_recovered = hub_recall_top_k(bigbird_edges, top_k_hubs)
            
            ris_dist = calc_mean_edge_distance(ris_edges, ris_recovered)
            lf_dist = calc_mean_edge_distance(longformer_edges, lf_recovered)
            bb_dist = calc_mean_edge_distance(bigbird_edges, bb_recovered)
            
            trial_result = {
                "trial": trial + 1,
                "budget_edges": budget,
                "density": density,
                "mass_captured": {
                    "RIS": float(ris_mass),
                    "Longformer": float(lf_mass),
                    "BigBird": float(bb_mass)
                },
                "centrality_correlation": {
                    "RIS": float(ris_corr) if not np.isnan(ris_corr) else 0.0,
                    "Longformer": float(lf_corr) if not np.isnan(lf_corr) else 0.0,
                    "BigBird": float(bb_corr) if not np.isnan(bb_corr) else 0.0
                },
                "hub_recall": {
                    "RIS": float(ris_hub_recall),
                    "Longformer": float(lf_hub_recall),
                    "BigBird": float(bb_hub_recall)
                },
                "mean_hub_distance": {
                    "RIS": float(ris_dist),
                    "Longformer": float(lf_dist),
                    "BigBird": float(bb_dist)
                }
            }
            results["trials"].append(trial_result)
            
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\nFinal Summary Table for {seq_len} tokens:")
        print(f"{'Method':<15} | {'Mass (Avg)':<12} | {'Corr (Avg)':<12} | {'Recall (%)':<12} | {'Mean Dist':<10}")
        print("-" * 75)
        for method in ["RIS", "Longformer", "BigBird"]:
            avg_mass = np.mean([t['mass_captured'][method] for t in results['trials']])
            avg_corr = np.mean([t['centrality_correlation'][method] for t in results['trials']])
            avg_rec = np.mean([t['hub_recall'][method] for t in results['trials']]) * 100
            avg_dist = np.mean([t['mean_hub_distance'][method] for t in results['trials']])
            print(f"{method:<15} | {avg_mass:<12.1f} | {avg_corr:<12.4f} | {avg_rec:<12.2f}% | {avg_dist:<10.1f}")
        print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
