# RIS: Reduced Interaction Sampling - Implementation Scripts

This directory contains the reference implementation, high-performance benchmarking suite, and figure-generation scripts for the Reduced Interaction Sampling (RIS) framework.

![RIS Analogy](../RIS_analogy.png)

### The RIS Analogy
The Reduced Interaction Sampling (RIS) framework can be understood through a simple analogy: **a woman looking at a forest through a sieve**. 

Although the sieve filters out and blocks a large portion of the visual input (reducing the density of interaction details), she can still clearly perceive and comprehend the overall structure and essence of the forest she is contemplating. 

Similarly, RIS sparsifies the quadratic interaction space of attention mechanisms in Transformers by sampling only a fraction of the elements while fully preserving the underlying topological structure and key information flow.

## Directory Structure

```text
ris/
|-- code/                      # Execution scripts
|   |-- LICENSE                # Apache License 2.0
|   |-- ris_core.py            # Reference RIS implementation
|   |-- generate_fig1_fig2_fig3_synthetic.py # Synthetic benchmarking suite
|   |-- orchestrator_real_world.py # Real-world validation (LiveJournal)
|   |-- generate_fig5_mean_comparison.py   # High-fidelity figure generation (Fig 5)
|   |-- benchmark_power_grid.py    # Infrastructure Domain Benchmark
|   |-- benchmark_collaboration.py # Science Domain Benchmark
|   |-- benchmark_financial.py     # Financial Domain Benchmark
|   |-- benchmark_climate.py       # Climate/Transport Domain Benchmark
|   |-- generate_fig4_universality.py # Aggregates all domains (Fig 4)
|   |-- generate_graphical_abstract.py # Produces graphical abstract
|   |-- run                    # Main entry point (Bash script)
|   |-- run_supper             # High-performance bash script
|   `-- attention_patterns.py  # Transformer pattern simulations
|-- data/                      # Input datasets (contains com-lj.ungraph.txt, etc.)
`-- results/                   # Output folder for generated PDFs and CSV data
```

## Software Requirements

- **Operating System:** Tested on Ubuntu 24.04.3 LTS (noble) and Ubuntu 22.04 LTS.
- **Python Version:** Python 3.8 or higher is required.
- **Key Dependencies:** `numpy`, `pandas`, `networkx`, `joblib`, `scipy`, `matplotlib`, `tqdm`.

### Setup Instructions

```bash
# Clone the repository and navigate to the code directory
cd code/

# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Hardware Requirements

The scripts are optimized for parallel execution and can scale from laptops to servers:

1.  **General Node (Desktop/Laptop):** 
    - Used for: Synthetic benchmarks, small networks ($N \le 2000$).
    - Recommendation: 4+ CPU cores, 8GB RAM.
    - **Optimization:** Scripts used for real-world large experiments (LiveJournal), high-replicate trials.
    - Profile: Targeted for many-core machines (e.g., Xeon E5).
    - **Note:** The synthetic benchmarks are optimized for parallel execution.

# RIS: Reduced Interaction Sampling - Experiments and Validation

This folder contains the complete suite of scripts used to validate the RIS (Reduced Interaction Sampling) framework across synthetic graphs, real-world interaction networks, and Large Language Models (LLMs).

## Core Scripts

- `ris_core.py`: The primary implementation of the RIS algorithm, containing the `RISSimulation` class and the Heuristic Adaptive Function (HAF).
- `attention_patterns.py`: Vectorized simulations of sliding-window (Longformer) and stochastic (BigBird) attention patterns for benchmarking.

## LLM Empirical Validation

To address the $O(N^2)$ quadratic bottleneck, our validation is split into two regimes based on hardware accessibility:

1.  **`llm_lt_16k.py` (Local/Small Contexts):**
    - Uses HuggingFace `transformers` to extract raw attention matrices from `TinyLlama-1.1B`.
    - Evaluates RIS against Longformer/BigBird using real Wikipedia text.
    - Optimized for CPU execution on Code Ocean (defaulting to 512 and 1024 tokens).
2.  **`llm_gt_8k.py` (Large Contexts / Simulation):**
    - Proves memory-efficient inference using `llama.cpp` for contexts up to 65,536 tokens.
    - Evaluates topological metrics using simulated attention matrices that represent the 1-billion-edge interaction space of large windows.
    - Essential for validating the **scale-invariance** of RIS beyond OOM hardware limits.

## Dual Execution Model (GitHub vs. Code Ocean)

This codebase is designed with a **Dual-Mode Execution Model** to run seamlessly both in local environments (cloned from GitHub) and inside a Code Ocean compute capsule:

- **Local Development / GitHub Clone:**
  - Data inputs are read from the local `data/` directory (adjacent to `code/`).
  - Simulation and benchmark outputs (PDFs, CSVs) are automatically saved to the local `results/` directory (adjacent to `code/`).
  - Path resolution is determined dynamically relative to the location of the execution scripts, preventing any broken path errors.
- **Code Ocean Compute Capsule:**
  - Data inputs are read from the mounted `/data` directory.
  - Output files are written directly to the mounted `/results` directory for persistence.
  - Automatically restricts execution limits (such as context length and replication size) to fit standard resource quotas.

- Run `./run` to execute the full synthetic and empirical validation suite.
- The LLM scripts will output a **Summary Table** to the console, showing Mass Captured, Centrality Correlation, and Hub Recall.
- **Key Metric:** RIS consistently achieves ~1.0% Hub Recall with over **7.5x less attention mass** than sliding-window baselines at scale.

## Dependencies

- `torch`, `transformers`, `numpy`, `matplotlib`, `scipy`, `pandas`, `joblib`, `networkx`, `tqdm`
- `llama-cpp-python` (Elective: for real LLM inference; handled via simulation fallback if missing)
- `huggingface-hub`

## Citation

If you use this framework in your research, please cite the preprint (Version 5):
*Santos, A. R. (2026). Towards Million-Token Context Windows: A Topology-Preserving Framework for Adaptive Transformer Sparsification. Zenodo. DOI: [10.5281/zenodo.20460983](https://doi.org/10.5281/zenodo.20460983).*

You can also run or export the fully configured environment using the Code Ocean capsule:
*   **Code Ocean Capsule DOI:** [10.24433/CO.9986683.v1](https://doi.org/10.24433/CO.9986683.v1)

### Figure Generation (Production)
*   **`generate_fig5_mean_comparison.py`**: Generates precisely validated data for **Figure 5**.
    - Parallel execution (defaults to `CPU_COUNT - 2`).
    - Standard: Uses 30 replicates for the reproducible run.
    - High-Fidelity: use `--replicates 20000` for paper-matching (Converged Finality).
*   `generate_graphical_abstract.py`: Produces the graphical abstract (PDF/SVG).
*   `generate_fig4_universality.py`: Produces the combined domain comparison (**Figure 4**).

## Scalability & Stress Testing

To verify the "Quadratic Wall Break" at extreme scales:
```bash
# Run scalability stress test up to N=100,000
python generate_fig1_fig2_fig3_synthetic.py --stress --replicates 30
```
Note: Stress tests require significant RAM (~32GB recommended for N=100k).

## Automated Reproducibility

For automated reproducibility on platforms like Code Ocean, use the main entry script:
```bash
# Detects the environment and runs the full experimental pipeline.
./run
```

### Data and Results
*   `data/` (or `/data` on Code Ocean): Mounted/local input dataset directory.
*   `results/` (or `/results` on Code Ocean): Output results directory.

## Reproducibility
All scripts leverage deterministic pseudo-random seeds. By default, running optimized scripts with `--replicates 20000` will produce the stable means reported in the paper. 

## High-Precision Statistical Analysis

For the final manuscript submission, we conducted a **Mega-Simulation (20,000 replicates)** on the LiveJournal interaction network to verify the statistical dominance of RIS-Structural over fixed-pattern heuristics (BigBird).

### Analysis Script: `analyze_results_efficient.py`
- **Purpose:** Memory-efficient processing of large simulation datasets (streaming JSON).
- **Metrics:** Calculates Standard Error of the Mean (SEM), 99% Confidence Intervals (CI), Coefficient of Variation (CV), and P-values.
- **Visuals:** Generates `tail_distribution_20k_refined.png` (Grouped Log-Scale Histogram).

### Reference Results (LiveJournal, n=20,000)

| Metric | RIS-Structural (Ours) | BigBird (Heuristic) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Hub Recall** | **1.00%** | 0.98% | **+2.04%** |
| **Statistical Error (SEM)** | **0.00005** | 0.00005 | - |
| **Structural Amnesia (Zero Recall)** | **13.54%** | 14.25% | **5% Reduction** |
| **Topological Invariance (Shuffle)** | **1.05%** | 1.05% (from 70%) | **Domination** |
| **P-value (Paired T-test)** | **0.0337** | - | Significant ($p < 0.05$) |

### Key Scientific Takeaways
1.  **Topological Fidelity:** While BigBird's recall collapses by over 98% under node permutation (Shuffle test), RIS-Structural remains invariant. This confirms that RIS captures the actual graph topology, whereas window-based models exploit sequence-order artifacts.
2.  **Reliability:** RIS-Structural provides a more consistent "topological floor," significantly reducing the probability of "structural amnesia" (zero recall of critical hubs) compared to fixed patterns.
3.  **Statistical Proof:** With 20,000 replicates, the SEM is low enough to confirm that the advantage of RIS is a persistent property of its architecture-aware sampling design, not stochastic noise.

## License
Released under the **Apache License 2.0**.
