# Real-World Data: Universality Benchmarking Suite

To verify the universality of RIS, the following datasets are used across different domains. For **Code Ocean**, these files must be present in the `/data` directory.

### 1. Social (LiveJournal)
- **File:** `com-lj.ungraph.txt` (Uncompressed)
- **Source:** [SNAP - LiveJournal](https://snap.stanford.edu/data/com-LiveJournal.html)
- **Nodes:** 3,997,962 | **Edges:** 34,681,189.
- **Use:** Figure 4 (Hub Recall) and Real-World Pearson Correlation.

### 2. Science (ca-GrQc)
- **File:** `ca-GrQc.txt.gz` (Compressed)
- **Source:** [SNAP - Collaboration Network](https://snap.stanford.edu/data/ca-GrQc.html)
- **Use:** Universality Benchmark (Figure 5).

### 3. Financial (Bitcoin Alpha)
- **File:** `soc-sign-bitcoin-alpha.csv.gz` (Compressed)
- **Source:** [SNAP - Bitcoin Alpha](https://snap.stanford.edu/data/soc-sign-bitcoinalpha.html)
- **Use:** Universality Benchmark (Figure 5).

### 4. Infrastructure (US Power Grid)
- **File:** `opsahl-powergrid.tar.bz2` (Compressed)
- **Source:** [Opsahl - Power Grid](https://toreopsahl.com/datasets/#networks)
- **Use:** Universality Benchmark (Figure 5 boundary case - Spatial Lattice).

### 5. Climate / Transport (Global Flights)
- **File:** `routes.dat` (Uncompressed CSV)
- **Source:** [OpenFlights - Routes](https://openflights.org/data.html)
- **Use:** Universality Benchmark (Figure 5).

---

## How to Obtain Large Datasets

Due to file size constraints, some datasets are not included directly in Git.

### LiveJournal (Direct Download)
```bash
wget https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
gunzip com-lj.ungraph.txt.gz
```

### Other Datasets
Most other datasets are small or the scripts (`benchmark_*.py`) will attempt to download them automatically if they are missing from the local `data/` directory. However, for a fully offline Reproducible Run (Code Ocean), please ensure all files listed above are uploaded to the `/data` directory.

## File Formats
- **LiveJournal/ca-GrQc:** SNAP edge list (tab-separated, comments start with #).
- **Bitcoin Alpha:** CSV (source, target, rating, time).
- **Power Grid:** Opsahl standard edge list.
- **Global Flights:** OpenFlights routes CSV format.
