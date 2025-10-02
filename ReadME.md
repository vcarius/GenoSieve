```
# Genomic Subsampler: Intelligent Sequence Downsampling

**Genomic Subsampler (GenoSieve)** is a Python tool for **intelligent subsampling of large genomic sequence datasets**.  
It reduces dataset size to a target number while ensuring that the resulting subset is both:  
- **Representative** – preserves geographic and temporal distributions.  
- **Genetically diverse** – maximizes variability among selected sequences.  

This tool is designed for researchers in **genomic epidemiology, virology, and evolutionary biology** who need representative subsets for **phylogenetic inference** or other **computationally intensive analyses**.

---

## ✨ Features

- **Proportional Allocation**: Ensures fair sampling per region/country with smoothing (`alpha`).  
- **Diversity Adjustment**: Optionally favors regions with higher genetic diversity (`beta`).  
- **Diversity Maximization**:  
  - Genetic Algorithm (**GA**) optimization (`avg`, `min`, `sumlogdet`).  
  - Hybrid Sampling (fast heuristic with unique-sequence prioritization).  
- **Sequence Vectorization**: Converts sequences into numerical vectors using **k-mers + TF-IDF**.  
- **Stratified Sampling**: Preserves **temporal** (day, month, year) and **clade** structure.  
- **Command-Line Interface (CLI)**: Fully configurable with multiple options.  

---

## ⚙️ Installation

Requirements:  
- [Git](https://git-scm.com/)  
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)  

### Setup
```bash
# Clone the repository
git clone https://github.com/vcarius/GenoSieve.git
cd GenoSieve

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate GenoSieve
````

---

## 🚀 Usage

Run via command line:

```bash
python GenoSieve.py --METADATA <path_to_metadata.tsv> \
                  --FASTA_ALN <path_to_alignment.fasta> \
                  --target_N <samples_per_group> \
                  [OPTIONS]
```

### Example 1 – Hybrid Sampling (default, fast)

```bash
python GenoSieve.py --METADATA data/metadata.tsv \
                  --FASTA_ALN data/sequences.fasta \
                  --target_N 500 \
                  --date_freq M \
                  --alpha 0.5
```

### Example 2 – Genetic Algorithm (advanced)

```bash
python GenoSieve.py --METADATA data/metadata.tsv \
                  --FASTA_ALN data/sequences.fasta \
                  --target_N 300 \
                  --use_GA \
                  --objective_function min \
                  --generations 150 \
                  --mutation_rate 0.3 \
                  --verbose
```

---

## 📌 Main Command-Line Arguments

| Argument               | Description                                                                   | Default  |
| ---------------------- | ----------------------------------------------------------------------------- | -------- |
| `--FASTA_ALN`          | **Required**. Input alignment (FASTA).                                        | `None`   |
| `--METADATA`           | **Required**. Metadata TSV with columns: `name`, `date`, `region`, `clade`.   | `None`   |
| `--target_N`           | **Required**. Number of sequences per group (time/clade).                     | `None`   |
| `--dedup`              | Remove duplicates by region, keeping oldest.                                  | `False`  |
| `--date_freq`          | Date grouping frequency (`D` = day, `M` = month, `Y` = year).                 | `M`      |
| `--alpha`              | Proportional allocation smoothing exponent (0–1).                             | `0.5`    |
| `--preserve_max_share` | Max fraction of `target_N` per country to enforce "at least one per country". | `0.8`    |
| `--use_diversity`      | Adjust allocations with diversity weights.                                    | `False`  |
| `--beta`               | Weight factor for diversity adjustment.                                       | `0.2`    |
| `--use_GA`             | Enable Genetic Algorithm optimization.                                        | `False`  |
| `--objective_function` | GA fitness function (`avg`, `min`, `sumlogdet`).                              | `min`    |
| `--generations`        | Number of GA generations.                                                     | `200`    |
| `--mutation_rate`      | GA mutation rate.                                                             | `0.25`   |
| `--crossover_rate`     | GA crossover rate.                                                            | `0.9`    |
| `--precompute_metric`  | Distance metric (`cosine`, `euclidean`, etc.).                                | `cosine` |
| `--verbose`            | Verbose mode (show optimization progress).                                    | `False`  |

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Developed by **Vinicius Carius de Souza**.

```


---
