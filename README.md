<div align="center">

# GenoSieve 🧬

**Intelligent Subsampling for Genomic Epidemiology**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**GenoSieve** is a Python tool for the intelligent subsampling of large genomic sequence datasets. It strategically reduces a dataset to a target size while ensuring the subset is both **representative** (preserving geographic and temporal distributions) and **genetically diverse** (maximizing sequence variability).

This tool is designed for researchers in genomic epidemiology, virology, and evolutionary biology who need to create meaningful, high-quality subsets for phylogenetic inference, modeling, or other computationally intensive analyses.

---

## 🤔 Why GenoSieve?

Randomly downsampling a large dataset is fast but risky. You might lose rare variants, under-represent emerging clades, or skew the geographic distribution of your samples.

GenoSieve solves this problem by using a multi-step, objective-driven approach. It treats subsampling not as a random process, but as an **optimization problem**: finding the best possible subset that preserves the most critical information from the original data.

## 🔬 How It Works

GenoSieve follows a three-stage pipeline for each user-defined group (e.g., by month and clade):

1.  **Stratify**: The dataset is first grouped by a time period (`--date_freq`) and clade. The subsampling process runs independently on each group.
2.  **Allocate**: For each group, GenoSieve determines *how many* sequences to select from each geographic region. This allocation is proportional to the number of available sequences but can be fine-tuned with diversity-aware weights.
3.  **Select**: Finally, the tool selects the *specific* sequences from each region. It uses a **Genetic Algorithm** or a **Hybrid Heuristic** to find the most genetically diverse subset that matches the allocated size.

## ✨ Key Features

-   **Smart Proportional Allocation**: Guarantees fair representation for each region using a smoothing factor (`--alpha`) and a "one-per-region" guarantee.
-   **Diversity-Weighted Allocation**: Optionally adjusts allocation to prioritize regions with higher intra-clade genetic diversity (`--use_diversity`).
-   **Advanced Diversity Maximization**:
    -   **Genetic Algorithm (GA)**: A powerful optimization engine to find the subset that best satisfies a diversity objective (`--objective_function`).
    -   **Hybrid Sampling**: A fast and effective heuristic that prioritizes unique sequences (singletons).
-   **Sequence Vectorization**: Transforms sequences into numerical vectors using k-mers and TF-IDF, enabling robust distance calculations.
-   **Stratified Workflow**: Preserves the temporal and phylogenetic structure of your data by processing groups independently.
-   **Flexible Command-Line Interface (CLI)**: Offers granular control over every step of the pipeline.

---

## ⚙️ Installation

**Requirements:**
*   [Git](https://git-scm.com/)
*   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

Follow these steps in your terminal:

```bash
# 1. Clone the repository
git clone https://github.com/vcarius/genosieve.git
cd genosieve

# 2. Create the Conda environment from the YAML file
# This creates an environment named "genosieve"
conda env create -f environment.yml

# 3. Activate the environment before running the script
conda activate genosieve
```
You are now ready to run GenoSieve!

---

## 🚀 Usage

GenoSieve is run from the command line. The script is named `genosieve.py`.

### Basic Command Structure

```bash
python genosieve.py --METADATA <path_to_metadata.tsv> \
                    --FASTA_ALN <path_to_alignment.fasta> \
                    --target_N <samples_per_group> \
                    [OPTIONS]
```

### Example 1: Fast Hybrid Sampling (Default)

This command subsamples to 500 sequences per month/clade group using the default fast heuristic.

```bash
python genosieve.py --METADATA data/metadata.tsv \
                    --FASTA_ALN data/sequences.fasta \
                    --target_N 500 \
                    --date_freq M
```

### Example 2: Advanced Genetic Algorithm Optimization

This command uses the Genetic Algorithm to select the 300 most diverse sequences, maximizing the minimum distance between any two sequences (`--objective_function min`).

```bash
python genosieve.py --METADATA data/metadata.tsv \
                    --FASTA_ALN data/sequences.fasta \
                    --target_N 300 \
                    --use_GA \
                    --objective_function min \
                    --generations 150 \
                    --verbose
```

---

## 📌 Command-Line Arguments

Here are the most important arguments. For a full list, run `python genosieve.py --help`.

| Argument | Description | Default |
|---|---|---|
| `--METADATA` | **Required**. Path to metadata TSV file. Must contain `name`, `date`, `region`, `clade`. | `None` |
| `--FASTA_ALN` | **Required**. Path to the sequence alignment file in FASTA format. | `None` |
| `--target_N` | **Required**. The total number of sequences to select per group (time/clade). | `None` |
| `--use_GA` | Enable the Genetic Algorithm for diversity maximization. If false, uses the faster hybrid method. | `False` |
| `--objective_function` | The fitness function for the GA. `min` (maximize minimum distance) is recommended. | `min` |
| `--alpha` | Controls the smoothness of proportional allocation (0=uniform, 1=proportional). | `0.5` |
| `--date_freq` | The frequency for grouping dates (`D`=day, `W`=week, `M`=month, `Y`=year). | `M` |
| `--dedup` | If enabled, removes identical sequences within the same region before subsampling. | `False` |
| `--use_diversity` | If enabled, uses sequence diversity to adjust allocation weights. | `False` |
| `--verbose` | Print detailed progress, especially useful for monitoring the GA. | `False` |

---

## 📄 Citation

If you use GenoSieve in your research, please cite this repository:

> Souza, V.C. (2025). *GenoSieve: Intelligent Subsampling for Genomic Epidemiology*. GitHub. https://github.com/vcarius/genosieve

*(Please update the year and details as needed when you publish.)*

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author & Maintainer

Developed by **Vinicius Carius de Souza**.

*   **GitHub**: [@vcarius](https://github.com/vcarius)
*   **LinkedIn**: [@vcarius]([https://www.linkedin.com/](https://www.linkedin.com/in/vinicius-carius-computational-biology/)````
