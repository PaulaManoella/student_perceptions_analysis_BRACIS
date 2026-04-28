# Analyzing Student Perceptions in Course Evaluation: A Hybrid Approach Using Topic Modeling and LLM-Based Automatic Labeling

This repository contains the source code and supplementary materials for the research paper submitted to the **Brazilian Conference on Intelligent Systems (BRACIS)**. The study applies multiple topic modeling algorithms to qualitative student feedback integrated with large language model (LLM) based automatic labeling.

## Overview

The goal of this project is to identify and analyze the latent thematic structures within student perceptions. We employ four topic modeling algorithms — **LSA**, **LDA**, **NMF**, and **BERTopic** — and conduct a qualitative inter-model comparative analysis to identify both cross-model recurring themes and model-specific exclusive clusters. Additionally, we leverage the **Llama 3.1 LLM** for automatic generative topic labeling.

## Results and Discussion Notebook

A dedicated Jupyter Notebook is provided for the comprehensive presentation and analysis of all results:

> **[`notebook/results_and_discussion.ipynb`](notebook/results_and_discussion.ipynb)**

This notebook serves as a self-contained technical report that consolidates all pipeline outputs, including:

- **Coherence and IRBO metric visualizations** for determining the optimal number of topics ($k$).
- **Topic keyword tables** for each of the four models (LSA, LDA, NMF, BERTopic).
- **Qualitative inter-model comparative analysis**, featuring common topics identified via Louvain community detection and model-exclusive clusters.
- **Automatic generative topic labels** produced by the Llama 3.1 LLM for the BERTopic results.

## Project Structure

```
student_perceptions_analysis_BRACIS/
├── data/                              # Raw dataset files (not tracked by git)
│   └── {xxxxxx}-*.csv / .xlsx           # Institutional evaluation data (multiple semesters)
├── notebook/
│   └── results_and_discussion.ipynb   # Results presentation and analysis notebook
├── output/
│   ├── figure/
│   │   ├── coherence_plot.pdf/.png    # Coherence (Cv) metric plot across models
│   │   ├── irbo_plot.pdf/.png         # IRBO diversity metric plot across models
│   │   └── super_topics.pdf/.png      # Louvain community detection graph
│   ├── llm_labeling/
│   │   ├── bertopic_llm_labels_en_ver.csv  # LLM-generated labels (English)
│   │   └── bertopic_llm_labels_pt_ver.csv  # LLM-generated labels (Portuguese)
│   └── topic_modeling/
│       ├── all_models_topics_en.csv   # Consolidated topics from all models (English)
│       ├── all_models_topics_pt.csv   # Consolidated topics from all models (Portuguese)
│       ├── exclusive_topics.csv       # Model-exclusive topics identified
│       └── super_topics.csv           # Common topics across models (communities)
├── src/
│   ├── data/
│   │   ├── df_bertopic_topics.pkl     # Serialized BERTopic topic results
│   │   ├── df_lda_topics.pkl          # Serialized LDA topic results
│   │   ├── df_lsa_topics.pkl          # Serialized LSA topic results
│   │   └── df_nmf_topics.pkl          # Serialized NMF topic results
│   ├── data_mining.py                 # Topic modeling algorithms (LSA, LDA, NMF, BERTopic)
│   ├── evaluation.py                  # Coherence, IRBO, cosine similarity, and Louvain
│   ├── preprocessing.py               # Text preprocessing and cleaning
│   ├── selection.py                   # Data filtering 
│   ├── transformation.py             # BoW, TF-IDF, dictionary construction
│   └── utils.py                       # Utility functions 
├── main.py                            # Main pipeline: preprocessing → topic modeling
├── coherence_irbo_evaluation.py       # Coherence and IRBO evaluation across k topics
├── inter-model_alignment.py           # Inter-model topic alignment
├── LLM_labeling.py                    # Automatic topic labeling with Llama 3.1
├── pyproject.toml                     # Project dependencies (Poetry)
├── poetry.lock                        # Locked dependency versions
└── README.md
```

## Pipeline Stages

The analysis is organized into four sequential pipeline stages:

### 1. Coherence & IRBO Evaluation (`coherence_irbo_evaluation.py`)

Evaluates each model across a range of $k$ values ($2 \leq k \leq 25$) using:
- **Coherence (Cv):** Measures the semantic interpretability of discovered topics.
- **IRBO (Inverted Rank-Biased Overlap):** Measures topic diversity across the model's output.


### 2. Topic Modeling (`main.py`)

Executes the full data processing and topic modeling pipeline:
- Loads and filters the raw dataset.
- Applies text preprocessing.
- Generates text vector representations.
- Runs four topic modeling algorithms with pre-defined number of topics ($k$):
  - **LSA** ($k = 25$)
  - **LDA** ($k = 13$)
  - **NMF** ($k = 13$)
  - **BERTopic** ($k = 14$)


### 3. Inter-model Alignment (`inter-model_alignment.py`)

Performs a qualitative cross-model analysis:
- Generates topic vector embeddings using a Portuguese sentence-transformer model (PORTULAN/serafim).
- Computes pairwise cosine similarity across all topics from all models.
- Generates a non-directed graph where nodes are topics and edges are cosine similarities.
- Applies the **Louvain community detection algorithm** to identify common and exclusive topic clusters.

### 4. Automatic Topic Labeling (`LLM_labeling.py`)

Generates concise, descriptive labels for BERTopic clusters using a local **Llama 3.1 8B** model:
- Processes each topic's keywords and representative documents.
- Produces short labels capturing the theme of each topic.

## Requirements

- **Python** ≥ 3.12, < 3.14
- **Poetry** for dependency management

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `nltk`, `spacy` | NLP preprocessing |
| `gensim` | LSA, LDA, and coherence evaluation |
| `scikit-learn` | NMF and TF-IDF |
| `bertopic` | Neural topic modeling |
| `sentence-transformers` | Text embeddings |
| `umap-learn` | Dimensionality reduction |
| `llama-cpp-python` | Local LLM inference |
| `python-louvain` | Community detection |
| `rbo` | Rank-biased overlap metric |
| `matplotlib`, `seaborn` | Visualization |

## Getting Started

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Run the topic modeling pipeline:**
   ```bash
   poetry run python main.py
   ```

3. **Run the coherence and IRBO evaluation:**
   ```bash
   poetry run python coherence_irbo_evaluation.py
   ```

4. **Run the inter-model alignment:**
   ```bash
   poetry run python inter-model_alignment.py
   ```

5. **Run the LLM labeling** (requires the Llama 3.1 model file):
   ```bash
   poetry run python LLM_labeling.py
   ```

6. **View the results notebook:**
   ```bash
   poetry run jupyter notebook notebook/results_and_discussion.ipynb
   ```

## Data Availability

The raw dataset (`data/`) is not included in this repository to protect student and faculty privacy. The `output/` directory contains all pre-computed results necessary to reproduce the analysis presented in the notebook.

## License

This project is part of an academic research submission. Please refer to the accompanying paper for citation guidelines.
