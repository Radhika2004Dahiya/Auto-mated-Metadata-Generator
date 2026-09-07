# Complaint Theme Mining & Auto-Labeling Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![SBERT](https://img.shields.io/badge/Model-SBERT%20all--MiniLM--L6--v2-green.svg)](https://www.sbert.net/)
[![HDBSCAN](https://img.shields.io/badge/Clustering-HDBSCAN-orange.svg)](https://hdbscan.readthedocs.io/)
[![Ollama Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2%20(Ollama)-purple.svg)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end NLP pipeline and interactive Streamlit dashboard designed to analyze 128,000+ Consumer Financial Protection Bureau (CFPB) complaint narratives. The pipeline leverages **SBERT embeddings**, **HDBSCAN clustering**, **c-TF-IDF keyword extraction**, and local **Llama 3.2 (via Ollama)** for automated theme labeling and summary generation.

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Hardware & Software Prerequisites](#-hardware--software-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Local LLM (Ollama) Configuration](#-local-llm-ollama-configuration)
- [Running the Pipeline & Dashboard](#-running-the-pipeline--dashboard)
- [Sampling & Scale Strategy](#-sampling--scale-strategy)
- [HDBSCAN Soft Clustering & Noise Mitigation](#-hdbscan-soft-clustering--noise-mitigation)
- [Running Automated Tests](#-running-automated-tests)
- [License](#-license)

---

## ✨ Features

- **Batched SBERT Vectorization:** Efficient dense embedding generation using `all-MiniLM-L6-v2` with batch size controls (`batch_size=64`).
- **Density-Based Clustering:** HDBSCAN identifies naturally occurring complaint themes without specifying arbitrary cluster counts.
- **Soft Clustering & Membership Vectors:** Reassigns unclustered noise points (-1) to their highest probability cluster using HDBSCAN soft membership scores.
- **c-TF-IDF Keyword Extraction:** Class-based TF-IDF extracts top representative keywords per theme cluster.
- **Local LLM Auto-Labeling:** Integration with local Llama 3.2 (Ollama) to produce executive titles and summaries for each cluster, with automatic keyword-based fallback if LLM services are offline.
- **Interactive Streamlit Dashboard:** 2D PCA semantic projection scatter plots, theme filtering, metric summary cards, and narrative search with full Streamlit caching (`@st.cache_data`).

---

## 📁 Project Architecture & Directory Structure

```text
complaint-theme-mining/
├── data/
│   ├── raw/                  # Raw complaint CSV files (Git-ignored)
│   ├── processed/            # Processed outputs with cluster assignments (Git-ignored)
│   └── sample_data.csv       # Tracked sample dataset (10 records) for demonstration
├── src/
│   ├── __init__.py
│   ├── config.py             # Global pipeline parameters, model paths & API timeouts
│   ├── nlp_pipeline.py       # SBERT + HDBSCAN + c-TF-IDF + Soft Clustering pipeline
│   ├── llm_labeler.py        # Ollama Llama 3.2 auto-labeler & fallback engine
│   └── dashboard.py          # Interactive Streamlit dashboard application
├── tests/
│   └── test_pipeline.py      # Automated pytest suite
├── .gitignore                # Excludes large CSVs, models, and virtual environments
├── requirements.txt          # Explicit Python library dependencies
├── README.md                 # Complete repository documentation
└── SUMMARY.md                # Detailed refactoring analysis & benchmarks
```

---

## 💻 Hardware & Software Prerequisites

- **Operating System:** Linux, macOS, or Windows WSL2
- **Python:** Python 3.10, 3.11, or 3.12
- **RAM Requirements:**
  - Minimum 8 GB RAM (for running SBERT + HDBSCAN on sample data)
  - Recommended 16 GB+ RAM (for running 30,000+ complaint batches)
- **Optional GPU / VRAM:** 4 GB+ VRAM (Accelerates SBERT embedding generation and Ollama Llama 3.2 inference)

---

## 🚀 Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-org/complaint-theme-mining.git
cd complaint-theme-mining

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🤖 Local LLM (Ollama) Configuration

The pipeline integrates with **Ollama** running **Llama 3.2** to automatically generate human-readable titles and summaries for each complaint cluster.

### 1. Install Ollama
Download and install Ollama from [https://ollama.ai/download](https://ollama.ai/download).

### 2. Pull Llama 3.2 Model
Pull the lightweight 3B Llama 3.2 model:
```bash
ollama pull llama3.2
```

### 3. Start Ollama Server
```bash
ollama serve
```
*Note: If Ollama is offline or uninstalled, the pipeline gracefully falls back to generating clean theme titles from c-TF-IDF keywords.*

---

## 📊 Running the Pipeline & Dashboard

### 1. Launch Streamlit Dashboard

Run the Streamlit application:
```bash
streamlit run src/dashboard.py
```
Open your browser at `http://localhost:8501`.

### 2. Dashboard UI Overview

1. **KPI Header Cards:** Displays Total Complaints, Discovered Themes, Noise Points, and Silhouette Validation Score.
2. **2D Semantic Space Visualization:** Interactive Plotly scatter plot projecting SBERT embeddings into 2D PCA space, color-coded by LLM theme title.
3. **Cluster Summaries & LLM Labels:** Expandable cards displaying top keywords, executive summaries, and sample sizes per theme.
4. **Narrative Explorer:** Searchable and filterable table allowing deep inspection of individual consumer complaint narratives.

---

## ⚖️ Sampling & Scale Strategy

Processing full 128,000+ complaint datasets with exact SBERT embeddings and pairwise distance matrices can impose high computational and memory bounds. When sampling down (e.g., to 30,000 rows), sampling bias is mitigated through the following strategies:

1. **Stratified Sampling:** Samples are stratified across `product` and `issue` categories to preserve the original distribution of consumer complaints.
2. **Temporal Windowing:** Sampling complaints across distinct quarterly/monthly time slices to prevent seasonal bias.
3. **Chunked Embedding Generation:** Embeddings are generated in mini-batches (`batch_size=64`), keeping peak memory consumption low.

---

## 🧩 HDBSCAN Soft Clustering & Noise Mitigation

HDBSCAN often categorizes ambiguous or boundary complaints as noise points (`cluster = -1`). To handle high noise ratios without losing valuable data:

1. **Membership Vectors:** The pipeline utilizes HDBSCAN's `all_points_membership_vectors()` to calculate soft probability distributions across all valid clusters for every data point.
2. **Noise Reassignment (`reassign_noise=True`):** Unclustered points (`-1`) can be assigned to the cluster with their highest membership probability if it exceeds a confidence threshold (default `0.10`).

---

## 🧪 Running Automated Tests

Execute the pytest suite to verify dataset loading, embeddings, HDBSCAN clustering, soft clustering, c-TF-IDF keyword extraction, and LLM fallback logic:

```bash
python3 -m pytest
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
