# Complaint Theme Mining & Auto-Labeling Pipeline

An end-to-end NLP pipeline and interactive dashboard for analyzing Consumer Financial Protection Bureau (CFPB) complaints using **SBERT embeddings**, **HDBSCAN clustering**, **c-TF-IDF keyword extraction**, and **Llama 3.2 (via Ollama)** for local LLM theme auto-labeling.

---

## 📌 Features

- **Memory-Efficient Vectorization:** SBERT (`all-MiniLM-L6-v2`) dense embeddings generated in configurable batch sizes.
- **Density-Based Clustering:** HDBSCAN for discovering naturally occurring complaint themes without specifying fixed cluster counts.
- **c-TF-IDF Keyword Extraction:** Automatic extraction of top topic terms per complaint cluster.
- **Local LLM Auto-Labeling:** Integration with local Llama 3.2 via Ollama to generate executive titles and summaries for each complaint cluster, with automatic keyword-based fallback if LLM is offline.
- **Interactive Streamlit Dashboard:** 2D PCA projection scatter plot, theme filtering, summary cards, and complaint narrative inspection with full Streamlit caching (`@st.cache_data`).
- **Silhouette Validation:** Dynamic calculation of cluster validation metrics and noise ratios.

---

## 📁 Project Structure

```
complaint-theme-mining/
├── data/
│   ├── raw/                  # Raw complaint CSV files (Git ignored)
│   ├── processed/            # Processed output datasets with cluster assignments
│   └── sample_data.csv       # Tracked sample dataset for testing/demo
├── src/
│   ├── __init__.py
│   ├── config.py             # Global project configuration & parameters
│   ├── nlp_pipeline.py       # SBERT + HDBSCAN + c-TF-IDF NLP pipeline
│   ├── llm_labeler.py        # Ollama Llama 3.2 auto-labeler & fallback logic
│   └── dashboard.py          # Streamlit web interface
├── tests/
│   └── test_pipeline.py      # Pytest unit and integration test suite
├── .gitignore                # Excludes large CSVs, models, and virtual environments
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── SUMMARY.md                # Comprehensive refactoring & optimization analysis
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/complaint-theme-mining.git
cd complaint-theme-mining
pip install -r requirements.txt
```

### 2. Running the Local LLM (Optional)

If you want LLM auto-generated theme summaries using Llama 3.2, ensure [Ollama](https://ollama.ai/) is installed and running:

```bash
ollama run llama3.2
```
*Note: If Ollama is not running, the pipeline gracefully falls back to c-TF-IDF keyword-driven theme titles.*

### 3. Run the Streamlit Dashboard

Launch the interactive web application:

```bash
streamlit run src/dashboard.py
```

---

## 🧪 Running Tests

Run the full automated pytest suite to verify all pipeline components and fallbacks:

```bash
python3 -m pytest
```

---

## 📊 Pipeline Overview

```
[ Raw Complaints Data ]
          │
          ▼
 [ SBERT Embedding (all-MiniLM-L6-v2) ]
          │
          ▼
   [ HDBSCAN Clustering ]
          │
  ┌───────┴────────┐
  ▼                ▼
[ c-TF-IDF ]  [ 2D PCA Mapping ]
  │                │
  ▼                │
[ Ollama Llama 3.2 Labeler ] ◄─ (Fallback to Keywords if offline)
  │                │
  └───────┬────────┘
          ▼
[ Streamlit Dashboard Visuals ]
```

---

## 📄 License
MIT License
