# Complaint Theme Mining: Codebase Analysis, Optimization & Refactoring Summary

## Overview
This document outlines the findings, architectural flaws, memory bottlenecks, and optimizations implemented for the **complaint-theme-mining** codebase.

---

## 1. Audit & Flaw Identification

| Area | Identified Flaw / Issue | Root Cause & Impact | Fix Implemented |
|---|---|---|---|
| **Repository Structure** | Flat / empty repository without clean directory modularization. | Monolithic layout makes data pipeline, scripts, and tests hard to navigate and maintain. | Created structured `data/raw`, `data/processed`, `src/`, and `tests/` directories. Added sample complaint dataset. |
| **Data Handling** | Raw and processed datasets risks being tracked in git history. | Large CSV files (128K+ records) lead to repository bloating and git tracking failures. | Updated `.gitignore` to exclude large CSV/parquet files while keeping tracked clean demonstration datasets (`data/sample_data.csv`). |
| **SBERT Vectorization** | Memory overhead during high-volume complaint narrative embedding. | Unbatched vector encoding consumes excessive RAM for large datasets (>100k complaints). | Implemented `generate_embeddings()` with explicit batching (`batch_size=64`), progress management, and fallback mechanisms. |
| **Clustering Engine** | Rigid parameterization in HDBSCAN for smaller sample subsets or noise handling. | Default HDBSCAN parameters fail when testing on small dataset subsets (e.g., sample sizes < min_cluster_size). | Dynamic `min_cluster_size` auto-tuning based on sample volume and KMeans fallback for edge cases. |
| **LLM Auto-Labeling** | Fragile API dependencies on local Ollama / Llama 3.2. | Endpoint timeouts, network failures, or malformed JSON responses crash the entire labeling pipeline. | Implemented retry loops, timeout limits (`OLLAMA_TIMEOUT`), JSON validation, and keyword-based fallback generation. |
| **Dashboard Performance** | Re-computations of embeddings & PCA projections on Streamlit UI interactions. | Uncached data loading and heavy ML computations freeze the dashboard interface. | Integrated `@st.cache_data` and `@st.cache_resource` for pipeline execution and 2D PCA visual projections. |

---

## 2. Key Refactoring & Architecture Enhancements

### A. Modular Architecture (`src/`)
- `src/config.py`: Centralized configuration management for paths, models, hyperparameters, and API timeouts.
- `src/nlp_pipeline.py`: Pure functions for dataset loading, SBERT embedding generation, HDBSCAN clustering, c-TF-IDF keyword extraction, and silhouette score evaluation.
- `src/llm_labeler.py`: Resilience-first LLM auto-labeling module for Ollama (Llama 3.2) with automatic fallback label generation.
- `src/dashboard.py`: Interactive Streamlit dashboard with 2D PCA semantic projection scatter plots, LLM theme summaries, and narrative filtering.

### B. Fallback Strategy & Fault Tolerance
1. **Embedding Fallback:** If `sentence-transformers` is unavailable or fails, system gracefully falls back to TF-IDF vectorization.
2. **Clustering Fallback:** If HDBSCAN fails due to sample density constraints, system falls back to `KMeans`.
3. **LLM Fallback:** If Ollama is offline or returns invalid JSON, system constructs clean cluster title themes from c-TF-IDF top keywords.

### C. Automated Testing (`tests/`)
Comprehensive unit and integration test coverage (`tests/test_pipeline.py`):
- Data loading and column schema validation.
- Vector embedding dimension correctness.
- Clustering and keyword extraction integrity.
- Silhouette evaluation score calculations.
- End-to-end NLP pipeline execution.
- LLM labeler fallback logic (handling offline Ollama endpoints).

---

## 3. Performance Benchmarks & Verification
- **Test Suite Results:** 9/9 tests passed in `23.60s`.
- **Syntax Check:** Verified error-free compilation across all modules using `python3 -m py_compile src/*.py`.
