# Evaluating Conversational AI Architectures for Higher Education

## A Comparison of Traditional NLP and Transformer-Based Approaches

**Sheffield Hallam University — MSc Computing Research Project**

---

### Overview

This repository contains the complete implementation and evaluation pipeline for a comparative study of two conversational AI architectures designed to support student interactions in higher education. The project implements a Traditional NLP pipeline (TF-IDF + Cosine Similarity) and a Transformer-based pipeline (Sentence-BERT + Context Enrichment + Re-Ranking), both trained and evaluated on the same dataset using identical quantitative metrics.

---

### Research Question

*RQ1: What is the comparative performance of traditional NLP-based and transformer-based conversational AI architectures in responding to students in a higher education setting in terms of response quality, contextual coherence, intent classification accuracy, and computational speed?*

---

### Dataset

- **Source:** [StudyChat (wmcnicho/StudyChat)]
- **Licence:** CC BY 4.0
- **Size:** 16,851 anonymised student–ChatGPT dialogues collected across two semesters at the University of Massachusetts Amherst
- **Split:** 7,998 training / 2,000 test samples (stratified by topic), with 300 samples used for comparative evaluation

---

### Repository Structure

```
├── Conversational_AI_Architecture_Comparison_StudyChat.ipynb   # Full pipeline notebook
├── README.md
├── comparison_results.csv          # Exported comparative results (generated at runtime)
├── tfidf_detailed_metrics.csv      # Per-sample Traditional NLP metrics (generated at runtime)
└── sbert_detailed_metrics.csv      # Per-sample Transformer metrics (generated at runtime)
```

---

### Architecture Summary

| Component | Traditional NLP Pipeline | Transformer Pipeline |
|---|---|---|
| **Feature Representation** | TF-IDF (10,000 features, bigrams, sublinear TF) | Sentence-BERT (all-MiniLM-L6-v2, 384-dim dense embeddings) |
| **Context Handling** | Single-turn query only | Multi-turn context enrichment via [SEP] concatenation (up to 3 prior turns) |
| **Retrieval Method** | Top-1 cosine similarity | Top-5 retrieval + response-relevance re-ranking (50/50 weighted) |
| **Intent Classification** | Logistic Regression on TF-IDF features | Logistic Regression on SBERT embeddings |

---

### Evaluation Metrics

- **Response Quality:** BLEU-1, BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L
- **Semantic Understanding:** Semantic Similarity, Contextual Coherence
- **Classification:** Intent Classification Accuracy (8 dialogue act types)
- **Efficiency:** Response Time (ms)
- **Statistical Testing:** Paired t-tests with Cohen's d effect sizes

---

### Key Results

| Metric | Traditional NLP | Transformer | Winner |
|---|---|---|---|
| BLEU-1 | Higher | — | Traditional NLP |
| BLEU-2, BLEU-4 | — | Higher | Transformer |
| ROUGE-1, ROUGE-2, ROUGE-L | — | Higher | Transformer |
| Semantic Similarity | 0.5885 | 0.6294 | **Transformer** (p=0.015, d=0.17) |
| Contextual Coherence | 0.4813 | 0.4833 | Transformer (marginal) |
| Intent Classification Accuracy | 57.45% | 46.60% | **Traditional NLP** |
| Response Time | 555.6 ms | Slower | **Traditional NLP** |

- **Overall:** Transformer dominated 7 of 11 metrics; Traditional NLP led in intent accuracy, retrieval similarity, and speed.
- **Statistical Significance:** Only semantic similarity achieved significance (p=0.015, Cohen's d=0.17); all other differences were non-significant.

---

### Notebook Pipeline (21 Cells)

1. **Install Dependencies** — automated pip install
2. **Import Libraries** — scikit-learn, sentence-transformers, NLTK, matplotlib, seaborn
3. **Load Dataset** — StudyChat from HuggingFace
4. **Exploratory Data Analysis** — distribution visualisations across topics and dialogue acts
5. **Data Preprocessing** — lowercasing, code/URL replacement, stopword removal, lemmatisation
6. **Train/Test Split** — stratified 80/20 split, capped at 10,000 samples
7. **Traditional NLP Chatbot** — TF-IDF vectoriser + cosine similarity retrieval
8. **Traditional NLP Intent Classification** — Logistic Regression on TF-IDF features
9. **Transformer Chatbot** — Sentence-BERT encoding with context enrichment
10. **Transformer Retrieval with Re-Ranking** — top-5 candidates, response-relevance scoring
11. **Transformer Intent Classification** — Logistic Regression on SBERT embeddings
12. **Evaluation Metric Functions** — BLEU, ROUGE, semantic similarity, contextual coherence
13. **Full Comparative Evaluation** — 300-sample head-to-head evaluation
14. **Results Comparison Table** — tabular summary of all metrics
15. **Bar Chart Comparison** — side-by-side performance visualisation
16. **Response Time & Intent Accuracy Comparison** — boxplot and bar chart
17. **Metric Distributions** — histograms for score distribution analysis
18. **Performance by Topic** — topic-level semantic similarity and coherence breakdown
19. **Multi-Turn Context Evaluation** — coherence by conversation length buckets
20. **Statistical Significance Testing** — paired t-tests with Cohen's d
21. **Final Summary & Export** — results exported to CSV; qualitative sample comparisons
22. **User Acceptance Testing** — 38 unit tests across 10 test classes validating all requirements

---

### Requirements

```
datasets
sentence-transformers
scikit-learn
nltk
rouge-score
matplotlib
seaborn
pandas
numpy
scipy
tqdm
```

---

### How to Run

1. Open `Conversational_AI_Architecture_Comparison_StudyChat.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially — dependencies are installed automatically in Cell 1.
3. A HuggingFace API token is required to access the StudyChat dataset (Cell 3).
4. The full pipeline executes within approximately 30 minutes on the Google Colab free tier.
5. Results are exported to CSV files and visualisations are generated inline.

---

### Reproducibility

- All experiments use a fixed random seed (`seed=42`) across all operations.
- The pipeline is fully self-contained within a single notebook with no local dependencies.
- Only open-source libraries and CC BY 4.0 licensed data are used.

---

### Ethics

No human participants were involved. The study used a publicly available, anonymised, PII-filtered dataset under a CC BY 4.0 licence. Ethical approval was obtained via Sheffield Hallam University's UREC1 process.

---

### Licence

This project is submitted as part of the MSc Computing programme at Sheffield Hallam University.

---

### Author

MSc Computing Research Project — Sheffield Hallam University
College of Business, Technology and Engineering (BTE)
