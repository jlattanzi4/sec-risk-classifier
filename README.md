# SEC 10-K Risk Factor Intelligence

An end-to-end NLP pipeline analyzing 79,000+ corporate risk disclosures from SEC 10-K filings. This project demonstrates multi-class text classification, transformer fine-tuning, topic modeling, semantic similarity analysis, and model explainability.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Interactive Report

**[View the Full Interactive Analysis](https://jlattanzi4.github.io/sec-risk-classifier/report.html)**

---

## Key Results

| Metric | Value |
|--------|-------|
| Risk Paragraphs Analyzed | 79,415 |
| Unique Companies | 13,970 |
| Classification F1 Score | **71.2%** |
| Topics Discovered | 21 |
| Semantic vs Lexical Gap | **2.4x** |

### Headline Finding

Companies share **51% semantic similarity** but only **21% lexical overlap** in their risk disclosures. This 2.4x gap reveals that companies use different words to express similar risk concepts - a form of "paraphrased boilerplate" invisible to traditional text analysis.

---

## NLP Techniques Demonstrated

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| **Text Classification** | TF-IDF + Ensemble (LR, SVM, RF, XGBoost, LightGBM) | Categorize risks into 10 domains |
| **Transformer Fine-tuning** | DistilBERT | Compare deep learning vs traditional ML |
| **Topic Modeling** | BERTopic (UMAP + HDBSCAN) | Unsupervised discovery of risk themes |
| **Sentence Embeddings** | SBERT (all-MiniLM-L6-v2) | Semantic similarity analysis |
| **Model Explainability** | SHAP | Identify which words drive predictions |

---

## Project Structure

```
sec-risk-classifier/
├── data/
│   └── processed/           # Processed parquet files
├── src/
│   ├── preprocessing.py     # Data cleaning and segmentation
│   ├── train_optimized.py   # TF-IDF ensemble classifier
│   ├── train_distilbert.py  # Transformer comparison
│   ├── topic_modeling.py    # BERTopic analysis
│   ├── model_explainability.py  # SHAP analysis
│   ├── boilerplate_detection.py # Semantic similarity
│   └── generate_report.py   # Interactive HTML report
├── models/                  # Trained model artifacts
├── outputs/                 # Visualizations
├── report.html              # Interactive analysis report
└── README.md
```

---

## Methodology

### 1. Data Acquisition and Preprocessing
- Source: [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) from Hugging Face
- Filtered to 10-K filings with Item 1A risk factor sections (2006-2020)
- Segmented into 79,415 individual risk paragraphs
- Memory-optimized processing with PyArrow for 1.6GB dataset

### 2. Multi-Class Classification
- **Best Model**: Soft Voting Ensemble achieving **71.2% F1 (macro)**
- Compared 5 algorithms: Logistic Regression, SVM, Random Forest, XGBoost, LightGBM
- Class balancing via strategic up/downsampling
- TF-IDF vectorization with 10,000 features and bigrams

### 3. Transformer Comparison
- Fine-tuned DistilBERT achieved **57.8% F1**
- Key insight: TF-IDF outperforms transformers on this task due to document length
- SEC filings average 48K characters, exceeding transformer token limits (512)

### 4. Topic Modeling
- BERTopic discovered **21 distinct risk themes**
- Industry-specific patterns emerged (Oil & Gas, Real Estate, Biotech)
- Only 14% of topics align with manual categories, revealing hidden sub-themes

### 5. Semantic Similarity Analysis
- Compared TF-IDF (lexical) vs SBERT (semantic) similarity
- Mean lexical similarity: 21.3%
- Mean semantic similarity: 51.0%
- Correlation: 0.31 (low) - capturing different dimensions

### 6. Model Explainability
- SHAP analysis reveals which words drive each classification
- Critical for regulated industries requiring interpretable decisions
- Visualized feature importance across all 10 risk categories

---

## Risk Categories

| Category | Example Risks |
|----------|--------------|
| Regulatory/Legal | Compliance, litigation, government action |
| Cybersecurity | Data breaches, privacy, information security |
| Competitive/Market | Competition, market share, pricing pressure |
| Macroeconomic | Recession, inflation, currency fluctuations |
| Operational/Supply | Supply chain disruption, manufacturing issues |
| Financial/Liquidity | Cash flow constraints, debt obligations |
| Environmental/Climate | ESG requirements, emissions, natural disasters |
| Personnel/Labor | Key employee retention, workforce issues |
| Reputational | Brand damage, public perception |
| Technology/Innovation | R&D risks, technological obsolescence |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jlattanzi4/sec-risk-classifier.git
cd sec-risk-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0
numpy<2
scikit-learn>=1.3
torch>=2.0
transformers>=4.30
sentence-transformers>=2.2
bertopic>=0.15
shap>=0.42
plotly>=5.18
pyarrow>=14.0
matplotlib
seaborn
xgboost
lightgbm
```

---

## Usage

```bash
# Run the main classifier
python src/train_optimized.py

# Run topic modeling
python src/topic_modeling.py

# Run explainability analysis
python src/model_explainability.py

# Run boilerplate detection
python src/boilerplate_detection.py

# Generate interactive report
python src/generate_report.py
```

---

## Data

This project uses the [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) dataset containing pre-parsed SEC filings from 1993-2020.

**Note:** Data files are not included due to size (1.6GB). The preprocessing pipeline can regenerate them from the source dataset.

---

## Key Visualizations

The interactive report includes:
- Model performance comparison across 7 algorithms
- Risk category distribution analysis
- Lexical vs semantic similarity comparison
- Topic discovery and temporal trends
- SHAP feature importance by category

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Joseph Lattanzi**
Data Scientist

- Portfolio: [jlattanzi4.github.io](https://jlattanzi4.github.io/)
- GitHub: [@jlattanzi4](https://github.com/jlattanzi4)
- LinkedIn: [Joseph Lattanzi](https://www.linkedin.com/in/jlattanzi4/)
