# SEC 10-K Risk Factor Classifier

An NLP pipeline that classifies corporate risk disclosures from SEC 10-K filings into meaningful categories and identifies boilerplate vs. materially substantive risks.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Public companies are required to disclose risk factors in their annual 10-K filings (Item 1A). This project builds a multi-label text classification system that:

1. **Categorizes risks** into 10 domains (regulatory, cybersecurity, operational, etc.)
2. **Detects boilerplate** language vs. material, company-specific disclosures
3. **Tracks risk evolution** across 72,000+ filings from 2006-2020

## Key Results

| Metric | Value |
|--------|-------|
| Filings Analyzed | 72,613 |
| Risk Paragraphs | 79,415 |
| Classification Categories | 10 |
| Date Range | 2006-2020 |

*Model performance metrics will be added as phases complete.*

## Tech Stack

- **Data**: Hugging Face Datasets, Pandas, NumPy
- **NLP**: NLTK, Scikit-learn, Transformers
- **Deep Learning**: PyTorch, FinBERT
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit

## Project Structure

```
├── data/
│   ├── raw/              # Original EDGAR-CORPUS data
│   └── processed/        # Cleaned datasets
├── notebooks/
│   ├── 01_data_acquisition_eda.ipynb
│   ├── 02_preprocessing_segmentation.ipynb
│   └── ...
├── src/
│   └── preprocessing.py  # Reusable preprocessing module
├── models/               # Trained model artifacts
├── outputs/              # Visualizations and reports
└── app/                  # Streamlit application (Phase 8)
```

## Risk Categories

| Category | Description |
|----------|-------------|
| Regulatory/Legal | Compliance, litigation, government action |
| Cybersecurity | Data breaches, privacy, information security |
| Competitive/Market | Competition, market share, pricing |
| Macroeconomic | Recession, inflation, currency risks |
| Operational/Supply | Supply chain, manufacturing, logistics |
| Financial/Liquidity | Cash flow, debt, credit access |
| Environmental/Climate | ESG, emissions, natural disasters |
| Personnel/Labor | Key employees, workforce, unions |
| Reputational | Brand, public perception, trust |
| Technology/Innovation | R&D, obsolescence, digital disruption |

## Methodology

### Phase 1: Data Acquisition ✅
- Loaded EDGAR-CORPUS dataset from Hugging Face
- Filtered to 10-K filings with Item 1A sections (2006-2020)
- Identified 200% growth in disclosure length over the period

### Phase 2: Preprocessing ✅
- Text cleaning and normalization
- Risk paragraph segmentation
- Preliminary keyword-based classification

### Phase 3-4: Classification (In Progress)
- TF-IDF baseline with logistic regression
- Word embedding approaches (Word2Vec, GloVe)

### Phase 5-6: Deep Learning (Planned)
- Fine-tuned FinBERT for financial text
- Multi-label classification architecture

### Phase 7: Boilerplate Detection (Planned)
- Identify generic vs. material disclosures
- Temporal analysis of risk language evolution

### Phase 8: Deployment (Planned)
- Interactive Streamlit dashboard
- Real-time classification of new filings

## Installation

```bash
# Clone the repository
git clone https://github.com/jlattanzi4/sec-risk-classifier.git
cd sec-risk-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

This project uses the [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) dataset from Hugging Face, which contains pre-parsed SEC filings from 1993-2020.

**Note:** Due to size constraints, data files are not included in this repository. Run the data acquisition notebook to download and process the dataset.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Joseph Lattanzi**
- Portfolio: [jlattanzi4.github.io](https://jlattanzi4.github.io/)
- GitHub: [@jlattanzi4](https://github.com/jlattanzi4)
