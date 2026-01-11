# World Bank Document Classification - 1st Place Solution

[![Competition](https://img.shields.io/badge/DrivenData-1st%20Place-gold)](https://www.drivendata.org/)
[![F1 Score](https://img.shields.io/badge/F1--micro-0.6925-blue)]()

First place solution for the [World Bank Group: Document Classification Challenge](https://www.drivendata.org/) on DrivenData, sponsored by the World Bank.

## Challenge Overview

Multi-label, multi-class NLP classification to identify topics in World Bank publications. The task involved classifying documents across 29 development sector categories including Macroeconomics, Poverty Reduction, Technology, Agriculture, and Education.

| Dataset | Documents |
|---------|-----------|
| Training | 18,660 |
| Test | 18,738 |
| Labels | 29 categories |

## Solution Highlights

- **Final Score**: F1-micro 0.6925 (1st place)
- **Key Challenges**: Significant data quality issues, severe class imbalance, noisy text data
- **Approach**: Ensemble of deep learning (LSTM/GRU) and gradient boosting (LightGBM) with extensive text preprocessing

## Methodology

### Data Preprocessing

The raw data required extensive cleaning to handle real-world document noise:

- **Text normalization**: Expanded contractions, removed URLs/emails/IP addresses, stripped Wikipedia markup and CSS artifacts
- **Custom vocabulary**: Developed typo correction mappings (`cleanwords.txt`) for domain-specific terms
- **Tokenization**: Lemmatization with NLTK, punctuation normalization, stopword removal
- **Quality filtering**: Removed duplicates and documents with insufficient content

### Feature Engineering

| Technique | Implementation |
|-----------|----------------|
| TF-IDF | N-grams (1-3), vocabulary size 10K-200K |
| Word Embeddings | Word2Vec, GloVe, custom-trained vectors (300-dim) |
| Sequence Encoding | Keras Tokenizer with padding (150-2000 tokens) |

### Modeling Approach

**Deep Learning (Keras/TensorFlow):**
- Bidirectional GRU with attention mechanisms
- RCNN (Recurrent CNN) combining sequence and local features
- Attention-weighted CNN with multiple filter sizes
- CuDNNGRU layers for GPU acceleration

**Gradient Boosting:**
- LightGBM and XGBoost on TF-IDF features
- Stacking ensemble with Logistic Regression meta-learner

### Handling Class Imbalance

- SMOTE oversampling for minority classes
- Class-weighted loss functions
- Threshold tuning for binary predictions

## Key Insights

### Adversarial Validation

Adversarial validation revealed approximately 4,000 training observations that were inadvertently present in the test data. This train/test overlap was identified by training a classifier to distinguish between train and test distributions - a technique that proved invaluable for understanding the data.

### Data Quality Matters

A significant portion of the performance gains came from preprocessing rather than model architecture:
- Domain-specific typo corrections improved embedding coverage
- Aggressive noise removal (markup, templates) reduced vocabulary pollution
- Document deduplication prevented overfitting to repeated content

### Ensemble Diversity

The final solution combined fundamentally different approaches:
- **Neural networks** captured semantic relationships and long-range dependencies
- **Gradient boosting** excelled at exploiting TF-IDF statistical patterns
- **Stacking** with cross-validation prevented leakage between base models and meta-learner

## Project Structure

```
├── utils/
│   ├── data_loader.py        # Dataset and embedding loading utilities
│   ├── data_transformer.py   # Text preprocessing pipeline
│   ├── model_zoo.py          # Neural network architectures (CNN, RNN, RCNN)
│   └── stacking_zoo.py       # Ensemble model configurations
│
├── Data Prep.ipynb           # Text cleaning and preprocessing
├── EDA.ipynb                 # Exploratory data analysis
├── Model Iteration.ipynb     # Keras model experiments
├── TFIDF-DNN *.ipynb         # TF-IDF + Dense network approach
├── Tree Model.ipynb          # Tree-based classifiers
├── Stacking Classifiers.ipynb# Ensemble stacking
├── Adversarial Validation.ipynb # Train/test distribution analysis
└── Submission.ipynb          # Final predictions and output
```

## Setup

### Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for neural network training)

### Dependencies

```bash
pip install numpy pandas scikit-learn
pip install tensorflow keras
pip install xgboost lightgbm
pip install nltk gensim
pip install mlxtend
```

### Data

Place competition data in `../data/`:
- `train_values.csv`, `test_values.csv` - Raw documents
- `train_labels.csv` - Target labels
- `vectors.kv` - Pre-trained Word2Vec embeddings (optional)

### Running the Pipeline

1. Run `Data Prep.ipynb` to generate preprocessed pickle files
2. Run model notebooks in any order to train individual models
3. Run `Submission.ipynb` to generate final predictions

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK, Gensim, Word2Vec, GloVe |
| Gradient Boosting | LightGBM, XGBoost |
| Ensemble | scikit-learn, mlxtend |
| Data Processing | pandas, NumPy |

## Acknowledgments

- [DrivenData](https://www.drivendata.org/) for hosting the competition
- [World Bank](https://www.worldbank.org/) for sponsoring and providing the dataset
- Competition participants for fostering a collaborative learning environment

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Cristopher Benge
