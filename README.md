# Complaint Topic Analysis

## Introduction

This project performs **topic modeling on consumer complaint narratives** from the [CFPB (Consumer Financial Protection Bureau)](https://www.consumerfinance.gov/data-research/consumer-complaints/) dataset. The goal is to automatically discover recurring themes and topics within consumer complaints using Natural Language Processing (NLP) and unsupervised machine learning techniques.

## Dataset

- **Source:** CFPB Consumer Complaint Database (`rows.csv`)
- **Size:** 1,282,355 complaints with 18 features
- **Key Columns:** Date received, Product, Sub-product, Issue, Sub-issue, Consumer complaint narrative, Company, State, Submitted via, Company response, Complaint ID
- **Narratives Used:** 383,564 complaints with non-null consumer complaint narratives (366,852 unique after deduplication)

## What We Did

### 1. Data Loading & Exploration
- Loaded the full CFPB complaint dataset using Pandas
- Inspected the structure, columns, and data types
- Filtered the dataset to retain only rows with consumer complaint narratives

### 2. Text Preprocessing
- Converted all text to lowercase and stripped whitespace
- Removed duplicate complaint narratives
- Resulted in **366,852 unique complaints** for analysis

### 3. NLP Cleaning with spaCy
- Loaded the `en_core_web_sm` spaCy language model
- Defined a `clean_text` function that:
  - Tokenizes the text using spaCy
  - Removes stopwords, punctuation, whitespace, and non-alphabetic tokens
  - Applies lemmatization to reduce words to their base forms
- Applied the cleaning function to a random sample of **10,000 complaints**

### 4. Feature Extraction — TF-IDF Vectorization
- Used `TfidfVectorizer` from scikit-learn to convert cleaned text into numerical features
- Parameters: `max_df=0.9`, `min_df=10`, `max_features=5000`
- Resulting matrix shape: **10,000 × 3,201**

### 5. Topic Modeling — NMF (Non-Negative Matrix Factorization)
- Trained an NMF model with **5 topics** on the TF-IDF matrix
- Extracted the top 10 keywords per topic to interpret themes:
  - **Topic 1:** Legal/administrative complaints (court, file, case, sell)
  - **Topic 2:** Loan & payment issues (payment, loan, bank, charge)
  - **Topic 3:** Late payment & communication (late, letter, send)
  - **Topic 4:** Credit reporting disputes (report, credit, equifax, experian)
  - **Topic 5:** Debt collection harassment (debt, collection, agency, call)

### 6. Topic Modeling — LDA (Latent Dirichlet Allocation)
- Created a Document-Term Matrix using `CountVectorizer` (same parameters as TF-IDF)
- Trained an LDA model with **5 topics** on the count matrix
- Extracted the top 10 keywords per topic:
  - **Topic 1:** Debt & account disputes
  - **Topic 2:** Credit report errors
  - **Topic 3:** Banking & card issues
  - **Topic 4:** Mortgage & loan payments
  - **Topic 5:** Payment & insurance issues

## Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python 3 | Programming language |
| Pandas | Data loading & manipulation |
| spaCy (`en_core_web_sm`) | NLP preprocessing (tokenization, lemmatization, stopword removal) |
| scikit-learn | TF-IDF, CountVectorizer, NMF, LDA |
| Google Colab (T4 GPU) | Execution environment |

## Project Structure

```
complaint-topic-analysis/
├── data/
│   └── rows.csv              # CFPB consumer complaints dataset
├── notebooks/
│   └── Untitled4.ipynb       # Main analysis notebook
├── src/                      # Source code (reserved for future scripts)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shreyashpatel00714/complaint-topic-analysis.git
   cd complaint-topic-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   - Open `notebooks/Untitled4.ipynb` in Google Colab or Jupyter Notebook
   - Upload `rows.csv` to the environment if running on Colab
   - Execute all cells sequentially

## Key Results

Both NMF and LDA successfully identified **5 distinct complaint topics** covering the major consumer financial complaint categories: credit reporting disputes, debt collection, loan/mortgage issues, banking problems, and legal/administrative matters.