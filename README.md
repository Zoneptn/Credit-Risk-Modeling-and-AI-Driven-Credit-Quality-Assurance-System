# Credit-Risk-Modeling-and-RAG-Enhanced-Explainable-Decision-Support-System
## 📖 Overview

This project builds an end-to-end credit risk decision support system using machine-learning and Retrieval-Augmented Generation (RAG).
A LightGBM Probability-of-Default (PD) model is trained on LendingClub data and deployed via Streamlit for real-time loan pre-screening.
RAG is used to provide policy-aware, explainable recommendations to support Credit Quality Assurance (QA) and underwriting decisions.

The goal is not just to predict default, but to support consistent, explainable, and policy-aligned credit decisions.

## 🏦 Business Problem

Traditional credit pre-screening often relies on:

- Hard rules (e.g., DTI < 40%)
- Manual reviews
- Inconsistent judgment across analysts

These approaches do not scale well and can miss emerging risk patterns.

This project addresses these issues by combining:

- Machine-learning based PD scoring
- Business risk banding
- LLM-based policy and market interpretation

## 🧠 System Architecture

```
User (Streamlit)
         ↓
Lite PD Model (LightGBM)
         ↓
PD + Risk Band
         ↓
RAG (Credit Policy)
         ↓
Explainable Credit Recommendation
```

## ⚙️ Model Selection

Three models were evaluated for Probability of Default (PD) prediction:

- Logistic Regression (baseline)
- XGBoost
- LightGBM

Each model was trained and evaluated using ROC–AUC and KS statistics. 
Hyperparameter tuning was applied using cross-validation.

The tuned LightGBM model achieved the strongest discriminatory power and was therefore selected as the final PD engine for deployment.


## Model Performanc Comparision

| Model                | Precision | Recall | F1-Score | ROC-AUC | KS |
|---------------------|-----------|--------|----------|--------|----|
| **LightGBM (Grid)**  | **0.36**   | **0.66** | **0.46** | **0.71** | **0.31** |
| XGBoost (Grid)      | 0.36      | 0.66   | 0.46     | 0.71   | 0.30 |
| LightGBM            | 0.35      | 0.67   | 0.46     | 0.71   | 0.30 |
| XGBoost             | 0.35      | 0.64   | 0.45     | 0.70   | 0.29 |
| Logistic Regression | 0.35      | 0.63   | 0.45     | 0.70   | 0.29 |
