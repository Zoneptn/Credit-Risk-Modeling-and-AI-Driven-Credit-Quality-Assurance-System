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


## Model Performance Comparision

| Model                | Precision | Recall | F1-Score | ROC-AUC | KS |
|---------------------|-----------|--------|----------|--------|--------|
| **LightGBM (Grid)**  | **0.3552** | **0.6632** | **0.4626** | **0.7108** | **0.3069** |
| XGBoost (Grid)      | 0.3552 | 0.6588 | 0.4615 | 0.7103 | 0.3047 |
| LightGBM            | 0.3528 | 0.6683 | 0.4618 | 0.7092 | 0.3050 |
| XGBoost             | 0.3540 | 0.6351 | 0.4546 | 0.7015 | 0.2921 |
| Logistic Regression | 0.3529 | 0.6316 | 0.4528 | 0.6960 | 0.2881 |

Among all tested models, the tuned LightGBM achieved the highest KS and ROC-AUC, indicating superior risk separation, and was therefore selected as the final Probability-of-Default engine

*** This is not the actual credit risk policy.

## 📐 Probability Calibration

In credit risk modeling, well-calibrated probabilities are important because PD values are used for:
- Risk banding
- Portfolio risk estimates
- Policy thresholds

After selecting the best-performing LightGBM model, probability calibration (e.g., isotonic or Platt scaling) can be applied to align predicted PDs with observed default rates.

In this project, model discrimination (ROC-AUC, KS) was the primary selection criterion, and the resulting PDs are used as inputs to the business risk bands and RAG-based decision logic.


### Risk Bands & Credit Policy

Model outputs are converted into business decisions using a four-tier risk policy:

| PD Range | Decision |
|---------|--------|
| PD ≥ 30% | Reject |
| 20% ≤ PD < 30% | QA Review |
| 5% ≤ PD < 20% | Approve with Caution |
| PD < 5% | Approve |

This transforms raw model probabilities into operational credit actions suitable for underwriting and Credit QA workflows.

## RAG-Based Explainable Credit Decisions

Retrieval-Augmented Generation (RAG) is used to ensure that credit decisions are aligned with bank policies and market context.

Credit policy documents and market outlooks are embedded into a vector database.  
When a loan is scored, the system retrieves the most relevant policy text and uses a large language model to generate an explanation based on those documents.

This allows the system to answer questions such as:
- Why is this loan considered risky?
- Which credit policy rules apply?
- What the recommended action should be?

*** This is not the acutual bank policy

## 🌐 Streamlit Application

A Streamlit web app is provided to simulate a real-time loan pre-screening tool.

Users can input:
- Annual income
- Debt-to-income ratio
- Loan amount
- Interest rate

The app then:
1. Computes the Probability of Default using the Lite PD model  
2. Assigns a risk band  
3. Calls the RAG engine to generate a policy-aware credit recommendation  

This replicates how credit analysts and QA teams interact with risk models in practice.

### Demo
### Loan Risk Scoreing
![Loan Scoring](/screenshot/app.png)

### RAG-Based Credit Recommendation
![RAG](/screenshot/rag.png)


### How to run Locally

## 🚀 How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```
Start the RAG backend:
```
python -m uvicorn rag_server:app --reload
```
Start the Streamlit app:
```
python -m streamlit run app.py
```

## 🛠 Tech Stack
- Python
- LightGBM
- XGBoost
- Scikit-learn
- Streamlit
- FastAPI
- LlamaIndex / RAG
- OpenAI Embeddings

⚠ This project uses simulated credit policy and does not represent any real bank’s underwriting rules.
