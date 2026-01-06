# Credit-Risk-Modeling-and-RAG-Enhanced-Explainable-Decision-Support-System
## 📖 Overview

This project builds an end-to-end credit risk decision support system using machine-learning and Retrieval-Augmented Generation (RAG).
A LightGBM Probability-of-Default (PD) model is trained on LendingClub data and deployed via Streamlit for real-time loan pre-screening.
RAG is used to provide policy-aware, explainable recommendations to support Credit Quality Assurance (QA) and underwriting decisions.

The goal is not just to predict default, but to support consistent, explainable, and policy-aligned credit decisions.

## 🏦 Business Problem

Banks must ensure that lending decisions are:
- **Risk-aware**
- **Policy-compliant**
- **Explainable to auditors and regulators**

Traditional credit workflows rely on:
- PD models without clear explanations  
- Static rule-based systems  
- Manual and inconsistent QA reviews  

These approaches do not scale well and make it difficult to demonstrate why a loan was approved or rejected.

This project addresses these issues by integrating:
- Machine-learning based PD scoring  
- Rule-based credit policy segmentation  
- LLM-powered policy interpretation and reasoning  

to produce **consistent, transparent, and auditable credit decisions**.


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

# Validation Set
| Model                      | Precision  | Recall     | F1-Score   | ROC-AUC    | KS Statistic |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ------------ |
| **LightGBM (Grid Search)** | **0.3558** | 0.6558     | **0.4613** | **0.7111** | **0.3072**   |
| LightGBM                   | 0.3541     | **0.6612** | 0.4612     | 0.7101     | 0.3067       |
| XGBoost (Grid Search)      | 0.3510     | **0.6669** | 0.4599     | 0.7076     | 0.3017       |
| XGBoost                    | 0.3549     | 0.6308     | 0.4543     | 0.7031     | 0.2970       |
| Logistic Regression        | 0.3522     | 0.6246     |            |            |              |

### Testing Set
| Model                      | Precision  | Recall     | F1-Score   | ROC-AUC    | KS Statistic |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ------------ |
| **LightGBM (Grid Search)** | **0.3568** | 0.6606     | **0.4633** | **0.7097** | **0.3086**   |
| LightGBM                   | 0.3552     | **0.6653** | 0.4631     | 0.7092     | 0.3064       |
| XGBoost (Grid Search)      | 0.3512     | **0.6671** | 0.4602     | 0.7067     | 0.3011       |
| XGBoost                    | 0.3543     | 0.6326     | 0.4542     | 0.6992     | 0.2912       |
| Logistic Regression        | 0.3528     | 0.6305     | 0.4524     | 0.6952     | 0.2891       |


From a credit-risk perspective, risk ranking quality and stability matter more than raw recall.
Therefore, LightGBM (Grid Search) is selected as the final Probability-of-Default model because it provides the best balance of discrimination power, stability, and robustness for downstream QA, Expected Loss, and policy-based decisioning.

## 📐 Probability Calibration

In credit risk modeling, well-calibrated probabilities are important because PD values are used for:
- Risk banding
- Portfolio risk estimates
- Policy thresholds

After selecting the best-performing LightGBM model, probability calibration (e.g., isotonic or Platt scaling) can be applied to align predicted PDs with observed default rates.

In this project, model discrimination (ROC-AUC, KS) was the primary selection criterion, and the resulting PDs are used as inputs to the business risk bands and RAG-based decision logic.

### Model Calibration Result
| Model                             | ROC-AUC    | KS Statistic | Brier Score |
| --------------------------------- | ---------- | ------------ | ----------- |
| **LightGBM (Before Calibration)** | **0.7097** | **0.3086**   | 0.2148      |
| **LightGBM (After Calibration)**  | 0.7036     | 0.2969       | **0.1599**  |

Although calibration slightly reduces ROC-AUC and KS, it substantially improves the Brier score, which measures how accurate the predicted probabilities are. This means the calibrated LightGBM model produces more reliable and realistic PD estimates, making it suitable for policy thresholds, Expected Loss calculations, and regulatory-style credit decisioning. As a result, the calibrated model is used for all downstream risk, QA, and financial impact analysis, while the uncalibrated model remains useful for pure risk ranking and explainability.


### Risk Bands & Credit Policy
This project applies a policy-driven Quality Assurance (QA) framework on top of the machine-learning Probability of Default (PD) model to simulate how banks make real credit decisions.

Each loan is evaluated using:

- PD (likelihood of default)

- LGD (loss severity)

- Expected Loss (financial impact)

- Policy risk flags (DTI, utilization, stability, verification)

Loans are segmented into four decision groups:
| Segment                  | Meaning                                 |
| ------------------------ | --------------------------------------- |
| **Fast Track**           | Very low risk, auto-approve             |
| **Approve with Caution** | Acceptable risk                         |
| **QA**                   | Borderline risk, requires manual review |
| **Review / Reject**      | High-risk or high-loss loans            |

### Loss-Aware Risk Control
Expected Loss is calculated as:

$$Expected Loss=𝑃𝐷 ×𝐿𝐺𝐷×Loan Amount$$

This allows the system to prioritize not just who is likely to default, but how costly a default would be.

Loans with:

- High PD

- High Expected Loss

- adverse risk flags

are routed to **QA or rejection**, even if PD alone is moderate.

### Decision Logic

- **Fast Track** → automatic approval

- **Approve with Caution** → standard underwriting

- **QA** → manual review

- **Review / Reject** → decline or escalate

This mirrors how banks combine risk models, financial loss, and credit policy to make safe, explainable decisions.


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
