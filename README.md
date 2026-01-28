# Credit Risk Modeling & AI-Driven Credit Quality Assurance System
## 📖 Overview

This project implements an end-to-end, AI-assisted credit risk and Quality Assurance (QA) system that combines:

- A machine-learning Probability of Default (PD) model (LightGBM),

- Rule-based credit policy segmentation,

- Retrieval-Augmented Generation (RAG) over bank credit policy documents, and

- An interactive Streamlit dashboard for human-in-the-loop review.

The goal is not only to predict default, but to support consistent, explainable, and policy-aligned credit decisions suitable for auditors and regulators.

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
LLM Reasoning (Explainable Decision) + Audit Log (Governance & Traceability)
```
## Workflow Orchestration (LangGraph)

This project uses **LangGraph** to orchestrate the end-to-end decision workflow as a directed state machine. Each step of the pipeline (risk screening, policy retrieval, LLM reasoning, parsing, and auditing) is implemented as a node, with shared state passed between nodes to ensure traceability and reproducibility of decisions.

- LangGraph manages the multi-step decision pipeline.  
- Ensures structured state passing between nodes (`case`, `policy`, `decision`, `decision_source`).  
- Makes the system modular, debuggable, and auditable.

## ⚙️ Model Selection

Three models were evaluated for Probability of Default (PD) prediction:

- Logistic Regression (baseline)
- XGBoost
- LightGBM

Each model was trained and evaluated using:
- ROC–AUC
- KS statistic
- Cross-validated hyperparameter tuning
  
The LightGBM (Grid Search) model was selected as the final PD engine due to its superior discrimination, stability, and suitability for downstream risk analysis.

## Model Performance Comparision

### Validation Set
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

### Shap Comparision
![Shap](/screenshot/shap.png)

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


## Risk Bands & Credit Policy
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

This project uses **Retrieval-Augmented Generation (RAG)** to produce **policy-aware**, **explainable credit decisions** for each loan.

For every QA case, the system combines:

- **Model outputs**:

         - Probability of Default (PD)

         - Loss Given Default (LGD)

         - Expected Loss

- **Risk signals**:

         - Policy segment

         - Segment default rate

         - SHAP-based risk drivers

-**Bank credit policy** (retrieved from PDF documents)

These inputs are passed to a large language model that acts as a Credit Quality Assurance analyst.

## 🖥 Interactive Credit QA Dashboard (Streamlit)

This project includes an interactive **Credit Quality Assurance dashboard** built with Streamlit, simulating how banks review and validate loan decisions.

The dashboard integrates:

- Machine-learning risk outputs (PD)

- Economic loss estimates (Expected Loss)

- Policy-based risk segmentation

- LLM-powered credit explanations

### What the dashboard provides

Credit analysts can:

- Browse a portfolio of loan cases

- Select individual loans for detailed review

- See key borrower metrics (DTI, LTI, utilization, employment, verification)

- View model-based risk (PD)

- See financial impact (Expected Loss)

- Review policy classification (QA / Reject)

- Receive AI-generated explanations and next-step recommendations
### Demo

#### Demo1
![Demo1](/screenshot/app2.png)


#### Demo2
![Demo2](/screenshot/app1.png)


### How to run Locally

## 🚀 How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```
Start the Streamlit app:
```
python -m streamlit run app.py
```

### **A) Data & Core Python**
- pandas  
- numpy (implicit with pandas/ML stack)  
- python standard libraries:
  - `re`
  - `json`
  - `os`
  - `datetime`

### **B) ML & Explainability**
- scikit-learn  
- xgboost  
- lightgbm  
- scipy  
- shap  

### **C) LLM + RAG + Workflow**
- **langchain**
- **langchain-community**
  - `PyPDFLoader`
  - `DirectoryLoader`
  - `Chroma` (vector store)
- **langchain-openai**
  - `OpenAI`
  - `OpenAIEmbeddings`
  - `ChatOpenAI`
- **langchain-huggingface**
  - `HuggingFaceEmbeddings`
- **langchain-text-splitters**
  - `RecursiveCharacterTextSplitter`
- **ChromaDB** (vector database)
- **LangGraph**
  - `StateGraph` for workflow orchestration

### **D) App & Deployment**
- streamlit  
- joblib  
- joblib
---
⚠ This project uses simulated credit policy and does not represent any real bank’s underwriting rules.
