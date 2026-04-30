# DLCV Project — Evidential Causal Alignment (ECA)

This workspace contains the proposal plus a **runnable** minimal experiment suite that implements the proposal’s required evaluations on **Adult** and **ToxiGen**.

## Files

- [Project_Proposal.pdf](Project_Proposal.pdf) — original 2‑page proposal.
- [run_experiments.py](run_experiments.py) — trains baselines + ECA and computes metrics.
- [requirements.txt](requirements.txt) — minimal Python dependencies.
- [results_adult.json](results_adult.json) — latest Adult run output.
- [results_toxigen.json](results_toxigen.json) — latest ToxiGen run output.

# Evidential Causal Alignment (ECA) for Trustworthy Language Models

## Overview

Large language models often make **biased predictions with high confidence**, which is problematic in sensitive applications. Traditional methods either focus on improving accuracy or fairness, but fail to account for **model uncertainty**.

This project implements **Evidential Causal Alignment (ECA)** — a framework that combines:
- **Evidential Deep Learning (EDL)** for uncertainty estimation
- **Counterfactual fairness** for bias detection
- **Uncertainty-guided training** to reduce biased predictions

---

## Key Idea

Instead of treating all training samples equally, ECA:

1. Estimates **uncertainty** using a Dirichlet distribution
2. Measures **causal bias** using counterfactual inputs
3. Reweights training to penalize:
   - uncertain predictions
   - biased predictions

👉 This leads to models that are:
- More **fair**
- More **reliable**
- Better at knowing when they are wrong

---

## Methodology

### 1. Evidential Modeling

Model outputs **evidence** \( e_k \), converted to Dirichlet parameters:

\[
\alpha_k = e_k + 1
\]

Prediction:
\[
p_k = \frac{\alpha_k}{\sum_j \alpha_j}
\]

Uncertainty:
\[
u = \frac{K}{\sum_j \alpha_j}
\]

---

### 2. Counterfactual Bias

We measure bias by flipping sensitive attributes:

\[
\text{Bias}(x) = |p(x) - p(x_{cf})|
\]

---

### 3. ECA Training

Loss is weighted using uncertainty and bias:

\[
w = u^2 \cdot (1 + \gamma \cdot \text{Bias})
\]

Additional terms:
- Penalize **overconfident wrong predictions**
- Encourage **confident correct predictions**
- Prevent **uncertainty collapse**

---

## Datasets

### 🔹 Adult Dataset
- Structured dataset (income prediction)
- Sensitive attribute: gender

### 🔹 ToxiGen Dataset
- Text dataset for toxicity classification
- Multiple demographic groups

---

## Models

- `bert-base-uncased`
- `distilbert-base-uncased`

---

## Experiments

We compare:

| Method | Description |
|--------|------------|
| Softmax | Standard training |
| Reweight Class | Class imbalance handling |
| Reweight Group | Fairness-aware baseline |
| Causal Prompting | Prompt-based debiasing |
| **ECA (ours)** | Uncertainty + causal alignment |

---

## Results Summary

### Adult Dataset

- ECA achieves:
  - ✔ Lowest bias
  - ✔ High uncertainty quality (AUROC ~0.84)
  - ✔ Maintains accuracy (~0.70)

---

### ToxiGen Dataset

- ECA achieves:
  - ✔ Best calibration (lowest ECE)
  - ✔ Strong uncertainty (AUROC ~0.70)
  - ⚠ Slight drop in accuracy vs best baseline

---

## Key Insights

- Uncertainty can effectively guide bias mitigation  
- There is a **tradeoff between accuracy and uncertainty**  
- ECA performs better on **structured data** than complex language tasks  

---

## 🛠 Installation

```bash
pip install torch transformers datasets scikit-learn numpy matplotlib
# Adult
python run_experiments.py --dataset adult --epochs 1

# ToxiGen
python run_experiments.py --dataset toxigen --epochs 1
