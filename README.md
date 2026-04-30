# DLCV Project — Evidential Causal Alignment (ECA)

This workspace contains the proposal plus a **runnable** minimal experiment suite that implements the proposal’s required evaluations on **Adult** and **ToxiGen**.

## Files

- [Project_Proposal.pdf](Project_Proposal.pdf) — original 2‑page proposal.
- [proposal_extracted.txt](proposal_extracted.txt) — text extracted from the PDF.
- [extract_proposal.py](extract_proposal.py) — extracts the PDF into text.
- [run_experiments.py](run_experiments.py) — trains baselines + ECA and computes metrics.
- [requirements.txt](requirements.txt) — minimal Python dependencies.
- [results_adult.json](results_adult.json) — latest Adult run output.
- [results_toxigen.json](results_toxigen.json) — latest ToxiGen run output.

## Proposal requirements (what we ran)

From the proposal text in [proposal_extracted.txt](proposal_extracted.txt):

### Datasets
- Adult dataset
- ToxiGen dataset

### Metrics
1. **Expected Calibration Error (ECE)**
2. **Counterfactual Fairness** (output change when sensitive attribute is flipped), additionally reported **weighted by evidential uncertainty** $u$
3. **Epistemic Coverage** (implemented as uncertainty’s ability to predict errors)

### Baselines / comparison
- **Softmax baseline** (standard fine-tuning with cross-entropy)
- **“Causal prompting” baseline** (approximation): an instruction prefix telling the model to ignore sensitive attributes
- **ECA**: evidential (Dirichlet) head + uncertainty-triggered causal reweighting

## What was implemented (matching the proposal at a minimal level)

### Evidential head (Dirichlet)
For $K=2$ classes, the ECA model outputs evidence $e_k\ge 0$ and Dirichlet parameters:

$$\alpha_k = e_k + 1$$

The predictive mean probability is:

$$p_k = \frac{\alpha_k}{\sum_j \alpha_j}$$

Uncertainty is the standard Dirichlet “total evidence” form:

$$u = \frac{K}{\sum_j \alpha_j}$$

### Loss + uncertainty-triggered causal reweighting
We use an evidential negative log-likelihood (via digamma terms) plus a KL regularizer to the uniform Dirichlet:

$$L = \mathbb{E}[w_i\,L_{\text{EDL}}] + \lambda\,\mathrm{KL}(\mathrm{Dir}(\alpha)\|\mathrm{Dir}(\mathbf{1}))$$

Causal influence weight is approximated using a counterfactual flip and the trained softmax baseline:

$$\text{causal\_w}(x) = |p(y{=}1\mid x) - p(y{=}1\mid x_{cf})|$$

Training weight:

$$w_i = (1 + \gamma\,\text{causal\_w}(x_i))\cdot \mathrm{norm}(u(x_i))$$

## How to reproduce

From this folder:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Re-extract the proposal text (optional):

```bash
./.venv/bin/python extract_proposal.py
```

Run experiments (CPU):

```bash
# Adult
./.venv/bin/python run_experiments.py \
  --dataset adult \
  --model distilbert-base-uncased \
  --train-size 400 --eval-size 200 \
  --epochs 1 --batch-size 8 --max-steps 100 \
  --out results_adult.json

# ToxiGen
./.venv/bin/python run_experiments.py \
  --dataset toxigen \
  --model distilbert-base-uncased \
  --train-size 400 --eval-size 200 \
  --epochs 1 --batch-size 8 --max-steps 100 \
  --out results_toxigen.json
```

Notes:
- These runs are **small-sample smoke experiments** (subsets + 1 epoch) so they complete on CPU.
- For the full report-quality run: increase `--train-size`, `--eval-size`, and `--epochs`.

## Counterfactual definitions used

- **Adult**: counterfactual $x_{cf}$ is created by swapping `sex: male` $\leftrightarrow$ `sex: female` in the text representation.
- **ToxiGen**: counterfactual $x_{cf}$ is created by conservative token swaps inside the text (e.g., `women` $\leftrightarrow$ `men`, `black people` $\leftrightarrow$ `white people`). If no safe swap is found, $x_{cf}=x$ (diff $=0$).

## Metrics (how to read them)

- **Accuracy**: standard classification accuracy.
- **ECE (15 bins)**: calibration gap between confidence and accuracy.
- **Counterfactual diff**: mean of $|p(y{=}1\mid x) - p(y{=}1\mid x_{cf})|$.
- **Counterfactual weighted by $u$** (ECA only): mean of $u(x)\cdot |p_1(x)-p_1(x_{cf})|$.
- **Epistemic coverage** (ECA only): AUROC of $u(x)$ for predicting “error” (1 if misclassified). Higher means uncertainty is better at flagging mistakes.

## Results (latest runs)

### Adult (from [results_adult.json](results_adult.json))

| Method | Accuracy | ECE | Counterfactual diff |
|---|---:|---:|---:|
| Softmax | 0.755 | 0.0378 | 0.000180 |
| Causal prompting | 0.755 | 0.1463 | 0.003397 |
| ECA | 0.755 | 0.1353 | 0.001552 |

ECA extras:
- $\mathbb{E}[u]$ = 0.2252
- Epistemic coverage (AUROC $u\rightarrow$error) = 0.2394
- $\mathbb{E}[u\cdot$ counterfactual diff$]$ = 0.000302

### ToxiGen (from [results_toxigen.json](results_toxigen.json))

**Label binarization:** `toxicity_human >= 3.0` is treated as toxic ($y{=}1$), else non-toxic ($y{=}0$).

| Method | Accuracy | ECE | Counterfactual diff |
|---|---:|---:|---:|
| Softmax | 0.620 | 0.0705 | 0.000739 |
| Causal prompting | 0.560 | 0.1973 | 0.000861 |
| ECA | 0.560 | 0.1760 | 0.000537 |

ECA extras:
- $\mathbb{E}[u]$ = 0.2230
- Epistemic coverage (AUROC $u\rightarrow$error) = 0.3449
- $\mathbb{E}[u\cdot$ counterfactual diff$]$ = 0.000113

Per-group breakdowns (accuracy/ECE by demographic) are included in the JSON outputs under `by_group`.
