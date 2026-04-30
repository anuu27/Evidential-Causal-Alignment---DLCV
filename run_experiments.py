from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str) -> Any:
    """Load a HuggingFace tokenizer, preferring the fast implementation.

    Some checkpoints don't provide a compatible fast tokenizer; in that case we
    fall back to the slow tokenizer so experiments still run.
    """

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[warn] Fast tokenizer unavailable for {model_name!r}: {type(e).__name__}: {e}")
        print("[warn] Falling back to slow tokenizer (use_fast=False).")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def _first_present(d: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        if k in d:
            return k
    return None


def _normalize_label(v: Any) -> int:
    # Best-effort normalization for common dataset label encodings.
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        # Many datasets store binary labels as 0.0/1.0 or probabilities in [0, 1].
        if 0.0 <= fv <= 1.0:
            return 1 if fv >= 0.5 else 0
        return int(fv)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", ">50k", ">50k.", "yes", "toxic", "true"}:
            return 1
        if s in {"0", "<=50k", "<=50k.", "no", "non-toxic", "false"}:
            return 0
    raise ValueError(f"Unsupported label value: {v!r}")


def _flip_sex(text: str) -> str:
    # Simple counterfactual flip for Adult prompts.
    # Keep it robust to punctuation (",", ".") by matching substrings, not whole tokens.
    t = text
    if "sex: male" in t:
        return t.replace("sex: male", "sex: female")
    if "sex: female" in t:
        return t.replace("sex: female", "sex: male")
    if "sex=male" in t:
        return t.replace("sex=male", "sex=female")
    if "sex=female" in t:
        return t.replace("sex=female", "sex=male")
    return text


def _try_flip_group(text: str) -> str | None:
    # Conservative counterfactual for ToxiGen-like text (only if an obvious group token exists).
    # Returns None if no safe flip was found.
    replacements = [
        (" women ", " men "),
        (" men ", " women "),
        (" female ", " male "),
        (" male ", " female "),
        (" black people ", " white people "),
        (" white people ", " black people "),
        (" gay people ", " straight people "),
        (" straight people ", " gay people "),
    ]
    t = f" {text.strip().lower()} "
    for a, b in replacements:
        if a in t:
            return t.replace(a, b).strip()
    return None


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    # probs: [N, K]
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += abs(acc - avg_conf) * float(mask.mean())
    return float(ece)


def safe_auc(scores: np.ndarray, errors: np.ndarray) -> float | None:
    # Returns None when AUC is undefined.
    if errors.min() == errors.max():
        return None
    try:
        return float(roc_auc_score(errors, scores))
    except Exception:
        return None
    
def temperature_scale(logits, T):
    return logits / T


def plot_uncertainty_histogram(
    *,
    uncertainties: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """Plot uncertainty distributions for correct vs incorrect predictions."""

    preds = probs.argmax(axis=1)
    correct_mask = preds == labels
    incorrect_mask = ~correct_mask

    u_correct = uncertainties[correct_mask]
    u_incorrect = uncertainties[incorrect_mask]
    print(probs[:5])
    print(preds[:5])

    plt.figure(figsize=(7.5, 4.5))
    bins = 30
    if len(u_correct) > 0:
        plt.hist(u_correct, bins=bins, alpha=0.65, label=f"Correct (n={len(u_correct)})", density=True)
    if len(u_incorrect) > 0:
        plt.hist(u_incorrect, bins=bins, alpha=0.65, label=f"Incorrect (n={len(u_incorrect)})", density=True)

    plt.xlabel("Evidential uncertainty (u)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    group: list[str]
    input_ids_cf: torch.Tensor | None
    attention_mask_cf: torch.Tensor | None
    causal_w: torch.Tensor | None
    sample_w: torch.Tensor | None


def collate_fn(tokenizer: Any, examples: list[dict[str, Any]]) -> Batch:
    text = [ex["text"] for ex in examples]
    enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")

    labels = torch.tensor([_normalize_label(ex["label"]) for ex in examples], dtype=torch.long)
    group = [str(ex.get("group", "unknown")) for ex in examples]

    # We normalize datasets so every example always has a counterfactual text.
    text_cf = [ex["text_cf"] for ex in examples]
    enc_cf = tokenizer(text_cf, truncation=True, padding=True, max_length=128, return_tensors="pt")

    causal_w = None
    if all("causal_w" in ex for ex in examples):
        causal_w = torch.tensor([float(ex["causal_w"]) for ex in examples], dtype=torch.float32)

    sample_w = None
    if all("sample_w" in ex for ex in examples):
        sample_w = torch.tensor([float(ex["sample_w"]) for ex in examples], dtype=torch.float32)

    return Batch(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=labels,
        group=group,
        input_ids_cf=enc_cf["input_ids"],
        attention_mask_cf=enc_cf["attention_mask"],
        causal_w=causal_w,
        sample_w=sample_w,
    )


class EvidentialHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        evidence_temperature: float = 2.0,
        max_evidence: float = 30.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_labels)
        self.evidence_temperature = float(max(evidence_temperature, 1e-3))
        self.max_evidence = float(max(max_evidence, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # evidence >= 0
        # Temperature scaling + clamping helps prevent evidence saturation.
        evidence = F.softplus(self.linear(x) / self.evidence_temperature)
        return torch.clamp(evidence, max=self.max_evidence)


def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    # KL(Dir(alpha) || Dir(1)) per sample.
    k = alpha.size(-1)
    sum_alpha = alpha.sum(dim=-1)

    log_b_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1)
    # B(1) = Γ(1)^K / Γ(K) = 1 / Γ(K)  => log B(1) = -log Γ(K)
    log_b_uni = -torch.lgamma(torch.tensor(float(k), device=alpha.device))

    dig = torch.digamma(alpha)
    dig_sum = torch.digamma(sum_alpha).unsqueeze(-1)
    kl = log_b_alpha - log_b_uni + ((alpha - 1.0) * (dig - dig_sum)).sum(dim=-1)
    return kl


def edl_nll(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Expected negative log-likelihood under a Dirichlet (Sensoy et al.).
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    y = F.one_hot(labels, num_classes=alpha.size(-1)).float()
    loss = (y * (torch.digamma(sum_alpha) - torch.digamma(alpha))).sum(dim=-1)
    return loss


class EcaClassifier(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        hidden_size: int,
        num_labels: int,
        evidence_temperature: float = 2.0,
        max_evidence: float = 30.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.head = EvidentialHead(
            hidden_size,
            num_labels,
            evidence_temperature=evidence_temperature,
            max_evidence=max_evidence,
        )
        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        evidence = self.head(pooled)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        probs = alpha / strength.unsqueeze(-1)
        u_raw = float(self.num_labels) / strength

        # Normalize uncertainty to [0, 1] within the current batch.
        u = float(self.num_labels) / (strength + 1e-8)
        u = torch.clamp(u, max=1.0)
        u = torch.sigmoid(u)

        return {"alpha": alpha, "probs": probs, "u": u, "u_raw": u_raw}


@torch.no_grad()
def predict_softmax(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
) -> dict[str, Any]:

    model.eval()
    all_probs = []
    all_labels = []
    all_group = []
    all_probs_cf = []

    T = 1.0  # keep stable

    with torch.no_grad():
        for b in dl:
            input_ids = b.input_ids.to(device)
            attn = b.attention_mask.to(device)

            input_ids_cf = b.input_ids_cf.to(device)
            attn_cf = b.attention_mask_cf.to(device)

            # forward
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            logits = temperature_scale(logits, T)
            probs = F.softmax(logits, dim=-1)

            # 🔥 FIX: convert to numpy properly
            all_probs.append(probs.cpu().numpy())

            # 🔥 FIX: ensure labels on CPU
            all_labels.extend(b.labels.cpu().numpy().tolist())

            all_group.extend(b.group)

            # counterfactual
            logits_cf = model(input_ids=input_ids_cf, attention_mask=attn_cf).logits
            logits_cf = temperature_scale(logits_cf, T)
            probs_cf = F.softmax(logits_cf, dim=-1)

            all_probs_cf.append(probs_cf.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.array(all_labels, dtype=np.int64)

    preds = probs.argmax(axis=1)

    out = {
        "probs": probs,
        "preds": preds,   
        "labels": labels,
        "group": np.array(all_group, dtype=object),
        "probs_cf": np.concatenate(all_probs_cf, axis=0),
    }

    return out


@torch.no_grad()
def predict_evidential(model: nn.Module, dl: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    all_labels: list[int] = []
    all_group: list[str] = []
    all_probs_cf: list[np.ndarray] | None = []

    for b in dl:
        out = model(input_ids=b.input_ids.to(device), attention_mask=b.attention_mask.to(device))
        all_probs.append(out["probs"].cpu().numpy())
        all_u.append(out["u"].cpu().numpy())
        all_labels.extend(b.labels.numpy().tolist())
        all_group.extend(b.group)

        if b.input_ids_cf is None:
            all_probs_cf = None
        elif all_probs_cf is not None:
            out_cf = model(input_ids=b.input_ids_cf.to(device), attention_mask=b.attention_mask_cf.to(device))
            all_probs_cf.append(out_cf["probs"].cpu().numpy())

    outd: dict[str, Any] = {
        "probs": np.concatenate(all_probs, axis=0),
        "u": np.concatenate(all_u, axis=0),
        "labels": np.array(all_labels, dtype=np.int64),
        "group": np.array(all_group, dtype=object),
    }
    if all_probs_cf is not None:
        outd["probs_cf"] = np.concatenate(all_probs_cf, axis=0)
    return outd


def summarize_predictions(pred: dict[str, Any], name: str) -> dict[str, Any]:
    probs = pred["probs"]
    labels = pred["labels"]
    group = pred["group"]


    if "preds" in pred:
        preds = pred["preds"]
    else:
        preds = probs.argmax(axis=1)

    acc = float((preds == labels).mean())
    ece = ece_score(probs, labels)

    out: dict[str, Any] = {
        "name": name,
        "accuracy": acc,
        "ece": ece,
    }

    
    groups = sorted({str(g) for g in group.tolist()})
    group_stats: dict[str, Any] = {}

    for g in groups:
        mask = group == g
        if int(mask.sum()) < 5:
            continue

        group_preds = preds[mask]
        group_labels = labels[mask]

        group_stats[g] = {
            "n": int(mask.sum()),
            "accuracy": float((group_preds == group_labels).mean()),
            "ece": ece_score(probs[mask], group_labels),
        }

    out["by_group"] = group_stats

    
    if "probs_cf" in pred:
        probs_cf = pred["probs_cf"]
        diff = np.abs(probs[:, 1] - probs_cf[:, 1])
        out["counterfactual_diff_p1"] = float(diff.mean())

    if "u" in pred:
        u = pred["u"]
        errors = (preds != labels).astype(np.int32)

        out["uncertainty_mean"] = float(u.mean())
        out["uncertainty_auc_error"] = safe_auc(u, errors)

        if "probs_cf" in pred:
            probs_cf = pred["probs_cf"]
            diff = np.abs(probs[:, 1] - probs_cf[:, 1])
            out["counterfactual_weighted_by_u"] = float((u * diff).mean())

    return out


def _take_split(ds: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def inverse_frequency_weights(values: list[Any]) -> list[float]:
    counts: dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    weights = [1.0 / float(counts.get(v, 1)) for v in values]
    mean_w = float(np.mean(weights)) if weights else 1.0
    if mean_w <= 0:
        return weights
    return [float(w) / mean_w for w in weights]


def make_balanced_sampler(train_ds: Dataset) -> WeightedRandomSampler:
    """Create a sampler that balances class labels and demographic groups.

    For Adult this directly helps reduce male/female skew. If male/female are
    not present, we still balance across available groups.
    """

    labels = [int(x) for x in train_ds["label"]]
    groups = [str(g).strip().lower() for g in train_ds["group"]]

    label_counts: dict[int, int] = {}
    group_counts: dict[str, int] = {}
    for y in labels:
        label_counts[y] = label_counts.get(y, 0) + 1
    for g in groups:
        group_counts[g] = group_counts.get(g, 0) + 1

    has_male = "male" in group_counts
    has_female = "female" in group_counts

    weights: list[float] = []
    for y, g in zip(labels, groups):
        w_label = 1.0 / float(label_counts.get(y, 1))

        # Prioritize male/female balancing when those groups exist.
        if has_male and has_female and g in {"male", "female"}:
            w_group = 1.0 / float(group_counts[g])
        else:
            w_group = 1.0 / float(group_counts.get(g, 1))

        # Product gives joint class/group balancing pressure.
        weights.append(w_label * w_group)

    weights_t = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_t, num_samples=len(train_ds), replacement=True)


def load_adult_dataset(seed: int) -> DatasetDict:
    last_err: Exception | None = None
    for name in ["adult", "scikit-learn/adult-census-income", "uciml/adult"]:
        try:
            return load_dataset(name)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load Adult dataset from known sources. Last error: {last_err}")


def prepare_adult(ds: DatasetDict, seed: int) -> DatasetDict:
    # Find a split.
    if "train" in ds and "test" in ds:
        dsd = DatasetDict(train=ds["train"], test=ds["test"])
    else:
        base = ds[list(ds.keys())[0]]
        dsd = base.train_test_split(test_size=0.2, seed=seed)

    label_key = _first_present(dsd["train"].features, ["income", "label", "class", "target"])
    if label_key is None:
        raise RuntimeError(f"Adult dataset label column not found. Columns: {dsd['train'].column_names}")

    sex_key = _first_present(dsd["train"].features, ["sex", "gender"])

    def to_prompt(ex: dict[str, Any]) -> dict[str, Any]:
        label = _normalize_label(ex[label_key])

        # Create a stable textual representation of tabular features.
        parts: list[str] = []
        for k, v in ex.items():
            if k == label_key:
                continue
            if v is None:
                continue
            parts.append(f"{k}: {str(v).strip().lower()}")
        text = ("Adult Census record — " + ", ".join(parts) + ". Predict if income is >50K.").strip()

        group = str(ex.get(sex_key, "unknown")).strip().lower() if sex_key else "unknown"
        text_cf = _flip_sex(text) if sex_key else text
        return {"text": text, "label": label, "group": group, "text_cf": text_cf}

    dsd = dsd.map(to_prompt, remove_columns=dsd["train"].column_names)
    return dsd


def load_toxigen_dataset(seed: int) -> DatasetDict:
    last_err: Exception | None = None
    for name in ["toxigen/toxigen-data", "toxigen", "allenai/toxigen"]:
        try:
            return load_dataset(name)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load ToxiGen dataset from known sources. Last error: {last_err}")


def prepare_toxigen(ds: DatasetDict, seed: int) -> DatasetDict:
    # Pick splits.
    if "train" in ds and "test" in ds:
        dsd = DatasetDict(train=ds["train"], test=ds["test"])
    else:
        base = ds[list(ds.keys())[0]]
        dsd = base.train_test_split(test_size=0.2, seed=seed)

    cols = set(dsd["train"].column_names)
    text_key = next((k for k in ["text", "generation", "sentence", "prompt"] if k in cols), None)
    label_key = next((k for k in ["label", "toxicity", "toxic", "toxicity_human", "toxicity_ai"] if k in cols), None)
    group_key = next((k for k in ["target_group", "group", "target", "demographic"] if k in cols), None)

    if text_key is None or label_key is None:
        raise RuntimeError(f"ToxiGen columns not recognized. Columns: {sorted(cols)}")

    def normalize(ex: dict[str, Any]) -> dict[str, Any]:
        raw_text = str(ex[text_key])

        raw_label = ex[label_key]
        if str(label_key).startswith("toxicity_"):
            # toxigen/toxigen-data uses a 1..5 toxicity score.
            label = 1 if float(raw_label) >= 3.0 else 0
        else:
            label = _normalize_label(raw_label)

        group = str(ex.get(group_key, "unknown")).strip().lower() if group_key else "unknown"

        flipped = _try_flip_group(raw_text)
        raw_text_cf = raw_text if flipped is None else flipped

        # Keep the sensitive attribute out of the input string so a token-level edit is a real counterfactual.
        text = raw_text
        text_cf = raw_text_cf

        return {"text": text, "label": label, "group": group, "text_cf": text_cf}

    dsd = dsd.map(normalize, remove_columns=dsd["train"].column_names)
    return dsd


@torch.no_grad()
def compute_causal_weights_softmax(
    model: nn.Module,
    ds: Dataset,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    # w_i := |p(y=1|x) - p(y=1|x_cf)|  (0 if no counterfactual available)
    model.eval()

    def local_collate(examples: list[dict[str, Any]]) -> Batch:
        return collate_fn(tokenizer, examples)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=local_collate)

    weights: list[float] = []
    for b in dl:
        if b.input_ids_cf is None:
            weights.extend([0.0] * int(b.labels.size(0)))
            continue

        logits = model(input_ids=b.input_ids.to(device), attention_mask=b.attention_mask.to(device)).logits
        probs = F.softmax(logits, dim=-1)[:, 1]
        logits_cf = model(input_ids=b.input_ids_cf.to(device), attention_mask=b.attention_mask_cf.to(device)).logits
        probs_cf = F.softmax(logits_cf, dim=-1)[:, 1]
        diff = (probs - probs_cf).abs().detach().cpu().numpy().tolist()
        weights.extend([float(x) for x in diff])

    if len(weights) != len(ds):
        raise RuntimeError("Causal weight length mismatch")
    return weights


def train_softmax_model(
    model_name: str,
    num_labels: int,
    train_ds: Dataset,
    tokenizer: Any,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_steps: int | None,
    balance_sampling: bool,
    sample_weighting: bool = False,
) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    model.train()

    if balance_sampling:
        sampler = make_balanced_sampler(train_ds)
        dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=lambda ex: collate_fn(tokenizer, ex))
    else:
        dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda ex: collate_fn(tokenizer, ex))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    steps = 0
    for _ in range(epochs):
        for b in dl:
            steps += 1
            opt.zero_grad(set_to_none=True)
            logits = model(
                input_ids=b.input_ids.to(device),
                attention_mask=b.attention_mask.to(device),
            ).logits
            if sample_weighting and b.sample_w is not None:
                loss_per = F.cross_entropy(logits, b.labels.to(device), reduction="none")
                w = b.sample_w.to(device)
                loss = (loss_per * w).mean()
            else:
                loss = F.cross_entropy(logits, b.labels.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
            

            if max_steps is not None and steps >= max_steps:
                return model

    return model


def train_eca_model(
    model_name: str,
    num_labels: int,
    train_ds: Dataset,
    tokenizer: Any,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_steps: int | None,
    lambda_kl: float,
    kl_start_factor: float,
    gamma_causal: float,
    overconf_penalty: float,
    evidence_floor: float,
    collapse_penalty: float,
    correct_confidence_bonus: float,
    correct_uncertainty_penalty: float,
    evidence_temperature: float,
    max_evidence: float,
    balance_sampling: bool,
) -> nn.Module:
    base = AutoModel.from_pretrained(model_name)
    hidden_size = int(getattr(base.config, "hidden_size"))
    model = EcaClassifier(
        base=base,
        hidden_size=hidden_size,
        num_labels=num_labels,
        evidence_temperature=evidence_temperature,
        max_evidence=max_evidence,
    )
    model.to(device)
    model.train()

    if balance_sampling:
        sampler = make_balanced_sampler(train_ds)
        dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=lambda ex: collate_fn(tokenizer, ex))
    else:
        dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda ex: collate_fn(tokenizer, ex))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    steps = 0
    kl_start_factor = float(min(max(kl_start_factor, 0.0), 1.0))
    for epoch_idx in range(epochs):
        if epochs <= 1:
            epoch_progress = 1.0
        else:
            epoch_progress = float(epoch_idx) / float(epochs - 1)
        # Epoch-wise KL annealing: start small and increase over epochs.
        lambda_kl_t = float(lambda_kl) * (kl_start_factor + (1.0 - kl_start_factor) * epoch_progress)

        for b in dl:
            steps += 1
            opt.zero_grad(set_to_none=True)

            out = model(input_ids=b.input_ids.to(device), attention_mask=b.attention_mask.to(device))
            alpha = out["alpha"]
            u = out["u"].detach()  # avoid the model gaming the weighting

            nll = edl_nll(alpha, b.labels.to(device))
            kl = kl_dirichlet_to_uniform(alpha)
            probs = out["probs"]
            strength = alpha.sum(dim=-1)
            evidence = alpha - 1.0

            # data-dependent causal influence weight (fixed, precomputed)
            if b.causal_w is None:
                causal = torch.ones_like(nll)
            else:
                causal = b.causal_w.to(device).detach()

            # Requested weighting: make high-uncertainty samples matter more.
            # uncertainty is normalized and clamped in the model forward pass.
            w = torch.minimum(u, torch.tensor(1.0, device=u.device)) * (1 + gamma_causal * causal)

            # Strong penalty when the model is confidently wrong.
            labels = b.labels.to(device)
            pred = probs.argmax(dim=-1)
            wrong = (pred != labels).float()
            correct = 1.0 - wrong
            conf = probs.max(dim=-1).values
            overconf = wrong * (conf**2) * torch.log1p(torch.relu(strength - float(num_labels)))
            overconf_term = overconf.sum() / (wrong.sum() + 1e-6)

            # Prevent evidence collapse (all evidence driven too close to zero).
            collapse_term = torch.relu(float(evidence_floor) - evidence).mean()

            # Encourage confident predictions when the model is correct.
            p_true = probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            correct_conf_term = (correct * (1.0 - p_true)).sum() / (correct.sum() + 1e-6)
            correct_uncertainty_term = (correct * u).sum() / (correct.sum() + 1e-6)

            loss = (
                (w * nll).mean()
                + lambda_kl_t * kl.mean()
                + float(overconf_penalty) * overconf_term
                + float(collapse_penalty) * collapse_term
                + float(correct_confidence_bonus) * correct_conf_term
                + float(correct_uncertainty_penalty) * correct_uncertainty_term
            )
            loss.backward()
            opt.step()

            if max_steps is not None and steps >= max_steps:
                return model

    return model


def run_one_dataset(
    *,
    dataset_name: str,
    dsd: DatasetDict,
    model_name: str,
    seed: int,
    train_size: int,
    eval_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    max_steps: int | None,
    lambda_kl: float,
    kl_start_factor: float,
    gamma_causal: float,
    overconf_penalty: float,
    evidence_floor: float,
    collapse_penalty: float,
    correct_confidence_bonus: float,
    correct_uncertainty_penalty: float,
    evidence_temperature: float,
    max_evidence: float,
    balance_sampling: bool,
    plot_dir: str | None,
    device: torch.device,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(model_name)

    train_ds = _take_split(dsd["train"], train_size, seed)
    test_ds = _take_split(dsd["test"], eval_size, seed + 1)

    softmax_model = train_softmax_model(
        model_name=model_name,
        num_labels=2,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        balance_sampling=balance_sampling,
    )

    class_w = inverse_frequency_weights([_normalize_label(x) for x in train_ds["label"]])
    group_w = inverse_frequency_weights([str(g).strip().lower() for g in train_ds["group"]])

    train_ds_class_w = train_ds.add_column("sample_w", class_w)
    train_ds_group_w = train_ds.add_column("sample_w", group_w)

    reweight_class_model = train_softmax_model(
        model_name=model_name,
        num_labels=2,
        train_ds=train_ds_class_w,
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        balance_sampling=balance_sampling,
        sample_weighting=True,
    )

    reweight_group_model = train_softmax_model(
        model_name=model_name,
        num_labels=2,
        train_ds=train_ds_group_w,
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        balance_sampling=balance_sampling,
        sample_weighting=True,
    )

    # Baseline: "standard causal prompting" approximation.
    # (We implement it as an instruction prefix that explicitly blocks sensitive-attribute use.)
    if dataset_name == "adult":
        prompt_prefix = "Instruction: ignore sex/gender when predicting income. "
    else:
        prompt_prefix = "Instruction: ignore the target group/demographics when predicting toxicity. "

    def add_prefix(ex: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": prompt_prefix + ex["text"],
            "text_cf": prompt_prefix + ex["text_cf"],
            "label": ex["label"],
            "group": ex.get("group", "unknown"),
        }

    train_ds_prompt = train_ds.map(add_prefix)
    test_ds_prompt = test_ds.map(add_prefix)

    causal_prompt_model = train_softmax_model(
        model_name=model_name,
        num_labels=2,
        train_ds=train_ds_prompt,
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        balance_sampling=balance_sampling,
    )

    # Precompute causal weights on the *training* set using the trained baseline.
    causal_w = compute_causal_weights_softmax(
        model=softmax_model,
        ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    train_ds_w = train_ds.add_column("causal_w", causal_w)

    eca_model = train_eca_model(
        model_name=model_name,
        num_labels=2,
        train_ds=train_ds_w,
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        lambda_kl=lambda_kl,
        kl_start_factor=kl_start_factor,
        gamma_causal=gamma_causal,
        overconf_penalty=overconf_penalty,
        evidence_floor=evidence_floor,
        collapse_penalty=collapse_penalty,
        correct_confidence_bonus=correct_confidence_bonus,
        correct_uncertainty_penalty=correct_uncertainty_penalty,
        evidence_temperature=evidence_temperature,
        max_evidence=max_evidence,
        balance_sampling=balance_sampling,
    )

    eval_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda ex: collate_fn(tokenizer, ex))
    eval_dl_prompt = DataLoader(
        test_ds_prompt, batch_size=batch_size, shuffle=False, collate_fn=lambda ex: collate_fn(tokenizer, ex)
    )

    soft_pred = predict_softmax(softmax_model, eval_dl, device)
    reweight_class_pred = predict_softmax(reweight_class_model, eval_dl, device)
    reweight_group_pred = predict_softmax(reweight_group_model, eval_dl, device)
    causal_prompt_pred = predict_softmax(causal_prompt_model, eval_dl_prompt, device)
    eca_pred = predict_evidential(eca_model, eval_dl, device)

    plot_file: str | None = None
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, f"uncertainty_hist_{dataset_name}_eca.png")
        plot_uncertainty_histogram(
            uncertainties=eca_pred["u"],
            probs=eca_pred["probs"],
            labels=eca_pred["labels"],
            out_path=plot_file,
            title=f"{dataset_name.upper()} - Uncertainty for Correct vs Incorrect",
        )

    result = {
        "dataset": dataset_name,
        "n_train": int(len(train_ds)),
        "n_eval": int(len(test_ds)),
        "model": model_name,
        "softmax": summarize_predictions(soft_pred, "softmax"),
        "reweight_class": summarize_predictions(reweight_class_pred, "reweight_class"),
        "reweight_group": summarize_predictions(reweight_group_pred, "reweight_group"),
        "causal_prompting": summarize_predictions(causal_prompt_pred, "causal_prompting"),
        "eca": summarize_predictions(eca_pred, "eca"),
    }
    if plot_file is not None:
        result["uncertainty_histogram"] = plot_file
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["adult", "toxigen", "all"], default="all")
    # Default to a widely supported checkpoint that reliably ships a fast tokenizer.
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--eval-size", type=int, default=400)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lambda-kl", type=float, default=0.02)
    p.add_argument("--kl-start-factor", type=float, default=0.1)
    p.add_argument("--gamma-causal", type=float, default=3.0)
    p.add_argument("--overconf-penalty", type=float, default=0.5)
    p.add_argument("--evidence-floor", type=float, default=0.2)
    p.add_argument("--collapse-penalty", type=float, default=0.2)
    p.add_argument("--correct-confidence-bonus", type=float, default=0.15)
    p.add_argument("--correct-uncertainty-penalty", type=float, default=0.2)
    p.add_argument("--evidence-temperature", type=float, default=2.0)
    p.add_argument("--max-evidence", type=float, default=30.0)
    p.add_argument("--balance-sampling", dest="balance_sampling", action="store_true")
    p.add_argument("--no-balance-sampling", dest="balance_sampling", action="store_false")
    p.set_defaults(balance_sampling=True)
    p.add_argument("--plot-dir", default="plots")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[str, Any] = {
        "run": {
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "train_size": args.train_size,
            "eval_size": args.eval_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "lambda_kl": args.lambda_kl,
            "kl_start_factor": args.kl_start_factor,
            "gamma_causal": args.gamma_causal,
            "overconf_penalty": args.overconf_penalty,
            "evidence_floor": args.evidence_floor,
            "collapse_penalty": args.collapse_penalty,
            "correct_confidence_bonus": args.correct_confidence_bonus,
            "correct_uncertainty_penalty": args.correct_uncertainty_penalty,
            "evidence_temperature": args.evidence_temperature,
            "max_evidence": args.max_evidence,
            "balance_sampling": args.balance_sampling,
            "plot_dir": args.plot_dir,
            "device": str(device),
        },
        "experiments": [],
    }

    todo: list[tuple[str, DatasetDict]] = []
    if args.dataset in {"adult", "all"}:
        todo.append(("adult", prepare_adult(load_adult_dataset(args.seed), args.seed)))
    if args.dataset in {"toxigen", "all"}:
        todo.append(("toxigen", prepare_toxigen(load_toxigen_dataset(args.seed), args.seed)))

    for name, dsd in todo:
        print(f"\n=== Running: {name} ===")
        exp = run_one_dataset(
            dataset_name=name,
            dsd=dsd,
            model_name=args.model,
            seed=args.seed,
            train_size=args.train_size,
            eval_size=args.eval_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            lambda_kl=args.lambda_kl,
            kl_start_factor=args.kl_start_factor,
            gamma_causal=args.gamma_causal,
            overconf_penalty=args.overconf_penalty,
            evidence_floor=args.evidence_floor,
            collapse_penalty=args.collapse_penalty,
            correct_confidence_bonus=args.correct_confidence_bonus,
            correct_uncertainty_penalty=args.correct_uncertainty_penalty,
            evidence_temperature=args.evidence_temperature,
            max_evidence=args.max_evidence,
            balance_sampling=args.balance_sampling,
            plot_dir=args.plot_dir,
            device=device,
        )
        results["experiments"].append(exp)

        print(json.dumps(exp, indent=2))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
