import numpy as np


def compute_reliability(confidences, correct, n_bins=10):
    """Bins predictions by confidence and computes per-bin accuracy."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = correct[mask].mean()
            bin_confs[i] = confidences[mask].mean()
    return bin_edges, bin_accs, bin_confs, bin_counts


def ece_mce(bin_accs, bin_confs, bin_counts):
    """Expected and Maximum Calibration Error from a reliability binning."""
    total = bin_counts.sum()
    gaps = np.abs(bin_accs - bin_confs)
    weights = bin_counts / total if total > 0 else bin_counts
    ece = float(np.sum(weights * gaps))
    nonzero = bin_counts > 0
    mce = float(np.max(gaps[nonzero])) if nonzero.any() else 0.0
    return ece, mce


def brier_score(probs, labels, n_classes):
    """Mean squared error between predicted probs and one-hot true labels."""
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def nll(probs, labels):
    eps = 1e-12
    p_true = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def temperature_scale(probs, T):
    """
    Applies temperature T to already-softmaxed probabilities. Using log(probs)
    as a logit surrogate is exact here: softmax is invariant to the per-sample
    additive constant that log(softmax(z)) differs from z by, so this
    reproduces true temperature scaling without needing raw logits.
    """
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def find_best_temperature(probs, labels, t_min=0.3, t_max=3.0, step=0.02):
    """Grid search for the T that minimizes NLL. Demo-grade: fit and evaluated
    on the same test set. In a rigorous pipeline this fit would use a held-out
    calibration split, separate from the reported test metrics."""
    t_range = np.arange(t_min, t_max + step, step)
    best_T, best_nll = 1.0, nll(probs, labels)
    for T in t_range:
        n = nll(temperature_scale(probs, T), labels)
        if n < best_nll:
            best_nll, best_T = n, T
    return float(best_T), best_nll