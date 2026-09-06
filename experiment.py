"""
Federated learning on interpretable speech biomarkers for Parkinson's disease.

This module contains the experiment only: synthetic data generation, the
federated / centralized / local-only training regimes, and the statistical
tests.

Run directly to reproduce the numbers reported in the paper:

    python experiment.py

"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------
# Configuration 
# ---------------------------------------------------------------------
N_SEEDS = 30            # independent repetitions of the whole pipeline
FL_ROUNDS = 20          # T: federated communication rounds
LOCAL_EPOCHS = 15       # E: local epochs per round
LEARNING_RATE = 0.30
L2 = 1e-3
TEST_FRACTION = 0.30    # 70/30 train/test split per site

EQUIVALENCE_BOUND = 0.02   # TOST bound on AUC 

RESULTS_FILE = "results.npz"

FEATURE_NAMES = ["VSA", "jitter_pct", "shimmer_db", "hnr_db", "f0_std_hz"]

EXPECTED_SIGN = {
    "VSA": -1,          # larger vowel space  -> healthy
    "jitter_pct": +1,   # more perturbation   -> PD
    "shimmer_db": +1,
    "hnr_db": -1,       # higher harmonics-to-noise -> healthy
    "f0_std_hz": -1,    # more pitch variation      -> healthy
}

# Canonical corner-vowel formant targets (F1, F2) in Hz.
VOWEL_TARGETS = {
    "a": np.array([730.0, 1090.0]),
    "i": np.array([270.0, 2290.0]),
    "u": np.array([300.0, 870.0]),
}
VOWEL_CENTER = np.mean(list(VOWEL_TARGETS.values()), axis=0)

# Three simulated institutions. Non-IID heterogeneity comes from
# site_scale (vowel-space size) and site_shift (recording conditions).
SITES = {
    "Site A": {"n": 200, "pd_rate": 0.45, "site_scale": 1.00,
               "site_shift": np.array([0.0, 0.0])},
    "Site B": {"n": 130, "pd_rate": 0.50, "site_scale": 1.15,
               "site_shift": np.array([10.0, 80.0])},
    "Site C": {"n": 200, "pd_rate": 0.55, "site_scale": 0.85,
               "site_shift": np.array([-8.0, -60.0])},
}

# Class-conditional phonatory distributions: (PD mean, healthy mean, sd).
PHONATORY = {
    "jitter_pct": (0.50, 0.42, 0.16),
    "shimmer_db": (0.30, 0.26, 0.09),
    "hnr_db": (19.5, 21.0, 3.3),
    "f0_std_hz": (19.0, 23.0, 6.5),
}


# ---------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------
def triangle_area(fa: np.ndarray, fi: np.ndarray, fu: np.ndarray) -> float:
    """Vowel space area: area of the /a/-/i/-/u/ triangle in the F1-F2 plane."""
    return 0.5 * abs(
        fa[0] * (fi[1] - fu[1])
        + fi[0] * (fu[1] - fa[1])
        + fu[0] * (fa[1] - fi[1])
    )


def generate_speaker(rng, is_pd, site_scale, site_shift):
    """One speaker: five biomarkers plus their corner-vowel positions.
    """
    undershoot = np.clip(
        rng.normal(0.27 if is_pd else 0.10, 0.15 if is_pd else 0.13), 0, 0.85
    )
    center = VOWEL_CENTER + site_shift

    vowels = {}
    for v, target in VOWEL_TARGETS.items():
        base = center + site_scale * (target - VOWEL_CENTER)
        vowels[v] = base + undershoot * (center - base) + rng.normal(0, 34, size=2)

    vsa = triangle_area(vowels["a"], vowels["i"], vowels["u"])
    features = [vsa]
    for name in FEATURE_NAMES[1:]:
        pd_mean, hc_mean, sd = PHONATORY[name]
        features.append(rng.normal(pd_mean if is_pd else hc_mean, sd))

    return np.array(features), vowels


def generate_site(cfg, seed):
    """All speakers for one institution."""
    rng = np.random.default_rng(seed)
    X, y, vowel_records = [], [], []
    for _ in range(cfg["n"]):
        is_pd = rng.random() < cfg["pd_rate"]
        feats, vowels = generate_speaker(
            rng, is_pd, cfg["site_scale"], cfg["site_shift"]
        )
        X.append(feats)
        y.append(int(is_pd))
        vowel_records.append((is_pd, vowels))
    return np.array(X), np.array(y, dtype=float), vowel_records


def train_test_split(X, y, frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - frac))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class LogisticModel:
    """Logistic regression trained by full-batch gradient descent.
    """

    def __init__(self, n_features, l2=0.0):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.l2 = l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, epochs, lr):
        n = len(y)
        for _ in range(epochs):
            err = self._sigmoid(X @ self.w + self.b) - y
            self.w -= lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= lr * err.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def get_params(self):
        """Full parameter vector (weights + bias), as sent to the server."""
        return np.concatenate([self.w, [self.b]])

    def set_params(self, vec):
        self.w = vec[:-1].copy()
        self.b = float(vec[-1])
        return self


def federated_average(param_vectors, sample_counts):
    """FedAvg: sample-weighted mean of the client parameter vectors."""
    return np.average(
        np.vstack(param_vectors), axis=0, weights=np.asarray(sample_counts, float)
    )


class Standardizer:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def evaluate(model, X, y):
    p = model.predict_proba(X)
    return roc_auc_score(y, p), accuracy_score(y, (p >= 0.5).astype(int))


# ---------------------------------------------------------------------
# One seed: all three training regimes on identical data
# ---------------------------------------------------------------------
def run_single_seed(seed):
    clients, vowel_records = {}, []
    for i, (name, cfg) in enumerate(SITES.items()):
        X, y, vrec = generate_site(cfg, seed * 100 + i)
        Xtr, ytr, Xte, yte = train_test_split(X, y, TEST_FRACTION, seed * 100 + i)
        clients[name] = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        vowel_records.extend(vrec)

    scaler = Standardizer().fit(np.vstack([c["Xtr"] for c in clients.values()]))
    for c in clients.values():
        c["Xtr_s"] = scaler.transform(c["Xtr"])
        c["Xte_s"] = scaler.transform(c["Xte"])

    X_test = np.vstack([c["Xte_s"] for c in clients.values()])
    y_test = np.concatenate([c["yte"] for c in clients.values()])
    d = len(FEATURE_NAMES)

    # 1. Centralized upper bound (requires pooling the training data).
    central = LogisticModel(d, L2).fit(
        np.vstack([c["Xtr_s"] for c in clients.values()]),
        np.concatenate([c["ytr"] for c in clients.values()]),
        FL_ROUNDS * LOCAL_EPOCHS,
        LEARNING_RATE,
    )
    central_auc, _ = evaluate(central, X_test, y_test)

    # 2. Local-only baseline: mean over three isolated site models.
    local_aucs = [
        evaluate(
            LogisticModel(d, L2).fit(
                c["Xtr_s"], c["ytr"], FL_ROUNDS * LOCAL_EPOCHS, LEARNING_RATE
            ),
            X_test,
            y_test,
        )[0]
        for c in clients.values()
    ]

    # 3. Federated (FedAvg). Only parameter vectors leave a site.
    global_model = LogisticModel(d, L2)
    history = []
    for _ in range(FL_ROUNDS):
        params, sizes = [], []
        for c in clients.values():
            local = LogisticModel(d, L2).set_params(global_model.get_params())
            local.fit(c["Xtr_s"], c["ytr"], LOCAL_EPOCHS, LEARNING_RATE)
            params.append(local.get_params())
            sizes.append(len(c["ytr"]))
        global_model.set_params(federated_average(params, sizes))
        history.append(evaluate(global_model, X_test, y_test)[0])

    fed_auc, fed_acc = evaluate(global_model, X_test, y_test)

    # Mean corner-vowel positions, for the vowel-triangle figure.
    mean_vowels = {}
    for label, want_pd in (("healthy", False), ("pd", True)):
        mean_vowels[label] = {
            v: np.mean([rec[1][v] for rec in vowel_records if rec[0] == want_pd], axis=0)
            for v in VOWEL_TARGETS
        }

    return {
        "local_auc": float(np.mean(local_aucs)),
        "fed_auc": fed_auc,
        "fed_acc": fed_acc,
        "central_auc": central_auc,
        "history": history,
        "weights": global_model.w.copy(),
        "mean_vowels": mean_vowels,
    }


# ---------------------------------------------------------------------
# Benchmark over all seeds + statistics
# ---------------------------------------------------------------------
def ci95(a):
    """Half-width of the 95% confidence interval of the mean."""
    return 1.96 * stats.sem(a)


def run_benchmark(n_seeds=N_SEEDS, verbose=True):
    if verbose:
        print(f"Running benchmark over {n_seeds} seeds...")
    results = [run_single_seed(s) for s in range(1, n_seeds + 1)]

    local = np.array([r["local_auc"] for r in results])
    fed = np.array([r["fed_auc"] for r in results])
    central = np.array([r["central_auc"] for r in results])
    weights = np.array([r["weights"] for r in results])
    histories = np.array([r["history"] for r in results])
    fed_acc = np.array([r["fed_acc"] for r in results])

    # Paired comparisons: every regime sees identical data within a seed.
    d_local = fed - local
    d_central = fed - central
    p_vs_local = stats.ttest_rel(fed, local).pvalue

    # TOST: is federated equivalent to centralized within +/- EQUIVALENCE_BOUND?
    p_tost = max(
        stats.ttest_1samp(d_central, -EQUIVALENCE_BOUND, alternative="greater").pvalue,
        stats.ttest_1samp(d_central, EQUIVALENCE_BOUND, alternative="less").pvalue,
    )

    # Explanation stability: how often is each feature ranked most important,
    # and does every coefficient carry its physiologically expected sign?
    top_counts = {name: 0 for name in FEATURE_NAMES}
    for w in weights:
        top_counts[FEATURE_NAMES[int(np.argmax(np.abs(w)))]] += 1
    mean_w = weights.mean(axis=0)
    signs_correct = sum(
        np.sign(mean_w[i]) == EXPECTED_SIGN[name]
        for i, name in enumerate(FEATURE_NAMES)
    )

    # Mean vowel triangle, averaged over seeds.
    mean_vowels = {
        label: {
            v: np.mean([r["mean_vowels"][label][v] for r in results], axis=0)
            for v in VOWEL_TARGETS
        }
        for label in ("healthy", "pd")
    }
    vsa = {
        label: triangle_area(tri["a"], tri["i"], tri["u"])
        for label, tri in mean_vowels.items()
    }

    summary = {
        "local_auc": local, "fed_auc": fed, "central_auc": central,
        "fed_acc": fed_acc, "weights": weights, "histories": histories,
        "d_local": d_local, "d_central": d_central,
        "p_vs_local": p_vs_local, "p_tost": p_tost,
        "top_counts": top_counts, "signs_correct": signs_correct,
        "mean_vowels": mean_vowels, "vsa": vsa,
    }

    if verbose:
        report(summary)
    return summary


def report(s):

    print()
    print("AUC (mean +/- 95% CI over seeds)")
    print(f"  Local-only   {s['local_auc'].mean():.3f} +/- {ci95(s['local_auc']):.3f}")
    print(f"  Federated    {s['fed_auc'].mean():.3f} +/- {ci95(s['fed_auc']):.3f}")
    print(f"  Centralized  {s['central_auc'].mean():.3f} +/- {ci95(s['central_auc']):.3f}")
    print(f"  Federated accuracy {s['fed_acc'].mean():.3f}")
    print()
    print("Paired comparisons")
    d = s["d_local"]
    print(f"  Federated - local:      {d.mean():+.3f} +/- {ci95(d):.3f}, "
          f"p = {s['p_vs_local']:.2e}, positive in {(d > 0).sum()}/{len(d)} seeds")
    dc = s["d_central"]
    print(f"  Federated - centralized: {dc.mean():+.4f}, "
          f"TOST (+/-{EQUIVALENCE_BOUND}) p = {s['p_tost']:.2e}")
    print()
    print("Explanation stability")
    top = max(s["top_counts"], key=s["top_counts"].get)
    print(f"  Top-ranked feature: {top} in {s['top_counts'][top]}/{len(s['fed_auc'])} seeds")
    print(f"  Coefficient signs matching physiology: {s['signs_correct']}/{len(FEATURE_NAMES)}")
    mean_w = s["weights"].mean(axis=0)
    ci_w = 1.96 * stats.sem(s["weights"], axis=0)
    for name, m, c in zip(FEATURE_NAMES, mean_w, ci_w):
        print(f"    {name:12s} {m:+.3f} +/- {c:.3f}")
    print()
    print("Vowel space area (mean triangle over seeds)")
    print(f"  Healthy {s['vsa']['healthy']/1000:.0f}k Hz^2, "
          f"PD {s['vsa']['pd']/1000:.0f}k Hz^2 "
          f"({100 * (1 - s['vsa']['pd'] / s['vsa']['healthy']):.0f}% contraction)")


# ---------------------------------------------------------------------
# Caching, so figures can be redrawn without re-running the benchmark
# ---------------------------------------------------------------------
def save_results(s, path=RESULTS_FILE):
    flat = {
        "local_auc": s["local_auc"], "fed_auc": s["fed_auc"],
        "central_auc": s["central_auc"], "fed_acc": s["fed_acc"],
        "weights": s["weights"], "histories": s["histories"],
        "p_vs_local": s["p_vs_local"], "p_tost": s["p_tost"],
        "signs_correct": s["signs_correct"],
        "top_counts": np.array([s["top_counts"][n] for n in FEATURE_NAMES]),
    }
    for label in ("healthy", "pd"):
        for v in VOWEL_TARGETS:
            flat[f"vowel_{label}_{v}"] = s["mean_vowels"][label][v]
    np.savez(path, **flat)
    print(f"\nSaved results to {path}")


def load_results(path=RESULTS_FILE):
    z = np.load(path)
    s = {k: z[k] for k in z.files}
    s["p_vs_local"] = float(s["p_vs_local"])
    s["p_tost"] = float(s["p_tost"])
    s["signs_correct"] = int(s["signs_correct"])
    s["top_counts"] = {n: int(c) for n, c in zip(FEATURE_NAMES, s["top_counts"])}
    s["d_local"] = s["fed_auc"] - s["local_auc"]
    s["d_central"] = s["fed_auc"] - s["central_auc"]
    s["mean_vowels"] = {
        label: {v: s[f"vowel_{label}_{v}"] for v in VOWEL_TARGETS}
        for label in ("healthy", "pd")
    }
    s["vsa"] = {
        label: triangle_area(t["a"], t["i"], t["u"])
        for label, t in s["mean_vowels"].items()
    }
    return s


if __name__ == "__main__":
    save_results(run_benchmark())
