import numpy as np
import random
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================
# Data & base model utils
# =========================
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data, data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target
    )

def build_tfidf(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=5,
        sublinear_tf=True,
        norm="l2"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

def train_ls_svm(X_train, y_train, C: float = 10.0):
    """Multi-class least-squares SVM via ridge regression (one-shot)."""
    classes = np.unique(y_train)
    n_classes = len(classes)
    Y = np.zeros((X_train.shape[0], n_classes))
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1.0
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver="auto")
    ridge.fit(X_train, Y)
    return ridge.coef_  # (K, d)

# ================
# Unlearning step
# ================
def influence_removal_ls_svm(X_train, y_train, theta_orig,
                             removal_indices: np.ndarray, C: float = 10.0):
    """
    Influence-style downdate for LS-SVM.
    Solve (I + C X^T X) v = grad_removal, then theta_unlearn = theta_orig - v
    """
    n_classes, n_features = theta_orig.shape

    # aggregate gradient contribution of removed samples
    grad_sum = np.zeros_like(theta_orig)
    for idx in removal_indices:
        x = X_train[idx].toarray().ravel()
        one_hot = np.zeros(n_classes); one_hot[y_train[idx]] = 1
        scores = theta_orig @ x
        error = scores - one_hot  # LS residual
        grad_sum += C * np.outer(error, x)

    def hess_matvec(v):
        Xv = X_train.dot(v)
        return v + C * (X_train.T.dot(Xv))

    H_op = LinearOperator((n_features, n_features), matvec=hess_matvec)

    theta_unlearn = np.zeros_like(theta_orig)
    for c in range(n_classes):
        v_c, _ = cg(H_op, grad_sum[c])
        theta_unlearn[c] = theta_orig[c] - v_c
    return theta_unlearn

# ==========================================
# Threshold theory + empirical probe (no τ)
# ==========================================
def compute_margins_and_sigma_th(theta, X, y):
    """
    margins: m_i = f_{y_i}(x_i) - max_{j!=y_i} f_j(x_i)
    sigma_th(i) ≈ m_i / (sqrt(2) * ||x_i||)
    """
    logits = X @ theta.T  # (N,K)
    true = logits[np.arange(len(y)), y]
    logits_copy = logits.copy()
    logits_copy[np.arange(len(y)), y] = -np.inf
    runnerup = np.max(logits_copy, axis=1)
    margins = true - runnerup
    x_norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    sigma_th = np.divide(np.maximum(margins, 0.0),
                         np.sqrt(2.0) * np.maximum(x_norms, 1e-12))
    return margins, x_norms, sigma_th

def accuracy_and_flip_rate(theta, X, y, sigma, rng):
    """
    Accuracy and flip-rate (vs sigma=0) on the same X,y.
    """
    base_logits = X @ theta.T
    base_preds  = base_logits.argmax(axis=1)

    noise = rng.normal(0.0, sigma, size=theta.shape)
    logits = X @ (theta + noise).T
    preds  = logits.argmax(axis=1)

    acc = (preds == y).mean()
    flip = (preds != base_preds).mean()
    return float(acc), float(flip)

def find_plateau_sigma(sigmas, accs, min_span=3, slope_eps=2e-3):
    """
    In log10(sigma) space, find first index where |d acc / d log10(sigma)| < slope_eps
    for 'min_span' consecutive intervals. Returns (idx or None).
    """
    s = np.log10(np.maximum(sigmas, 1e-16))
    slopes = np.diff(accs) / np.maximum(np.diff(s), 1e-12)
    count = 0
    for i, g in enumerate(np.abs(slopes)):
        count = count + 1 if g < slope_eps else 0
        if count >= min_span:
            return i - (min_span - 1)
    return None

def plot_threshold_for_C(C_value, sigma_grid, accs, flips, q25, q50, q75, sigma_plateau):
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ln1 = ax1.plot(sigma_grid, accs, marker='o', linewidth=2, color="#1f77b4",
                   label="Accuracy (retained set)")
    ax1.set_xscale('log'); ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel(r"Noise $\sigma$ (log)"); ax1.set_ylabel("Accuracy", color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.grid(True, ls=":", alpha=0.35)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(sigma_grid, flips, marker='s', linestyle='--', linewidth=2, color="#ff7f0e",
                   label="Flip rate")
    ax2.set_ylabel("Flip rate", color="#ff7f0e")
    ax2.tick_params(axis='y', labelcolor="#ff7f0e")

    # Shade C-specific threshold band (theory)
    for ax in (ax1, ax2):
        ax.axvspan(max(q25,1e-12), max(q75,1e-12), color="#ffddaa", alpha=0.35,
                   label="margin band (Q25–Q75)")
        ax.axvline(max(q50,1e-12), color="#ffb347", linestyle=":", alpha=0.8)

    # Plateau marker
    if sigma_plateau is not None:
        for ax in (ax1, ax2):
            ax.axvline(sigma_plateau, color="#2ca02c", linestyle=":", alpha=0.9,
                       label="plateau start")

    # Legend (dedup)
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)

    title_plateau = f"{sigma_plateau:.3g}" if sigma_plateau is not None else "n/a"
    ax1.set_title(f"Threshold @ C={C_value}  |  σ_th band≈[{q25:.3g}, {q75:.3g}]  |  plateau≈{title_plateau}")
    plt.tight_layout(); plt.show()

# ===========
# Main sweep (NO τ — per-C only)
# ===========
def main():
    # --- data & tf-idf (same as before) ---
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Fix class_to_unlearn once for reproducibility across all C (like before)
    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    # Your C grid (unchanged)
    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]

    # For summaries
    summary_rows = []

    for C in C_list:
        print("\n" + "="*72)
        print(f"[C={C}] training baseline & performing unlearning (Hessian downdate only)")

        # Train & unlearn (identical flow)
        theta_orig = train_ls_svm(X_train, y_train, C)
        removal_indices = np.where(y_train == class_to_unlearn)[0]
        theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, removal_indices, C)

        theta_base = theta_un.copy()
        theta_base[class_to_unlearn, :] = 0.0

        # Retained set
        keep = (y_test != class_to_unlearn)
        Xte_keep = X_test[keep]
        yte_keep = y_test[keep]

        # Theory band for this C
        _, _, sigma_th = compute_margins_and_sigma_th(theta_base, Xte_keep, yte_keep)
        q25 = np.percentile(sigma_th, 25)
        q50 = np.median(sigma_th)
        q75 = np.percentile(sigma_th, 75)
        print(f"[C={C}] σ_th band: Q25={q25:.4g}, Q50={q50:.4g}, Q75={q75:.4g}")

        # Empirical sweep (centered at median threshold)
        center = max(q50, 1e-6)
        sigma_grid = np.logspace(np.log10(center/50), np.log10(center*50), 41)
        rng = np.random.RandomState(123)
        accs, flips = [], []
        for s in sigma_grid:
            a, f = accuracy_and_flip_rate(theta_base, Xte_keep, yte_keep, s, rng)
            accs.append(a); flips.append(f)

        # Plateau detection
        idx_plateau = find_plateau_sigma(sigma_grid, accs, min_span=3, slope_eps=2e-3)
        sigma_plateau = sigma_grid[idx_plateau] if idx_plateau is not None else None
        print(f"[C={C}] plateau σ≈{(sigma_plateau if sigma_plateau else 'n/a')}")

        # Plot per-C threshold figure
        plot_threshold_for_C(C, sigma_grid, accs, flips, q25, q50, q75, sigma_plateau)

        # Save summary
        summary_rows.append({
            "C": C, "reg": 1.0/C, "q25": q25, "q50": q50, "q75": q75,
            "sigma_plateau": (sigma_plateau if sigma_plateau else np.nan),
            "acc_at_q25": float(np.interp(q25, sigma_grid, accs)),
            "acc_at_q50": float(np.interp(q50, sigma_grid, accs)),
            "acc_at_q75": float(np.interp(q75, sigma_grid, accs)),
        })

    # ------- Optional: compact summary plot over C -------
    # Show median threshold and plateau σ versus C (log y)
    Cs = [row["C"] for row in summary_rows]
    q50s = [row["q50"] for row in summary_rows]
    plats = [row["sigma_plateau"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(Cs, q50s, marker='o', linewidth=2, label="Median threshold (Q50)")
    ax.plot(Cs, plats, marker='s', linestyle='--', linewidth=2, label="Detected plateau σ")
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel("C (regularization = 1/C)")
    ax.set_ylabel("σ (log)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("C vs intrinsic noise threshold & plateau σ")
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend()
    plt.tight_layout(); plt.show()

    # Print table
    print("\n=== Per-C threshold summary ===")
    print("   C       1/C        Q25         Q50         Q75      plateauσ")
    for r in summary_rows:
        print(f"{r['C']:>5}   {r['reg']:<10.6f} {r['q25']:<11.4g} {r['q50']:<11.4g} "
              f"{r['q75']:<11.4g} {r['sigma_plateau']:<.4g}")

if __name__ == "__main__":
    main()


"""
import numpy as np
import random
import scipy.sparse as sp
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================
# Data & base model utils
# =========================
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data, data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target
    )

def build_tfidf(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=5,
        sublinear_tf=True,
        norm="l2"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

def train_ls_svm(X_train, y_train, C: float = 10.0):
    """
    #Multi-class least-squares SVM via ridge regression (one-shot closed form).
"""
    classes = np.unique(y_train)
    n_classes = len(classes)
    Y = np.zeros((X_train.shape[0], n_classes))
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1.0
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver="auto")
    ridge.fit(X_train, Y)
    return ridge.coef_  # shape (K, d)

# ================
# Unlearning step
# ================
def influence_removal_ls_svm(X_train, y_train, theta_orig,
                             removal_indices: np.ndarray, C: float = 10.0):
    """
    #Influence-style Hessian solve for LS-SVM (same as before).
    #Solves (I + C X^T X) v = grad_removal, then theta_unlearn = theta_orig - v
"""
    n_classes, n_features = theta_orig.shape

    # aggregate gradient contribution of removed samples
    grad_sum = np.zeros_like(theta_orig)
    for idx in removal_indices:
        x = X_train[idx].toarray().ravel()
        one_hot = np.zeros(n_classes); one_hot[y_train[idx]] = 1
        scores = theta_orig @ x
        error = scores - one_hot  # LS residual
        grad_sum += C * np.outer(error, x)

    def hess_matvec(v):
        Xv = X_train.dot(v)
        return v + C * (X_train.T.dot(Xv))

    H_op = LinearOperator((n_features, n_features), matvec=hess_matvec)

    theta_unlearn = np.zeros_like(theta_orig)
    for c in range(n_classes):
        v_c, _ = cg(H_op, grad_sum[c])
        theta_unlearn[c] = theta_orig[c] - v_c
    return theta_unlearn

# ================
# Attack features
# ================
def build_attack_features(P: np.ndarray, labels: np.ndarray):
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p  = P[np.arange(len(labels)), labels].reshape(-1,1)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap  = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce_loss, gap])

def train_shadow_attack(theta_target, vectorizer,
                        train_docs, train_labels, C: float = 1.0):
    """
    #Simple shadow attack (no unlearning inside): fixed attack we will use to *evaluate*
    #retained-class AUC while we only add noise to weights.
"""
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels,
        test_size=0.5, random_state=0, stratify=train_labels
    )
    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)
    theta_sh = train_ls_svm(Xs, s_lbls, C)
    Ps, Ph = softmax(Xs @ theta_sh.T, axis=1), softmax(Xh @ theta_sh.T, axis=1)
    A_s, A_h = build_attack_features(Ps, s_lbls), build_attack_features(Ph, h_lbls)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))])
    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf

# ==========================================
# AUC under *weight noise* (no fine-tuning)
# ==========================================
def eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                         train_docs, train_labels,
                                         test_docs,  test_labels,
                                         attack_clf,
                                         sigma, retained_mask_test,
                                         trials=5, seed=0):
    """
    #Add N(0, sigma^2) noise to weights, compute retained-class MIA AUC, averaged over trials.
    #The attack model is fixed (no retraining).
"""
    rng = np.random.RandomState(seed)
    aucs = []
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)

    for t in range(trials):
        noise = rng.normal(loc=0.0, scale=sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise

        Ptr = softmax(Xtr @ theta_noisy.T, axis=1)
        Pte = softmax(Xte @ theta_noisy.T, axis=1)
        A_tr = build_attack_features(Ptr, train_labels)
        A_te = build_attack_features(Pte, test_labels)

        # members = all train; non-members = retained test only
        keep_tr = np.ones(len(train_labels), dtype=bool)
        keep_te = retained_mask_test
        X_att = np.vstack([A_tr[keep_tr], A_te[keep_te]])
        y_att = np.concatenate([np.ones(keep_tr.sum()), np.zeros(keep_te.sum())])

        aucs.append(roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1]))

    return float(np.mean(aucs))

def _stable_uint32_from_key(key_tuple):
    """
    #Turn any small tuple into a reproducible uint32 for RNG seeding.
"""
    return hash(key_tuple) & 0xFFFFFFFF

def eval_utility_under_weight_noise(theta_base, vectorizer,
                                    test_docs, test_labels, class_to_unlearn,
                                    sigma, trials=5, seed_base=0, key=None):
    """
    #Accuracy on test *excluding* the unlearned class, averaged over trials.

    #DETERMINISTIC NOISE FIX:
    #Noise RNG is keyed by (key, trial_idx). If you pass the same key—for example
    #key=("C_sigma", C, round(sigma, 12))—then any time you reuse that (C,σ)
    #you will get identical utility numbers across different runs (e.g., τ=0.6 vs τ=0.7).
"""
    Xte  = vectorizer.transform(test_docs)
    keep = (test_labels != class_to_unlearn)
    K, d = theta_base.shape

    if key is None:
        key = ("default", float(sigma))

    accs = []
    for t in range(trials):
        seed_t = (seed_base ^ _stable_uint32_from_key((key, t))) % (2**32)
        rng = np.random.RandomState(seed_t)
        noise = rng.normal(loc=0.0, scale=sigma, size=(K, d))
        theta_noisy = theta_base + noise
        preds = (Xte @ theta_noisy.T).argmax(axis=1)
        accs.append(accuracy_score(test_labels[keep], preds[keep]))

    return float(np.mean(accs))

# ==========================================
# Unbounded noise calibration (grow + bisect)
# ==========================================
def calibrate_noise_for_tau_noft_unbounded(theta_base, vectorizer,
                                           train_docs, train_labels,
                                           test_docs,  test_labels,
                                           attack_clf,
                                           retained_mask_test,
                                           tau, auc_tol=1e-3,
                                           max_grow_steps=60,   # can reach insanely large sigma
                                           bisection_steps=40,
                                           trials=5, seed=0):
    """
    #Find sigma >= 0 such that retained-class AUC ≈ tau (within auc_tol), with no fine-tuning.
    #Strategy: compute AUC(0); if already <= tau, return sigma=0. Else grow sigma by 2x
    #until AUC <= tau or we hit growth cap; then bisection on [low, high].
    #Returns dict: {"sigma": σ*, "auc": auc(σ*), "hit_boundary": bool}
"""
    # AUC at zero noise
    auc0 = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                train_docs, train_labels,
                                                test_docs,  test_labels,
                                                attack_clf, sigma=0.0,
                                                retained_mask_test=retained_mask_test,
                                                trials=trials, seed=seed)
    if auc0 <= tau + auc_tol:
        return {"sigma": 0.0, "auc": auc0, "hit_boundary": False}

    # exponential growth
    sigma_low, auc_low = 0.0, auc0
    sigma_high = 1.0
    for _ in range(max_grow_steps):
        auc_high = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                        train_docs, train_labels,
                                                        test_docs,  test_labels,
                                                        attack_clf, sigma=sigma_high,
                                                        retained_mask_test=retained_mask_test,
                                                        trials=trials, seed=seed)
        if auc_high <= tau:
            break
        sigma_low, auc_low = sigma_high, auc_high
        sigma_high *= 2.0
    else:
        # still above tau even at huge sigma → return boundary as best effort
        return {"sigma": sigma_high, "auc": auc_high, "hit_boundary": True}

    # bisection search
    left, right = sigma_low, sigma_high
    best_sigma, best_auc = sigma_high, auc_high
    for _ in range(bisection_steps):
        mid = 0.5 * (left + right)
        auc_mid = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                       train_docs, train_labels,
                                                       test_docs,  test_labels,
                                                       attack_clf, sigma=mid,
                                                       retained_mask_test=retained_mask_test,
                                                       trials=trials, seed=seed)
        if auc_mid > tau:
            left = mid
        else:
            right = mid
            best_sigma, best_auc = mid, auc_mid
        if abs(auc_mid - tau) <= auc_tol:
            best_sigma, best_auc = mid, auc_mid
            break

    return {"sigma": best_sigma, "auc": best_auc, "hit_boundary": False}

# =========================
# Pretty plotting (per-τ)
# =========================
def plot_utility_and_noise_vs_C(C_list, utilities, sigmas, tau):
    x = list(range(len(C_list)))
    fig, ax1 = plt.subplots(figsize=(9, 5.2))

    color_u = "#1f77b4"
    ln1 = ax1.plot(
        x, utilities, marker='o', linewidth=2, color=color_u,
        label="Utility (accuracy w/o unlearned)"
    )
    ax1.set_xlabel("C (regularization = 1/C)")
    ax1.set_ylabel("Utility (Accuracy w/o unlearned)", color=color_u)
    ax1.tick_params(axis='y', labelcolor=color_u)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in C_list], rotation=0)

    ax2 = ax1.twinx()
    color_s = "#d62728"
    req_sigmas = [max(1e-12, s) for s in sigmas]
    ln2 = ax2.plot(
        x, req_sigmas, marker='s', linestyle='--', linewidth=2, color=color_s,
        label=r"Required noise $\sigma^{\ast}$"
    )
    ax2.set_ylabel(r"Required noise $\sigma^{\ast}$ (log scale)", color=color_s)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_yscale('log')

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)

    ax1.set_title(f"Fix retained-class AUC τ={tau:.2f}: Utility vs C and required noise")
    plt.tight_layout()
    plt.show()

# =========================
# Pretty plotting (per-C)
# =========================
def plot_utility_and_noise_vs_tau(C_value, taus, utilities, sigmas):
    """
    #NEW: For a fixed C, show Utility (left y) and required noise σ* (right y, log) across τ.
"""
    x = list(range(len(taus)))
    fig, ax1 = plt.subplots(figsize=(9, 5.2))

    color_u = "#1f77b4"
    ln1 = ax1.plot(
        x, utilities, marker='o', linewidth=2, color=color_u,
        label="Utility (accuracy w/o unlearned)"
    )
    ax1.set_xlabel("Target retained-class AUC (τ)")
    ax1.set_ylabel("Utility (Accuracy w/o unlearned)", color=color_u)
    ax1.tick_params(axis='y', labelcolor=color_u)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t:.2f}" for t in taus], rotation=0)

    ax2 = ax1.twinx()
    color_s = "#d62728"
    req_sigmas = [max(1e-12, s) for s in sigmas]
    ln2 = ax2.plot(
        x, req_sigmas, marker='s', linestyle='--', linewidth=2, color=color_s,
        label=r"Required noise $\sigma^{\ast}$"
    )
    ax2.set_ylabel(r"Required noise $\sigma^{\ast}$ (log scale)", color=color_s)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_yscale('log')

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)

    ax1.set_title(f"Fixed C = {C_value}: Utility and required noise vs τ")
    plt.tight_layout()
    plt.show()

# ===========
# Main sweep
# ===========
def main():
    # --- data & tf-idf ---
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # pick one class to unlearn for comparability across C
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    # C grid and target AUCs
    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]

    # Your original taus:
    # taus = [0.51, 0.60, 0.70, 0.80, 0.90]

    # Optional: denser tau grid for smoother per-C curves (uncomment to use)
    taus = [0.51, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70,
            0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]

    # retained mask on test
    retained_mask_test = (y_test != class_to_unlearn)

    # storage A (existing): results_by_tau[tau] = rows across C
    results_by_tau = {tau: [] for tau in taus}
    # storage B (NEW): results_by_C[C] = rows across tau
    results_by_C = {C: [] for C in C_list}

    for C in C_list:
        print("\n" + "="*72)
        print(f"[C={C}] training baseline & performing unlearning (Hessian downdate only)")

        theta_orig = train_ls_svm(X_train, y_train, C)
        removal_indices = np.where(y_train == class_to_unlearn)[0]
        theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, removal_indices, C)

        # base release: zero-out removed class
        theta_base = theta_un.copy()
        theta_base[class_to_unlearn, :] = 0.0

        # fixed attack (no retraining during noise calibration)
        attack_clf = train_shadow_attack(theta_orig, vectorizer, train_docs, y_train, C)

        # sweep taus
        for tau in taus:
            cal = calibrate_noise_for_tau_noft_unbounded(
                theta_base=theta_base,
                vectorizer=vectorizer,
                train_docs=train_docs, train_labels=y_train,
                test_docs=test_docs,  test_labels=y_test,
                attack_clf=attack_clf,
                retained_mask_test=retained_mask_test,
                tau=tau, auc_tol=1e-3,
                max_grow_steps=60, bisection_steps=40,
                trials=5, seed=0
            )

            # utility at calibrated sigma (DETERMINISTIC; keyed by (C, σ))
            util = eval_utility_under_weight_noise(
                theta_base, vectorizer, test_docs, y_test, class_to_unlearn,
                sigma=cal["sigma"], trials=5, seed_base=1,
                key=("C_sigma", float(C), float(np.round(cal["sigma"], 12)))
            )

            reg_strength = 1.0 / C
            print(f"[τ={tau:.2f}, C={C:>5}] 1/C={reg_strength:.5f}  σ*={cal['sigma']:.12g}  "
                  f"AUC={cal['auc']:.3f}  Acc_wo={util:.4f}  boundary={cal['hit_boundary']}")

            row = {
                "C": C,
                "tau": tau,
                "reg": reg_strength,
                "sigma": cal["sigma"],
                "auc": cal["auc"],
                "acc_wo": util,
                "boundary": cal["hit_boundary"],
            }
            results_by_tau[tau].append(row)
            results_by_C[C].append(row)

    # ================
    # Visualization 1 (existing): per-τ, x-axis = C
    # ================
    for tau in taus:
        rows = sorted(results_by_tau[tau], key=lambda r: C_list.index(r["C"]))  # keep original C order
        Cs     = [r["C"] for r in rows]
        accs   = [r["acc_wo"] for r in rows]
        sigmas = [max(r["sigma"], 1e-12) for r in rows]
        plot_utility_and_noise_vs_C(Cs, accs, sigmas, tau=tau)

    # ================
    # Visualization 2 (NEW): per-C, x-axis = τ
    # ================
    for C in C_list:
        rows = sorted(results_by_C[C], key=lambda r: taus.index(r["tau"]))
        tau_axis = [r["tau"] for r in rows]
        accs     = [r["acc_wo"] for r in rows]
        sigmas   = [max(r["sigma"], 1e-12) for r in rows]
        plot_utility_and_noise_vs_tau(C, tau_axis, accs, sigmas)

    # Optional: print tables per C
    for C in C_list:
        rows = sorted(results_by_C[C], key=lambda r: taus.index(r["tau"]))
        print(f"\n=== Results for fixed C={C} across τ (retained-class AUC target) ===")
        print("   τ        σ* (noise)         AUC_hit   Acc_wo   boundary")
        for r in rows:
            print(f"  {r['tau']:.2f}    {r['sigma']:<16.12g}   {r['auc']:.3f}    {r['acc_wo']:.4f}   {str(r['boundary']):>8}")

if __name__ == "__main__":
    main()
"""


