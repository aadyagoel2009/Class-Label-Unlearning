import numpy as np
import random
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
    classes = np.unique(y_train)
    Y = np.zeros((X_train.shape[0], len(classes)), dtype=float)
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1.0
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver="auto")
    ridge.fit(X_train, Y)
    return ridge.coef_, classes  # theta, classes

# ==========================================
# Random relabel baseline
# ==========================================
def random_relabel_train(X_train, y_train, C, class_to_unlearn, seed=0):
    rng = np.random.RandomState(seed)
    y_rlb = y_train.copy()
    mask_u = (y_train == class_to_unlearn)
    remain_classes = np.array(sorted(list(set(y_train) - {class_to_unlearn})))
    y_rlb[mask_u] = rng.choice(remain_classes, size=mask_u.sum(), replace=True)
    theta_base, classes_base = train_ls_svm(X_train, y_rlb, C)
    return theta_base, classes_base, y_rlb

# ================
# Attack features
# ================
def build_attack_features(P: np.ndarray, local_labels: np.ndarray):
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p  = P[np.arange(len(local_labels)), local_labels].reshape(-1, 1)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap  = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    return np.hstack([P, entropy, ce_loss, gap])

def to_local(labels, classes_):
    mp = {int(c): i for i, c in enumerate(classes_)}
    return np.array([mp[int(y)] for y in labels], dtype=int)

def remap_probs_to_target(P_src, classes_src, classes_tgt):
    out = np.zeros((P_src.shape[0], len(classes_tgt)), dtype=P_src.dtype)
    pos = {int(c): j for j, c in enumerate(classes_src)}
    for k, c in enumerate(classes_tgt):
        j = pos.get(int(c), None)
        if j is not None:
            out[:, k] = P_src[:, j]
    return out

def train_shadow_attack_aligned(vectorizer, train_docs, train_labels,
                                classes_tgt, C_for_shadow: float = 1.0, seed: int = 0):
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels, test_size=0.5, random_state=0, stratify=train_labels
    )
    keep_s = np.isin(s_lbls, classes_tgt)
    keep_h = np.isin(h_lbls, classes_tgt)
    s_docs = [d for d, k in zip(s_docs, keep_s) if k]
    h_docs = [d for d, k in zip(h_docs, keep_h) if k]
    s_lbls = s_lbls[keep_s]
    h_lbls = h_lbls[keep_h]

    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)
    theta_sh, classes_sh = train_ls_svm(Xs, s_lbls, C_for_shadow)

    Ps_sh = softmax(Xs @ theta_sh.T, axis=1)
    Ph_sh = softmax(Xh @ theta_sh.T, axis=1)

    Ps = remap_probs_to_target(Ps_sh, classes_sh, classes_tgt)
    Ph = remap_probs_to_target(Ph_sh, classes_sh, classes_tgt)

    s_local = to_local(s_lbls, classes_tgt)
    h_local = to_local(h_lbls, classes_tgt)

    A_s = build_attack_features(Ps, s_local)
    A_h = build_attack_features(Ph, h_local)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_local)), np.zeros(len(h_local))])

    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf

# ==================================================
# AUC under weight noise + calibration for target τ
# ==================================================
def eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                         vectorizer,
                                         train_docs, train_labels,
                                         test_docs,  test_labels,
                                         attack_clf,
                                         sigma, retained_mask_test,
                                         trials=5, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)

    keep_tr = np.isin(train_labels, classes_base)
    keep_te = retained_mask_test & np.isin(test_labels, classes_base)

    ytr_local = to_local(train_labels[keep_tr], classes_base)
    yte_local = to_local(test_labels[keep_te], classes_base)

    for _ in range(trials):
        noise = rng.normal(loc=0.0, scale=sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise

        Ptr = softmax((Xtr @ theta_noisy.T)[keep_tr], axis=1)
        Pte = softmax((Xte @ theta_noisy.T)[keep_te], axis=1)

        A_tr = build_attack_features(Ptr, ytr_local)
        A_te = build_attack_features(Pte, yte_local)

        X_att = np.vstack([A_tr, A_te])
        y_att = np.concatenate([np.ones(A_tr.shape[0]), np.zeros(A_te.shape[0])])

        aucs.append(roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:, 1]))

    return float(np.mean(aucs))

def calibrate_noise_for_tau_noft_unbounded(theta_base, classes_base,
                                           vectorizer,
                                           train_docs, train_labels,
                                           test_docs,  test_labels,
                                           attack_clf,
                                           retained_mask_test,
                                           tau, auc_tol=1e-3,
                                           max_grow_steps=60,
                                           bisection_steps=40,
                                           trials=5, seed=0):
    auc0 = eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                                vectorizer,
                                                train_docs, train_labels,
                                                test_docs,  test_labels,
                                                attack_clf, sigma=0.0,
                                                retained_mask_test=retained_mask_test,
                                                trials=trials, seed=seed)
    if auc0 <= tau + auc_tol:
        return {"sigma": 0.0, "auc": auc0, "hit_boundary": False}

    sigma_low, auc_low = 0.0, auc0
    sigma_high = 1.0
    for _ in range(max_grow_steps):
        auc_high = eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                                        vectorizer,
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
        return {"sigma": sigma_high, "auc": auc_high, "hit_boundary": True}

    left, right = sigma_low, sigma_high
    best_sigma, best_auc = sigma_high, auc_high
    for _ in range(bisection_steps):
        mid = 0.5 * (left + right)
        auc_mid = eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                                       vectorizer,
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

# ========================================================
# Gradient stability under random relabeling
# ========================================================
def row_l2_norms(X):
    if sp.issparse(X):
        return np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    else:
        return np.linalg.norm(X, axis=1)

def per_sample_residuals(theta, X, y_local):
    logits = X @ theta.T
    r = logits.copy()
    r[np.arange(X.shape[0]), y_local] -= 1.0
    xnorm = row_l2_norms(X)
    return r, xnorm

def cosine(a, b, eps=1e-12):
    num = np.sum(a*b, axis=1)
    den = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1) + eps
    return num/den

def gradient_stability_random_relabel(X_train, y_train, C, class_to_unlearn,
                                      seeds=(0,1,2,3,4), max_points=None):
    N = X_train.shape[0]
    if (max_points is not None) and (max_points < N):
        rng = np.random.RandomState(123)
        idx = np.sort(rng.choice(N, size=max_points, replace=False))
        Xs, ys = X_train[idx], y_train[idx]
    else:
        idx = np.arange(N)
        Xs, ys = X_train, y_train

    Rs = []
    xnorm_common = None
    classes_list = []
    for sd in seeds:
        theta_s, classes_s, y_rlb = random_relabel_train(X_train, y_train, C, class_to_unlearn, seed=sd)
        yloc = to_local(y_rlb[idx], classes_s)
        r_s, xnorm = per_sample_residuals(theta_s, Xs, yloc)
        Rs.append((r_s, classes_s))
        classes_list.append(classes_s)
        if xnorm_common is None:
            xnorm_common = xnorm

    all_classes = np.array(sorted(list(set(np.concatenate(classes_list)))))
    Rstack = []
    for r_s, classes_s in Rs:
        out = np.zeros((r_s.shape[0], len(all_classes)), dtype=r_s.dtype)
        pos = {int(c): j for j, c in enumerate(classes_s)}
        for k, c in enumerate(all_classes):
            j = pos.get(int(c), None)
            if j is not None:
                out[:, k] = r_s[:, j]
        Rstack.append(out)

    S = len(seeds)
    Rstack = np.stack(Rstack, axis=0)
    r_mean = np.mean(Rstack, axis=0)
    cos_all = []
    gradnorm_all = []
    for s in range(S):
        cos_s = cosine(Rstack[s], r_mean)
        cos_all.append(cos_s)
        gn_s = C * np.linalg.norm(Rstack[s], axis=1) * xnorm_common
        gradnorm_all.append(gn_s)
    cos_mat = np.stack(cos_all, axis=1)
    gn_mat  = np.stack(gradnorm_all, axis=1)

    cos_mean = cos_mat.mean(axis=1)
    gn_var   = gn_mat.var(axis=1)

    return {
        "cosine_mean_global": float(np.mean(cos_mean)),
        "gradnorm_var_global": float(np.mean(gn_var)),
    }

# =========================
# Plot with σ* min at 1e-12
# =========================
def plot_sigma_vs_instability(C_vals, sigma_vals, cos_global, gnvar_global, tau,
                              annotate=True):
    fig, ax1 = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(C_vals))

    # Plot σ* on log scale; always show down to 1e-12
    sig_raw = np.asarray(sigma_vals, dtype=float)
    sig_plot = np.clip(sig_raw, 1e-12, None)

    pos = sig_plot[sig_plot > 0]
    smax = pos.max() if pos.size else 1e-12
    lo = 1e-12
    hi = 10**(np.ceil(np.log10(max(smax*1.6, 1.2e-12))))

    ax1.plot(x, sig_plot, marker='o', lw=2, label="Required noise σ*")
    ax1.set_xticks(x); ax1.set_xticklabels([str(c) for c in C_vals])
    ax1.set_xlabel("C")
    ax1.set_ylabel("σ* (log)")
    ax1.set_yscale('log')
    ax1.set_ylim(lo, hi)
    ax1.grid(ls=":", alpha=0.35)

    if annotate:
        for xi, yi, raw in zip(x, sig_plot, sig_raw):
            txt = f"{raw:.3g}" if raw > 0 else "0"
            ax1.text(xi, yi, txt, fontsize=9, ha="center", va="bottom")

    ax2 = ax1.twinx()
    ax2.plot(x, [1-c for c in cos_global], marker='s', ls='--', lw=2,
             label="Instability (1 − mean cosine)", color="#d62728")
    ax2.plot(x, gnvar_global, marker='^', ls='--', lw=2,
             label="Grad-norm variance", color="#2ca02c")
    ax2.set_ylabel("Instability metrics")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="best")
    ax1.set_title(f"Gradient instability vs required noise at τ={tau}")
    plt.tight_layout()
    plt.show()

# ===========
# Main sweep
# ===========
def main():
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    taus   = [0.51, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70,
              0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]

    retained_mask_test = (y_test != class_to_unlearn)

    for tau in taus:
        sigma_star = []
        cos_global = []
        gnvar_global = []
        print("\n" + "="*80)
        print(f"[τ={tau}] gradient stability and σ* across C")
        for C in C_list:
            # stability across random relabelings
            stab = gradient_stability_random_relabel(
                X_train, y_train, C, class_to_unlearn,
                seeds=(0,1,2,3,4), max_points=8000
            )
            cos_global.append(stab["cosine_mean_global"])
            gnvar_global.append(stab["gradnorm_var_global"])

            # baseline model for σ* calibration
            theta_base, classes_base, y_rlb = random_relabel_train(X_train, y_train, C, class_to_unlearn, seed=0)
            attack_clf = train_shadow_attack_aligned(
                vectorizer=vectorizer, train_docs=train_docs, train_labels=y_rlb,
                classes_tgt=classes_base, C_for_shadow=C, seed=0
            )

            cal = calibrate_noise_for_tau_noft_unbounded(
                theta_base=theta_base, classes_base=classes_base,
                vectorizer=vectorizer,
                train_docs=train_docs, train_labels=y_train,
                test_docs=test_docs,  test_labels=y_test,
                attack_clf=attack_clf,
                retained_mask_test=retained_mask_test,
                tau=tau, auc_tol=1e-3,
                max_grow_steps=60, bisection_steps=40,
                trials=5, seed=0
            )
            sigma_star.append(cal["sigma"])
            print(f"  C={C:>6} | sigma*={cal['sigma']:.12g} | AUC≈{cal['auc']:.3f} | "
                  f"1-mean_cos={1-stab['cosine_mean_global']:.4f} | "
                  f"gradnorm_var={stab['gradnorm_var_global']:.4e}")

        # correlations (across C) for info
        sig = np.array(sigma_star)
        instab1 = 1.0 - np.array(cos_global)
        instab2 = np.array(gnvar_global)
        if len(np.unique(sig)) > 1:
            r1, p1 = pearsonr(sig, instab1)
            r2, p2 = pearsonr(sig, instab2)
            print(f"  corr(sigma*, 1-mean_cos) = {r1:+.3f} (p={p1:.3g})")
            print(f"  corr(sigma*, gradnorm_var) = {r2:+.3f} (p={p2:.3g})")
        else:
            print("  sigma* constant across C at this τ (correlation not meaningful)")

        # figure for this τ (σ* axis min forced to 1e-12)
        plot_sigma_vs_instability(C_list, sigma_star, cos_global, gnvar_global, tau, annotate=True)

if __name__ == "__main__":
    main()
"""
import numpy as np
import random
import scipy.sparse as sp
from scipy.special import softmax
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
    #Multi-class least-squares SVM via ridge regression.
    #Returns (theta, classes), where theta rows are aligned to `classes`.
"""
    classes = np.unique(y_train)
    n_classes = len(classes)
    Y = np.zeros((X_train.shape[0], n_classes), dtype=float)
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1.0
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver="auto")
    ridge.fit(X_train, Y)
    theta = ridge.coef_  # shape (K, d) ordered as `classes`
    return theta, classes

# ==========================================
# Random relabel "unlearning" baseline
# ==========================================
def random_relabel_train(X_train, y_train, C, class_to_unlearn, seed=0):
    """
    #Take all samples from `class_to_unlearn` and randomly scatter them
    #into the remaining classes. Retrain LS-SVM on these *relabelled* targets.
    #Returns:
    #  theta_base, classes_base, y_rlb (the relabelled training labels)
"""
    rng = np.random.RandomState(seed)
    y_rlb = y_train.copy()
    mask_u = (y_train == class_to_unlearn)
    remain_classes = np.array(sorted(list(set(y_train) - {class_to_unlearn})))
    # randomly assign each removed sample to a remaining class
    y_rlb[mask_u] = rng.choice(remain_classes, size=mask_u.sum(), replace=True)
    theta_base, classes_base = train_ls_svm(X_train, y_rlb, C)
    return theta_base, classes_base, y_rlb

# ================
# Attack features
# ================
def build_attack_features(P: np.ndarray, local_labels: np.ndarray):
    """
    #Attack features built against *local class indices* in [0..K-1].
"""
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p  = P[np.arange(len(local_labels)), local_labels].reshape(-1, 1)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap  = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    return np.hstack([P, entropy, ce_loss, gap])

def to_local(labels, classes_):
    """
    #Map global labels to local indices [0..K-1] consistent with 'classes_'.
"""
    mp = {int(c): i for i, c in enumerate(classes_)}
    return np.array([mp[int(y)] for y in labels], dtype=int)

def remap_probs_to_target(P_src, classes_src, classes_tgt):
    """
    #Reindex columns of P_src (aligned to classes_src) into the order of classes_tgt.
    #Missing classes are zero-filled.
"""
    out = np.zeros((P_src.shape[0], len(classes_tgt)), dtype=P_src.dtype)
    pos = {int(c): j for j, c in enumerate(classes_src)}
    for k, c in enumerate(classes_tgt):
        j = pos.get(int(c), None)
        if j is not None:
            out[:, k] = P_src[:, j]
    return out

def train_shadow_attack_aligned(vectorizer,
                                train_docs, train_labels,
                                classes_tgt,               # align attacker to these classes
                                C_for_shadow: float = 1.0,
                                seed: int = 0):
    """
    #Shadow attacker whose attack features are constructed in the same class basis
    #as 'classes_tgt' (active classes of the released model).
"""
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels, test_size=0.5, random_state=0, stratify=train_labels
    )
    # keep only labels present in classes_tgt
    keep_s = np.isin(s_lbls, classes_tgt)
    keep_h = np.isin(h_lbls, classes_tgt)
    s_docs = [d for d, k in zip(s_docs, keep_s) if k]
    h_docs = [d for d, k in zip(h_docs, keep_h) if k]
    s_lbls = s_lbls[keep_s]
    h_lbls = h_lbls[keep_h]

    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)
    theta_sh, classes_sh = train_ls_svm(Xs, s_lbls, C_for_shadow)

    # shadow probabilities
    Ps_sh = softmax(Xs @ theta_sh.T, axis=1)
    Ph_sh = softmax(Xh @ theta_sh.T, axis=1)

    # remap into target class basis
    Ps = remap_probs_to_target(Ps_sh, classes_sh, classes_tgt)
    Ph = remap_probs_to_target(Ph_sh, classes_sh, classes_tgt)

    # labels → local indices of target basis
    s_local = to_local(s_lbls, classes_tgt)
    h_local = to_local(h_lbls, classes_tgt)

    A_s = build_attack_features(Ps, s_local)
    A_h = build_attack_features(Ph, h_local)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_local)), np.zeros(len(h_local))])

    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf

# ==========================================
# AUC / Utility under *weight noise* (no FT)
# ==========================================
def eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                         vectorizer,
                                         train_docs, train_labels,
                                         test_docs,  test_labels,
                                         attack_clf,
                                         sigma, retained_mask_test,
                                         trials=5, seed=0):
    """
   # Add N(0, sigma^2) to weights (in `classes_base` ordering) and compute
   # retained-class MIA AUC (members=train, non-members=test excluding unlearned class),
   # averaged over trials. The attacker is fixed and aligned to `classes_base`.
"""
    rng = np.random.RandomState(seed)
    aucs = []
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)

    # filter labels to those in classes_base & map to local for feat building
    keep_tr = np.isin(train_labels, classes_base)
    keep_te = retained_mask_test & np.isin(test_labels, classes_base)

    ytr_local = to_local(train_labels[keep_tr], classes_base)
    yte_local = to_local(test_labels[keep_te], classes_base)

    for _ in range(trials):
        noise = rng.normal(loc=0.0, scale=sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise

        Ptr = softmax((Xtr @ theta_noisy.T)[keep_tr], axis=1)
        Pte = softmax((Xte @ theta_noisy.T)[keep_te], axis=1)

        A_tr = build_attack_features(Ptr, ytr_local)
        A_te = build_attack_features(Pte, yte_local)

        X_att = np.vstack([A_tr, A_te])
        y_att = np.concatenate([np.ones(A_tr.shape[0]), np.zeros(A_te.shape[0])])

        aucs.append(roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:, 1]))

    return float(np.mean(aucs))

def eval_utility_under_weight_noise(theta_base, classes_base,
                                    vectorizer,
                                    test_docs, test_labels, class_to_unlearn,
                                    sigma, trials=3, seed=0):
    """
   # Accuracy on test excluding the originally unlearned class.
   # IMPORTANT: map local argmax indices back to global labels via `classes_base`.
"""
    rng = np.random.RandomState(seed)
    accs = []
    Xte = vectorizer.transform(test_docs)

    # keep only test points that are in the model's class basis (i.e., not the unlearned class)
    in_basis = np.isin(test_labels, classes_base)
    keep = in_basis & (test_labels != class_to_unlearn)  # second term is redundant but explicit

    y_true = test_labels[keep]

    for _ in range(trials):
        noise = rng.normal(loc=0.0, scale=sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise

        # scores over the active classes (ordered as classes_base)
        scores = Xte @ theta_noisy.T
        preds_local = np.argmax(scores, axis=1)

        # map local indices -> global labels
        preds_global = np.asarray(classes_base)[preds_local]

        accs.append(accuracy_score(y_true, preds_global[keep]))

    return float(np.mean(accs))

# ==========================================
# Unbounded noise calibration (grow + bisect)
# ==========================================
def calibrate_noise_for_tau_noft_unbounded(theta_base, classes_base,
                                           vectorizer,
                                           train_docs, train_labels,
                                           test_docs,  test_labels,
                                           attack_clf,
                                           retained_mask_test,
                                           tau, auc_tol=1e-3,
                                           max_grow_steps=60,   # can reach very large sigma
                                           bisection_steps=40,
                                           trials=5, seed=0):
    """
   # Find sigma ≥ 0 such that retained-class AUC ≈ tau (within auc_tol), no fine-tuning.
"""
    auc0 = eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                                vectorizer,
                                                train_docs, train_labels,
                                                test_docs,  test_labels,
                                                attack_clf, sigma=0.0,
                                                retained_mask_test=retained_mask_test,
                                                trials=trials, seed=seed)
    if auc0 <= tau + auc_tol:
        return {"sigma": 0.0, "auc": auc0, "hit_boundary": False}

    sigma_low, auc_low = 0.0, auc0
    sigma_high = 1.0
    # grow
    for _ in range(max_grow_steps):
        auc_high = eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                                        vectorizer,
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
        return {"sigma": sigma_high, "auc": auc_high, "hit_boundary": True}

    # bisection
    left, right = sigma_low, sigma_high
    best_sigma, best_auc = sigma_high, auc_high
    for _ in range(bisection_steps):
        mid = 0.5 * (left + right)
        auc_mid = eval_retained_auc_under_weight_noise(theta_base, classes_base,
                                                       vectorizer,
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
# Pretty plotting
# =========================
def plot_utility_and_noise_vs_C(C_list, utilities, sigmas, tau):
    """
  #  Discrete x-axis at the exact C values, with two series:
  #    - Utility (accuracy w/o unlearned class) on left y-axis
  #    - Required noise σ* on right y-axis (log scale)
"""
    x = list(range(len(C_list)))
    fig, ax1 = plt.subplots(figsize=(9, 5.2))

    # Left axis: utility
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

    # Right axis: required noise σ* (log scale)
    ax2 = ax1.twinx()
    color_s = "#d62728"
    req_sigmas = [max(1e-12, s) for s in sigmas]
    ln2 = ax2.plot(
        x, req_sigmas, marker='s', linestyle='--', linewidth=2, color=color_s,
        label="Required noise (σ*)"
    )
    ax2.set_ylabel("Required noise (σ*) [log scale]", color=color_s)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_yscale('log')

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)

    ax1.set_title(f"Fixed retained-class AUC τ={tau:.2f}: Utility vs C and required noise")
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

    # C grid and target retained-class AUCs
    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    taus   = [0.51, 0.60, 0.70, 0.80, 0.90]

    # retained test mask (exclude the *original* unlearned class)
    retained_mask_test = (y_test != class_to_unlearn)

    # storage: results[tau] = list of dicts per C
    results = {tau: [] for tau in taus}

    for C in C_list:
        print("\n" + "="*72)
        print(f"[C={C}] random relabel baseline + attacker alignment")

        # random relabel & train base model
        theta_base, classes_base, y_rlb = random_relabel_train(X_train, y_train, C, class_to_unlearn, seed=0)

        # attacker aligned to base classes & relabeled distribution
        attack_clf = train_shadow_attack_aligned(
            vectorizer=vectorizer,
            train_docs=train_docs,
            train_labels=y_rlb,          # <<< relabeled labels
            classes_tgt=classes_base,    # <<< align to base model classes
            C_for_shadow=C,
            seed=0
        )

        for tau in taus:
            cal = calibrate_noise_for_tau_noft_unbounded(
                theta_base=theta_base, classes_base=classes_base,
                vectorizer=vectorizer,
                train_docs=train_docs, train_labels=y_train,   # full original train (members)
                test_docs=test_docs,  test_labels=y_test,      # test
                attack_clf=attack_clf,
                retained_mask_test=retained_mask_test,
                tau=tau, auc_tol=1e-3,
                max_grow_steps=60, bisection_steps=40,
                trials=5, seed=0
            )

            util = eval_utility_under_weight_noise(
                theta_base, classes_base, vectorizer,
                test_docs, y_test, class_to_unlearn,
                sigma=cal["sigma"], trials=5, seed=1
            )

            reg_strength = 1.0 / C
            print(f"[τ={tau:.2f}, C={C:>5}] 1/C={reg_strength:.5f}  σ*={cal['sigma']:.4g}  "
                  f"AUC={cal['auc']:.3f}  Acc_wo={util:.4f}  boundary={cal['hit_boundary']}")

            results[tau].append({
                "C": C,
                "reg": reg_strength,
                "sigma": cal["sigma"],
                "auc": cal["auc"],
                "acc_wo": util,
                "boundary": cal["hit_boundary"],
            })

    # ================
    # Visualization
    # ================
    # For each tau: x-axis is categorical C values in the original order.
    for tau in taus:
        rows = sorted(results[tau], key=lambda r: C_list.index(r["C"]))
        Cs     = [r["C"] for r in rows]
        accs   = [r["acc_wo"] for r in rows]
        sigmas = [max(r["sigma"], 1e-12) for r in rows]
        plot_utility_and_noise_vs_C(Cs, accs, sigmas, tau=tau)

    # Optional: print tables
    for tau in taus:
        rows = sorted(results[tau], key=lambda r: C_list.index(r["C"]))
        print(f"\n=== Results for τ={tau:.2f} (retained-class AUC target) ===")
        print("  C         1/C (reg)    σ* (noise)      AUC_hit   Acc_wo   boundary")
        for r in rows:
            print(f"  {str(r['C']):>6}   {r['reg']:>10.6f}   {r['sigma']:<12.5g}   {r['auc']:.3f}    {r['acc_wo']:.4f}   {str(r['boundary']):>8}")

if __name__ == "__main__":
    main()\
"""