import numpy as np
import random
import scipy.sparse as sp
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from scipy import stats
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, accuracy_score

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
    return ridge.coef_  # shape (K, d)

# ================
# Unlearning step
# ================
def influence_removal_ls_svm(X_train, y_train, theta_orig,
                             removal_indices: np.ndarray, C: float = 10.0):
    """
    Influence-style Hessian solve for LS-SVM:
      Solve H v = grad_sum, H = I + C X^T X, then theta_unlearn = theta_orig - v
    """
    n_classes, n_features = theta_orig.shape

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

# ==========================
# Attack & calibration utils
# ==========================
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
    Simple shadow attack; we only use it to *evaluate* retained-class AUC,
    while we add Gaussian noise to weights during calibration.
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

def eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                         train_docs, train_labels,
                                         test_docs,  test_labels,
                                         attack_clf,
                                         sigma, retained_mask_test,
                                         trials=5, seed=0):
    """
    Add N(0, sigma^2) to weights, compute retained-class MIA AUC.
    Members = all train; non-members = retained test only (exclude unlearned class).
    """
    rng = np.random.RandomState(seed)
    aucs = []
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)

    keep_tr = np.ones(len(train_labels), dtype=bool)
    keep_te = retained_mask_test

    ytr = train_labels[keep_tr]
    yte = test_labels[keep_te]

    for _ in range(trials):
        noise = rng.normal(loc=0.0, scale=sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise

        Ptr = softmax((Xtr @ theta_noisy.T)[keep_tr], axis=1)
        Pte = softmax((Xte @ theta_noisy.T)[keep_te], axis=1)

        A_tr = build_attack_features(Ptr, ytr)
        A_te = build_attack_features(Pte, yte)

        X_att = np.vstack([A_tr, A_te])
        y_att = np.concatenate([np.ones(A_tr.shape[0]), np.zeros(A_te.shape[0])])

        aucs.append(roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1]))

    return float(np.mean(aucs))

def calibrate_noise_for_tau_noft_unbounded(theta_base, vectorizer,
                                           train_docs, train_labels,
                                           test_docs,  test_labels,
                                           attack_clf,
                                           retained_mask_test,
                                           tau, auc_tol=1e-3,
                                           max_grow_steps=60,
                                           bisection_steps=40,
                                           trials=5, seed=0):
    """
    Find sigma >= 0 such that retained-class AUC ≈ tau (within auc_tol), no FT.
    """
    auc0 = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
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
        return {"sigma": sigma_high, "auc": auc_high, "hit_boundary": True}

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

# =========================================
# (5) Variance & (6) Sameness utilities
# =========================================
def ks_2samp_robust(x, y):
    """Compat across SciPy versions (no 'method' kw)."""
    return stats.ks_2samp(np.asarray(x), np.asarray(y))

def cohens_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2 + 1e-12))
    return (x.mean() - y.mean()) / (s + 1e-12)

def gmm_k_bic(x):
    """BIC-based modality flag (1 vs 2)."""
    X = np.asarray(x).reshape(-1,1)
    best_k, best_bic = None, np.inf
    for k in (1,2):
        g = GaussianMixture(k, covariance_type='full', random_state=0).fit(X)
        b = g.bic(X)
        if b < best_bic:
            best_bic, best_k = b, k
    return best_k

def top1_margins_for_unlearned(theta, X_train, y_train, class_to_unlearn, sigma=0.0, seed=0):
    """
    Top-1 margin = (true-class logit) - (largest incorrect-class logit)
    for TRAIN docs of the class we unlearned. Optionally add N(0,σ^2) to θ.
    """
    rng = np.random.RandomState(seed)
    idx = np.where(y_train == class_to_unlearn)[0]
    Xc  = X_train[idx]
    y   = y_train[idx]
    W   = theta if sigma <= 0 else theta + rng.normal(0, sigma, size=theta.shape)
    logits = Xc @ W.T                  # [Nc, K]
    true = logits[np.arange(logits.shape[0]), y]
    logits[np.arange(logits.shape[0]), y] = -np.inf
    comp = logits.max(axis=1)
    return (true - comp).astype(np.float32)   # positive = confident correct

def summarize_same_vs_diff(x_before, x_after, alpha=0.05):
    """Return a compact ‘sameness’ summary with effect sizes."""
    x_before = np.asarray(x_before); x_after = np.asarray(x_after)
    # distances / tests
    ks_stat, ks_p = ks_2samp_robust(x_before, x_after)
    wdist = stats.wasserstein_distance(x_before, x_after)
    # energy distance (if available)
    try:
        endist = stats.energy_distance(x_before, x_after)
    except Exception:
        endist = np.nan
    # means/variances
    t_stat, t_p = stats.ttest_ind(x_before, x_after, equal_var=False)
    lev_stat, lev_p = stats.levene(x_before, x_after, center='median')
    d = cohens_d(x_before, x_after)
    var_b, var_a = float(np.var(x_before, ddof=1)), float(np.var(x_after, ddof=1))
    # simple “same?” heuristic: all three must hold
    same = (ks_p > alpha) and (abs(d) < 0.2) and (lev_p > alpha)
    return {
        "ks_p": float(ks_p),
        "wasser": float(wdist),
        "energy": float(endist) if endist==endist else None,
        "t_p": float(t_p),
        "levene_p": float(lev_p),
        "cohens_d": float(d),
        "var_before": var_b,
        "var_after": var_a,
        "same_dist": bool(same)
    }

def report_before_after_stats(theta_before, theta_after,
                              X_train, y_train,
                              class_to_unlearn,
                              sigma_star, C, tau, seed=0):
    # margins
    mB  = top1_margins_for_unlearned(theta_before, X_train, y_train, class_to_unlearn, sigma=0.0,   seed=seed)
    mA0 = top1_margins_for_unlearned(theta_after,  X_train, y_train, class_to_unlearn, sigma=0.0,   seed=seed)
    mAs = top1_margins_for_unlearned(theta_after,  X_train, y_train, class_to_unlearn, sigma=sigma_star, seed=seed)

    # modality (1 vs 2)
    kB  = gmm_k_bic(mB)
    kA0 = gmm_k_bic(mA0)
    kAs = gmm_k_bic(mAs)

    # sameness summaries
    rep0 = summarize_same_vs_diff(mB, mA0)   # before vs after (σ=0)
    reps = summarize_same_vs_diff(mB, mAs)   # before vs after (σ=σ*)

    # compact one-liner
    print(f"C={str(C):>6} | τ={tau:.2f} | σ*={sigma_star:.5g} | "
          f"k: {kB}→{kA0}→{kAs} | "
          f"var: {rep0['var_before']:.4g}→{rep0['var_after']:.4g}→{reps['var_after']:.4g} | "
          f"KS_p@σ0={rep0['ks_p']:.2e} KS_p@σ*={reps['ks_p']:.2e} | "
          f"W@σ*={reps['wasser']:.4f} | d@σ*={reps['cohens_d']:.3f} | "
          f"same@σ*={reps['same_dist']}")
    return {"before": mB, "after0": mA0, "after_sigma": mAs,
            "gmm": (kB, kA0, kAs), "rep0": rep0, "reps": reps}

# ===========
# Main sweep
# ===========
def main():
    # --- data & tf-idf ---
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # pick one class to unlearn for comparability across C
    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}\n")

    # grids
    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    taus   = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # retained test set mask (exclude the unlearned class)
    retained_mask_test = (y_test != class_to_unlearn)

    print("=== BEFORE vs AFTER Top-1 Margin on TRAIN members of the unlearned class ===")
    print("Columns: C | tau | sigma* | k_b→k_a0→k_aσ | var b→a0→aσ | KS p (σ=0, σ*) | W@σ* | d@σ* | same@σ*")

    for C in C_list:
        # 1) BEFORE: baseline model
        theta_before = train_ls_svm(X_train, y_train, C)

        # 2) UNLEARN: downdate + FT on retained set; reconstruct final AFTER (σ=0)
        removal_indices = np.where(y_train == class_to_unlearn)[0]
        theta_un = influence_removal_ls_svm(X_train, y_train, theta_before, removal_indices, C)

        keep_mask = np.ones(len(y_train), bool); keep_mask[removal_indices] = False
        X_red, y_red = X_train[keep_mask], y_train[keep_mask]
        classes_red = np.unique(y_red)
        theta_init = theta_un[classes_red, :]

        ft = LogisticRegression(
            penalty='l2', C=C, solver='lbfgs',
            fit_intercept=False, warm_start=True,
            max_iter=5, random_state=42
        )
        ft.coef_ = theta_init.copy()
        ft.classes_ = classes_red
        ft.fit(X_red, y_red)

        # Reconstruct full θ with zero row for removed class (consistent shape)
        n_classes, n_features = theta_before.shape
        theta_after = np.zeros_like(theta_before)
        for i, cls in enumerate(classes_red):
            theta_after[cls, :] = ft.coef_[i]
        # ensure the unlearned class row stays zero
        theta_after[class_to_unlearn, :] = 0.0

        # 3) Build "base" release for attack calibration (same as theta_after)
        theta_base = theta_after.copy()

        # 4) Attack & calibrate σ* to reach tau on retained-class AUC
        attack_clf = train_shadow_attack(theta_before, vectorizer, train_docs, y_train, C)

        for tau in taus:
            cal = calibrate_noise_for_tau_noft_unbounded(
                theta_base=theta_base, vectorizer=vectorizer,
                train_docs=train_docs, train_labels=y_train,
                test_docs=test_docs,  test_labels=y_test,
                attack_clf=attack_clf,
                retained_mask_test=retained_mask_test,
                tau=tau, auc_tol=1e-3,
                max_grow_steps=60, bisection_steps=40,
                trials=5, seed=0
            )
            sigma_star = float(cal["sigma"])

            # 5 & 6) Variance + sameness (before vs after (σ=0) vs after (σ=σ*))
            _ = report_before_after_stats(
                theta_before=theta_before,
                theta_after=theta_after,
                X_train=X_train, y_train=y_train,
                class_to_unlearn=class_to_unlearn,
                sigma_star=sigma_star, C=C, tau=tau, seed=0
            )

if __name__ == "__main__":
    main()