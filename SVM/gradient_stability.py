import numpy as np
import random
import scipy.sparse as sp
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================================
# Data & base model utilities (unchanged from your code)
# =====================================================
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
    Y = np.zeros((X_train.shape[0], len(classes)))
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1.0
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver="auto")
    ridge.fit(X_train, Y)
    return ridge.coef_  # shape (K, d)

def influence_removal_ls_svm(X_train, y_train, theta_orig,
                             removal_indices: np.ndarray, C: float = 10.0):
    """
    Influence-style Hessian solve for LS-SVM:
      H v = grad_sum,   H = I + C X^T X
      theta_unlearn = theta_orig - v
    """
    n_classes, n_features = theta_orig.shape

    grad_sum = np.zeros_like(theta_orig)
    for idx in removal_indices:
        x = X_train[idx].toarray().ravel()
        one_hot = np.zeros(n_classes); one_hot[y_train[idx]] = 1
        scores = theta_orig @ x
        error = scores - one_hot
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

# =========================================
# Instability utilities (direction & magnitude)
# =========================================
def row_l2_norms(X):
    if sp.issparse(X):
        return np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return np.linalg.norm(X, axis=1)

def to_local(labels, classes_):
    mp = {int(c): i for i, c in enumerate(classes_)}
    return np.array([mp[int(y)] for y in labels], dtype=int)

def per_sample_residuals(theta_act, X_eval, y_eval_local):
    """
    Residual in class space for LS-SVM: r_i = logits_i - onehot(y_i)
    where logits_i = theta_act @ x_i
    Returns:
      r: [N_eval, K_act],  xnorm: [N_eval]
    """
    logits = X_eval @ theta_act.T
    r = logits.copy()
    r[np.arange(X_eval.shape[0]), y_eval_local] -= 1.0
    xnorm = row_l2_norms(X_eval)
    return r, xnorm

def cosine_rows(A, B, eps=1e-12):
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + eps
    return num / den

def stratified_subsample_indices(y, frac=0.9, seed=0):
    """
    Indices (in y’s indexing) of a stratified subsample of size ≈ frac*len(y).
    Guarantees at least 1 sample per class.
    """
    rng = np.random.RandomState(seed)
    idx_per_class = defaultdict(list)
    for i, c in enumerate(y):
        idx_per_class[int(c)].append(i)
    chosen = []
    for c, idxs in idx_per_class.items():
        m = max(1, int(np.ceil(frac * len(idxs))))
        chosen.extend(rng.choice(idxs, size=m, replace=False))
    return np.array(sorted(chosen), dtype=int)

# =========================================
# Your deterministic pipeline + FT subsampling hook
# =========================================
def fit_pipeline_downdate_then_ft_active_block(X_train, y_train, C, class_to_unlearn,
                                               ft_indices=None, ft_max_iter=5):
    """
    Run: baseline LS-SVM -> influence removal -> FT on retained set (or its subsample).
    Returns only the ACTIVE block (rows for retained classes), the active class list,
    and a fixed evaluation set (the full retained training set mapped to local indices).
    """
    # Baseline & downdate
    theta_orig = train_ls_svm(X_train, y_train, C)
    rem_idx = np.where(y_train == class_to_unlearn)[0]
    theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, rem_idx, C)

    # Retained set for FT
    keep_mask = np.ones(len(y_train), bool); keep_mask[rem_idx] = False
    X_keep, y_keep = X_train[keep_mask], y_train[keep_mask]
    classes_red = np.unique(y_keep)

    # Init FT from downdated rows (aligned to classes_red)
    theta_init = theta_un[classes_red, :]

    ft = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs',
        fit_intercept=False, warm_start=True,
        max_iter=ft_max_iter, random_state=42
    )
    ft.coef_ = theta_init.copy()
    ft.classes_ = classes_red

    if ft_indices is None:
        X_ft, y_ft = X_keep, y_keep
    else:
        X_ft, y_ft = X_keep[ft_indices], y_keep[ft_indices]

    ft.fit(X_ft, y_ft)

    # Active block only (rows ordered as classes_red)
    theta_act = ft.coef_.copy()

    # Fixed evaluation set = full retained training set, mapped to local indices
    y_eval_local = to_local(y_keep, classes_red)

    return theta_act, classes_red, X_keep, y_eval_local  # (K_act,d), [K_act], [N_keep, d], [N_keep]

def instability_for_C_via_FT_subsampling(X_train, y_train, C, class_to_unlearn,
                                         S=5, frac=0.9, seed_base=123, ft_max_iter=5):
    """
    Build S nearby models by changing ONLY the FT subset (stratified subsamples).
    Compute:
      - directional instability = 1 - mean cosine to across-run mean residual
      - magnitude instability = mean over samples of Var_s[ C * ||r_i^(s)|| * ||x_i|| ]
    """
    # Get retained set labels once to index subsamples consistently
    rem_idx = np.where(y_train == class_to_unlearn)[0]
    keep_mask = np.ones(len(y_train), bool); keep_mask[rem_idx] = False
    y_keep = y_train[keep_mask]

    runs = []
    X_eval = None
    y_eval_local = None
    xnorm_eval = None
    classes_ref = None

    for s in range(S):
        ft_idx = stratified_subsample_indices(y_keep, frac=frac, seed=seed_base + s)
        theta_act, classes_s, X_keep, yloc = fit_pipeline_downdate_then_ft_active_block(
            X_train, y_train, C, class_to_unlearn,
            ft_indices=ft_idx, ft_max_iter=ft_max_iter
        )
        runs.append((theta_act, classes_s))
        if X_eval is None:
            X_eval = X_keep
            y_eval_local = yloc
            xnorm_eval = row_l2_norms(X_eval)
            classes_ref = classes_s

    # Sanity: same active classes across runs
    assert all(np.array_equal(cs, classes_ref) for _, cs in runs), \
        "Active class sets differ across runs; ensure stratified subsamples keep all classes."

    # Residual stacks: [S, N_eval, K_act]
    Rstack = []
    for theta_act, _ in runs:
        r_s, _ = per_sample_residuals(theta_act, X_eval, y_eval_local)
        Rstack.append(r_s)
    Rstack = np.stack(Rstack, axis=0)

    # Directional instability
    r_mean = Rstack.mean(axis=0)                           # [N_eval, K_act]
    cos_all = [cosine_rows(Rstack[s], r_mean) for s in range(Rstack.shape[0])]
    cos_mat = np.stack(cos_all, axis=1)                    # [N_eval, S]
    cosine_mean_global = float(np.mean(cos_mat.mean(axis=1)))
    directional_instability = 1.0 - cosine_mean_global

    # Magnitude instability: g_i^(s) = C * ||r_i^(s)|| * ||x_i||
    G = np.linalg.norm(Rstack, axis=2)                  # [S, N_eval]
    gradnorms = C * G * xnorm_eval[None, :]             # [S, N_eval]
    gn_var_per_sample = gradnorms.var(axis=0) 
    magnitude_instability = float(np.mean(gn_var_per_sample))

    return directional_instability, magnitude_instability

# =========================
# Plotting helper (optional)
# =========================
def plot_instability_vs_C(C_list, dir_vals, mag_vals, title="Instability vs C"):
    x = np.arange(len(C_list))
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(x, dir_vals, marker='o', lw=2, label="Directional (1 − mean cosine)")
    ax1.plot(x, mag_vals, marker='s', lw=2, label="Magnitude (Var grad-norm)")
    ax1.set_xticks(x); ax1.set_xticklabels([str(c) for c in C_list])
    ax1.set_xlabel("C (regularization = 1/C)")
    ax1.set_ylabel("Instability")
    ax1.grid(ls=":", alpha=0.35)
    ax1.legend(loc="best")
    ax1.set_title(title)
    plt.tight_layout(); plt.show()

# ===========
# Main driver
# ===========
def main():
    # --- data ---
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # choose one class to unlearn (fix it for reproducibility)
    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    # C grid you were using
    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]

    # Instability via FT subsampling (S runs per C; ≈90% each, stratified)
    S = 5
    FRAC = 0.40
    dir_vals, mag_vals = [], []

    print(f"[instability] S={S} stratified FT subsamples, frac={FRAC}")
    for C in C_list:
        dI, mI = instability_for_C_via_FT_subsampling(
            X_train, y_train, C, class_to_unlearn,
            S=S, frac=FRAC, seed_base=123, ft_max_iter=5
        )
        dir_vals.append(dI); mag_vals.append(mI)
        print(f"C={C:>6} | directional (1−cos)={dI:.4f} | magnitude var={mI:.4e}")

    # Plot
    plot_instability_vs_C(C_list, dir_vals, mag_vals,
                          title=f"Gradient instability vs C (class_to_unlearn={class_to_unlearn})")

if __name__ == "__main__":
    main()