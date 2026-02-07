import numpy as np
import random
import scipy.sparse as sp
from copy import deepcopy
from scipy.special import softmax, rel_entr
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
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

# =========================
# Logistic Regression target
# =========================
def train_logreg(X_train, y_train, C: float = 10.0, max_iter: int = 2000, seed: int = 42):
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        multi_class="multinomial",
        fit_intercept=False,
        max_iter=max_iter,
        random_state=seed
    )
    lr.fit(X_train, y_train)
    return lr   # .coef_ (K,d), .classes_

def evaluate_accuracy(lr_model, vectorizer, docs, labels):
    X = vectorizer.transform(docs)
    preds = lr_model.predict(X)
    return accuracy_score(labels, preds)

def compute_probs(lr_model, X):
    logits = X @ lr_model.coef_.T  # fit_intercept=False
    return softmax(logits, axis=1)

def probs_from_theta(theta, X):
    logits = X @ theta.T
    return softmax(logits, axis=1)

# ==========================================
# LR Hessian pieces (influence-style)
# ==========================================
def _per_sample_grad_multinomial_lr(p_vec, y_row, x_row):
    g = p_vec.copy()
    g[y_row] -= 1.0
    return g[:, None] * x_row[None, :]

def _summed_grad_removed_class(W, X, y, classes, class_to_unlearn):
    Z = X @ W.T
    P = softmax(Z, axis=1)
    rem_idx = np.where(y == class_to_unlearn)[0]
    G = np.zeros_like(W)
    if len(rem_idx) == 0:
        return G, P
    class_to_row = {int(c): i for i, c in enumerate(classes)}
    y_row = class_to_row[int(class_to_unlearn)]
    for i in rem_idx:
        x = X[i].toarray().ravel() if sp.issparse(X) else np.asarray(X[i]).ravel()
        G += _per_sample_grad_multinomial_lr(P[i], y_row, x)
    return G, P

def _hessian_vec_prod_lr(W, V, X, P, C):
    lam = 1.0 / C
    U = X @ V.T                      # (n,K)
    rowdot = np.sum(P * U, axis=1)   # (n,)
    M = P * U - P * rowdot[:, None]  # (n,K)
    HV = (M.T @ X)
    if sp.issparse(HV):
        HV = HV.A
    HV = HV + lam * V
    return HV

def unlearn_class_via_hessian_lr(lr_model, X_train, y_train, class_to_unlearn,
                                 C: float, cg_tol=1e-3, cg_max_iter=300):
    """
    Influence-style Hessian downdate for removing ALL samples of a class:
      W' = W - H^{-1} * g_removed
    Then zero the removed class row so the released model carries no usable
    parameters for that class.
    """
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()           # (K,d)
    classes = lr_model.classes_
    K, d = W.shape

    G_rem, P = _summed_grad_removed_class(W, X_train, y_train, classes, class_to_unlearn)
    if not np.any(G_rem):
        row = np.where(classes == class_to_unlearn)[0]
        if len(row) > 0:
            W[row[0], :] = 0.0
            lr_new.coef_ = W
        return lr_new

    def matvec(vec):
        V = vec.reshape(K, d)
        HV = _hessian_vec_prod_lr(W, V, X_train, P, C)
        return HV.reshape(-1)

    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)
    b = G_rem.reshape(-1)
    delta, info = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    Delta = delta.reshape(K, d)

    W_un = W - Delta
    row = np.where(classes == class_to_unlearn)[0]
    if len(row) > 0:
        W_un[row[0], :] = 0.0
    lr_new.coef_ = W_un
    return lr_new

# =========================
# Locality experiment utils
# =========================
def cosine_topk_neighbors(X, idx, k=50):
    """
    X: L2-normalized features (csr or dense) shape (n,d).
    Returns (neighbor_indices, similarities) for the top-k most similar
    rows to X[idx], excluding idx itself.
    """
    n = X.shape[0]
    k_eff = min(k, n - 1)

    if sp.issparse(X):
        # cosine(x_i, x_j) = (x_i · x_j) when rows are L2-normalized
        xi = X[idx]                              # (1, d) csr
        sims = (xi @ X.T).toarray().ravel()      # dense (n,)
    else:
        xi = X[idx]
        if xi.ndim == 1:
            xi = xi.reshape(1, -1)               # make it 2D
        sims = 1.0 - cdist(xi, X, metric="cosine").ravel()  # dense (n,)

    sims[idx] = -np.inf                          # exclude self
    topk_idx = np.argpartition(-sims, k_eff)[:k_eff]
    topk_idx = topk_idx[np.argsort(-sims[topk_idx])]        # sort by sim desc
    return topk_idx, sims[topk_idx]

def pick_anchors_threshold(P_test, y_test_labels, removed_label, inv_max=0.5, sens_min=0.5):
    """
    K-agnostic anchor picker using your rule:
      - invariant:  p(removed) < inv_max  AND true_label != removed_label
      - sensitive:  p(removed) >= sens_min AND true_label != removed_label
    Returns (idx_invariant, idx_sensitive, inv_pool, sens_pool) and prints candidate counts.
    Picks the item closest to the 0.5 boundary within each bucket for interpretability.
    """
    # Map removed_label to column index is handled outside; here we need its column index:
    # The caller passes 'removed_label' as the TRAIN LABEL value, not row.
    # We need the column index of removed_label in P_test. Caller must pass removed_row for P selection.
    # To keep the signature clean, we’ll compute p_rem outside and pass it here via closure if needed.
    raise_if = False  # dummy placeholder so we keep signature stable

def pick_anchors_threshold_with_probs(p_removed, y_test_labels, removed_label, inv_max=0.5, sens_min=0.5):
    """
    Same as above, but takes precomputed p_removed (vector).
    """
    true_is_removed = (y_test_labels == removed_label)
    inv_mask  = (p_removed <  inv_max) & (~true_is_removed)
    sens_mask = (p_removed >= sens_min) & (~true_is_removed)

    inv_idx = np.where(inv_mask)[0]
    sens_idx = np.where(sens_mask)[0]
    print(f"Candidates — invariant: {len(inv_idx)}, sensitive: {len(sens_idx)}")

    inv = sens = None
    if len(inv_idx) > 0:
        inv = int(inv_idx[np.argmin(np.abs(p_removed[inv_idx] - 0.5))])  # closest from below
    if len(sens_idx) > 0:
        sens = int(sens_idx[np.argmin(np.abs(p_removed[sens_idx] - 0.5))])  # closest from above
    return inv, sens, inv_idx, sens_idx

def kl_nonremoved(P_before, P_after, removed_idx):
    """
    KL divergence between BEFORE and AFTER after dropping the removed class,
    with per-sample renormalization over remaining classes (works for any K).
    """
    keep = [j for j in range(P_before.shape[1]) if j != removed_idx]
    Qb = P_before[:, keep]
    Qa = P_after[:,  keep]
    Qb = Qb / (Qb.sum(axis=1, keepdims=True) + 1e-12)
    Qa = Qa / (Qa.sum(axis=1, keepdims=True) + 1e-12)
    return np.sum(rel_entr(Qb, Qa), axis=1)

def top1_margin_against_removed(P, removed_idx):
    """
    margin = (max_j≠removed p_j) - p_removed
    """
    keep = [j for j in range(P.shape[1]) if j != removed_idx]
    p_removed = P[:, removed_idx]
    p_best_keep = P[:, keep].max(axis=1)
    return p_best_keep - p_removed

def summarize_neighborhood(tag, P_b, P_a, sims, removed_idx):
    pr_b = P_b[:, removed_idx]
    pr_a = P_a[:, removed_idx]
    dpr  = pr_a - pr_b

    # margin = best non-removed prob - removed prob
    K = P_b.shape[1]
    keep = np.delete(np.arange(K), removed_idx)
    m_b = P_b[:, keep].max(axis=1) - pr_b
    m_a = P_a[:, keep].max(axis=1) - pr_a
    dm  = m_a - m_b

    # KL between BEFORE/AFTER restricted to non-removed classes
    Qb = P_b[:, keep]
    Qa = P_a[:, keep]
    Qb = Qb / (Qb.sum(axis=1, keepdims=True) + 1e-12)
    Qa = Qa / (Qa.sum(axis=1, keepdims=True) + 1e-12)
    kl = np.sum(rel_entr(Qb, Qa), axis=1)

    # argmax over non-removed classes
    pred_b = keep[np.argmax(P_b[:, keep], axis=1)]
    pred_a = keep[np.argmax(P_a[:, keep], axis=1)]
    switch_rate = float(np.mean(pred_b != pred_a))

    r, rp = (np.nan, np.nan)
    if len(sims) > 1:
        r, rp = pearsonr(sims, -dpr)

    def mstd(x): return float(np.mean(x)), float(np.std(x))
    mp, sp = mstd(dpr)
    mm, sm = mstd(dm)
    mk, sk = mstd(kl)

    print(f"\n[{tag}] n={len(dpr)}  (mean cos sim={np.mean(sims):.3f})")
    print(f"  Δp(removed):      mean={mp:+.4f} ± {sp:.4f}   (negative = drop)")
    print(f"  Δmargin vs rem.:  mean={mm:+.4f} ± {sm:.4f}   (positive = better separation)")
    print(f"  KL_nonremoved:    mean={mk:.4f} ± {sk:.4f}")
    print(f"  Non-removed argmax switch rate: {switch_rate:.3f}")
    print(f"  corr(sim, -Δp_removed) = {r:.3f}  (p={rp:.2e})")

# =========================
# Plotting helpers (paper-ready)
# =========================
def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_anchor_snapshots(P_before, P_after, idx_inv, idx_sens, removed_row, classes, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

    def _one(ax, P_b, P_a, title):
        x = np.arange(len(classes))
        ax.bar(x, P_b, alpha=0.75, label="Before")
        ax.bar(x, -P_a, alpha=0.75, label="After (negative bars)")
        ax.set_title(title)
        ax.set_xlabel("Class index")
        ax.set_xticks(x)
        ax.set_ylabel("Probability (After plotted as negative)")
        ax.axvline(removed_row, ls="--", lw=1.2, color="k", alpha=0.6)
        ax.legend(frameon=True)

    _one(axes[0], P_before[idx_inv], P_after[idx_inv], f"Invariant anchor (idx={idx_inv})")
    _one(axes[1], P_before[idx_sens], P_after[idx_sens], f"Sensitive anchor (idx={idx_sens})")
    fig.suptitle("Anchor probability snapshots (removed class column marked)")
    _savefig(outpath)

def plot_hist_dpr(dpr_inv, dpr_sens, removed_label, outpath, bins=40):
    vmin = min(dpr_inv.min(), dpr_sens.min())
    vmax = max(dpr_inv.max(), dpr_sens.max())
    edges = np.linspace(vmin, vmax, bins+1)

    plt.figure(figsize=(6, 4))
    plt.hist(dpr_inv,  bins=edges, alpha=0.6, density=True, label="Invariant nbrs")
    plt.hist(dpr_sens, bins=edges, alpha=0.6, density=True, label="Sensitive nbrs")
    plt.axvline(0.0, color="k", lw=1, ls=":")
    plt.title(f"Δp_removed for neighbors (removed class = {removed_label})")
    plt.xlabel("Δp_removed = p_removed_after − p_removed_before")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    _savefig(outpath)

def plot_effect_size_abs_dpr(dpr_inv, dpr_sens, tstat, pval, outpath):
    data = [np.abs(dpr_inv), np.abs(dpr_sens)]
    labels = ["Invariant", "Sensitive"]

    plt.figure(figsize=(5.5, 3.8))
    parts = plt.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
    means = [np.mean(v) for v in data]
    xs = [1, 2]
    plt.scatter(xs, means, zorder=3)
    plt.xticks(xs, labels)
    plt.ylabel("|Δp_removed|")
    plt.title(f"Effect size: |Δp_removed| (t={tstat:.2f}, p={pval:.2e})")
    _savefig(outpath)

def plot_locality_scatter(sim_inv, dpr_inv, sim_sens, dpr_sens, r_inv, rp_inv, r_sens, rp_sens, outpath):
    plt.figure(figsize=(6.2, 4.4))
    plt.scatter(sim_inv,  -dpr_inv,  s=18, alpha=0.6, label=f"Invariant (r={r_inv:.2f}, p={rp_inv:.1e})")
    plt.scatter(sim_sens, -dpr_sens, s=18, alpha=0.6, label=f"Sensitive (r={r_sens:.2f}, p={rp_sens:.1e})")
    plt.xlabel("Cosine similarity to anchor (TF–IDF)")
    plt.ylabel("−Δp_removed  (larger = bigger drop)")
    plt.title("Locality of unlearning effect")
    plt.legend(frameon=True, loc="best")
    _savefig(outpath)

# =========================
# Full experiment (any K)
# =========================
def run_locality_experiment_anyK(train_docs, test_docs, y_train, y_test,
                                 C=10.0, class_to_unlearn=None, *,
                                 k_neighbors=50,
                                 inv_max=0.5,
                                 sens_min=0.5,
                                 cg_tol=1e-3, cg_max_iter=300, seed=42,
                                 save_figs=True):
    """
    Works for ANY number of classes K >= 2.
    1) Train LR on full K classes; get BEFORE probs on test.
    2) Unlearn target class via Hessian + zeroing its weight row; get AFTER probs.
    3) Pick invariant & sensitive anchors by p(removed) threshold rule; pull cosine-k neighbors.
    4) Summarize neighborhood shifts; t-test on |Δp_removed| between neighborhoods.
    """
    # TF-IDF
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # choose class to unlearn (label space)
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes.")
    if class_to_unlearn is None:
        class_to_unlearn = int(np.random.RandomState(seed).choice(classes))
    if class_to_unlearn not in classes:
        raise ValueError("class_to_unlearn not present in training labels.")

    # Train + BEFORE probs
    lr = train_logreg(X_train, y_train, C=C, seed=seed)
    P_test_before = compute_probs(lr, X_test)

    # Map removed label -> row index in coefficient matrix
    removed_row = int(np.where(lr.classes_ == class_to_unlearn)[0][0])

    # Hessian unlearning + zero removed row + AFTER probs
    lr_un = unlearn_class_via_hessian_lr(lr, X_train, y_train, class_to_unlearn,
                                         C=C, cg_tol=cg_tol, cg_max_iter=cg_max_iter)
    theta = lr_un.coef_.copy()
    theta[removed_row, :] = 0.0
    P_test_after = probs_from_theta(theta, X_test)

    # Cosine neighbors in TF-IDF space
    Xn = normalize(X_test, norm="l2", copy=True)

    # Anchors using your thresholds, with counts (use p_removed from BEFORE)
    p_removed_before = P_test_before[:, removed_row]
    idx_inv, idx_sens, inv_pool, sens_pool = pick_anchors_threshold_with_probs(
        p_removed_before, y_test, class_to_unlearn, inv_max=inv_max, sens_min=sens_min
    )

    print(f"\nK={len(classes)} classes, chosen class_to_unlearn={class_to_unlearn} (row={removed_row})")
    print(f"  Invariant anchor idx = {idx_inv}")
    print(f"  Sensitive  anchor idx = {idx_sens}")

    if idx_inv is None or idx_sens is None:
        print("Could not find both anchors under the specified thresholds; try adjusting inv_max/sens_min.")
        return

    # Neighbor sets
    nbr_inv, sims_inv   = cosine_topk_neighbors(Xn, idx_inv,  k=k_neighbors)
    nbr_sens, sims_sens = cosine_topk_neighbors(Xn, idx_sens, k=k_neighbors)

    # BEFORE/AFTER subsets for neighborhoods
    Pb_inv  = P_test_before[nbr_inv]
    Pa_inv  = P_test_after[nbr_inv]
    Pb_sens = P_test_before[nbr_sens]
    Pa_sens = P_test_after[nbr_sens]

    # Summaries
    summarize_neighborhood("INVARIANT neighborhood", Pb_inv, Pa_inv, sims_inv, removed_row)
    summarize_neighborhood("SENSITIVE neighborhood", Pb_sens, Pa_sens, sims_sens, removed_row)

    # Hypothesis test: sensitive |Δp_removed| > invariant |Δp_removed|
    dpr_inv  = (Pa_inv[:,  removed_row] - Pb_inv[:,  removed_row])
    dpr_sens = (Pa_sens[:, removed_row] - Pb_sens[:, removed_row])
    tstat, pval = ttest_ind(np.abs(dpr_sens), np.abs(dpr_inv), equal_var=False)
    print(f"\n|Δp_removed| t-test: sensitive vs invariant  t={tstat:.3f}, p={pval:.2e}")

    # Show the two anchors themselves (BEFORE → AFTER)
    print("\nAnchor snapshots (BEFORE → AFTER):")
    print(f"  Invariant: {P_test_before[idx_inv]}  ->  {P_test_after[idx_inv]}")
    print(f"  Sensitive : {P_test_before[idx_sens]} ->  {P_test_after[idx_sens]}")

    # ===== Optional figures =====
    if save_figs:
        plot_anchor_snapshots(
            P_test_before, P_test_after,
            idx_inv, idx_sens, removed_row,
            classes=np.arange(P_test_before.shape[1]),
            outpath="fig_anchor_snapshots.png"
        )

        plot_hist_dpr(
            dpr_inv, dpr_sens,
            removed_label=class_to_unlearn,
            outpath="fig_hist_delta_p_removed.png"
        )

        # correlations for locality scatter
        r_inv = r_sens = np.nan
        rp_inv = rp_sens = np.nan
        if len(sims_inv) > 1:
            r_inv, rp_inv = pearsonr(sims_inv, -dpr_inv)
        if len(sims_sens) > 1:
            r_sens, rp_sens = pearsonr(sims_sens, -dpr_sens)

        plot_locality_scatter(
            sims_inv, dpr_inv, sims_sens, dpr_sens,
            r_inv, rp_inv, r_sens, rp_sens,
            outpath="fig_locality_scatter.png"
        )

        plot_effect_size_abs_dpr(
            dpr_inv, dpr_sens, tstat, pval,
            outpath="fig_effect_size_abs_delta_p_removed.png"
        )

# ===========
# Main
# ===========
def main():
    # Data (any K from 20NG)
    train_docs, test_docs, y_train, y_test = prepare_data()

    C = 10.0
    # Pick a class to unlearn (label, not row)
    class_to_unlearn = random.randint(0, int(y_train.max()))

    run_locality_experiment_anyK(
        train_docs, test_docs, y_train, y_test,
        C=C,
        class_to_unlearn=class_to_unlearn,
        k_neighbors=50,
        inv_max=0.49,     # invariant: p_removed < 0.5
        sens_min=0.51,    # sensitive: p_removed >= 0.5
        cg_tol=1e-3, cg_max_iter=300, seed=42,
        save_figs=True
    )

if __name__ == "__main__":
    main()