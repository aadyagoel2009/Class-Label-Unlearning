import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# -----------------------------
# Data prep (20 Newsgroups)
# -----------------------------
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data, data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target
    )

def build_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=5,
        sublinear_tf=True,
        norm="l2"
    )
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    return vec, Xtr, Xte

# -----------------------------
# Target model = Multinomial LR
# -----------------------------
def train_logreg(X, y, C=10.0, max_iter=2000, seed=42):
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        multi_class="multinomial",
        fit_intercept=False,
        max_iter=max_iter,
        random_state=seed
    )
    lr.fit(X, y)
    return lr

def compute_probs(lr_model, X):
    logits = X @ lr_model.coef_.T
    return softmax(logits, axis=1)

# -----------------------------
# Hessian downweight (LR)
# -----------------------------
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
    U = X @ V.T                    # (n,K)
    rowdot = np.sum(P * U, axis=1) # (n,)
    M = P * U - P * rowdot[:, None]
    HV = (M.T @ X)
    if sp.issparse(HV):
        HV = HV.A
    return HV + lam * V

def unlearn_class_via_hessian_lr(lr_model, X_train, y_train, class_to_unlearn,
                                 C=10.0, cg_tol=1e-3, cg_max_iter=300):
    """
    Hessian downweight: W' = W - H^{-1} * g_removed, then zero the removed class row.
    """
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()
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
    delta, _ = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    Delta = delta.reshape(K, d)

    W_un = W - Delta
    row = np.where(classes == class_to_unlearn)[0]
    if len(row) > 0:
        W_un[row[0], :] = 0.0
    lr_new.coef_ = W_un
    return lr_new

# -----------------------------
# Top-1 margin utilities
# -----------------------------
def compute_margins_from_probs(lr_model, vectorizer, docs, labels):
    """
    margin = p_true - max_{j≠true} p_j   (only for examples whose true label
    exists in lr_model.classes_)
    """
    X = vectorizer.transform(docs)
    P = compute_probs(lr_model, X)

    class_to_col = {int(c): i for i, c in enumerate(lr_model.classes_)}
    idx = np.array([class_to_col.get(int(y), -1) for y in labels])
    ok = (idx >= 0)
    P_ok = P[ok]
    idx_ok = idx[ok]
    rows = np.arange(len(idx_ok))

    p_true = P_ok[rows, idx_ok]
    P_other = P_ok.copy()
    P_other[rows, idx_ok] = -np.inf
    max_other = P_other.max(axis=1)
    return p_true - max_other, ok

# -----------------------------
# Main: make the four figures
# -----------------------------
def main():
    # Data
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Choose a class to unlearn (no fixed seed here)
    class_to_unlearn = int(np.random.randint(0, y_train.max() + 1))
    C = 10.0

    # Train + unlearn
    lr_before = train_logreg(X_train, y_train, C=C, seed=42)
    lr_after  = unlearn_class_via_hessian_lr(
        lr_model=lr_before, X_train=X_train, y_train=y_train,
        class_to_unlearn=class_to_unlearn, C=C
    )

    # Margins before/after
    m_before, _ = compute_margins_from_probs(lr_before, vectorizer, test_docs, y_test)
    m_after,  _ = compute_margins_from_probs(lr_after,  vectorizer, test_docs, y_test)

    # Split retain vs target
    retain_mask = (y_test != class_to_unlearn)
    target_mask = ~retain_mask

    bins = np.linspace(-1.0, 1.0, 61)

    # 1) BEFORE — retain classes
    plt.figure(figsize=(6,4))
    plt.hist(m_before[retain_mask], bins=bins, density=True, alpha=0.9)
    plt.title(f"Before — Retain Classes")
    plt.xlabel("Top-1 margin  (p_true − max p(other))"); plt.ylabel("Density")
    plt.tight_layout(); plt.show()

    # 2) AFTER — retain classes
    plt.figure(figsize=(6,4))
    plt.hist(m_after[retain_mask], bins=bins, density=True, alpha=0.9)
    plt.title(f"After — Retain Classes")
    plt.xlabel("Top-1 margin  (p_true − max p(other))"); plt.ylabel("Density")
    plt.tight_layout(); plt.show()

    # 3) BEFORE — target class
    plt.figure(figsize=(6,4))
    plt.hist(m_before[target_mask], bins=bins, density=True, alpha=0.9)
    plt.title(f"After — Target Class")
    plt.xlabel("Top-1 margin  (p_true − max p(other))"); plt.ylabel("Density")
    plt.tight_layout(); plt.show()

    # 4) AFTER — target class
    plt.figure(figsize=(6,4))
    plt.hist(m_after[target_mask], bins=bins, density=True, alpha=0.9)
    plt.title(f"After — Target Class")
    plt.xlabel("Top-1 margin  (p_true − max p(other))"); plt.ylabel("Density")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()