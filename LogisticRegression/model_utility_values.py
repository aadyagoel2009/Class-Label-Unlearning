import numpy as np
import random
import scipy.sparse as sp
from copy import deepcopy
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Data loaders
# =========================
def load_20ng(test_size=0.2, seed=42):
    data = fetch_20newsgroups(subset="all")
    train_docs, test_docs, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size,
        random_state=seed, stratify=data.target
    )
    return train_docs, test_docs, np.asarray(y_train), np.asarray(y_test)

def load_ag_news(test_size=0.2, seed=42):
    # Requires: pip install datasets
    from datasets import load_dataset
    ds = load_dataset("ag_news")
    train_texts = [r["text"] for r in ds["train"]]
    train_labels = np.asarray([r["label"] for r in ds["train"]])
    test_texts  = [r["text"] for r in ds["test"]]
    test_labels = np.asarray([r["label"] for r in ds["test"]])
    # (Already split; keep as-is for consistency)
    return train_texts, test_texts, train_labels, test_labels

def load_dbpedia(test_size=0.2, seed=42):
    # Requires: pip install datasets
    from datasets import load_dataset
    ds = load_dataset("dbpedia_14")
    train_texts = [r["content"] for r in ds["train"]]
    train_labels = np.asarray([r["label"] for r in ds["train"]])
    test_texts  = [r["content"] for r in ds["test"]]
    test_labels = np.asarray([r["label"] for r in ds["test"]])
    return train_texts, test_texts, train_labels, test_labels

# =========================
# TF-IDF + LR utilities
# =========================
def build_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=5,
        sublinear_tf=True,
        norm="l2"
    )
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
    return vec, X_train, X_test

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

def compute_probs_from_coef(theta, X):
    logits = X @ theta.T
    return softmax(logits, axis=1)

def compute_probs(model, X):
    return softmax(X @ model.coef_.T, axis=1)

def predict_wo_class(theta, X, classes, removed_label):
    """
    Argmax prediction forbidding the removed class.
    """
    probs = compute_probs_from_coef(theta, X)
    keep_mask = (classes != removed_label)
    keep_cols = np.where(keep_mask)[0]
    # argmax among kept columns
    k_idx = keep_cols[np.argmax(probs[:, keep_cols], axis=1)]
    return classes[k_idx]

def accuracy_excluding_label(preds, y_true, removed_label):
    mask = (y_true != removed_label)
    return float(accuracy_score(y_true[mask], preds[mask]))

# =========================
# Hessian downweight pieces
# =========================
def _per_sample_grad_mult_lr(p_vec, y_row, x_row):
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
    row_map = {int(c): i for i, c in enumerate(classes)}
    y_row = row_map[int(class_to_unlearn)]
    for i in rem_idx:
        x = X[i].toarray().ravel() if sp.issparse(X) else np.asarray(X[i]).ravel()
        G += _per_sample_grad_mult_lr(P[i], y_row, x)
    return G, P

def _hvp_lr(W, V, X, P, C):
    lam = 1.0 / C
    U = X @ V.T                  # (n,K)
    rowdot = np.sum(P * U, axis=1)  # (n,)
    M = P * U - P * rowdot[:, None]
    HV = (M.T @ X)
    if sp.issparse(HV):
        HV = HV.A
    return HV + lam * V

def unlearn_class_via_hessian_lr(lr_model, X_train, y_train, class_to_unlearn,
                                 C=10.0, cg_tol=1e-3, cg_max_iter=300):
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()           # (K,d)
    classes = lr_model.classes_
    K, d = W.shape

    G_rem, P = _summed_grad_removed_class(W, X_train, y_train, classes, class_to_unlearn)
    if not np.any(G_rem):
        # still zero row for safety
        row = np.where(classes == class_to_unlearn)[0]
        if len(row) > 0:
            W[row[0], :] = 0.0
            lr_new.coef_ = W
        return lr_new

    def matvec(vec):
        V = vec.reshape(K, d)
        HV = _hvp_lr(W, V, X_train, P, C)
        return HV.reshape(-1)

    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)
    b = G_rem.reshape(-1)
    delta, info = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    Delta = delta.reshape(K, d)

    W_un = W - Delta
    # zero the removed row at release
    row = np.where(classes == class_to_unlearn)[0]
    if len(row) > 0:
        W_un[row[0], :] = 0.0

    lr_new.coef_ = W_un
    return lr_new

# =========================
# Golden standard & baseline label transforms
# =========================
def next_top1_labels_for_deleted(model, X, y, deleted_label):
    """
    For every training sample with y==deleted_label, pick argmax among classes != deleted_label.
    Returns a new y' array with deterministic next-top-1 for the deleted samples.
    """
    classes = model.classes_
    probs = compute_probs(model, X)
    keep_mask = (classes != deleted_label)
    keep_cols = np.where(keep_mask)[0]
    next_idx = keep_cols[np.argmax(probs[:, keep_cols], axis=1)]
    next_labels = classes[next_idx]
    y_new = y.copy()
    sel = (y == deleted_label)
    y_new[sel] = next_labels[sel]
    return y_new

def random_relabel_deleted(y, deleted_label, classes, seed=0):
    rng = np.random.RandomState(seed)
    other = classes[classes != deleted_label]
    y_rr = y.copy()
    sel = (y_rr == deleted_label)
    y_rr[sel] = rng.choice(other, size=sel.sum(), replace=True)
    return y_rr

# =========================
# One dataset run
# =========================
def run_one_dataset(name, loader_fn, C=10.0, seed=42, rr_seed=0):
    print(f"\n=== Dataset: {name} ===")
    train_docs, test_docs, y_train, y_test = loader_fn()
    vec, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Base model
    base = train_logreg(X_train, y_train, C=C, seed=seed)
    acc_pre = float(accuracy_score(y_test, base.predict(X_test)))

    # Pick class to unlearn (sample from present training labels)
    classes = np.unique(y_train)
    class_to_unlearn = int(np.random.RandomState(seed).choice(classes))
    print(f"Unlearned class label: {class_to_unlearn}")

    # ---------- Golden standard (complete retraining with next-top-1) ----------
    y_train_gs = next_top1_labels_for_deleted(base, X_train, y_train, class_to_unlearn)
    gs = train_logreg(X_train, y_train_gs, C=C, seed=seed)

    # Retained-class accuracy (exclude y=c)
    preds_gs = gs.predict(X_test)
    acc_gs_retained = float(accuracy_score(y_test[y_test != class_to_unlearn],
                                           preds_gs[y_test != class_to_unlearn]))

    # ---------- Random relabel baseline (complete retraining) ----------
    y_train_rr = random_relabel_deleted(y_train, class_to_unlearn, classes, seed=rr_seed)
    rr = train_logreg(X_train, y_train_rr, C=C, seed=seed)
    preds_rr = rr.predict(X_test)
    acc_rr_retained = float(accuracy_score(y_test[y_test != class_to_unlearn],
                                           preds_rr[y_test != class_to_unlearn]))

    # ---------- Our algorithm (Hessian downweight + zero deleted row) ----------
    ours = unlearn_class_via_hessian_lr(base, X_train, y_train, class_to_unlearn, C=C,
                                        cg_tol=1e-3, cg_max_iter=300)
    # forbid choosing c at inference
    preds_ours = predict_wo_class(ours.coef_, X_test, ours.classes_, class_to_unlearn)
    acc_ours_retained = float(accuracy_score(y_test[y_test != class_to_unlearn],
                                             preds_ours[y_test != class_to_unlearn]))

    # ---------- Agreement on deleted-class test items: GS vs Ours ----------
    mask_c = (y_test == class_to_unlearn)
    if mask_c.any():
        # GS predicts only over retained classes (since c was removed from its training labels)
        gs_on_c = gs.predict(X_test[mask_c])
        # Our algorithm predicts with c forbidden
        ours_on_c = preds_ours[mask_c]
        agree_deleted = float(np.mean(gs_on_c == ours_on_c))
    else:
        agree_deleted = float("nan")

    # Print results
    print(f"(a) Pre-unlearning accuracy (overall):        {acc_pre:.4f}")
    print(f"(b) Random relabel accuracy (retained):       {acc_rr_retained:.4f}")
    print(f"(c) Golden standard accuracy (retained):      {acc_gs_retained:.4f}")
    print(f"(d) Our algorithm accuracy (retained):        {acc_ours_retained:.4f}")
    print(f"(e) Agreement on deleted class (GS vs Ours):  {agree_deleted:.4f}")

    # Return for optional aggregation
    return dict(
        dataset=name,
        unlearned_class=class_to_unlearn,
        acc_pre=acc_pre,
        acc_rr_retained=acc_rr_retained,
        acc_gs_retained=acc_gs_retained,
        acc_ours_retained=acc_ours_retained,
        agree_deleted=agree_deleted
    )

# =========================
# Main: loop datasets
# =========================
def main():
    results = []
    results.append(run_one_dataset("20 Newsgroups", load_20ng, C=10.0, seed=42, rr_seed=0))
    try:
        results.append(run_one_dataset("AG News", load_ag_news, C=10.0, seed=42, rr_seed=0))
    except Exception as e:
        print("\n[AG News] Skipped due to import/data issue:", e)
    try:
        results.append(run_one_dataset("DBPedia-14", load_dbpedia, C=10.0, seed=42, rr_seed=0))
    except Exception as e:
        print("\n[DBPedia-14] Skipped due to import/data issue:", e)

    # (Optional) pretty print a final summary row-wise
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['dataset']:>12} | c={r['unlearned_class']:>2} | "
              f"pre={r['acc_pre']:.4f} | rr={r['acc_rr_retained']:.4f} | "
              f"gs={r['acc_gs_retained']:.4f} | ours={r['acc_ours_retained']:.4f} | "
              f"agree_c={r['agree_deleted']:.4f}")

if __name__ == "__main__":
    main()