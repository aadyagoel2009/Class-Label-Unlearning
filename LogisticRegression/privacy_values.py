import numpy as np
import random
import scipy.sparse as sp
from copy import deepcopy
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# -----------------------------
# Dataset loaders
# -----------------------------
def load_20newsgroups(test_size=0.2, seed=42):
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all")
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)

def _load_hf_dataset(name, text_col, label_col, test_size=0.2, seed=42):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            f"Install `datasets` to load {name} or replace this loader."
        )
    ds = load_dataset(name)
    texts = list(ds["train"][text_col]) + list(ds["test"][text_col])
    labels = np.array(list(ds["train"][label_col]) + list(ds["test"][label_col]), dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return X_tr, X_te, y_tr, y_te

def load_agnews(test_size=0.2, seed=42):
    # AG News has 4 labels {0..3}
    return _load_hf_dataset("ag_news", "text", "label", test_size, seed)

def load_dbpedia14(test_size=0.2, seed=42):
    # DBPedia-14 has 14 labels {0..13}
    return _load_hf_dataset("dbpedia_14", "content", "label", test_size, seed)

# -----------------------------
# Vectorizer + model
# -----------------------------
def build_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1,3),
        max_df=0.85, min_df=5, sublinear_tf=True, norm="l2"
    )
    Xtr = vec.fit_transform(train_texts); Xte = vec.transform(test_texts)
    return vec, Xtr, Xte

def train_logreg(X, y, C=10.0, max_iter=2000, seed=42):
    lr = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        multi_class="multinomial", fit_intercept=False,
        max_iter=max_iter, random_state=seed
    )
    lr.fit(X, y)
    return lr

def compute_probs(lr_model, X):
    return softmax(X @ lr_model.coef_.T, axis=1)

# -----------------------------
# Attack features + AUC
# -----------------------------
def build_attack_features_aligned(P, labels, classes_present):
    n, k = P.shape
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)

    class_to_col = {int(c): i for i, c in enumerate(classes_present)}
    col_idx = np.array([class_to_col.get(int(y), -1) for y in labels])
    rows = np.arange(n)
    true_p = np.zeros((n,1))
    m = (col_idx >= 0)
    if np.any(m):
        true_p[m,0] = P[rows[m], col_idx[m]]
    ce = -np.log(true_p + 1e-12)

    top2 = np.sort(P, axis=1)[:, -2:]
    gap = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce, gap])

def get_attack_dataset(target_model, vectorizer, docs, labels):
    X = vectorizer.transform(docs)
    P = compute_probs(target_model, X)
    return build_attack_features_aligned(P, labels, target_model.classes_)

def compute_mia_auc(attack_clf, target_model, vectorizer,
                    train_docs, train_labels, test_docs, test_labels):
    A_tr = get_attack_dataset(target_model, vectorizer, train_docs, train_labels)
    A_te = get_attack_dataset(target_model, vectorizer, test_docs,  test_labels)
    X_att = np.vstack([A_tr, A_te])
    y_att = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))])
    return roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1])

def compute_class_mia_auc(attack_clf, target_model, vectorizer,
                          train_docs, train_labels, test_docs, test_labels, cls):
    tr_idx = np.where(train_labels == cls)[0]
    te_idx = np.where(test_labels  == cls)[0]
    docs_tr_k = [train_docs[i] for i in tr_idx]
    docs_te_k = [test_docs[i]  for i in te_idx]
    labs_tr_k = train_labels[tr_idx]
    labs_te_k = test_labels[te_idx]
    return compute_mia_auc(attack_clf, target_model, vectorizer,
                           docs_tr_k, labs_tr_k, docs_te_k, labs_te_k)

# -----------------------------
# Hessian utilities (same as yours)
# -----------------------------
def _hvp_lr(W, V, X, P, C):
    lam = 1.0 / C
    U = X @ V.T
    rowdot = np.sum(P * U, axis=1)
    M = P * U - P * rowdot[:, None]
    HV = (M.T @ X)
    if sp.issparse(HV): HV = HV.A
    return HV + lam * V

def _sum_grad_remove_class(W, X, y, classes, c):
    Z = X @ W.T
    P = softmax(Z, axis=1)
    idx = np.where(y == c)[0]
    G = np.zeros_like(W)
    if len(idx) == 0:
        return G, P
    class_to_row = {int(k): i for i, k in enumerate(classes)}
    y_row = class_to_row[int(c)]
    for i in idx:
        x = X[i].toarray().ravel() if sp.issparse(X) else np.asarray(X[i]).ravel()
        g = np.zeros_like(W)
        g[y_row, :] = x  # (e_y ⊗ x)
        G += g
    # Note: this builds (e_y ⊗ x); we’ll subtract with H^{-1} later using the exact formula below
    return G, P

# Precise gradient for original label:
def _grad_sample_for_label(x_row, y_row, K):
    g = np.zeros((K, x_row.shape[0]), dtype=float)
    g[y_row, :] = x_row
    return g

# -----------------------------
# Our algorithm (Hessian downweight)
# -----------------------------
def unlearn_class_hessian(lr_model, Xtr, ytr, cls_to_unlearn, C=10.0, cg_tol=1e-3, cg_max_iter=300):
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()
    classes = lr_model.classes_
    K, d = W.shape

    # Sum true gradients over removed class (as in your code)
    #   g_c = sum_i ∇ℓ(x_i, y_i)  at current W
    # Implement directly using softmax form
    Z = Xtr @ W.T
    P = softmax(Z, axis=1)
    rem_idx = np.where(ytr == cls_to_unlearn)[0]
    G = np.zeros_like(W)
    class_to_row = {int(c): i for i, c in enumerate(classes)}
    y_row = class_to_row[int(cls_to_unlearn)]
    for i in rem_idx:
        x = Xtr[i].toarray().ravel() if sp.issparse(Xtr) else np.asarray(Xtr[i]).ravel()
        p_vec = P[i].copy(); p_vec[y_row] -= 1.0   # (p - e_y)
        G += p_vec[:, None] * x[None, :]

    # HVP operator at current W
    def matvec(vec):
        V = vec.reshape(K, d)
        HV = _hvp_lr(W, V, Xtr, P, C)
        return HV.reshape(-1)
    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)

    b = G.reshape(-1)                  # g_c
    delta, _ = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    Delta = delta.reshape(K, d)

    W_un = W - Delta                   # W' = W - H^{-1} g_c
    # zero removed row for release
    row = np.where(classes == cls_to_unlearn)[0]
    if len(row) > 0: W_un[row[0], :] = 0.0
    lr_new.coef_ = W_un
    return lr_new

# -----------------------------
# Random relabel unlearning (Hessian step)
# -----------------------------
def unlearn_class_random_relabel(lr_model, Xtr, ytr, cls_to_unlearn, C=10.0,
                                 cg_tol=1e-3, cg_max_iter=300, seed=123):
    """
    Reassign y=c samples to random labels r!=c, then apply one Newton step
    for gradient change: sum_i [∇ℓ(x_i, r_i) - ∇ℓ(x_i, y_i)].
      For multinomial LR, this equals sum_i [(e_y - e_{r_i}) ⊗ x_i].
    Finally, zero the removed class row for release.
    """
    rng = np.random.RandomState(seed)
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()
    classes = lr_model.classes_
    K, d = W.shape

    # Build gradient change G_diff = sum_i (e_y - e_r) ⊗ x_i
    row_y = int(np.where(classes == cls_to_unlearn)[0][0])
    G_diff = np.zeros_like(W)
    idx = np.where(ytr == cls_to_unlearn)[0]
    if len(idx) == 0:
        # still zero removed row
        W[row_y, :] = 0.0
        lr_new.coef_ = W
        return lr_new

    other_rows = [int(r) for r in range(K) if r != row_y]
    for i in idx:
        x = Xtr[i].toarray().ravel() if sp.issparse(Xtr) else np.asarray(Xtr[i]).ravel()
        r_row = rng.choice(other_rows)
        G_diff[row_y, :] += x
        G_diff[r_row, :] -= x

    # HVP at current W
    Z = Xtr @ W.T
    P = softmax(Z, axis=1)
    def matvec(vec):
        V = vec.reshape(K, d)
        return _hvp_lr(W, V, Xtr, P, C).reshape(-1)
    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)

    delta, _ = cg(H_op, G_diff.reshape(-1), tol=cg_tol, maxiter=cg_max_iter)
    W_rr = W - delta.reshape(K, d)

    # zero removed row for release
    W_rr[row_y, :] = 0.0
    lr_new.coef_ = W_rr
    return lr_new

# -----------------------------
# Shadows (before / after variants)
# -----------------------------
def build_attack_dataset_from_shadows(vec, train_docs, train_labels, *, S=10, C_shadow=10.0):
    X_all, y_all = [], []
    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs = vec.transform(s_docs); Xh = vec.transform(h_docs)
        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed)
        Ps = compute_probs(shadow, Xs); Ph = compute_probs(shadow, Xh)
        A_s = build_attack_features_aligned(Ps, s_lbls, shadow.classes_)
        A_h = build_attack_features_aligned(Ph, h_lbls, shadow.classes_)
        X_all.append(np.vstack([A_s, A_h]))
        y_all.append(np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))]))
    return np.vstack(X_all), np.concatenate(y_all)

def build_shadows_after(vec, train_docs, train_labels, *, S=10, C_shadow=10.0,
                        class_to_unlearn=0, mode="ours", cg_tol=1e-3, cg_max_iter=300):
    """
    mode ∈ {"ours","random"} controls how each shadow unlearns its own chosen class (same label).
    """
    X_all, y_all = [], []
    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs = vec.transform(s_docs); Xh = vec.transform(h_docs)
        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed)
        if mode == "ours":
            shadow_u = unlearn_class_hessian(shadow, Xs, s_lbls, class_to_unlearn,
                                             C=C_shadow, cg_tol=cg_tol, cg_max_iter=cg_max_iter)
        else:
            shadow_u = unlearn_class_random_relabel(shadow, Xs, s_lbls, class_to_unlearn,
                                                    C=C_shadow, cg_tol=cg_tol, cg_max_iter=cg_max_iter, seed=seed)
        Ps = compute_probs(shadow_u, Xs); Ph = compute_probs(shadow_u, Xh)
        A_s = build_attack_features_aligned(Ps, s_lbls, shadow_u.classes_)
        A_h = build_attack_features_aligned(Ph, h_lbls, shadow_u.classes_)
        X_all.append(np.vstack([A_s, A_h]))
        y_all.append(np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))]))
    return np.vstack(X_all), np.concatenate(y_all)

# -----------------------------
# One run on a dataset
# -----------------------------
def run_privacy_report(name, loader_fn, C=10.0, S=10, seed=42):
    print(f"\n=== {name} ===")
    train_docs, test_docs, y_tr, y_te = loader_fn()
    random.seed(seed); np.random.seed(seed)

    # Vectorizer and base model
    vec, Xtr, Xte = build_tfidf(train_docs, test_docs)
    base = train_logreg(Xtr, y_tr, C=C, seed=seed)

    # choose target class to unlearn
    cls_to_unlearn = int(np.random.choice(np.unique(y_tr)))
    print(f"Target class to unlearn: {cls_to_unlearn}")

    # --- Build attackers ---
    # BEFORE attacker from standard shadows
    X_att_before, y_att_before = build_attack_dataset_from_shadows(vec, train_docs, y_tr, S=S, C_shadow=C)
    attacker_before = LogisticRegression(max_iter=5000, random_state=2).fit(X_att_before, y_att_before)

    # AFTER attacker for OURS
    X_att_after_ours, y_att_after_ours = build_shadows_after(
        vec, train_docs, y_tr, S=S, C_shadow=C, class_to_unlearn=cls_to_unlearn, mode="ours"
    )
    attacker_after_ours = LogisticRegression(max_iter=5000, random_state=3).fit(X_att_after_ours, y_att_after_ours)

    # AFTER attacker for RANDOM RELABEL
    X_att_after_rr, y_att_after_rr = build_shadows_after(
        vec, train_docs, y_tr, S=S, C_shadow=C, class_to_unlearn=cls_to_unlearn, mode="random"
    )
    attacker_after_rr = LogisticRegression(max_iter=5000, random_state=4).fit(X_att_after_rr, y_att_after_rr)

    # --- Target models ---
    model_pre   = base
    model_ours  = unlearn_class_hessian(base, Xtr, y_tr, cls_to_unlearn, C=C)
    model_rr    = unlearn_class_random_relabel(base, Xtr, y_tr, cls_to_unlearn, C=C)

    # --- AUCs ---
    def auc_ret(model, attacker):
        return compute_mia_auc(attacker, model, vec, train_docs, y_tr, test_docs, y_te)

    def auc_cls(model, attacker, cls):
        return compute_class_mia_auc(attacker, model, vec, train_docs, y_tr, test_docs, y_te, cls=cls)

    results = {
        "a_pre_retain": auc_ret(model_pre, attacker_before),
        "b_pre_target": auc_cls(model_pre, attacker_before, cls_to_unlearn),
        "c_rr_retain":  auc_ret(model_rr, attacker_after_rr),
        "d_rr_target":  auc_cls(model_rr, attacker_after_rr, cls_to_unlearn),
        "e_ours_retain": auc_ret(model_ours, attacker_after_ours),
        "f_ours_target": auc_cls(model_ours, attacker_after_ours, cls_to_unlearn),
    }

    # pretty print
    for k in ["a_pre_retain","b_pre_target","c_rr_retain","d_rr_target","e_ours_retain","f_ours_target"]:
        print(f"{k}: {results[k]:.4f}")
    return results

# -----------------------------
# Main: run all three datasets
# -----------------------------
if __name__ == "__main__":
    summaries = {}
    summaries["20newsgroups"] = run_privacy_report("20 Newsgroups", load_20newsgroups, C=10.0, S=10, seed=42)
    summaries["AGNews"]       = run_privacy_report("AG News",       load_agnews,       C=10.0, S=10, seed=43)
    summaries["DBPedia14"]    = run_privacy_report("DBPedia-14",    load_dbpedia14,    C=10.0, S=10, seed=44)

    print("\n=== Summary (ROC–AUC) ===")
    for ds, res in summaries.items():
        print(f"\n{ds}")
        for label, val in res.items():
            print(f"  {label}: {val:.4f}")