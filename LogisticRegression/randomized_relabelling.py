import numpy as np
import random
from copy import deepcopy
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
import scipy.sparse as sp
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# Data prep
# -----------------------------
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data, data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target
    )

def build_tfidf(train_texts: list, test_texts: list):
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

# -----------------------------
# Target model = Multinomial LR (no intercept)
# -----------------------------
def train_logreg(X, y, C: float = 10.0, max_iter: int = 2000, seed: int = 42):
    lr = LogisticRegression(
        penalty='l2',
        C=C,
        solver='lbfgs',
        multi_class='multinomial',
        fit_intercept=False,
        max_iter=max_iter,
        random_state=seed
    )
    lr.fit(X, y)
    return lr  # .coef_ shape (K, d), .classes_

def evaluate_accuracy(lr_model, vectorizer, docs: list, labels: np.ndarray):
    X = vectorizer.transform(docs)
    preds = lr_model.predict(X)
    return accuracy_score(labels, preds)

def compute_probs(lr_model, X):
    logits = X @ lr_model.coef_.T  # no intercept
    return softmax(logits, axis=1)

# -----------------------------
# Attack features (+ label alignment)
# -----------------------------
def build_attack_features_aligned(P: np.ndarray, labels: np.ndarray, classes_present: np.ndarray):
    """
    Robust to missing labels (if a class was removed in some pipeline).
    Features: [class probs | entropy | CE(true) | top-2 gap]
    """
    n, k = P.shape
    # entropy
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    # true_p aligned
    class_to_col = {int(c): i for i, c in enumerate(classes_present)}
    col_idx = np.array([class_to_col.get(int(y), -1) for y in labels])
    rows = np.arange(n)
    true_p = np.zeros((n, 1))
    mask = (col_idx >= 0)
    if np.any(mask):
        true_p[mask, 0] = P[rows[mask], col_idx[mask]]
    ce_loss = -np.log(true_p + 1e-12)
    # top-2 gap
    top2 = np.sort(P, axis=1)[:, -2:]
    gap = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    return np.hstack([P, entropy, ce_loss, gap])

def get_attack_dataset(lr_model, vectorizer, docs, labels):
    X = vectorizer.transform(docs)
    P = compute_probs(lr_model, X)
    return build_attack_features_aligned(P, labels, lr_model.classes_)

def compute_mia_auc(attack_clf, target_model, vectorizer,
                    train_docs, train_labels, test_docs, test_labels):
    A_tr = get_attack_dataset(target_model, vectorizer, train_docs, train_labels)
    A_te = get_attack_dataset(target_model, vectorizer, test_docs,  test_labels)
    X_att = np.vstack([A_tr, A_te])
    y_att = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))])
    return roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:, 1])

def compute_class_mia_auc(attack_clf, target_model, vectorizer,
                          train_docs, train_labels, test_docs, test_labels, cls: int):
    tr_idx = np.where(train_labels == cls)[0]
    te_idx = np.where(test_labels  == cls)[0]
    docs_tr_k = [train_docs[i] for i in tr_idx]
    docs_te_k = [test_docs[i] for i in te_idx]
    labs_tr_k = train_labels[tr_idx]
    labs_te_k = test_labels[te_idx]
    return compute_mia_auc(attack_clf, target_model, vectorizer,
                           docs_tr_k, labs_tr_k, docs_te_k, labs_te_k)

# -----------------------------
# Multinomial LR grads & HVP
# -----------------------------
def per_sample_grad_multinomial_lr(p_vec, y_row, x_row):
    """
    p_vec: (K,) softmax probs for one sample
    y_row: int row index in W for the label (aligned to model.classes_)
    x_row: (d,) dense feature vector
    returns grad wrt W shape (K, d) for that sample
    """
    g = p_vec.copy()
    g[y_row] -= 1.0
    return g[:, None] * x_row[None, :]  # (K,d)

def summed_grads_for_indices(W, X, y, classes, indices, labels_for_indices):
    """
    Sum per-sample grads over 'indices', but use 'labels_for_indices' for those samples'
    labels in the gradient (can be original or randomized labels).
    Returns (G_sum, P) where P are the softmax probs for all samples at W.
    """
    K, d = W.shape
    Z = X @ W.T
    P = softmax(Z, axis=1)

    class_to_row = {int(c): i for i, c in enumerate(classes)}
    G = np.zeros_like(W)
    if len(indices) == 0:
        return G, P

    for idx, new_lab in zip(indices, labels_for_indices):
        x = X[idx].toarray().ravel() if sp.issparse(X) else np.asarray(X[idx]).ravel()
        y_row = class_to_row[int(new_lab)]
        G += per_sample_grad_multinomial_lr(P[idx], y_row, x)
    return G, P

def hessian_vec_prod_lr(W, V, X, P, C):
    """
    HVP for multinomial LR with L2 (fit_intercept=False).
    H = X^T[Diag(Var[p]) ⊗ I_d]X + (1/C) I   (applied implicitly)
    """
    lam = 1.0 / C
    # U = X @ V^T -> (n,K)
    U = X @ V.T
    # dot = sum_k p_k * u_k
    dot = np.sum(P * U, axis=1)  # (n,)
    # M = (Diag(p) - p p^T) u  row-wise
    M = P * U - P * dot[:, None]
    HV = (M.T @ X)
    if sp.issparse(HV):
        HV = HV.A
    HV = HV + lam * V
    return HV

# -----------------------------
# Hessian-based randomized relabeling (TARGET & SHADOW)
# -----------------------------
def random_relabels_for_class(y, classes, class_to_unlearn, rng):
    """
    For each sample with y == class_to_unlearn, draw a new label uniformly
    from the OTHER classes. Returns (indices, new_labels_array).
    """
    all_classes = np.array(classes)
    other = all_classes[all_classes != class_to_unlearn]
    idx = np.where(y == class_to_unlearn)[0]
    if len(idx) == 0:
        return idx, np.array([], dtype=int)
    new_labs = rng.choice(other, size=len(idx), replace=True)
    return idx, new_labs

def unlearn_via_hessian_lr_random_relabel(lr_model, X_train, y_train,
                                          class_to_unlearn, C: float,
                                          cg_tol=1e-3, cg_max_iter=300, seed=0):
    """
    Logistic-regression Hessian update for RANDOMIZED RELABELING.
    W' = W - H^{-1} * sum_i [ ∇ℓ_i(new_label) - ∇ℓ_i(old_label) ].
    """
    rng = np.random.default_rng(seed)
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()          # (K,d)
    classes = lr_model.classes_
    K, d = W.shape

    # choose randomized new labels for all target-class training samples
    rem_idx = np.where(y_train == class_to_unlearn)[0]
    if len(rem_idx) == 0:
        return lr_new

    new_idx, new_labels = random_relabels_for_class(y_train, classes, class_to_unlearn, rng)

    # compute grads for new and old labels at current W
    G_new, P = summed_grads_for_indices(W, X_train, y_train, classes, new_idx, new_labels)
    G_old, _ = summed_grads_for_indices(W, X_train, y_train, classes, new_idx, y_train[new_idx])
    G_diff = G_new - G_old  # grad_new - grad_old

    if not np.any(G_diff):
        return lr_new

    # define linear operator H on vec(V) using HVP at W (with probs P)
    def matvec(vec):
        V = vec.reshape(K, d)
        HV = hessian_vec_prod_lr(W, V, X_train, P, C)
        return HV.reshape(-1)

    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)

    b = G_diff.reshape(-1)
    delta, info = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    if info != 0:
        # CG didn’t fully converge; apply current delta anyway
        pass
    Delta = delta.reshape(K, d)

    # Update weights
    W_un = W - Delta
    lr_new.coef_ = W_un
    return lr_new

# -----------------------------
# Multi-shadow builders (pool attack data)
# -----------------------------
def build_attack_dataset_from_shadows(vectorizer, train_docs, train_labels, *,
                                      S: int = 10, C_shadow, seed0: int = 0):
    """
    Build a pooled attack dataset from S shadow models (no unlearning inside shadows).
    """
    X_all, y_all = [], []
    for s in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed0 + s, stratify=train_labels
        )
        Xs = vectorizer.transform(s_docs)
        Xh = vectorizer.transform(h_docs)

        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed0 + s)
        Ps = compute_probs(shadow, Xs)
        Ph = compute_probs(shadow, Xh)

        A_s = build_attack_features_aligned(Ps, s_lbls, shadow.classes_)
        A_h = build_attack_features_aligned(Ph, h_lbls, shadow.classes_)

        X_all.append(np.vstack([A_s, A_h]))
        y_all.append(np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))]))

    X_att = np.vstack(X_all)
    y_att = np.concatenate(y_all)
    return X_att, y_att

def build_attack_dataset_from_shadows_with_rand_relabel(vectorizer, train_docs, train_labels, *,
                                                        S: int = 10, C_shadow, class_to_unlearn: int,
                                                        cg_tol=1e-3, cg_max_iter=300, seed0: int = 0):
    """
    Build a pooled attack dataset from S shadow models that each perform
    the SAME randomized-relabeling Hessian unlearning as the target.
    """
    X_all, y_all = [], []
    for s in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed0 + s, stratify=train_labels
        )
        Xs = vectorizer.transform(s_docs)
        Xh = vectorizer.transform(h_docs)

        # Train shadow
        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed0 + s)
        # Apply randomized-relabeling Hessian unlearning on the shadow
        shadow_un = unlearn_via_hessian_lr_random_relabel(
            lr_model=shadow, X_train=Xs, y_train=s_lbls,
            class_to_unlearn=class_to_unlearn, C=C_shadow,
            cg_tol=cg_tol, cg_max_iter=cg_max_iter, seed=seed0 + s
        )

        # Features from the unlearned shadow
        Ps = compute_probs(shadow_un, Xs)
        Ph = compute_probs(shadow_un, Xh)
        A_s = build_attack_features_aligned(Ps, s_lbls, shadow_un.classes_)
        A_h = build_attack_features_aligned(Ph, h_lbls, shadow_un.classes_)

        X_all.append(np.vstack([A_s, A_h]))
        y_all.append(np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))]))

    X_att = np.vstack(X_all)
    y_att = np.concatenate(y_all)
    return X_att, y_att

# -----------------------------
# Unlearning on the TARGET model (LR, Hessian randomized relabel)
# -----------------------------
def unlearn_class_via_rand_relabel_hessian_target(lr_orig, X_train, y_train, class_to_unlearn,
                                                  C: float, cg_tol=1e-3, cg_max_iter=300, seed: int = 0):
    return unlearn_via_hessian_lr_random_relabel(
        lr_model=lr_orig, X_train=X_train, y_train=y_train,
        class_to_unlearn=class_to_unlearn, C=C,
        cg_tol=cg_tol, cg_max_iter=cg_max_iter, seed=seed
    )

# -----------------------------
# Main
# -----------------------------
def main():
    # Data
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Pick a class to unlearn
    random.seed(7)
    target_class = random.randint(0, int(y_train.max()))
    C = 10.0

    # ----- Baseline logistic model
    lr_orig = train_logreg(X_train, y_train, C=C, seed=42)
    acc_before = evaluate_accuracy(lr_orig, vectorizer, test_docs, y_test)

    # ===== Multi-shadow attacker (BEFORE) =====
    X_att_before, y_att_before = build_attack_dataset_from_shadows(
        vectorizer, train_docs, y_train, S=10, C_shadow=C, seed0=0
    )
    attack_before = LogisticRegression(max_iter=5000, random_state=2).fit(X_att_before, y_att_before)
    auc_before = compute_mia_auc(attack_before, lr_orig, vectorizer,
                                 train_docs, y_train, test_docs, y_test)
    auc_before_cls = compute_class_mia_auc(attack_before, lr_orig, vectorizer,
                                           train_docs, y_train, test_docs, y_test,
                                           cls=target_class)

    # ----- Unlearn via HESSIAN randomized relabeling (no refit)
    lr_after = unlearn_class_via_rand_relabel_hessian_target(
        lr_orig, X_train, y_train, target_class, C=C, cg_tol=1e-3, cg_max_iter=300, seed=0
    )

    # ===== Multi-shadow attacker (AFTER; shadows also do rand-relabel + Hessian) =====
    X_att_after, y_att_after = build_attack_dataset_from_shadows_with_rand_relabel(
        vectorizer, train_docs, y_train, S=10, C_shadow=C,
        class_to_unlearn=target_class, cg_tol=1e-3, cg_max_iter=300, seed0=0
    )
    attack_after = LogisticRegression(max_iter=5000, random_state=2).fit(X_att_after, y_att_after)

    # --- Utilities
    acc_after_overall = evaluate_accuracy(lr_after, vectorizer, test_docs, y_test)

    keep_idx = np.where(y_test != target_class)[0]
    test_docs_wo = [test_docs[i] for i in keep_idx]
    y_test_wo = y_test[keep_idx]
    acc_after_wo = evaluate_accuracy(lr_after, vectorizer, test_docs_wo, y_test_wo)

    # --- Privacy scores
    auc_after = compute_mia_auc(attack_after, lr_after, vectorizer,
                                train_docs, y_train, test_docs, y_test)
    auc_after_cls = compute_class_mia_auc(attack_after, lr_after, vectorizer,
                                          train_docs, y_train, test_docs, y_test,
                                          cls=target_class)

    print(f"Target class randomized-relabelled: {target_class}")
    print(f"Accuracy BEFORE: {acc_before:.4f}")
    print(f"MIA AUC BEFORE (overall): {auc_before:.4f}")
    print(f"MIA AUC BEFORE (class {target_class}): {auc_before_cls:.4f}\n")

    print(f"Accuracy AFTER  (overall): {acc_after_overall:.4f}")
    print(f"Accuracy AFTER  (excl. class {target_class}): {acc_after_wo:.4f}")
    print(f"MIA AUC AFTER  (overall): {auc_after:.4f}")
    print(f"MIA AUC AFTER  (class {target_class}): {auc_after_cls:.4f}")

if __name__ == "__main__":
    main()