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
# Target model = Multinomial LR
# -----------------------------
def train_logreg(X, y, C, max_iter: int = 2000, seed: int = 42):
    """
    Multinomial Logistic Regression (L2), no intercept to match your previous shape logic.
    """
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
    return lr  # has .coef_ (K,d), .classes_

def evaluate_accuracy(lr_model, vectorizer, docs: list, labels: np.ndarray):
    X = vectorizer.transform(docs)
    preds = lr_model.predict(X)
    return accuracy_score(labels, preds)

# -----------------------------
# Attack features (+ label alignment)
# -----------------------------
def compute_probs(lr_model, X):
    logits = X @ lr_model.coef_.T  # fit_intercept=False
    return softmax(logits, axis=1)

def build_attack_features_aligned(P: np.ndarray, labels: np.ndarray, classes_present: np.ndarray):
    """
    Safely build attack features even if some true labels are not in classes_present.
    For missing labels, true_p is set to 0 (maximally surprising).
    Features: [probs | entropy | CE(true) | top-2 gap]
    """
    n, k = P.shape
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)

    class_to_col = {int(c): i for i, c in enumerate(classes_present)}
    col_idx = np.array([class_to_col.get(int(y), -1) for y in labels])
    rows = np.arange(n)
    true_p = np.zeros((n, 1))
    mask = (col_idx >= 0)
    if np.any(mask):
        true_p[mask, 0] = P[rows[mask], col_idx[mask]]
    ce_loss = -np.log(true_p + 1e-12)

    top2 = np.sort(P, axis=1)[:, -2:]
    gap = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)

    return np.hstack([P, entropy, ce_loss, gap])

def get_attack_dataset(lr_model, vectorizer, docs, labels):
    X = vectorizer.transform(docs)
    P = compute_probs(lr_model, X)
    A = build_attack_features_aligned(P, labels, lr_model.classes_)
    return A

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
# Logistic-regression Hessian unlearning pieces
# -----------------------------
def _per_sample_grad_multinomial_lr(p_vec, y_row, x_row):
    """
    Gradient wrt W (K,d) for one sample: (p - onehot(y)) ⊗ x
    """
    g = p_vec.copy()
    g[y_row] -= 1.0
    return g[:, None] * x_row[None, :]

def _summed_grad_removed_class(W, X, y, classes, class_to_unlearn):
    """
    Sum per-sample grads over all samples with label == class_to_unlearn,
    evaluated at current weights W. Returns (G_sum, P) where P is softmax(X W^T).
    """
    K, d = W.shape
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
    """
    HVP for multinomial LR with L2 (fit_intercept=False):
      H(V) = ∑_j X_j^T (Diag(p_j) - p_j p_j^T) (V X_j) + (1/C) V
    Implemented efficiently as: U = X V^T; M = P⊙U - P*(rowdot); HV = M^T X + (1/C) V
    """
    lam = 1.0 / C
    U = X @ V.T                    # (n,K)
    rowdot = np.sum(P * U, axis=1) # (n,)
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

    # Sum gradients of all samples in the removed class
    G_rem, P = _summed_grad_removed_class(W, X_train, y_train, classes, class_to_unlearn)
    if not np.any(G_rem):
        # nothing to remove
        # still zero the row to make the class inert in release
        row = np.where(classes == class_to_unlearn)[0]
        if len(row) > 0:
            W[row[0], :] = 0.0
            lr_new.coef_ = W
        return lr_new

    # Define linear operator for H using HVP
    def matvec(vec):
        V = vec.reshape(K, d)
        HV = _hessian_vec_prod_lr(W, V, X_train, P, C)
        return HV.reshape(-1)

    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)

    # Solve H * delta = G_rem  -> delta = H^{-1} G_rem
    b = G_rem.reshape(-1)
    delta, info = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    Delta = delta.reshape(K, d)

    # Apply downdate
    W_un = W - Delta

    # Zero the removed class row (final release)
    row = np.where(classes == class_to_unlearn)[0]
    if len(row) > 0:
        W_un[row[0], :] = 0.0

    lr_new.coef_ = W_un
    return lr_new

# -----------------------------
# Multi-shadow builders (pool attack data)
# -----------------------------
def build_attack_dataset_from_shadows(vectorizer, train_docs, train_labels, *,
                                      S: int = 10, C_shadow):
    """
    Pooled attack data from S shadows (no unlearning inside).
    """
    X_all, y_all = [], []
    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs = vectorizer.transform(s_docs)
        Xh = vectorizer.transform(h_docs)

        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed)
        Ps = compute_probs(shadow, Xs)
        Ph = compute_probs(shadow, Xh)

        A_s = build_attack_features_aligned(Ps, s_lbls, shadow.classes_)
        A_h = build_attack_features_aligned(Ph, h_lbls, shadow.classes_)

        X_all.append(np.vstack([A_s, A_h]))
        y_all.append(np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))]))

    X_att = np.vstack(X_all)
    y_att = np.concatenate(y_all)
    return X_att, y_att

def build_attack_dataset_from_shadows_with_unlearning(vectorizer, train_docs, train_labels, *,
                                                      S: int = 10, C_shadow,
                                                      class_to_unlearn: int,
                                                      cg_tol=1e-3, cg_max_iter=300):
    """
    Pooled attack data from S shadow models that each perform the SAME unlearning
    as the target: logistic-regression Hessian downdate + zeroing the removed row.
    """
    X_all, y_all = [], []
    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs = vectorizer.transform(s_docs)
        Xh = vectorizer.transform(h_docs)

        # Train shadow on s_docs
        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed)

        # Apply the SAME Hessian unlearning to the shadow
        shadow_un = unlearn_class_via_hessian_lr(
            shadow, Xs, s_lbls, class_to_unlearn,
            C=C_shadow, cg_tol=cg_tol, cg_max_iter=cg_max_iter
        )

        # Build attack features from unlearned shadow
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
# Main
# -----------------------------
def main():
    # Data
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Pick a class to unlearn
    class_to_unlearn = random.randint(0, int(y_train.max()))
    C = 10.0
    print(f"Class label unlearned: {class_to_unlearn}")


    # ----- Baseline logistic model
    lr_orig = train_logreg(X_train, y_train, C=C, seed=42)
    acc_before = evaluate_accuracy(lr_orig, vectorizer, test_docs, y_test)
    print(f"Accuracy before unlearning: {acc_before:.4f}")

    # ===== Multi-shadow attacker (BEFORE) =====
    X_att_before, y_att_before = build_attack_dataset_from_shadows(
        vectorizer, train_docs, y_train, S=10, C_shadow=C
    )
    attack_before = LogisticRegression(max_iter=5000, random_state=2).fit(X_att_before, y_att_before)
    auc_before = compute_mia_auc(attack_before, lr_orig, vectorizer,
                                 train_docs, y_train, test_docs, y_test)
    auc_before_cls = compute_class_mia_auc(attack_before, lr_orig, vectorizer,
                                           train_docs, y_train, test_docs, y_test,
                                           cls=class_to_unlearn)

    print(f"MIA AUC before unlearning (overall): {auc_before:.4f}")
    print(f"MIA AUC before unlearning (class {class_to_unlearn}): {auc_before_cls:.4f}\n")
    
    # ----- Unlearn selected class by HESSIAN downdate (+ zero removed row)
    lr_final = unlearn_class_via_hessian_lr(
        lr_model=lr_orig, X_train=X_train, y_train=y_train,
        class_to_unlearn=class_to_unlearn, C=C, cg_tol=1e-3, cg_max_iter=300
    )

    # ===== Multi-shadow attacker (AFTER; shadows also do HESSIAN unlearning) =====
    X_att_after, y_att_after = build_attack_dataset_from_shadows_with_unlearning(
        vectorizer, train_docs, y_train, S=10, C_shadow=C,
        class_to_unlearn=class_to_unlearn, cg_tol=1e-3, cg_max_iter=300
    )
    attack_after = LogisticRegression(max_iter=5000, random_state=2).fit(X_att_after, y_att_after)

    acc_after_overall = evaluate_accuracy(lr_final, vectorizer, test_docs, y_test)
    auc_after = compute_mia_auc(attack_after, lr_final, vectorizer,
                                train_docs, y_train, test_docs, y_test)
    auc_after_cls = compute_class_mia_auc(attack_after, lr_final, vectorizer,
                                          train_docs, y_train, test_docs, y_test,
                                          cls=class_to_unlearn)

    # Accuracy excluding the removed class from test (optional but informative)
    keep_idx = np.where(y_test != class_to_unlearn)[0]
    test_docs_wo = [test_docs[i] for i in keep_idx]
    y_test_wo = y_test[keep_idx]
    acc_after_wo = evaluate_accuracy(lr_final, vectorizer, test_docs_wo, y_test_wo)

    print(f"Class label unlearned: {class_to_unlearn}")
    print(f"Accuracy before unlearning: {acc_before:.4f}")
    print(f"MIA AUC before unlearning (overall): {auc_before:.4f}")
    print(f"MIA AUC before unlearning (class {class_to_unlearn}): {auc_before_cls:.4f}\n")

    print(f"Accuracy after unlearning (overall): {acc_after_overall:.4f}")
    print(f"Accuracy after unlearning (excl. class {class_to_unlearn}): {acc_after_wo:.4f}")
    print(f"MIA AUC after unlearning (overall): {auc_after:.4f}")
    print(f"MIA AUC after unlearning (class {class_to_unlearn}): {auc_after_cls:.4f}")

if __name__ == "__main__":
    main()