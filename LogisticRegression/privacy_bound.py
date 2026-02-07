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
    return lr

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
    Features: [probs | entropy | CE(true) | top-2 gap]
    Handles missing true labels in classes_present by setting true_p=0.
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

def compute_retained_mia_auc(attack_clf, target_model, vectorizer,
                             train_docs, train_labels, test_docs, test_labels, deleted_cls: int):
    tr_idx = np.where(train_labels != deleted_cls)[0]
    te_idx = np.where(test_labels  != deleted_cls)[0]
    docs_tr = [train_docs[i] for i in tr_idx]
    docs_te = [test_docs[i]  for i in te_idx]
    labs_tr = train_labels[tr_idx]
    labs_te = test_labels[te_idx]
    return compute_mia_auc(attack_clf, target_model, vectorizer,
                           docs_tr, labs_tr, docs_te, labs_te)


# -----------------------------
# Logistic-regression Hessian unlearning pieces
# -----------------------------
def _per_sample_grad_multinomial_lr(p_vec, y_row, x_row):
    g = p_vec.copy()
    g[y_row] -= 1.0
    return g[:, None] * x_row[None, :]

def _summed_grad_removed_class(W, X, y, classes, class_to_unlearn):
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
    lam = 1.0 / C
    U = X @ V.T                    # (n,K)
    rowdot = np.sum(P * U, axis=1) # (n,)
    M = P * U - P * rowdot[:, None]
    HV = (M.T @ X)
    if sp.issparse(HV):
        HV = HV.A
    HV = HV + lam * V
    return HV

def unlearn_class_via_hessian_lr(
    lr_model, X_train, y_train, class_to_unlearn,
    C: float, cg_tol=1e-3, cg_max_iter=300,
    sigma: float = 0.0, noise_seed: int = 0, keep_deleted_row_zero: bool = True
):
    """
    Influence-style Hessian downdate:
      W' = W - H^{-1} * G_removed
    Then zero the removed class row for release.
    Then (optional) add Gaussian noise to released parameters:
      W_release = W' + N(0, sigma^2 I)
    By default, the deleted class row stays exactly zero.
    """
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()   # (K,d)
    classes = lr_model.classes_
    K, d = W.shape

    G_rem, P = _summed_grad_removed_class(W, X_train, y_train, classes, class_to_unlearn)

    if not np.any(G_rem):
        row = np.where(classes == class_to_unlearn)[0]
        if len(row) > 0:
            W[row[0], :] = 0.0
        W_un = W
    else:
        def matvec(vec):
            V = vec.reshape(K, d)
            HV = _hessian_vec_prod_lr(W, V, X_train, P, C)
            return HV.reshape(-1)

        H_op = LinearOperator(shape=(K * d, K * d), matvec=matvec, dtype=W.dtype)

        b = G_rem.reshape(-1)
        delta, info = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
        Delta = delta.reshape(K, d)

        W_un = W - Delta

        row = np.where(classes == class_to_unlearn)[0]
        if len(row) > 0:
            W_un[row[0], :] = 0.0

    # Add Gaussian noise at release
    if sigma > 0.0:
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0.0, sigma, size=W_un.shape)

        if keep_deleted_row_zero:
            row = np.where(classes == class_to_unlearn)[0]
            if len(row) > 0:
                noise[row[0], :] = 0.0

        W_un = W_un + noise

        if keep_deleted_row_zero:
            row = np.where(classes == class_to_unlearn)[0]
            if len(row) > 0:
                W_un[row[0], :] = 0.0

    lr_new.coef_ = W_un
    return lr_new


# -----------------------------
# Multi-shadow builders (pool attack data)
# -----------------------------
def build_attack_dataset_from_shadows_with_unlearning(
    vectorizer, train_docs, train_labels, *,
    S: int = 10, C_shadow: float,
    class_to_unlearn: int,
    cg_tol=1e-3, cg_max_iter=300,
    sigma: float = 0.0, noise_seed_base: int = 1000
):
    """
    Pooled attack data from S shadow models that each perform the SAME unlearning
    and the SAME release noise sigma.
    """
    X_all, y_all = [], []

    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs = vectorizer.transform(s_docs)
        Xh = vectorizer.transform(h_docs)

        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed)

        shadow_un = unlearn_class_via_hessian_lr(
            shadow, Xs, s_lbls, class_to_unlearn,
            C=C_shadow, cg_tol=cg_tol, cg_max_iter=cg_max_iter,
            sigma=sigma, noise_seed=noise_seed_base + seed, keep_deleted_row_zero=True
        )

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
# Helpers for mean ± std formatting
# -----------------------------
def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

def fmt_ms(m, s, decimals=4):
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


# -----------------------------
# Main experiment runner
# -----------------------------
def main():
    # Reproducibility for class choice / other python randomness
    np.random.seed(0)
    random.seed(0)

    # Data
    train_docs, test_docs, y_train, y_test = prepare_data(random_state=42)
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Pick a class to unlearn (fixed for reproducibility)
    class_to_unlearn = int(np.random.default_rng(0).integers(0, int(y_train.max()) + 1))

    C = 10.0
    print(f"Deleted class label: {class_to_unlearn}")
    print(f"C (inverse reg strength): {C}")

    # Base model
    lr_orig = train_logreg(X_train, y_train, C=C, seed=42)
    acc_before = evaluate_accuracy(lr_orig, vectorizer, test_docs, y_test)
    print(f"Accuracy before unlearning: {acc_before:.4f}\n")

    # Sigma sweep + target noise seeds
    sigmas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    noise_seeds = [101, 202, 303, 404, 505]  # 5 different target release noise seeds

    # Precompute test-without-deleted indices (utility)
    keep_idx = np.where(y_test != class_to_unlearn)[0]
    test_docs_wo = [test_docs[i] for i in keep_idx]
    y_test_wo = y_test[keep_idx]

    results = []

    for sigma in sigmas:
        # Train attacker ONCE per sigma (shadow models use varied noise seeds already)
        X_att, y_att = build_attack_dataset_from_shadows_with_unlearning(
            vectorizer, train_docs, y_train,
            S=10, C_shadow=C,
            class_to_unlearn=class_to_unlearn,
            cg_tol=1e-3, cg_max_iter=300,
            sigma=sigma, noise_seed_base=2000
        )
        attack = LogisticRegression(max_iter=5000, random_state=2).fit(X_att, y_att)

        # Evaluate 5 different released noisy models for this sigma
        acc_overall_list = []
        acc_wo_list = []
        mia_overall_list = []
        mia_deleted_list = []
        mia_retained_list = []

        # For sigma=0, noise_seed is irrelevant; we still run once and replicate for clean std=0.
        seeds_to_run = noise_seeds if sigma > 0 else [noise_seeds[0]]

        for ns in seeds_to_run:
            lr_final = unlearn_class_via_hessian_lr(
                lr_model=lr_orig, X_train=X_train, y_train=y_train,
                class_to_unlearn=class_to_unlearn, C=C,
                cg_tol=1e-3, cg_max_iter=300,
                sigma=sigma, noise_seed=ns, keep_deleted_row_zero=True
            )

            # Utility
            acc_after_overall = evaluate_accuracy(lr_final, vectorizer, test_docs, y_test)
            acc_after_wo = evaluate_accuracy(lr_final, vectorizer, test_docs_wo, y_test_wo)

            # Privacy (MIA)
            auc_overall = compute_mia_auc(attack, lr_final, vectorizer,
                                          train_docs, y_train, test_docs, y_test)
            auc_deleted = compute_class_mia_auc(attack, lr_final, vectorizer,
                                                train_docs, y_train, test_docs, y_test,
                                                cls=class_to_unlearn)
            auc_retained = compute_retained_mia_auc(attack, lr_final, vectorizer,
                                                    train_docs, y_train, test_docs, y_test,
                                                    deleted_cls=class_to_unlearn)

            acc_overall_list.append(acc_after_overall)
            acc_wo_list.append(acc_after_wo)
            mia_overall_list.append(auc_overall)
            mia_deleted_list.append(auc_deleted)
            mia_retained_list.append(auc_retained)

        # If sigma=0, replicate single result so formatting stays uniform and std=0
        if sigma == 0.0:
            acc_overall_list *= 5
            acc_wo_list *= 5
            mia_overall_list *= 5
            mia_deleted_list *= 5
            mia_retained_list *= 5

        acc_m, acc_s = mean_std(acc_overall_list)
        accwo_m, accwo_s = mean_std(acc_wo_list)
        mia_m, mia_s = mean_std(mia_overall_list)
        miad_m, miad_s = mean_std(mia_deleted_list)
        miar_m, miar_s = mean_std(mia_retained_list)

        # Conservative theory curve (diverges at sigma=0)
        B_theory = (4.0 * (C ** 2) / (sigma ** 2)) if sigma > 0 else None

        results.append({
            "sigma": sigma,
            "acc_overall_m": acc_m, "acc_overall_s": acc_s,
            "acc_wo_m": accwo_m, "acc_wo_s": accwo_s,
            "mia_overall_m": mia_m, "mia_overall_s": mia_s,
            "mia_deleted_m": miad_m, "mia_deleted_s": miad_s,
            "mia_retained_m": miar_m, "mia_retained_s": miar_s,
            "B_theory": B_theory
        })

    # Print results table (mean ± std)
    print("\n=== Results (mean ± std over 5 noise seeds) ===")
    header = ("sigma", "acc_overall", "acc_excl_deleted", "mia_overall", "mia_deleted", "mia_retained", "B_theory")
    print("{:>10} {:>20} {:>22} {:>20} {:>20} {:>21} {:>12}".format(*header))

    for r in results:
        sigma_str = "0" if r["sigma"] == 0 else f"{r['sigma']:.1e}"
        b_str = "None" if r["B_theory"] is None else f"{r['B_theory']:.2e}"

        print("{:>10} {:>20} {:>22} {:>20} {:>20} {:>21} {:>12}".format(
            sigma_str,
            fmt_ms(r["acc_overall_m"], r["acc_overall_s"]),
            fmt_ms(r["acc_wo_m"], r["acc_wo_s"]),
            fmt_ms(r["mia_overall_m"], r["mia_overall_s"]),
            fmt_ms(r["mia_deleted_m"], r["mia_deleted_s"]),
            fmt_ms(r["mia_retained_m"], r["mia_retained_s"]),
            b_str
        ))


if __name__ == "__main__":
    main()
