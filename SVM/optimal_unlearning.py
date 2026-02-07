import numpy as np
import random
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ---------------------------- Data prep ----------------------------
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

# ---------------------------- LS-SVM (Ridge) ----------------------------
def train_ls_svm(X_train, y_train, C: float = 10.0):
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)
    Y = np.zeros((n_samples, n_classes))
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver='auto')
    ridge.fit(X_train, Y)
    return ridge.coef_  # (n_classes x n_features)

def evaluate_accuracy(theta, vectorizer, docs: list, labels: np.ndarray):
    X = vectorizer.transform(docs)
    scores = X @ theta.T
    preds = np.argmax(scores, axis=1)
    return accuracy_score(labels, preds)

# ---------------------------- MIA utilities ----------------------------
def build_attack_features(P: np.ndarray, labels: np.ndarray):
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p  = P[np.arange(len(labels)), labels].reshape(-1,1)
    ce_loss = -np.log(true_p + 1e-12)
    top2    = np.sort(P, axis=1)[:, -2:]
    gap     = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce_loss, gap])

def train_shadow_attack(theta_target, vectorizer, train_docs: list, train_labels: np.ndarray, C: float = 1.0):
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels,
        test_size=0.5,
        random_state=0,
        stratify=train_labels
    )
    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)
    theta_sh = train_ls_svm(Xs, s_lbls, C)
    Ps = softmax(Xs @ theta_sh.T, axis=1)
    Ph = softmax(Xh @ theta_sh.T, axis=1)
    A_s = build_attack_features(Ps, s_lbls)
    A_h = build_attack_features(Ph, h_lbls)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))])
    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf

def train_shadow_attack_with_unlearning_smw(theta_target, vectorizer, train_docs: list,
                                            train_labels: np.ndarray, C: float, class_to_unlearn: int,
                                            cg_tol: float = 1e-3, cg_max_iter: int = 300, verbose: bool = False):
    """Shadow attack where the shadow model performs the *same* SMW class unlearning."""
    # split into shadow‐train vs holdout
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels,
        test_size=0.5, random_state=0, stratify=train_labels
    )
    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)

    # train LS-SVM on shadow-train
    theta_sh = train_ls_svm(Xs, s_lbls, C)

    # SMW exact reduced-set downdate for the shadow (remove the target class)
    rem = np.where(s_lbls == class_to_unlearn)[0]
    theta_sh_final, _ = smw_downdate_ls_svm(
        Xs, s_lbls, theta_sh, rem, C,
        cg_tol=cg_tol, cg_max_iter=cg_max_iter, verbose=verbose
    )

    # build attack dataset
    Ps = softmax(Xs @ theta_sh_final.T, axis=1)
    Ph = softmax(Xh @ theta_sh_final.T, axis=1)
    A_s = build_attack_features(Ps, s_lbls)
    A_h = build_attack_features(Ph, h_lbls)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))])

    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf

def compute_mia_auc(attack_clf, theta, vectorizer,
                    train_docs: list, train_labels: np.ndarray,
                    test_docs: list, test_labels: np.ndarray):
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)
    Ptr = softmax(Xtr @ theta.T, axis=1)
    Pte = softmax(Xte @ theta.T, axis=1)
    A_tr = build_attack_features(Ptr, train_labels)
    A_te = build_attack_features(Pte,  test_labels)
    X_att = np.vstack([A_tr, A_te])
    y_att = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))])
    return roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1])

def compute_class_mia_auc(attack_clf, theta, vectorizer,
                          train_docs, train_labels, test_docs, test_labels, cls: int):
    tr_idx = np.where(train_labels == cls)[0]
    te_idx = np.where(test_labels  == cls)[0]
    docs_tr = [train_docs[i] for i in tr_idx]
    docs_te = [test_docs[i]  for i in te_idx]
    y_tr = train_labels[tr_idx]
    y_te = test_labels[te_idx]
    return compute_mia_auc(attack_clf, theta, vectorizer, docs_tr, y_tr, docs_te, y_te)

# ---------------------------- SMW exact downdate ----------------------------
def _one_hot(labels: np.ndarray, n_classes: int):
    Y = np.zeros((len(labels), n_classes))
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

def _ridge_loss_and_grad_norm(X, Y, W, alpha: float):
    # Loss = ||XW - Y||_F^2 + alpha ||W||_F^2
    R = X @ W.T - Y                     # (n x K)
    loss = float(np.sum(R*R) + alpha * np.sum(W*W))
    # Grad = 2(X^T R + alpha W)  (K x d)
    G = 2.0 * ( (X.T @ R).T + alpha * W )
    grad_norm = float(np.linalg.norm(G))
    return loss, grad_norm

def smw_downdate_ls_svm(X_train, y_train, theta_orig,
                        removal_indices: np.ndarray, C: float = 10.0,
                        cg_tol: float = 1e-3, cg_max_iter: int = 300,
                        verbose: bool = True):
    """
    Exact reduced-set ridge solution via Sherman–Morrison–Woodbury.

    We solve for W' that minimizes ||X_keep W - Y_keep||_F^2 + alpha ||W||_F^2
    using the identity for (A - UU^T)^{-1}, with A = alpha I + X^T X and U = X_R^T.

    Returns:
      - theta_new: (n_classes x n_features)
      - diag: dict with loss/grad diagnostics on the reduced set.
    """
    alpha = 1.0 / C
    W = theta_orig.copy()               # (K x d)
    K, d = W.shape

    keep_mask = np.ones(X_train.shape[0], bool)
    keep_mask[removal_indices] = False

    X_keep = X_train[keep_mask]        # (n_keep x d)
    y_keep = y_train[keep_mask]
    Y_keep = _one_hot(y_keep, K)       # (n_keep x K)

    X_R = X_train[removal_indices]     # (k x d)
    y_R = y_train[removal_indices]
    k = X_R.shape[0]

    if verbose:
        print(f"[SMW] Removing k={k} samples; feature dim d={d}; classes K={K}")

    # Linear operator for A v = alpha v + X^T (X v) built on FULL X (as required by SMW)
    def A_matvec(v):
        return alpha * v + X_train.T @ (X_train @ v)

    A_op = LinearOperator((d, d), matvec=A_matvec, dtype=np.float64)

    # --- Step 1: Build G = U^T A^{-1} U (k x k), and prepare U Y_R and U T RHS
    # Each u_j is a dense vector of length d (column j of U = X_R^T)
    # We'll assemble G column-by-column: G[:, j] = X_R @ z_j  where A z_j = u_j
    G = np.zeros((k, k), dtype=np.float64)

    # Precompute S = U^T W = X_R @ W^T  (k x K)
    S = (X_R @ W.T).toarray() if hasattr(X_R @ W.T, "toarray") else (X_R @ W.T)

    # Prepare U Y_R = X_R^T @ Y_R (d x K)
    Y_R_mat = _one_hot(y_R, K)         # (k x K)
    UY = X_R.T @ Y_R_mat               # (d x K), sparse @ dense -> dense

    # Solve V1 = A^{-1} (U Y_R)  (d x K), column-wise CG
    V1 = np.zeros((d, K), dtype=np.float64)
    for c in range(K):
        rhs = UY[:, c]
        z_c, info = cg(A_op, rhs, atol=0.0, tol=cg_tol, maxiter=cg_max_iter)
        if info != 0 and verbose:
            print(f"[SMW] CG for V1 col {c} ended with info={info}")
        V1[:, c] = z_c

    # Build G by k CG solves; also cheap because each RHS is u_j (a single doc vector)
    for j in range(k):
        u_j = X_R[j].toarray().ravel()             # (d,)
        z_j, info = cg(A_op, u_j, atol=0.0, tol=cg_tol, maxiter=cg_max_iter)
        if info != 0 and verbose:
            print(f"[SMW] CG for G col {j} ended with info={info}")
        # g_col = U^T z_j = X_R @ z_j  (k,)
        g_col = (X_R @ z_j).ravel()
        G[:, j] = g_col

    # --- Step 2: Assemble M = I - G  (k x k) and solve T = M^{-1} (S - G Y_R)
    M = np.eye(k, dtype=np.float64) - G
    RHS_T = S - G @ Y_R_mat           # (k x K)
    # Solve kxk linear system for each class column
    T = np.linalg.solve(M, RHS_T)     # (k x K)

    # --- Step 3: Compute V2 = A^{-1} (U T) with K CG solves, where U T = X_R^T @ T (d x K)
    UT = X_R.T @ T                    # (d x K)
    V2 = np.zeros((d, K), dtype=np.float64)
    for c in range(K):
        rhs = UT[:, c]
        v2_c, info = cg(A_op, rhs, atol=0.0, tol=cg_tol, maxiter=cg_max_iter)
        if info != 0 and verbose:
            print(f"[SMW] CG for V2 col {c} ended with info={info}")
        V2[:, c] = v2_c

    # --- Step 4: Final exact reduced-set weights:
    #     W' = W - (A^{-1} U Y_R)^T + (A^{-1} U T)^T  = W - V1.T + V2.T
    theta_new = W - V1.T + V2.T       # (K x d)

    # --- Diagnostics on the reduced set (loss/grad/KKT)
    loss_red, grad_norm_red = _ridge_loss_and_grad_norm(X_keep, Y_keep, theta_new, alpha)

    # KKT residual: R_c = alpha w_c + X_keep^T (X_keep w_c) - X_keep^T y_c  (stacked Fro norm)
    # (this is proportional to the gradient; should be small if optimal)
    kkt_resid = 0.0
    for c in range(K):
        w_c = theta_new[c]                                   # (d,)
        r_c = alpha * w_c + X_keep.T @ (X_keep @ w_c) - X_keep.T @ Y_keep[:, c]
        kkt_resid += float(np.dot(r_c, r_c))
    kkt_resid = float(np.sqrt(kkt_resid))

    diag = {
        "k_removed": int(k),
        "loss_reduced_set": loss_red,
        "grad_norm_reduced_set": grad_norm_red,
        "kkt_residual_norm": kkt_resid
    }
    if verbose:
        print(f"[SMW] Reduced-set loss={loss_red:.4f} | grad_norm={grad_norm_red:.3e} | KKT_resid={kkt_resid:.3e}")

    return theta_new, diag

# ---------------------------- Main ----------------------------
def main():
    # ---- data & features
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # pick class to unlearn
    class_to_unlearn = random.randint(0, int(np.max(y_train)))
    C = 10.0  # you can also try 100.0 per your earlier sweeps

    # ---- baseline model & attack
    theta_orig = train_ls_svm(X_train, y_train, C)
    acc_before = evaluate_accuracy(theta_orig, vectorizer, test_docs, y_test)
    attack_clf_before = train_shadow_attack(theta_orig, vectorizer, train_docs, y_train, C)
    auc_before = compute_mia_auc(attack_clf_before, theta_orig, vectorizer, train_docs, y_train, test_docs, y_test)
    auc_before_cls = compute_class_mia_auc(attack_clf_before, theta_orig, vectorizer, train_docs, y_train,
                                           test_docs, y_test, cls=class_to_unlearn)

    # ---- exact reduced-set downdate via SMW (remove ALL examples of the chosen class)
    removal_indices = np.where(y_train == class_to_unlearn)[0]
    theta_smw, smw_diag = smw_downdate_ls_svm(
        X_train, y_train, theta_orig, removal_indices, C,
        cg_tol=1e-4,  # was 1e-3
        cg_max_iter=600,  # was 300
        verbose=True
    )

    # ---- evaluate after unlearning
    # retrain a shadow attack that mirrors the unlearning (uses SMW inside)
    attack_clf_after = train_shadow_attack_with_unlearning_smw(
        theta_smw, vectorizer, train_docs, y_train, C,
        class_to_unlearn, cg_tol=1e-3, cg_max_iter=300, verbose=False
    )

    acc_after_all = evaluate_accuracy(theta_smw, vectorizer, test_docs, y_test)
    auc_after_all = compute_mia_auc(attack_clf_after, theta_smw, vectorizer, train_docs, y_train, test_docs, y_test)
    auc_after_cls = compute_class_mia_auc(attack_clf_after, theta_smw, vectorizer, train_docs, y_train,
                                          test_docs, y_test, cls=class_to_unlearn)

    # accuracy excluding the unlearned class in test set
    keep_idx_test = np.where(y_test != class_to_unlearn)[0]
    test_docs_wo = [test_docs[i] for i in keep_idx_test]
    y_test_wo = y_test[keep_idx_test]
    acc_after_wo = evaluate_accuracy(theta_smw, vectorizer, test_docs_wo, y_test_wo)

    # ---- print results
    print(f"\nClass label unlearned:                        {class_to_unlearn}")
    print(f"Accuracy before unlearning:                   {acc_before:.4f}")
    print(f"MIA AUC before unlearning:                    {auc_before:.4f}")
    print(f"MIA AUC before unlearning (class {class_to_unlearn}): {auc_before_cls:.4f}\n")

    print(f"Accuracy after unlearning (all classes):      {acc_after_all:.4f}")
    print(f"Accuracy after unlearning (wo unlearned):     {acc_after_wo:.4f}")
    print(f"Overall MIA AUC after unlearning:             {auc_after_all:.4f}")
    print(f"MIA AUC after unlearning (class {class_to_unlearn}):  {auc_after_cls:.4f}\n")

    print("[SMW convergence diagnostics]")
    print(f"  k removed:                  {smw_diag['k_removed']}")
    print(f"  reduced-set loss:          {smw_diag['loss_reduced_set']:.4f}")
    print(f"  reduced-set grad norm:     {smw_diag['grad_norm_reduced_set']:.3e}")
    print(f"  KKT residual norm:         {smw_diag['kkt_residual_norm']:.3e}")

if __name__ == "__main__":
    main()