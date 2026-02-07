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

def prepare_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load 20 Newsgroups data and split into train/test.

    Inputs:
      - test_size: fraction of data for the test split
      - random_state: seed for reproducibility

    Outputs:
      - train_texts: list of training documents
      - test_texts: list of test documents
      - train_labels: np.ndarray of shape (n_train,)
      - test_labels: np.ndarray of shape (n_test,)
    """
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target
    )

def build_tfidf(train_texts: list, test_texts: list):
    """
    Fit TF-IDF on training texts and transform both splits.

    Inputs:
      - train_texts: list of training documents
      - test_texts: list of test documents

    Outputs:
      - vectorizer: fitted TfidfVectorizer
      - X_train: sparse matrix (n_train x n_features)
      - X_test: sparse matrix (n_test x n_features)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=5,
        sublinear_tf=True,
        norm="l2"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

def train_ls_svm(X_train, y_train, C: float = 10.0):
    """
    Train a multi-class least-squares SVM via ridge regression.

    Inputs:
      - X_train: sparse matrix (n_samples x n_features)
      - y_train: np.ndarray of length n_samples
      - C: regularization parameter

    Outputs:
      - theta: np.ndarray of shape (n_classes x n_features)
    """
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)
    Y = np.zeros((n_samples, n_classes))
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver='auto')
    ridge.fit(X_train, Y)
    return ridge.coef_

def influence_removal_ls_svm(X_train, y_train, theta_orig,
                             removal_indices: np.ndarray, C: float = 10.0):
    """
    Remove influence of given samples via influence-function Hessian solve.

    Inputs:
      - X_train: sparse matrix (n_samples x n_features)
      - y_train: np.ndarray of length n_samples
      - theta_orig: np.ndarray (n_classes x n_features)
      - removal_indices: 1D array of sample indices to remove
      - C: regularization parameter

    Outputs:
      - theta_unlearn: np.ndarray (n_classes x n_features)
    """
    n_classes, n_features = theta_orig.shape
    grad_sum = np.zeros_like(theta_orig)
    for idx in removal_indices:
        x = X_train[idx].toarray().ravel()
        one_hot = np.zeros(n_classes); one_hot[y_train[idx]] = 1
        scores = theta_orig.dot(x)
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

def evaluate_accuracy(theta, vectorizer, docs: list, labels: np.ndarray):
    """
    Compute classification accuracy.

    Inputs:
      - theta: np.ndarray (n_classes x n_features)
      - vectorizer: fitted TfidfVectorizer
      - docs: list of str
      - labels: np.ndarray of length n_docs

    Outputs:
      - accuracy: float
    """
    X = vectorizer.transform(docs)
    scores = X @ theta.T
    preds = np.argmax(scores, axis=1)
    return accuracy_score(labels, preds)

def build_attack_features(P: np.ndarray, labels: np.ndarray):
    """
    Construct features for membership inference attack.

    Inputs:
      - P: softmax prob matrix (n_samples x n_classes)
      - labels: np.ndarray of length n_samples

    Outputs:
      - features: np.ndarray (n_samples x (n_classes + 3))
    """
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p = P[np.arange(len(labels)), labels].reshape(-1,1)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce_loss, gap])

def train_shadow_attack(theta_target, vectorizer, train_docs: list, train_labels: np.ndarray, C: float = 1.0):
    """
    Train a shadow-model-based membership inference attack.

    Inputs:
      - theta_target: np.ndarray (n_classes x n_features)
      - vectorizer: TfidfVectorizer
      - train_docs: list of training docs
      - train_labels: np.ndarray of training labels
      - C: regularization for shadow LS-SVM

    Outputs:
      - attack_clf: trained LogisticRegression attack model
    """
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels,
        test_size=0.5,
        random_state=0,
        stratify=train_labels
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

def train_shadow_attack_with_unlearning(theta_target, vectorizer, train_docs: list, train_labels: np.ndarray, C: float, class_to_unlearn: int):
    """
    Exactly the same as train_shadow_attack, but *inside* each shadow
    you also remove class `class_to_unlearn` and fine-tune—just like the real pipeline.
    """
    # split into shadow‐train vs holdout
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels,
        test_size=0.5, random_state=0, stratify=train_labels
    )
    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)

    # train LS-SVM on shadow-train
    theta_sh = train_ls_svm(Xs, s_lbls, C)

    # unlearn class_to_unlearn in the shadow model
    rem = np.where(s_lbls == class_to_unlearn)[0]
    theta_sh_un = influence_removal_ls_svm(Xs, s_lbls, theta_sh, rem, C)

    # fine-tune on remaining shadow-train
    keep = np.ones(len(s_lbls), bool)
    keep[rem] = False
    Xs_red, y_sred = Xs[keep], s_lbls[keep]
    classes_red = np.unique(y_sred)
    theta_init = theta_sh_un[classes_red]

    ft = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs',
        fit_intercept=False, warm_start=True,
        max_iter=5, random_state=42
    )
    ft.coef_ = theta_init.copy()
    ft.classes_ = classes_red
    ft.fit(Xs_red, y_sred)

    # reconstruct full K×d shadow weight matrix (freeze class_to_unlearn row=0)
    K, d = theta_sh.shape
    theta_sh_final = np.zeros((K, d))
    for i, c in enumerate(classes_red):
        theta_sh_final[c] = ft.coef_[i]

    # build attack dataset (same as train_shadow_attack)
    Ps = softmax(Xs @ theta_sh_final.T, axis=1)
    Ph = softmax(Xh @ theta_sh_final.T, axis=1)
    A_s, A_h = build_attack_features(Ps, s_lbls), build_attack_features(Ph, h_lbls)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))])

    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf

def compute_mia_auc(attack_clf, theta, vectorizer,
                    train_docs: list, train_labels: np.ndarray,
                    test_docs: list, test_labels: np.ndarray):
    """
    Compute ROC AUC for the membership inference attack.

    Inputs:
      - attack_clf: trained attack model
      - theta: np.ndarray (n_classes x n_features)
      - vectorizer: TfidfVectorizer
      - train_docs/train_labels: training split
      - test_docs/test_labels: test split

    Outputs:
      - auc: float ROC AUC
    """
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)
    Ptr = softmax(Xtr @ theta.T, axis=1)
    Pte = softmax(Xte @ theta.T, axis=1)
    A_tr = build_attack_features(Ptr, train_labels)
    A_te = build_attack_features(Pte, test_labels)
    X_att = np.vstack([A_tr, A_te])
    y_att = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))])
    return roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1])

def compute_class_mia_auc(attack_clf, theta, vectorizer, train_docs, train_labels,test_docs,  test_labels, cls: int):
    """
    Compute ROC-AUC of the attack for membership inference on class `cls` only.
    """
    train_idx = np.where(train_labels == cls)[0]
    test_idx = np.where(test_labels == cls)[0]
    docs_tr_k = [train_docs[i] for i in train_idx]
    docs_te_k = [test_docs[i] for i in test_idx]
    labs_tr_k = train_labels[train_idx]
    labs_te_k = test_labels[test_idx]
    return compute_mia_auc(
        attack_clf,
        theta,
        vectorizer,
        docs_tr_k, labs_tr_k,
        docs_te_k, labs_te_k
    )

from sklearn.preprocessing import normalize

def logistic_loss_and_grad(theta_weights, X, y, classes, C):
    """
    Multinomial logistic loss (1-vs-rest softmax) on reduced set + L2 (1/C).
    theta_weights: (K_active x d) numpy array for the active classes in `classes`
    X: sparse CSR (n x d) or dense (n x d)
    y: (n,) labels from the global label space
    classes: sorted unique labels present in this reduced set (length K_active)
    C: same C you use elsewhere (L2 strength is 1/C)
    """
    # map y to 0..K_active-1 for rows that are present
    class_to_row = {c:i for i,c in enumerate(classes)}
    y_local = np.array([class_to_row[yi] for yi in y])

    # scores and probabilities
    if hasattr(X, "tocsr"):
        S = X @ theta_weights.T           # (n x K)
    else:
        S = X.dot(theta_weights.T)
    S = S - S.max(axis=1, keepdims=True)  # numerical stability
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)

    # loss
    n = X.shape[0]
    row_idx = np.arange(n)
    loglik = -np.log(P[row_idx, y_local] + 1e-12).sum()
    l2 = 0.5 * (theta_weights**2).sum() / C
    loss = loglik + l2

    # gradient wrt theta (K x d)
    R = P
    R[row_idx, y_local] -= 1.0
    if hasattr(X, "tocsr"):
        grad = (R.T @ X)                  # (K x d), stays sparse*dense -> dense
        grad = np.asarray(grad)           # ensure ndarray
    else:
        grad = R.T.dot(X)
    grad /= 1.0                           # (no averaging; matches loss scaling)
    grad += theta_weights / C
    return float(loss), grad

def cosine(u, v, eps=1e-12):
    uu = np.linalg.norm(u); vv = np.linalg.norm(v)
    if uu < eps or vv < eps: return 0.0
    return float(np.dot(u, v) / (uu*vv))

def vectorize_theta(theta_matrix):
    """Flatten K x d weights to a single vector."""
    return theta_matrix.ravel()

def fine_tune_with_logging(theta_init, X_red, y_red, classes_red, C,
                           max_bursts=30, inner_max_iter=20,
                           tol_grad=1e-3, tol_rel=1e-6, verbose=False):
    """
    Repeated warm-start LBFGS bursts with gradient/loss logging and stopping on:
      (i) ||grad||_2 < tol_grad  OR  (ii) relative loss improvement < tol_rel
    Returns theta_final, loss_hist, grad_hist, converged.
    """
    # initialize a multinomial logistic model on the active classes
    ft = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs', multi_class='multinomial',
        fit_intercept=False, warm_start=True, max_iter=inner_max_iter,
        random_state=42
    )
    ft.classes_ = classes_red
    ft.coef_ = theta_init.copy()

    loss_hist, grad_hist = [], []
    prev_loss = None
    converged = False

    for b in range(max_bursts+1):
        # evaluate current loss/grad
        loss, grad = logistic_loss_and_grad(ft.coef_, X_red, y_red, classes_red, C)
        gnorm = float(np.linalg.norm(grad))
        loss_hist.append(loss); grad_hist.append(gnorm)

        if verbose:
            rel = float('nan') if prev_loss is None else (prev_loss - loss)/max(prev_loss,1.0)
            print(f"[FT step {b:02d}] loss={loss:.4f} | grad_norm={gnorm:.3e} | rel_impr={rel if not np.isnan(rel) else float('nan'):.3e}")

        # stopping tests (after first evaluation)
        if prev_loss is not None:
            rel_impr = (prev_loss - loss)/max(prev_loss, 1.0)
            if gnorm < tol_grad or rel_impr < tol_rel:
                converged = True
                break
        prev_loss = loss

        # one more LBFGS burst
        ft.fit(X_red, y_red)

    return ft.coef_.copy(), loss_hist, grad_hist, converged

def run_unlearning_once(train_docs, test_docs, y_train, y_test, vectorizer, C, class_to_unlearn, seed=0,
                        ft_max_bursts=30, ft_inner=20, tol_grad=1e-3, tol_rel=1e-6, verbose=False):
    """
    Your full pipeline for a single (C, seed): returns dict with theta_final,
    gradient history, last gradient vector, predictions, and convergence flag.
    """
    random.seed(seed); np.random.seed(seed)

    # Build full LS-SVM baseline
    X_train = vectorizer.transform(train_docs)
    X_test  = vectorizer.transform(test_docs)
    theta_orig = train_ls_svm(X_train, y_train, C)

    # Unlearn indices
    removal_indices = np.where(y_train == class_to_unlearn)[0]
    theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, removal_indices, C)

    # Reduced set
    keep = np.ones(len(y_train), bool); keep[removal_indices] = False
    X_red, y_red = X_train[keep], y_train[keep]
    classes_red = np.unique(y_red)
    theta_init = theta_un[classes_red, :]

    # Fine-tune with logging
    theta_ft, loss_hist, grad_hist, conv = fine_tune_with_logging(
        theta_init, X_red, y_red, classes_red, C,
        max_bursts=ft_max_bursts, inner_max_iter=ft_inner,
        tol_grad=tol_grad, tol_rel=tol_rel, verbose=verbose
    )

    # Reconstruct full Kxd with zeroed unlearned row
    K, d = theta_orig.shape
    theta_final = np.zeros_like(theta_orig)
    for i, c in enumerate(classes_red):
        theta_final[c] = theta_ft[i]

    # Predictions (agreement checks)
    scores = X_test @ theta_final.T
    preds  = np.argmax(scores, axis=1)

    # Final gradient vector on reduced set (actual values)
    _, grad_final = logistic_loss_and_grad(theta_ft, X_red, y_red, classes_red, C)

    return {
        "theta_final": theta_final,
        "theta_ft_active": theta_ft,           # K_active x d
        "classes_red": classes_red,
        "loss_hist": np.array(loss_hist),
        "grad_hist": np.array(grad_hist),
        "converged": conv,
        "preds_test": preds,
        "grad_final_active": grad_final,       # K_active x d
    }

def analyze_across_C(results_by_C, vectorizer, C_list, report_topk=10):
    """
    Print/report:
      - pairwise cosine of θ_final across C
      - prediction agreement across C
      - gradient histograms & top-k gradient coords (by magnitude) at final step
      - gradient stability between last two bursts (cosine)
    """
    # --- θ stability ---
    thetas = [vectorize_theta(results_by_C[C]["theta_final"]) for C in C_list]
    print("\n[Parameter stability: cosine(θ_C, θ_C')]")
    for i, Ci in enumerate(C_list):
        row = []
        for j, Cj in enumerate(C_list):
            row.append(f"{cosine(thetas[i], thetas[j]):.4f}")
        print(f"C={Ci:<6} : " + "  ".join(row))

    # --- prediction agreement ---
    preds = [results_by_C[C]["preds_test"] for C in C_list]
    print("\n[Prediction agreement on test set]")
    for i, Ci in enumerate(C_list):
        row = []
        for j, Cj in enumerate(C_list):
            agree = (preds[i] == preds[j]).mean()
            row.append(f"{agree:.4f}")
        print(f"C={Ci:<6} : " + "  ".join(row))

    # --- gradient diagnostics at final step ---
    feat_names = vectorizer.get_feature_names_out()
    print("\n[Final gradient vector stats per C (on reduced set, active classes)]")
    for C in C_list:
        G = results_by_C[C]["grad_final_active"]          # (K_active x d)
        gvec = G.ravel()
        print(f"  C={C:<6} ||g||2={np.linalg.norm(gvec):.3e}  ||g||inf={np.max(np.abs(gvec)):.3e}")

        # top-k coordinates (show class,row,feature)
        idx = np.argpartition(np.abs(gvec), -report_topk)[-report_topk:]
        idx = idx[np.argsort(-np.abs(gvec[idx]))]
        print("   top-|g_i| features:")
        Kact, d = G.shape
        for k in idx:
            r = k // d
            f = k % d
            print(f"     class_row={results_by_C[C]['classes_red'][r]}  feat='{feat_names[f]}'  g={G[r,f]:+.3e}")
        print()

    # --- gradient stability across bursts (cosine between last two) ---
    print("[Gradient stability across the last two bursts]")
    for C in C_list:
        hist = results_by_C[C]["grad_hist"]
        # we logged only norms; show end norms and burst count
        print(f"  C={C:<6} bursts={len(hist)-1}  final_norm={hist[-1]:.3e}  prev_norm={hist[-2] if len(hist)>1 else float('nan'):.3e}")

def main():
    # --- data & tf-idf ---
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # choose the same class_to_unlearn across C so releases are comparable
    class_to_unlearn = random.randint(0, int(y_train.max()))

    C_list = [0.00000000001, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]

    results_by_C = {}
    for C in C_list:
        print(f"\n=== Running unlearning with C={C} ===")
        res = run_unlearning_once(
            train_docs, test_docs, y_train, y_test, vectorizer, C, class_to_unlearn,
            seed=0, ft_max_bursts=30, ft_inner=20, tol_grad=1e-3, tol_rel=1e-6, verbose=True
        )
        results_by_C[C] = res
        print(f"   converged={res['converged']}  final grad norm={res['grad_hist'][-1]:.3e}")

    # analyze stability across those C values
    analyze_across_C(results_by_C, vectorizer, C_list, report_topk=10)

if __name__ == "__main__":
    main()