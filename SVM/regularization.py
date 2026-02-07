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
import scipy.sparse as sp

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

def _softmax_rows(Z):
    # Stable softmax row-wise
    Zmax = Z.max(axis=1, keepdims=True)
    EZ = np.exp(Z - Zmax)
    return EZ / EZ.sum(axis=1, keepdims=True)

def logistic_loss_and_grad(theta: np.ndarray, X, y: np.ndarray, classes: np.ndarray, C: float):
    """
    Multinomial logistic loss with L2 regularization scaled like your code.
    Objective:  J(θ) = (1/(2C)) * ||θ||_F^2 + (1/n) * Σ_i CE(softmax(X_i θ^T), y_i)
    Returns (loss, grad) where grad has shape like theta.
    """
    # map original labels -> local indices [0..K_present-1]
    class_to_local = {c: i for i, c in enumerate(classes)}
    y_local = np.array([class_to_local[yy] for yy in y], dtype=int)

    # logits: Z = X @ theta^T  -> shape (n, K_present)
    if sp.issparse(X):
        Z = X.dot(theta.T)
    else:
        Z = X @ theta.T

    # probabilities via softmax along classes
    P = softmax(Z, axis=1)

    # one-hot targets
    n_samples, Kp = P.shape
    Y = np.zeros((n_samples, Kp), dtype=P.dtype)
    Y[np.arange(n_samples), y_local] = 1.0

    # residuals
    R = P - Y

    # data loss: -sum log p_true
    p_true = P[np.arange(n_samples), y_local]
    data_loss = -np.log(p_true + 1e-12).sum()

    # L2 regularization
    reg_loss = 0.5 * (1.0 / C) * np.sum(theta * theta)
    loss = data_loss + reg_loss

    # gradient: R^T X  + (1/C) * theta
    if sp.issparse(X):
        # (d x n) · (n x K) = (d x K) -> transpose to (K x d)
        grad_w = (X.T.dot(R)).T
    else:
        # (K x n) · (n x d) = (K x d)
        grad_w = R.T @ X

    grad = grad_w + (1.0 / C) * theta
    return loss, grad

def run_finetune_with_monitor(X_red, y_red, classes_red, theta_init, C,
                              max_bursts=50, burst_iters=5,
                              grad_tol=1e-3, rel_impr_tol=1e-6, verbose=True):
    """
    Repeats LBFGS in short bursts with warm starts, tracking loss & grad norm.
    Stops when either grad_norm <= grad_tol or relative improvement <= rel_impr_tol.
    Returns: theta_final, history(dict), steps(int), converged(bool)
    """
    theta_curr = theta_init.copy()
    history = {"loss": [], "grad_norm": [], "rel_impr": []}

    # initial diagnostics (before any fine-tune step)
    loss0, grad0 = logistic_loss_and_grad(theta_curr, X_red, y_red, classes_red, C)
    gnorm0 = float(np.linalg.norm(grad0))
    history["loss"].append(loss0)
    history["grad_norm"].append(gnorm0)
    history["rel_impr"].append(np.nan)
    if verbose:
        print(f"[FT step 00] loss={loss0:.4f} | grad_norm={gnorm0:.3e} | rel_impr=nan")

    prev_loss = loss0
    steps = 0
    converged = False

    for t in range(1, max_bursts + 1):
        # one burst of LBFGS
        ft = LogisticRegression(
            penalty='l2', C=C, solver='lbfgs',
            fit_intercept=False, warm_start=True,
            max_iter=burst_iters, random_state=42
        )
        ft.coef_ = theta_curr.copy()
        ft.classes_ = classes_red
        ft.fit(X_red, y_red)

        theta_curr = ft.coef_.copy()

        # diagnostics
        loss_t, grad_t = logistic_loss_and_grad(theta_curr, X_red, y_red, classes_red, C)
        gnorm_t = float(np.linalg.norm(grad_t))
        rel_impr = max(0.0, (prev_loss - loss_t) / max(prev_loss, 1e-12))

        history["loss"].append(loss_t)
        history["grad_norm"].append(gnorm_t)
        history["rel_impr"].append(rel_impr)

        if verbose:
            print(f"[FT step {t:02d}] loss={loss_t:.4f} | grad_norm={gnorm_t:.3e} | rel_impr={rel_impr:.3e}")

        steps = t
        # stopping conditions
        if gnorm_t <= grad_tol or rel_impr <= rel_impr_tol:
            converged = True
            break
        prev_loss = loss_t

    return theta_curr, history, steps, converged

def main():
    # ===== data & tf-idf =====
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # fix a class to unlearn across all C to make curves comparable
    class_to_unlearn = random.randint(0, max(y_train))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    # C values to sweep (tweak as you like)
    C_grid = [0.00000000001, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]

    # storage for plots / table
    grad_histories = {}        # C -> (steps, converged, history dict)
    steps_to_conv = []         # (C, steps, converged)
    acc_wo_unlearned = []      # (C, accuracy)
    auc_unlearned = []         # (C, class AUC after)
    auc_overall = []           # (C, overall AUC after)

    # ===== loop over C =====
    for C in C_grid:
        print("\n" + "="*70)
        print(f"[C={C}] baseline training & attack")
        theta_orig = train_ls_svm(X_train, y_train, C)
        acc_before = evaluate_accuracy(theta_orig, vectorizer, test_docs, y_test)
        attack_clf_before = train_shadow_attack(theta_orig, vectorizer, train_docs, y_train, C)
        auc_before = compute_mia_auc(attack_clf_before, theta_orig, vectorizer, train_docs, y_train, test_docs,  y_test)
        auc_before_cls = compute_class_mia_auc(attack_clf_before, theta_orig, vectorizer, train_docs, y_train, test_docs,  y_test, cls=class_to_unlearn)

        print(f"[C={C}] acc_before={acc_before:.4f} | auc_before={auc_before:.4f} | auc_before_cls={auc_before_cls:.4f}")

        removal_indices = np.where(y_train == class_to_unlearn)[0]
        theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, removal_indices, C)

        # reduced set for fine-tune
        keep_mask = np.ones(len(y_train), bool)
        keep_mask[removal_indices] = False
        X_red, y_red = X_train[keep_mask], y_train[keep_mask]
        classes_red = np.unique(y_red)
        theta_init = theta_un[classes_red, :]

        # diagnostics right after Hessian update (before fine-tune)
        loss_h, grad_h = logistic_loss_and_grad(theta_init, X_red, y_red, classes_red, C)
        print(f"[C={C}] after Hessian downdate: loss={loss_h:.4f} | grad_norm={np.linalg.norm(grad_h):.3e}")

        # ===== monitored fine-tune =====
        theta_ft, hist, steps, converged = run_finetune_with_monitor(
            X_red, y_red, classes_red, theta_init, C,
            max_bursts=60, burst_iters=5, grad_tol=1e-3, rel_impr_tol=1e-6, verbose=True
        )
        grad_histories[C] = (steps, converged, hist)
        steps_to_conv.append((C, steps, converged))

        # reconstruct full theta with zero row for removed class
        n_classes, n_features = theta_orig.shape
        theta_final = np.zeros_like(theta_orig)
        for i, cls in enumerate(classes_red):
            theta_final[cls, :] = theta_ft[i]

        # retrain the attack on the *post-unlearning* model
        attack_clf_after = train_shadow_attack_with_unlearning(theta_final, vectorizer, train_docs, y_train, C, class_to_unlearn)

        # metrics after
        # overall acc
        acc_after_all = evaluate_accuracy(theta_final, vectorizer, test_docs, y_test)
        # acc excluding the unlearned class in test
        test_keep_idx = np.where(y_test != class_to_unlearn)[0]
        test_docs_wo = [test_docs[i] for i in test_keep_idx]
        y_test_wo = y_test[test_keep_idx]
        acc_after_wo = evaluate_accuracy(theta_final, vectorizer, test_docs_wo, y_test_wo)

        auc_after_all = compute_mia_auc(attack_clf_after, theta_final, vectorizer, train_docs, y_train, test_docs,  y_test)
        auc_after_cls = compute_class_mia_auc(attack_clf_after, theta_final, vectorizer, train_docs, y_train, test_docs,  y_test, cls=class_to_unlearn)

        acc_wo_unlearned.append((C, acc_after_wo))
        auc_overall.append((C, auc_after_all))
        auc_unlearned.append((C, auc_after_cls))

        print(f"[C={C}] acc_after_all={acc_after_all:.4f} | acc_after_wo_unlearned={acc_after_wo:.4f}")
        print(f"[C={C}] auc_after_all={auc_after_all:.4f} | auc_after_unlearned={auc_after_cls:.4f}")
        print(f"[C={C}] steps={steps} | converged={converged} | final_grad_norm={hist['grad_norm'][-1]:.3e} | final_rel_impr={hist['rel_impr'][-1]:.3e}")

    # ===== plots =====
    def _cat_plot(x_labels, y_values, ylabel, title):
        x_pos = np.arange(len(x_labels))
        plt.figure(figsize=(7.2, 4.4))
        plt.plot(x_pos, y_values, marker='o')
        plt.xticks(x_pos, x_labels, rotation=0)
        plt.xlabel("1/C")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _fmt_lam(val):
        # nice string for 1/C tick label
        if val == 0:
            return "0"
        if val >= 1000 or val < 1e-2:
            return f"{val:.0e}"           # scientific
        if val >= 1:
            return f"{val:.1f}"
        return f"{val:.3f}"

    # sort by lambda = 1/C (ascending) once
    rows_steps = sorted([(1.0/float(C), steps, conv) for (C, steps, conv) in steps_to_conv],
                        key=lambda t: t[0])
    lam = [r[0] for r in rows_steps]
    steps_sorted = [r[1] for r in rows_steps]
    conv_sorted  = [r[2] for r in rows_steps]
    lam_labels = [_fmt_lam(v) for v in lam]

    # 1) Steps to convergence vs 1/C (categorical x)
    _cat_plot(lam_labels, steps_sorted,
              ylabel="Bursts to convergence",
              title="Steps to convergence vs regularization strength (1/C)")

    # 2) Final gradient norm vs 1/C (categorical x, log y)
    rows_g = sorted([(1.0/float(C), float(grad_histories[C][2]["grad_norm"][-1]))
                     for C in grad_histories], key=lambda t: t[0])
    lam_g   = [r[0] for r in rows_g]
    g_final = [r[1] for r in rows_g]
    x_pos_g = np.arange(len(lam_g))
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(x_pos_g, g_final, marker='o')
    plt.yscale('log')
    plt.xticks(x_pos_g, [_fmt_lam(v) for v in lam_g])
    plt.xlabel("1/C")
    plt.ylabel("Final gradient norm (log)")
    plt.title("Final gradient norm vs regularization strength (1/C)")
    plt.tight_layout()
    plt.show()

    # 3) Utility (accuracy w/o unlearned class) vs 1/C
    rows_acc = sorted([(1.0/float(C), acc) for (C, acc) in acc_wo_unlearned], key=lambda t: t[0])
    lam_acc  = [r[0] for r in rows_acc]
    acc_vals = [r[1] for r in rows_acc]
    _cat_plot([_fmt_lam(v) for v in lam_acc], acc_vals,
              ylabel="Accuracy (excluding unlearned class)",
              title="Utility vs regularization strength (1/C)")

    # 4) Privacy AUCs vs 1/C
    rows_auc_un = sorted([(1.0/float(C), auc) for (C, auc) in auc_unlearned], key=lambda t: t[0])
    rows_auc_all= sorted([(1.0/float(C), auc) for (C, auc) in auc_overall],  key=lambda t: t[0])

    lam_u = [r[0] for r in rows_auc_un]
    auc_u = [r[1] for r in rows_auc_un]
    lam_o = [r[0] for r in rows_auc_all]
    auc_o = [r[1] for r in rows_auc_all]

    x_pos_u = np.arange(len(lam_u))
    x_pos_o = np.arange(len(lam_o))

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(x_pos_u, auc_u, marker='o', label="MIA AUC (unlearned class)")
    plt.plot(x_pos_o, auc_o, marker='o', label="MIA AUC (overall)")
    plt.xticks(x_pos_u, [_fmt_lam(v) for v in lam_u])
    plt.xlabel("1/C")
    plt.ylabel("AUC")
    plt.title("Privacy vs regularization strength (1/C)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()