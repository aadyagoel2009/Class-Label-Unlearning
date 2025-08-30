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

def train_ls_svm(X_train, y_train, C: float = 1.0):
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
                             removal_indices: np.ndarray, C: float = 1.0):
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

def summarize_vector(arr):
    if len(arr) == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {"n": int(len(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "p25": float(q25), "median": float(q50), "p75": float(q75),
            "min": float(np.min(arr)), "max": float(np.max(arr))}

def tfidf_distributions_for_class(vectorizer, docs, labels, class_id):
    """
    Summarize TF–IDF inputs (which do not change after unlearning).
    Returns per-doc sparsity and L2 norm stats for the selected class.
    """
    idx = np.where(labels == class_id)[0]
    if len(idx) == 0:
        return {"n_docs": 0}

    docs_k = [docs[i] for i in idx]
    Xk = vectorizer.transform(docs_k)
    nnz = np.diff(Xk.indptr)                                  # nonzeros per doc
    l2  = np.sqrt(Xk.multiply(Xk).sum(axis=1)).A1             # L2 norm per doc
    return {
        "n_docs": int(Xk.shape[0]),
        "nnz_per_doc": summarize_vector(nnz),
        "l2_norm_per_doc": summarize_vector(l2)
    }

def top_weighted_terms_for_class(vectorizer, theta, class_id, docs, labels, top_k=20):
    """
    Report top-|weight| terms for this class, with mean TF–IDF inside vs outside the class.
    """
    feature_names = vectorizer.get_feature_names_out()
    w = theta[class_id]                                       # weights for this class
    top_idx = np.argsort(np.abs(w))[-min(top_k, len(w)) :][::-1]

    idx_in  = np.where(labels == class_id)[0]
    idx_out = np.where(labels != class_id)[0]
    Xin  = vectorizer.transform([docs[i] for i in idx_in])    # inputs don’t change after unlearning
    Xout = vectorizer.transform([docs[i] for i in idx_out])
    mean_in  = np.asarray(Xin.mean(axis=0)).ravel()
    mean_out = np.asarray(Xout.mean(axis=0)).ravel()

    rows = []
    for j in top_idx:
        rows.append( (feature_names[j], float(w[j]), float(mean_in[j]), float(mean_out[j])) )
    return rows

def model_signal_distributions_for_class(theta, vectorizer, docs, labels, class_id):
    """
    Model-side signals that DO change after unlearning, evaluated on docs of class_id:
    - prob of that class, entropy, top-2 gap, and the logit for the removed class row.
    """
    idx = np.where(labels == class_id)[0]
    if len(idx) == 0:
        return {}
    X = vectorizer.transform([docs[i] for i in idx])
    Z = X @ theta.T
    P = softmax(Z, axis=1)

    true_prob = P[:, class_id]
    entropy   = -np.sum(P * np.log(P + 1e-12), axis=1)
    top2      = np.sort(P, axis=1)[:, -2:]
    gap       = top2[:, 1] - top2[:, 0]
    logit_k   = np.asarray(Z[:, class_id]).ravel()

    return {
        "true_prob":        summarize_vector(true_prob),
        "entropy":          summarize_vector(entropy),
        "top2_gap":         summarize_vector(gap),
        "logit_removedrow": summarize_vector(logit_k),
    }

def main():
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    class_to_unlearn = random.randint(0, max(y_train))
    C = 10.0

    # baseline training & attack
    theta_orig = train_ls_svm(X_train, y_train, C)
    acc_before = evaluate_accuracy(theta_orig, vectorizer, test_docs, y_test)
    attack_clf_before = train_shadow_attack(theta_orig, vectorizer, train_docs, y_train, C)
    auc_before = compute_mia_auc(attack_clf_before, theta_orig, vectorizer, train_docs, y_train, test_docs,  y_test)
    auc_before_cls = compute_class_mia_auc(attack_clf_before, theta_orig, vectorizer, train_docs, y_train, test_docs,  y_test, cls=class_to_unlearn)

    # unlearning class class_to_unlearn
    removal_indices = np.where(y_train == class_to_unlearn)[0]
    theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, removal_indices, C)

    # fine-tune on reduced set (freeze class_to_unlearn row)
    keep_mask = np.ones(len(y_train), bool)
    keep_mask[removal_indices] = False
    X_red, y_red = X_train[keep_mask], y_train[keep_mask]
    classes_red = np.unique(y_red)
    theta_init = theta_un[classes_red, :]

    ft = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs',
        fit_intercept=False, warm_start=True,
        max_iter=5, random_state=42
    )
    ft.coef_ = theta_init.copy()
    ft.classes_ = classes_red
    ft.fit(X_red, y_red)

    # reconstruct full theta with zero row for removed class
    n_classes, n_features = theta_orig.shape
    theta_final = np.zeros_like(theta_orig)
    for i, cls in enumerate(classes_red):
        theta_final[cls, :] = ft.coef_[i]

    # retrain the attack on new model
    attack_clf_after = train_shadow_attack_with_unlearning(theta_final, vectorizer, train_docs, y_train, C, class_to_unlearn)

    acc_after = evaluate_accuracy(theta_final, vectorizer, test_docs, y_test)
    auc_after = compute_mia_auc(attack_clf_after, theta_final, vectorizer, train_docs, y_train, test_docs,  y_test)
    auc_after_cls = compute_class_mia_auc(attack_clf_after, theta_final, vectorizer, train_docs, y_train, test_docs,  y_test, cls=class_to_unlearn)
    
    test_keep_idx = np.where(y_test != class_to_unlearn)[0]
    test_docs_wo = [test_docs[i] for i in test_keep_idx]
    y_test_wo = y_test[test_keep_idx]
    acc_after_wo_unlearned = evaluate_accuracy(theta_final, vectorizer, test_docs_wo, y_test_wo)

    print(f"Class label unlearned: {class_to_unlearn}")
    print(f"Accuracy before unlearning: {acc_before:.4f}")
    print(f"MIA AUC before unlearning: {auc_before:.4f}")
    print(f"MIA AUC before unlearning (class {class_to_unlearn}): {auc_before_cls:.4f}\n")
    print(f"Accuracy after unlearning: {acc_after_wo_unlearned:.4f}")
    print(f"Overall MIA AUC after unlearning: {auc_after:.4f}")
    print(f"MIA AUC after unlearning (class {class_to_unlearn}): {auc_after_cls:.4f}")

    tfidf_stats_before = tfidf_distributions_for_class(vectorizer, train_docs, y_train, class_to_unlearn)
    tfidf_stats_after  = tfidf_distributions_for_class(vectorizer, train_docs, y_train, class_to_unlearn)  # identical by design

    print("\n[TF–IDF input distributions for removed class (should be identical before/after)]")
    print(" before:", tfidf_stats_before)
    print("  after:", tfidf_stats_after)

    # Top weighted terms for the removed class (weights from BEFORE unlearning)
    top_terms = top_weighted_terms_for_class(vectorizer, theta_orig, class_to_unlearn, train_docs, y_train, top_k=15)
    print("\n[Top-|weight| terms for removed class (before unlearning)]")
    print(" term | weight | mean TF–IDF in-class | mean TF–IDF out-of-class")
    for term, wj, mi, mo in top_terms:
        print(f"  {term:30s} | {wj:+.3e} | {mi:.3e} | {mo:.3e}")

    # === Model-side signal distributions (these DO change) ===
    signals_train_before = model_signal_distributions_for_class(theta_orig,  vectorizer, train_docs, y_train, class_to_unlearn)
    signals_train_after  = model_signal_distributions_for_class(theta_final, vectorizer, train_docs, y_train, class_to_unlearn)
    signals_test_before  = model_signal_distributions_for_class(theta_orig,  vectorizer, test_docs,  y_test,  class_to_unlearn)
    signals_test_after   = model_signal_distributions_for_class(theta_final, vectorizer, test_docs,  y_test,  class_to_unlearn)

    print("\n[Model signals on removed-class TRAIN docs]")
    print(" before:", signals_train_before)
    print("  after:", signals_train_after)

    print("\n[Model signals on removed-class TEST docs]")
    print(" before:", signals_test_before)
    print("  after:", signals_test_after)


if __name__ == "__main__":
    main()