import numpy as np
import random
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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
    X_test  = vectorizer.transform(test_texts)
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
    classes_present = np.unique(y_train)
    n_classes = len(classes_present)

    Y_onehot = np.zeros((n_samples, n_classes))
    for class_index, class_id in enumerate(classes_present):
        Y_onehot[y_train == class_id, class_index] = 1

    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver='auto')
    ridge.fit(X_train, Y_onehot)
    return ridge.coef_

def influence_removal_ls_svm(X_train, y_train, theta_original,
                             removal_indices: np.ndarray, C: float = 1.0):
    """
    Remove influence of given samples via influence-function Hessian solve.

    Inputs:
      - X_train: sparse matrix (n_samples x n_features)
      - y_train: np.ndarray of length n_samples
      - theta_original: np.ndarray (n_classes x n_features)
      - removal_indices: 1D array of sample indices to remove
      - C: regularization parameter

    Outputs:
      - theta_unlearned: np.ndarray (n_classes x n_features)
    """
    n_classes, n_features = theta_original.shape

    # accumulate Δg over removed points
    grad_sum = np.zeros_like(theta_original)
    for idx in removal_indices:
        x_i = X_train[idx].toarray().ravel()
        one_hot = np.zeros(n_classes); one_hot[y_train[idx]] = 1
        scores = theta_original.dot(x_i)
        error  = scores - one_hot
        grad_sum += C * np.outer(error, x_i)

    # Hessian-vector: H v = v + C Xᵀ (X v)
    def hess_matvec(vec):
        Xv = X_train.dot(vec)
        return vec + C * (X_train.T.dot(Xv))

    H_op = LinearOperator((n_features, n_features), matvec=hess_matvec)

    theta_unlearned = np.zeros_like(theta_original)
    for class_row in range(n_classes):
        v_solution, _ = cg(H_op, grad_sum[class_row])  # default CG settings
        theta_unlearned[class_row] = theta_original[class_row] - v_solution

    return theta_unlearned

def evaluate_accuracy(theta_weights, vectorizer, docs: list, labels: np.ndarray):
    """
    Compute classification accuracy.

    Inputs:
      - theta_weights: np.ndarray (n_classes x n_features)
      - vectorizer: fitted TfidfVectorizer
      - docs: list of str
      - labels: np.ndarray of length n_docs

    Outputs:
      - accuracy: float
    """
    X = vectorizer.transform(docs)
    scores = X @ theta_weights.T
    preds = np.argmax(scores, axis=1)
    return accuracy_score(labels, preds)

def build_attack_features(prob_matrix: np.ndarray, labels: np.ndarray):
    """
    Construct features for membership inference attack.

    Inputs:
      - prob_matrix: softmax prob matrix (n_samples x n_classes)
      - labels: np.ndarray of length n_samples

    Outputs:
      - features: np.ndarray (n_samples x (n_classes + 3))
    """
    entropy = -np.sum(prob_matrix * np.log(prob_matrix + 1e-12), axis=1, keepdims=True)
    true_probs  = prob_matrix[np.arange(len(labels)), labels].reshape(-1, 1)
    ce_loss = -np.log(true_probs + 1e-12)
    top2    = np.sort(prob_matrix, axis=1)[:, -2:]
    margin  = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    return np.hstack([prob_matrix, entropy, ce_loss, margin])

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
    s_docs, h_docs, s_labels, h_labels = train_test_split(
        train_docs, train_labels,
        test_size=0.5,
        random_state=0,
        stratify=train_labels
    )
    X_shadow_train, X_shadow_holdout = vectorizer.transform(s_docs), vectorizer.transform(h_docs)
    theta_shadow = train_ls_svm(X_shadow_train, s_labels, C)

    P_in  = softmax(X_shadow_train @ theta_shadow.T, axis=1)
    P_out = softmax(X_shadow_holdout @ theta_shadow.T, axis=1)

    A_in  = build_attack_features(P_in,  s_labels)
    A_out = build_attack_features(P_out, h_labels)
    X_attack = np.vstack([A_in, A_out])
    y_attack = np.concatenate([np.ones(len(s_labels)), np.zeros(len(h_labels))])

    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_attack, y_attack)
    return attack_clf

def train_shadow_attack_with_unlearning(theta_target, vectorizer, train_docs: list, train_labels: np.ndarray, C: float, mask_cls: int):
    """
    Same as train_shadow_attack, but inside each shadow model we also
    remove class `mask_cls` and fine-tune—mirroring the real pipeline.
    """
    s_docs, h_docs, s_labels, h_labels = train_test_split(
        train_docs, train_labels,
        test_size=0.5, random_state=0, stratify=train_labels
    )
    X_shadow_train, X_shadow_holdout = vectorizer.transform(s_docs), vectorizer.transform(h_docs)

    # train LS-SVM on shadow-train
    theta_shadow = train_ls_svm(X_shadow_train, s_labels, C)

    # unlearn mask_cls in the shadow model
    removal_idx_shadow = np.where(s_labels == mask_cls)[0]
    theta_shadow_un = influence_removal_ls_svm(X_shadow_train, s_labels, theta_shadow, removal_idx_shadow, C)

    # fine-tune on remaining shadow-train
    keep_mask_shadow = np.ones(len(s_labels), dtype=bool)
    keep_mask_shadow[removal_idx_shadow] = False
    X_shadow_reduced, y_shadow_reduced = X_shadow_train[keep_mask_shadow], s_labels[keep_mask_shadow]
    classes_remaining = np.unique(y_shadow_reduced)
    theta_init_reduced = theta_shadow_un[classes_remaining]

    ft_shadow = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs',
        fit_intercept=False, warm_start=True,
        max_iter=5, random_state=42
    )
    ft_shadow.coef_    = theta_init_reduced.copy()
    ft_shadow.classes_ = classes_remaining
    ft_shadow.fit(X_shadow_reduced, y_shadow_reduced)

    # reconstruct full weight matrix (zero row for removed class)
    n_classes, n_features = theta_shadow.shape
    theta_shadow_final = np.zeros((n_classes, n_features))
    for row_idx, class_id in enumerate(classes_remaining):
        theta_shadow_final[class_id] = ft_shadow.coef_[row_idx]

    # build attack dataset
    P_in  = softmax(X_shadow_train  @ theta_shadow_final.T, axis=1)
    P_out = softmax(X_shadow_holdout @ theta_shadow_final.T, axis=1)

    A_in  = build_attack_features(P_in,  s_labels)
    A_out = build_attack_features(P_out, h_labels)
    X_attack = np.vstack([A_in, A_out])
    y_attack = np.concatenate([np.ones(len(s_labels)), np.zeros(len(h_labels))])

    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_attack, y_attack)
    return attack_clf

def compute_mia_auc(attack_clf, theta_weights, vectorizer,
                    train_docs: list, train_labels: np.ndarray,
                    test_docs: list, test_labels: np.ndarray):
    """
    Compute ROC AUC for the membership inference attack.
    """
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)
    Ptr = softmax(Xtr @ theta_weights.T, axis=1)
    Pte = softmax(Xte @ theta_weights.T, axis=1)
    A_tr = build_attack_features(Ptr, train_labels)
    A_te = build_attack_features(Pte, test_labels)
    X_attack = np.vstack([A_tr, A_te])
    y_attack = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))])
    return roc_auc_score(y_attack, attack_clf.predict_proba(X_attack)[:, 1])

def compute_class_mia_auc(attack_clf, theta_weights, vectorizer,
                          train_docs, train_labels, test_docs, test_labels, class_id: int):
    """
    Compute ROC-AUC of the attack for membership inference on class `class_id` only.
    """
    train_idx = np.where(train_labels == class_id)[0]
    test_idx  = np.where(test_labels  == class_id)[0]

    docs_tr_k = [train_docs[i] for i in train_idx]
    docs_te_k = [test_docs[i]  for i in test_idx]
    labs_tr_k = train_labels[train_idx]
    labs_te_k = test_labels[test_idx]

    return compute_mia_auc(
        attack_clf,
        theta_weights,
        vectorizer,
        docs_tr_k, labs_tr_k,
        docs_te_k, labs_te_k
    )

def fine_tune_to_convergence(theta_init_reduced, X_reduced, y_reduced, classes_remaining, C: float,
                             tol: float = 1e-4, max_iter: int = 1000):
    """
    Warm-start multinomial logistic fine-tune and let scikit run to convergence.

    Inputs:
      - theta_init_reduced: np.ndarray (n_remaining_classes x n_features)
      - X_reduced, y_reduced: reduced training split (removed class filtered out)
      - classes_remaining: sorted np.ndarray of remaining class ids
      - C: inverse L2 regularization
      - tol: LBFGS convergence tolerance
      - max_iter: iteration cap

    Outputs:
      - theta_reduced_final: np.ndarray (n_remaining_classes x n_features)
      - n_lbfgs_steps: int, number of LBFGS iterations scikit actually used
    """
    ft = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        fit_intercept=False, warm_start=True,
        max_iter=max_iter, tol=tol, random_state=42
    )
    ft.coef_    = theta_init_reduced.copy()
    ft.classes_ = np.array(classes_remaining)
    ft.fit(X_reduced, y_reduced)

    # n_iter_ may be array-like; take the max
    n_lbfgs_steps = int(np.max(np.atleast_1d(ft.n_iter_)))
    return ft.coef_.copy(), n_lbfgs_steps

def main():
    # data & features
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # choose class to unlearn & set regularization
    class_to_unlearn = random.randint(0, int(max(y_train)))
    C = 10.0

    # baseline model + attack
    theta_original = train_ls_svm(X_train, y_train, C)
    acc_before = evaluate_accuracy(theta_original, vectorizer, test_docs, y_test)

    attack_clf = train_shadow_attack(theta_original, vectorizer, train_docs, y_train, C)
    auc_before_overall = compute_mia_auc(attack_clf, theta_original, vectorizer, train_docs, y_train, test_docs, y_test)
    auc_before_class   = compute_class_mia_auc(attack_clf, theta_original, vectorizer, train_docs, y_train, test_docs, y_test, class_id=class_to_unlearn)

    # unlearning for the selected class
    removal_indices = np.where(y_train == class_to_unlearn)[0]
    theta_after_downdate = influence_removal_ls_svm(X_train, y_train, theta_original, removal_indices, C)

    # build reduced training set (removed class filtered out)
    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[removal_indices] = False
    X_reduced, y_reduced = X_train[keep_mask], y_train[keep_mask]

    # fine-tune to convergence on remaining classes; keep row order by original class ids
    classes_remaining = np.unique(y_reduced)
    theta_init_reduced = theta_after_downdate[classes_remaining, :]

    theta_reduced_final, n_lbfgs_steps = fine_tune_to_convergence(
        theta_init_reduced, X_reduced, y_reduced, classes_remaining, C,
        tol=1e-4, max_iter=1000
    )
    print(f"LBFGS iterations to convergence: {n_lbfgs_steps}")

    # reconstruct full (n_classes x n_features) matrix with a zero row for the unlearned class
    n_classes, n_features = theta_original.shape
    theta_final = np.zeros_like(theta_original)
    for row_idx, class_id in enumerate(classes_remaining):
        theta_final[class_id, :] = theta_reduced_final[row_idx, :]

    # retrain shadow attack to mirror unlearning pipeline
    attack_clf = train_shadow_attack_with_unlearning(theta_final, vectorizer, train_docs, y_train, C, class_to_unlearn)

    # metrics
    acc_after = evaluate_accuracy(theta_final, vectorizer, test_docs, y_test)
    auc_after_overall = compute_mia_auc(attack_clf, theta_final, vectorizer, train_docs, y_train, test_docs, y_test)
    auc_after_class   = compute_class_mia_auc(attack_clf, theta_final, vectorizer, train_docs, y_train, test_docs, y_test, class_id=class_to_unlearn)

    test_keep_idx = np.where(y_test != class_to_unlearn)[0]
    test_docs_wo = [test_docs[i] for i in test_keep_idx]
    y_test_wo = y_test[test_keep_idx]
    acc_after_wo_unlearned = evaluate_accuracy(theta_final, vectorizer, test_docs_wo, y_test_wo)

    # report
    print(f"Class label unlearned: {class_to_unlearn}")
    print(f"Accuracy before unlearning: {acc_before:.4f}")
    print(f"MIA AUC before unlearning: {auc_before_overall:.4f}")
    print(f"MIA AUC before unlearning (class {class_to_unlearn}): {auc_before_class:.4f}\n")
    print(f"Accuracy after unlearning: {acc_after_wo_unlearned:.4f}")
    print(f"Overall MIA AUC after unlearning: {auc_after_overall:.4f}")
    print(f"MIA AUC after unlearning (class {class_to_unlearn}): {auc_after_class:.4f}")

if __name__ == "__main__":
    main()