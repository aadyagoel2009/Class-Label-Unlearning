import numpy as np
import random
from scipy.special import softmax
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
def train_logreg(X, y, C: float = 10.0, max_iter: int = 2000, seed: int = 42):
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
    For missing labels, true_p is set to 0 (maximally surprising), which is a
    conservative choice for the attacker and avoids out-of-bounds indexing.
    """
    n, k = P.shape
    # entropy
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    # true_p with alignment
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

    # stack features: [P | entropy | ce_loss | gap]
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
# Multi-shadow builders (pool attack data)
# -----------------------------
def build_attack_dataset_from_shadows(vectorizer, train_docs, train_labels, *,
                                      S: int = 10, C_shadow):
    """
    Build a pooled attack dataset from S shadow models (no unlearning inside shadows).
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
                                                      class_to_unlearn: int):
    """
    Build a pooled attack dataset from S shadow models that each perform
    the same UNLEARNING procedure as the target (here: refit-without class).
    """
    X_all, y_all = [], []
    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs = vectorizer.transform(s_docs)
        Xh = vectorizer.transform(h_docs)

        # Train shadow
        shadow = train_logreg(Xs, s_lbls, C=C_shadow, seed=seed)

        # Refit shadow *without* the target class (mirrors your current target unlearning)
        keep = (s_lbls != class_to_unlearn)
        Xs_red, y_sred = Xs[keep], s_lbls[keep]
        shadow_un = train_logreg(Xs_red, y_sred, C=C_shadow, seed=seed)

        # Features from unlearned shadow
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
# Unlearning on the TARGET model (LR)
# -----------------------------
def unlearn_class_via_refit_lr(X_train, y_train, class_to_unlearn: int,
                               C: float, seed: int = 42):
    """
    Remove a class by refitting multinomial LR on the reduced dataset (fresh fit).
    """
    keep_mask = (y_train != class_to_unlearn)
    X_red, y_red = X_train[keep_mask], y_train[keep_mask]
    lr_new = train_logreg(X_red, y_red, C=C, seed=seed)
    return lr_new

# -----------------------------
# Main
# -----------------------------
def main():
    # Data
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Pick a class to unlearn
    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    C = 10.0

    # ----- Baseline logistic model
    lr_orig = train_logreg(X_train, y_train, C=C, seed=42)
    acc_before = evaluate_accuracy(lr_orig, vectorizer, test_docs, y_test)

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

    # ----- Unlearn selected class by refitting LR on reduced set (fresh)
    lr_final = unlearn_class_via_refit_lr(X_train, y_train,
                                          class_to_unlearn, C=C, seed=42)

    # ===== Multi-shadow attacker (AFTER; shadows also unlearn) =====
    X_att_after, y_att_after = build_attack_dataset_from_shadows_with_unlearning(
        vectorizer, train_docs, y_train, S=10, C_shadow=C,
        class_to_unlearn=class_to_unlearn
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