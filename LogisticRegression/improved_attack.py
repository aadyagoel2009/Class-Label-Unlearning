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
        min_df=5,          # <- lower to 2 if you want more overfit signal
        sublinear_tf=True,
        norm="l2"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

# -----------------------------
# Target model = Multinomial LR
# -----------------------------
def train_logreg(X, y, C: float = 10.0, max_iter: int = 2000, seed: int = 42, fit_intercept: bool=False):
    lr = LogisticRegression(
        penalty='l2',
        C=C,                       # try 50-100 for more overfit (higher MIA)
        solver='lbfgs',
        multi_class='multinomial',
        fit_intercept=fit_intercept,
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
# Probs & alignment helpers
# -----------------------------
def logits_from_model(lr_model, X):
    # works for both fit_intercept True/False
    Z = X @ lr_model.coef_.T
    if lr_model.fit_intercept:
        Z = Z + lr_model.intercept_[None, :]
    return Z

def probs_from_model(lr_model, X):
    Z = logits_from_model(lr_model, X)
    return softmax(Z, axis=1), lr_model.classes_

def align_probs(P, model_classes, union_classes):
    out = np.zeros((P.shape[0], len(union_classes)), dtype=P.dtype)
    m2u = {int(c): i for i, c in enumerate(union_classes)}
    for j, c in enumerate(model_classes):
        out[:, m2u[int(c)]] = P[:, j]
    return out

def extra_feats(P, labels, classes):
    """
    Build richer features from probabilities:
      - full P
      - true_p (aligned)
      - entropy
      - top-2 gap
    """
    n = P.shape[0]
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    idx = np.array([class_to_col.get(int(y), -1) for y in labels])
    rows = np.arange(n)
    true_p = np.zeros((n, 1))
    mask = idx >= 0
    if np.any(mask):
        true_p[mask, 0] = P[rows[mask], idx[mask]]
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    return np.hstack([P, true_p, entropy, gap])

# -----------------------------
# Build attack datasets
# -----------------------------
def make_before_only_attack_set(lr_before, vectorizer,
                                train_docs, y_train, test_docs, y_test,
                                union_classes, use_extras=True):
    """
    Features = before-only confidences (optionally with extras).
    Rows = [train (members=1); test (non-members=0)].
    """
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)
    P_tr_b_raw, cls_b = probs_from_model(lr_before, Xtr)
    P_te_b_raw, _     = probs_from_model(lr_before, Xte)

    P_tr_b = align_probs(P_tr_b_raw, cls_b, union_classes)
    P_te_b = align_probs(P_te_b_raw, cls_b, union_classes)

    if use_extras:
        X_tr = extra_feats(P_tr_b, y_train, union_classes)
        X_te = extra_feats(P_te_b, y_test,  union_classes)
    else:
        X_tr, X_te = P_tr_b, P_te_b

    X_att = np.vstack([X_tr, X_te])
    y_att = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_test))])

    idx_train_block = np.arange(len(y_train))
    idx_test_block  = np.arange(len(y_test)) + len(y_train)
    return X_att, y_att, idx_train_block, idx_test_block

def make_after_dual_attack_set(lr_before, lr_after, vectorizer,
                               train_docs, y_train, test_docs, y_test,
                               union_classes, use_deltas=True, use_extras=True):
    """
    Features = [before P | after P | (delta)] plus extras for both b & a.
    """
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)

    P_tr_b_raw, cls_b = probs_from_model(lr_before, Xtr)
    P_te_b_raw, _     = probs_from_model(lr_before, Xte)
    P_tr_a_raw, cls_a = probs_from_model(lr_after,  Xtr)
    P_te_a_raw, _     = probs_from_model(lr_after,  Xte)

    P_tr_b = align_probs(P_tr_b_raw, cls_b, union_classes)
    P_te_b = align_probs(P_te_b_raw, cls_b, union_classes)
    P_tr_a = align_probs(P_tr_a_raw, cls_a, union_classes)
    P_te_a = align_probs(P_te_a_raw, cls_a, union_classes)

    if use_extras:
        Eb = extra_feats(P_tr_b, y_train, union_classes)
        Ea = extra_feats(P_tr_a, y_train, union_classes)
        X_tr = np.hstack([Eb, Ea, (Ea - Eb)]) if use_deltas else np.hstack([Eb, Ea])

        Eb_te = extra_feats(P_te_b, y_test, union_classes)
        Ea_te = extra_feats(P_te_a, y_test, union_classes)
        X_te = np.hstack([Eb_te, Ea_te, (Ea_te - Eb_te)]) if use_deltas else np.hstack([Eb_te, Ea_te])
    else:
        X_tr = np.hstack([P_tr_b, P_tr_a, (P_tr_a - P_tr_b)]) if use_deltas else np.hstack([P_tr_b, P_tr_a])
        X_te = np.hstack([P_te_b, P_te_a, (P_te_a - P_te_b)]) if use_deltas else np.hstack([P_te_b, P_te_a])

    X_att = np.vstack([X_tr, X_te])
    y_att = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_test))])

    idx_train_block = np.arange(len(y_train))
    idx_test_block  = np.arange(len(y_test)) + len(y_train)
    return X_att, y_att, idx_train_block, idx_test_block

# -----------------------------
# Train+eval attacker (in-sample and held-out)
# -----------------------------
def train_eval_auc(X_att, y_att, heldout_frac=0.5):
    # in-sample AUC (upper bound)
    clf = LogisticRegression(max_iter=5000, random_state=0, class_weight='balanced')
    clf.fit(X_att, y_att)
    auc_in = roc_auc_score(y_att, clf.predict_proba(X_att)[:, 1])

    # held-out AUC (more realistic)
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X_att, y_att, test_size=heldout_frac, random_state=123, stratify=y_att
    )
    clf_ho = LogisticRegression(max_iter=5000, random_state=1, class_weight='balanced')
    clf_ho.fit(X_tr, y_tr)
    auc_ho = roc_auc_score(y_ho, clf_ho.predict_proba(X_ho)[:, 1])
    return (auc_in, clf), (auc_ho, clf_ho)

def masked_auc(clf, X_att, y_att, mask):
    if mask.sum() < 2 or mask.sum() == len(mask):
        return np.nan
    return roc_auc_score(y_att[mask], clf.predict_proba(X_att[mask])[:, 1])

# -----------------------------
# Unlearning on TARGET model (refit)
# -----------------------------
def unlearn_class_via_refit_lr(X_train, y_train, class_to_unlearn: int,
                               C: float, seed: int = 42, fit_intercept: bool=False):
    keep = (y_train != class_to_unlearn)
    return train_logreg(X_train[keep], y_train[keep], C=C, seed=seed, fit_intercept=fit_intercept)

# -----------------------------
# Main
# -----------------------------
def main():
    # Data
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Pick class to unlearn
    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    C = 100.0  # try 50-100 to increase membership leakage

    # Target models
    lr_before = train_logreg(X_train, y_train, C=C, seed=42, fit_intercept=False)
    lr_after  = unlearn_class_via_refit_lr(X_train, y_train, class_to_unlearn, C=C, seed=42, fit_intercept=False)

    # Accuracies (context)
    acc_b = evaluate_accuracy(lr_before, vectorizer, test_docs, y_test)
    acc_a = evaluate_accuracy(lr_after,  vectorizer, test_docs, y_test)
    print(f"Acc BEFORE: {acc_b:.4f} | Acc AFTER: {acc_a:.4f}")

    # Union classes for alignment
    union_classes = np.unique(np.concatenate([lr_before.classes_, lr_after.classes_]))

    # ===== BEFORE-ONLY ATTACK =====
    Xb_all, yb_all, b_tr_idx, b_te_idx = make_before_only_attack_set(
        lr_before, vectorizer,
        train_docs, y_train, test_docs, y_test,
        union_classes, use_extras=True
    )
    (auc_b_in, clf_b_in), (auc_b_ho, clf_b_ho) = train_eval_auc(Xb_all, yb_all, heldout_frac=0.5)

    # Per-class masks (from the SAME dataset)
    memb_mask_b = np.zeros_like(yb_all, dtype=bool)
    nonm_mask_b = np.zeros_like(yb_all, dtype=bool)
    memb_mask_b[b_tr_idx] = (y_train == class_to_unlearn)
    nonm_mask_b[b_te_idx] = (y_test  == class_to_unlearn)
    class_mask_b = memb_mask_b | nonm_mask_b
    print(f"[BEFORE] members(target)={memb_mask_b.sum()}, nonmembers(target)={nonm_mask_b.sum()}")

    auc_b_class_in  = masked_auc(clf_b_in,  Xb_all, yb_all, class_mask_b)
    auc_b_class_ho  = masked_auc(clf_b_ho,  Xb_all, yb_all, class_mask_b)

    # ===== AFTER DUAL-FEATURE ATTACK =====
    Xa_all, ya_all, a_tr_idx, a_te_idx = make_after_dual_attack_set(
        lr_before, lr_after, vectorizer,
        train_docs, y_train, test_docs, y_test,
        union_classes, use_deltas=True, use_extras=True
    )
    (auc_a_in, clf_a_in), (auc_a_ho, clf_a_ho) = train_eval_auc(Xa_all, ya_all, heldout_frac=0.5)

    memb_mask_a = np.zeros_like(ya_all, dtype=bool)
    nonm_mask_a = np.zeros_like(ya_all, dtype=bool)
    memb_mask_a[a_tr_idx] = (y_train == class_to_unlearn)
    nonm_mask_a[a_te_idx] = (y_test  == class_to_unlearn)
    class_mask_a = memb_mask_a | nonm_mask_a
    print(f"[AFTER]  members(target)={memb_mask_a.sum()}, nonmembers(target)={nonm_mask_a.sum()}")

    auc_a_class_in  = masked_auc(clf_a_in,  Xa_all, ya_all, class_mask_a)
    auc_a_class_ho  = masked_auc(clf_a_ho,  Xa_all, ya_all, class_mask_a)

    # ===== Diagnostics: are member vs non-member features different? =====
    # Compare true_p (before) for train vs test (quick sanity check)
    # (We know where those columns sit in the feature block: extra_feats appended [P | true_p | entropy | gap])
    d = len(union_classes)
    true_p_col = d  # after the first d probs
    tr_true_p_mean = Xb_all[b_tr_idx, true_p_col].mean()
    te_true_p_mean = Xb_all[b_te_idx, true_p_col].mean()
    print(f"[Diag BEFORE] mean true_p train={tr_true_p_mean:.4f}, test={te_true_p_mean:.4f}")

    # ===== Report =====
    print(f"\nClass unlearned: {class_to_unlearn}")
    print(f"Overall AUC (before-only)  IN-SAMPLE: {auc_b_in:.4f}")
    print(f"Overall AUC (before-only)  HELD-OUT:  {auc_b_ho:.4f}")
    print(f"Target class {class_to_unlearn} AUC (before-only) IN-SAMPLE: {auc_b_class_in:.4f}")
    print(f"Target class {class_to_unlearn} AUC (before-only) HELD-OUT:  {auc_b_class_ho:.4f}")

    print(f"\nOverall AUC (after dual)   IN-SAMPLE: {auc_a_in:.4f}")
    print(f"Overall AUC (after dual)   HELD-OUT:  {auc_a_ho:.4f}")
    print(f"Target class {class_to_unlearn} AUC (after dual)  IN-SAMPLE: {auc_a_class_in:.4f}")
    print(f"Target class {class_to_unlearn} AUC (after dual)  HELD-OUT:  {auc_a_class_ho:.4f}")

if __name__ == "__main__":
    main()