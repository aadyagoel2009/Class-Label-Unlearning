import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def prepare_model(test_size=0.2, random_state=42):
    data = fetch_20newsgroups(subset="all")
    docs, labels = data.data, data.target
    return train_test_split(
        docs, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

def build_tfidf(train_docs, test_docs):
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,3),
        max_df=0.85,
        min_df=5,
        sublinear_tf=True,
        norm="l2"
    )
    return tfidf, tfidf.fit_transform(train_docs), tfidf.transform(test_docs)

def train_svc(X, y):
    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10_000,
        dual=False,
        random_state=42
    )
    clf.fit(X, y)
    return clf

def evaluate(clf, X_test, y_test, mask_cls=None):
    """Mask mask_cls by setting its decision score to -inf, then argmax."""
    scores = clf.decision_function(X_test)
    if mask_cls is not None:
        scores[:, mask_cls] = -np.inf
    preds = scores.argmax(axis=1)
    return accuracy_score(y_test, preds)

def main():
    # Load & split
    train_docs, test_docs, y_train, y_test = prepare_model()
    tfidf, X_train, X_test = build_tfidf(train_docs, test_docs)

    # Pick a random class to unlearn
    num_classes = len(set(y_train))
    unlearn = random.randint(0, num_classes-1)
    print("→ Unlearning class:", unlearn)

    # Kang-style label-redistribution on TRAIN set
    remap = []
    other = [c for c in range(num_classes) if c != unlearn]
    for y in y_train:
        remap.append(random.choice(other) if y == unlearn else y)

    # Train original & unlearned models
    orig_clf = train_svc(X_train, y_train)
    kang_clf = train_svc(X_train, remap)

    # Evaluate on FULL test set
    acc_before = evaluate(orig_clf, X_test, y_test)
    acc_kang_full = evaluate(kang_clf, X_test, y_test)
    print(f"Accuracy before unlearning: {acc_before:.3f}")
    print(f" Accuracy after Kang-unlearn (full): {acc_kang_full:.3f}")

    # Evaluate on **remaining** classes only
    keep = np.array([i for i,y in enumerate(y_test) if y != unlearn])
    acc_kang_remain = evaluate(
        kang_clf,
        X_test[keep],
        y_test[keep],
        mask_cls=None  # no need to mask here since we never include unlearned class
    )
    print(f"Accuracy on remaining classes only: {acc_kang_remain:.3f}")

if __name__ == "__main__":
    main()