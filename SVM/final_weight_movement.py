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

# -----------------------------
# Data & base model utilities
# -----------------------------
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data,
        data.target,
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
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

def train_ls_svm(X_train, y_train, C: float = 10.0):
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
    X = vectorizer.transform(docs)
    scores = X @ theta.T
    preds = np.argmax(scores, axis=1)
    return accuracy_score(labels, preds)

# -----------------------------------
# MIA (kept for completeness; unused)
# -----------------------------------
def build_attack_features(P: np.ndarray, labels: np.ndarray):
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p = P[np.arange(len(labels)), labels].reshape(-1,1)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce_loss, gap])

def train_shadow_attack(theta_target, vectorizer, train_docs: list, train_labels: np.ndarray, C: float = 1.0):
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels, test_size=0.5, random_state=0, stratify=train_labels
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

def train_shadow_attack_with_unlearning(theta_target, vectorizer,
                                        train_docs: list, train_labels: np.ndarray,
                                        C: float, class_to_unlearn: int):
    # 1) split
    s_docs, h_docs, s_lbls, h_lbls = train_test_split(
        train_docs, train_labels, test_size=0.5, random_state=0, stratify=train_labels
    )
    Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)

    # 2) shadow LS-SVM
    theta_sh = train_ls_svm(Xs, s_lbls, C)

    # 3) unlearn target class in shadow
    rem = np.where(s_lbls == class_to_unlearn)[0]
    theta_sh_un = influence_removal_ls_svm(Xs, s_lbls, theta_sh, rem, C)

    # 4) fine-tune on remaining shadow-train
    keep = np.ones(len(s_lbls), dtype=bool)
    keep[rem] = False
    Xs_red, y_sred = Xs[keep], s_lbls[keep]
    classes_red = np.unique(y_sred)
    theta_init = theta_sh_un[classes_red]

    ft = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs', multi_class='multinomial',
        fit_intercept=False, warm_start=True, max_iter=5, random_state=42
    )
    ft.classes_ = classes_red
    ft.coef_ = theta_init.copy()
    ft.fit(Xs_red, y_sred)

    # reconstruct full K×d shadow weights (zero row for the removed class)
    K, d = theta_sh.shape
    theta_sh_final = np.zeros((K, d))
    for i, c in enumerate(classes_red):
        theta_sh_final[c] = ft.coef_[i]

    # 5) attack dataset
    Ps = softmax(Xs @ theta_sh_final.T, axis=1)  # members
    Ph = softmax(Xh @ theta_sh_final.T, axis=1)  # non-members
    A_s = build_attack_features(Ps, s_lbls)
    A_h = build_attack_features(Ph, h_lbls)
    X_att = np.vstack([A_s, A_h])
    y_att = np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))])

    # 6) train attack clf
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
    A_te = build_attack_features(Pte, test_labels)
    X_att = np.vstack([A_tr, A_te])
    y_att = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))])
    return roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1])

# -----------------------------------
# Fine-tune diagnostics
# -----------------------------------
def logistic_loss_and_grad(theta_weights, X, y, classes, C):
    class_to_row = {c:i for i,c in enumerate(classes)}
    y_local = np.array([class_to_row[yi] for yi in y])

    S = X @ theta_weights.T if hasattr(X, "tocsr") else X.dot(theta_weights.T)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)

    n = X.shape[0]
    row_idx = np.arange(n)
    loglik = -np.log(P[row_idx, y_local] + 1e-12).sum()
    l2 = 0.5 * (theta_weights**2).sum() / C
    loss = loglik + l2

    R = P
    R[row_idx, y_local] -= 1.0
    grad = (R.T @ X) if hasattr(X, "tocsr") else R.T.dot(X)
    grad = np.asarray(grad)
    grad += theta_weights / C
    return float(loss), grad

def fine_tune_with_logging(theta_init, X_red, y_red, classes_red, C,
                           max_bursts=30, inner_max_iter=20,
                           tol_grad=1e-3, tol_rel=1e-6, verbose=False):
    ft = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs', multi_class='multinomial',
        fit_intercept=False, warm_start=True, max_iter=inner_max_iter,
        random_state=42
    )
    ft.classes_ = classes_red
    ft.coef_ = theta_init.copy()

    loss_hist, grad_hist = [], []
    theta_hist = []
    prev_loss = None
    converged = False

    for b in range(max_bursts + 1):
        theta_hist.append(ft.coef_.copy())
        loss, grad = logistic_loss_and_grad(ft.coef_, X_red, y_red, classes_red, C)
        gnorm = float(np.linalg.norm(grad))
        loss_hist.append(loss)
        grad_hist.append(gnorm)

        if verbose:
            rel = float('nan') if prev_loss is None else (prev_loss - loss)/max(prev_loss, 1.0)
            print(f"[FT step {b:02d}] loss={loss:.4f} | grad_norm={gnorm:.3e} | rel_impr={rel if not np.isnan(rel) else float('nan'):.3e}")

        if prev_loss is not None:
            rel_impr = (prev_loss - loss)/max(prev_loss, 1.0)
            if gnorm < tol_grad or rel_impr < tol_rel:
                converged = True
                break
        prev_loss = loss
        ft.fit(X_red, y_red)

    return ft.coef_.copy(), np.array(loss_hist), np.array(grad_hist), theta_hist, converged

# -----------------------------
# Displacement & plotting
# -----------------------------
def frob_norm(A): 
    return float(np.linalg.norm(A))

def compute_displacement_curves(theta_hist):
    """Return cumulative displacement ||θ^t-θ^0||_F and weight norm ||θ^t||_F per burst."""
    theta0 = theta_hist[0]
    disp, wnorm = [], []
    for th in theta_hist:
        disp.append(frob_norm(th - theta0))
        wnorm.append(frob_norm(th))
    return np.array(disp), np.array(wnorm)

def plot_gradient_norms(results_by_C, title="Gradient norms during unlearning fine-tune"):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    C_list = list(results_by_C.keys())
    C_list.sort(key=lambda c: float(c))
    colors = cm.tab10(np.linspace(0, 1, min(10, len(C_list))))

    plt.figure(figsize=(9.5, 5.5))
    for i, C in enumerate(C_list):
        hist = results_by_C[C]["grad_hist"]
        plt.semilogy(range(len(hist)), hist, marker='o', linewidth=2,
                     color=colors[i % len(colors)],
                     label=f"C={C} (steps={len(hist)-1}, conv={results_by_C[C]['converged']})")
    plt.xlabel("Fine-tune burst index")
    plt.ylabel("Gradient norm (log scale)")
    plt.title(title)
    plt.grid(True, which="both", ls=":", alpha=0.35)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

def _fmt_lam(val):
    """Nice tick labels for lambda = 1/C."""
    if val == 0:
        return "0"
    if val >= 1000 or val < 1e-2:
        return f"{val:.0e}"
    if val >= 1:
        return f"{val:.1f}"
    return f"{val:.3f}"

def plot_final_weight_difference(results_by_C,
                                 title="Final weight displacement ||θ_T − θ_0||_F by regularization (1/C)"):
    """
    Bar chart of final parameter displacement vs regularization λ=1/C.
    Uses categorical x-ticks (ascending) to avoid scale distortion.
    """
    # collect (lambda, final_disp)
    rows = []
    for C_key, res in results_by_C.items():
        C = float(C_key)
        lam = 1.0 / C if C != 0.0 else 0.0
        theta_hist = res["theta_hist_active"]
        disp_curve, _ = compute_displacement_curves(theta_hist)
        rows.append((lam, disp_curve[-1]))

    # sort by lambda ascending
    rows.sort(key=lambda t: t[0])
    lam_vals  = [r[0] for r in rows]
    disp_vals = [r[1] for r in rows]
    lam_labels = [_fmt_lam(v) for v in lam_vals]

    # categorical bar plot
    x_pos = np.arange(len(lam_vals))
    plt.figure(figsize=(9.5, 5.0))
    plt.bar(x_pos, disp_vals, align='center', edgecolor='black', linewidth=0.6)
    plt.xticks(x_pos, lam_labels, rotation=0)
    plt.xlabel("1/C (regularization strength)")
    plt.ylabel("Final displacement  ||θ_T − θ_0||_F")
    plt.title(title)
    plt.grid(axis='y', ls=':', alpha=0.35)
    plt.tight_layout()
    plt.show()

def print_final_weight_table(results_by_C):
    print("\n[Final weight and displacement (active classes)]")
    print("C        | bursts | final ||θ||_F    | total disp ||θ_T-θ_0||_F")
    print("---------+--------+------------------+------------------------")
    for C, res in sorted(results_by_C.items(), key=lambda kv: float(kv[0])):
        th_hist = res["theta_hist_active"]
        disp_curve, wnorm_curve = compute_displacement_curves(th_hist)
        bursts = len(res["grad_hist"]) - 1
        print(f"{str(C):<8} | {bursts:<6d} | {wnorm_curve[-1]:<16.3e} | {disp_curve[-1]:<22.3e}")

# -----------------------------
# One full unlearning run
# -----------------------------
def run_unlearning_once(train_docs, test_docs, y_train, y_test, vectorizer, C, class_to_unlearn,
                        seed=0, ft_max_bursts=30, ft_inner=20, tol_grad=1e-3, tol_rel=1e-6, verbose=False):
    random.seed(seed); np.random.seed(seed)
    X_train = vectorizer.transform(train_docs)
    X_test  = vectorizer.transform(test_docs)

    theta_orig = train_ls_svm(X_train, y_train, C)

    removal_indices = np.where(y_train == class_to_unlearn)[0]
    theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, removal_indices, C)

    keep = np.ones(len(y_train), bool)
    keep[removal_indices] = False
    X_red, y_red = X_train[keep], y_train[keep]
    classes_red = np.unique(y_red)
    theta_init = theta_un[classes_red, :]

    theta_ft, loss_hist, grad_hist, theta_hist, conv = fine_tune_with_logging(
        theta_init, X_red, y_red, classes_red, C,
        max_bursts=ft_max_bursts, inner_max_iter=ft_inner,
        tol_grad=tol_grad, tol_rel=tol_rel, verbose=verbose
    )

    # reconstruct full Kxd with zero row for removed class
    K, d = theta_orig.shape
    theta_final = np.zeros_like(theta_orig)
    for i, cls in enumerate(classes_red):
        theta_final[cls, :] = theta_ft[i]

    scores = X_test @ theta_final.T
    preds  = np.argmax(scores, axis=1)

    return {
        "theta_final": theta_final,
        "theta_ft_active": theta_ft,
        "classes_red": classes_red,
        "loss_hist": loss_hist,
        "grad_hist": grad_hist,
        "theta_hist_active": theta_hist,
        "converged": conv,
        "preds_test": preds,
    }

# -----------------------------
# Main: sweep C, plot & table
# -----------------------------
def main():
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # fix one class to unlearn so releases are comparable across C
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print("Class to unlearn:", class_to_unlearn)

    C_list = [1e-11, 1e-5, 0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]

    results_by_C = {}
    for C in C_list:
        print(f"\n=== Running unlearning with C={C} ===")
        res = run_unlearning_once(
            train_docs, test_docs, y_train, y_test, vectorizer, C, class_to_unlearn,
            seed=0, ft_max_bursts=60, ft_inner=5, tol_grad=1e-3, tol_rel=1e-6, verbose=True
        )
        results_by_C[C] = res
        print(f"   converged={res['converged']}  final grad norm={res['grad_hist'][-1]:.3e}")

    # figure 1: gradient norms vs burst
    plot_gradient_norms(results_by_C,
        title="Gradient norms across bursts for different C")

    # figure 2: final weight displacement by C
    plot_final_weight_difference(results_by_C,
        title="Final parameter displacement ||θ_T − θ_0||_F vs Regularization Strength (1/C)")

    # compact table of final numbers
    print_final_weight_table(results_by_C)

if __name__ == "__main__":
    main()
