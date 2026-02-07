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
import matplotlib.pyplot as plt

# =========================
# Data & base model utils
# =========================
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data, data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target
    )

def build_tfidf(train_texts, test_texts):
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

# =========================
# Logistic Regression target
# =========================
def train_logreg(X_train, y_train, C: float = 10.0, max_iter: int = 2000, seed: int = 42):
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        multi_class="multinomial",
        fit_intercept=False,
        max_iter=max_iter,
        random_state=seed
    )
    lr.fit(X_train, y_train)
    return lr   # .coef_ (K,d), .classes_

def evaluate_accuracy(lr_model, vectorizer, docs, labels):
    X = vectorizer.transform(docs)
    preds = lr_model.predict(X)
    return accuracy_score(labels, preds)

def probs_from_theta(theta, X):
    logits = X @ theta.T
    return softmax(logits, axis=1)

# ================
# Attack features
# ================
def build_attack_features(P: np.ndarray, labels: np.ndarray, classes_present: np.ndarray):
    class_to_col = {int(c): i for i, c in enumerate(classes_present)}
    idx = np.array([class_to_col.get(int(y), -1) for y in labels])
    rows = np.arange(len(labels))
    true_p = np.zeros((len(labels), 1))
    mask = (idx >= 0)
    if np.any(mask):
        true_p[mask, 0] = P[rows[mask], idx[mask]]
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap  = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce_loss, gap])

# --------- pooled multi-shadow attacker ----------
def train_shadow_attack_lr(vectorizer, train_docs, train_labels, C, S: int = 20):
    """
    Train an LR attacker using pooled attack data from S shadow models.
    (No unlearning inside shadows; same feature set.)
    """
    X_all, y_all = [], []
    for seed in range(S):
        s_docs, h_docs, s_lbls, h_lbls = train_test_split(
            train_docs, train_labels, test_size=0.5, random_state=seed, stratify=train_labels
        )
        Xs, Xh = vectorizer.transform(s_docs), vectorizer.transform(h_docs)
        shadow = train_logreg(Xs, s_lbls, C=C, seed=seed)
        Ps, Ph = probs_from_theta(shadow.coef_, Xs), probs_from_theta(shadow.coef_, Xh)
        A_s = build_attack_features(Ps, s_lbls, shadow.classes_)
        A_h = build_attack_features(Ph, h_lbls, shadow.classes_)
        X_all.append(np.vstack([A_s, A_h]))
        y_all.append(np.concatenate([np.ones(len(s_lbls)), np.zeros(len(h_lbls))]))
    X_att = np.vstack(X_all)
    y_att = np.concatenate(y_all)
    attack_clf = LogisticRegression(max_iter=5000, random_state=2)
    attack_clf.fit(X_att, y_att)
    return attack_clf
# -----------------------------------------------------------

# ==========================================
# AUC under *weight noise* (no fine-tuning)
# ==========================================
def eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                         train_docs, train_labels,
                                         test_docs,  test_labels,
                                         attack_clf,
                                         sigma, retained_mask_test,
                                         trials=5, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    Xtr = vectorizer.transform(train_docs)
    Xte = vectorizer.transform(test_docs)
    K, d = theta_base.shape

    for _ in range(trials):
        noise = rng.normal(loc=0.0, scale=sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise
        Ptr = probs_from_theta(theta_noisy, Xtr)
        Pte = probs_from_theta(theta_noisy, Xte)
        A_tr = build_attack_features(Ptr, train_labels, np.arange(K))
        A_te = build_attack_features(Pte, test_labels,  np.arange(K))
        keep_tr = np.ones(len(train_labels), dtype=bool)
        keep_te = retained_mask_test
        X_att = np.vstack([A_tr[keep_tr], A_te[keep_te]])
        y_att = np.concatenate([np.ones(keep_tr.sum()), np.zeros(keep_te.sum())])
        aucs.append(roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1]))
    return float(np.mean(aucs))

def _stable_uint32_from_key(key_tuple):
    return hash(key_tuple) & 0xFFFFFFFF

def eval_utility_under_weight_noise(theta_base, vectorizer,
                                    test_docs, test_labels, class_to_unlearn,
                                    sigma, trials=5, seed_base=0, key=None):
    Xte  = vectorizer.transform(test_docs)
    keep = (test_labels != class_to_unlearn)
    K, d = theta_base.shape

    if key is None:
        key = ("default", float(sigma))

    accs = []
    for t in range(trials):
        seed_t = (seed_base ^ (hash((key, t)) & 0xFFFFFFFF)) % (2**32)
        rng = np.random.RandomState(seed_t)

        noise = rng.normal(loc=0.0, scale=sigma, size=(K, d))
        noise[class_to_unlearn, :] = 0.0                 # do NOT perturb removed class
        theta_noisy = theta_base + noise

        scores = Xte @ theta_noisy.T                     # (n_test, K)
        scores = np.asarray(scores)
        scores[:, class_to_unlearn] = -np.inf            # forbid picking removed class
        preds = scores.argmax(axis=1)

        accs.append(accuracy_score(test_labels[keep], preds[keep]))

    return float(np.mean(accs))

# ==========================================
# LR Hessian pieces (class removal)
# ==========================================
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
    U = X @ V.T
    dot = np.sum(P * U, axis=1)
    M = P * U - P * dot[:, None]
    HV = (M.T @ X)
    if sp.issparse(HV):
        HV = HV.A
    HV = HV + lam * V
    return HV

def unlearn_class_via_hessian_lr(lr_model, X_train, y_train,
                                 class_to_unlearn, C: float,
                                 cg_tol=1e-3, cg_max_iter=300):
    """
    Influence-style Hessian downdate to remove ALL samples of the class:
      W' = W - H^{-1} * g_removed, then zero the removed-class row.
    """
    lr_new = deepcopy(lr_model)
    W = lr_model.coef_.copy()
    classes = lr_model.classes_
    K, d = W.shape

    G_rem, P = _summed_grad_removed_class(W, X_train, y_train, classes, class_to_unlearn)
    if not np.any(G_rem):
        # nothing to remove; still zero the row in the release
        row = np.where(classes == class_to_unlearn)[0]
        if len(row) > 0:
            W[row[0], :] = 0.0
        lr_new.coef_ = W
        return lr_new

    def matvec(vec):
        V = vec.reshape(K, d)
        HV = _hessian_vec_prod_lr(W, V, X_train, P, C)
        return HV.reshape(-1)

    H_op = LinearOperator(shape=(K*d, K*d), matvec=matvec, dtype=W.dtype)
    b = G_rem.reshape(-1)
    delta, info = cg(H_op, b, tol=cg_tol, maxiter=cg_max_iter)
    Delta = delta.reshape(K, d)

    W_un = W - Delta

    # Zero the removed class row to make it unselectable
    row = np.where(classes == class_to_unlearn)[0]
    if len(row) > 0:
        W_un[row[0], :] = 0.0

    lr_new.coef_ = W_un
    return lr_new

# ==========================================
# Unbounded noise calibration (grow + bisect)
# ==========================================
def calibrate_noise_for_tau_noft_unbounded(theta_base, vectorizer,
                                           train_docs, train_labels,
                                           test_docs,  test_labels,
                                           attack_clf,
                                           retained_mask_test,
                                           tau, auc_tol=1e-3,
                                           max_grow_steps=60,
                                           bisection_steps=40,
                                           trials=5, seed=0):
    auc0 = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                train_docs, train_labels,
                                                test_docs,  test_labels,
                                                attack_clf, sigma=0.0,
                                                retained_mask_test=retained_mask_test,
                                                trials=trials, seed=seed)
    if auc0 <= tau + auc_tol:
        return {"sigma": 0.0, "auc": auc0, "hit_boundary": False}

    sigma_low, auc_low = 0.0, auc0
    sigma_high = 1.0
    for _ in range(max_grow_steps):
        auc_high = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                        train_docs, train_labels,
                                                        test_docs,  test_labels,
                                                        attack_clf, sigma=sigma_high,
                                                        retained_mask_test=retained_mask_test,
                                                        trials=trials, seed=seed)
        if auc_high <= tau:
            break
        sigma_low, auc_low = sigma_high, auc_high
        sigma_high *= 2.0
    else:
        return {"sigma": sigma_high, "auc": auc_high, "hit_boundary": True}

    left, right = sigma_low, sigma_high
    best_sigma, best_auc = sigma_high, auc_high
    for _ in range(bisection_steps):
        mid = 0.5 * (left + right)
        auc_mid = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                       train_docs, train_labels,
                                                       test_docs,  test_labels,
                                                       attack_clf, sigma=mid,
                                                       retained_mask_test=retained_mask_test,
                                                       trials=trials, seed=seed)
        if auc_mid > tau:
            left = mid
        else:
            right = mid
            best_sigma, best_auc = mid, auc_mid
        if abs(auc_mid - tau) <= auc_tol:
            best_sigma, best_auc = mid, auc_mid
            break

    return {"sigma": best_sigma, "auc": best_auc, "hit_boundary": False}

# =========================
# Plotting: lines + heatmaps
# =========================
def plot_utility_and_noise_vs_C(C_list, utilities, sigmas, tau):
    from matplotlib.ticker import MaxNLocator
    x = list(range(len(C_list)))
    fig, ax1 = plt.subplots(figsize=(9, 5.2))

    color_u = "#1f77b4"
    ln1 = ax1.plot(x, utilities, marker='o', linewidth=2, color=color_u,
                   label="Utility (accuracy w/o unlearned)")
    ax1.set_xlabel("C (regularization = 1/C)")
    ax1.set_ylabel("Utility (Accuracy w/o unlearned)", color=color_u)
    ax1.tick_params(axis='y', labelcolor=color_u)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in C_list], rotation=0)

    ax2 = ax1.twinx()
    color_s = "#d62728"
    req_sigmas = [max(1e-12, s) for s in sigmas]
    ln2 = ax2.plot(x, req_sigmas, marker='s', linestyle='--', linewidth=2, color=color_s,
                   label=r"Required noise $\sigma^{\ast}$")
    ax2.set_ylabel(r"Required noise $\sigma^{\ast}$ (log scale)", color=color_s)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_yscale('log')

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)
    ax1.set_title(f"Fix retained-class AUC τ={tau:.2f}: Utility vs C and required noise")
    plt.tight_layout()
    plt.show()

def plot_heatmaps(C_list, taus, util_grid, sigma_grid):
    """
    util_grid, sigma_grid: shape (len(taus), len(C_list))
    """
    Cs = np.array(C_list)
    Ts = np.array(taus)

    # Utility heatmap
    plt.figure(figsize=(8.5, 4.2))
    im = plt.imshow(util_grid, aspect='auto', origin='lower', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Utility (Accuracy w/o unlearned)")
    plt.xticks(np.arange(len(Cs)), [str(c) for c in Cs])
    plt.yticks(np.arange(len(Ts)), [f"{t:.2f}" for t in Ts])
    plt.xlabel("C")
    plt.ylabel("τ (target retained-class AUC)")
    plt.title("Utility heatmap")
    plt.tight_layout()
    plt.show()

    # Noise heatmap (log10 sigma for readability)
    sigma_plot = np.log10(np.maximum(sigma_grid, 1e-12))
    plt.figure(figsize=(8.5, 4.2))
    im2 = plt.imshow(sigma_plot, aspect='auto', origin='lower')
    cbar = plt.colorbar(im2)
    cbar.set_label(r"log10 required noise $\sigma^{\ast}$")
    plt.xticks(np.arange(len(Cs)), [str(c) for c in Cs])
    plt.yticks(np.arange(len(Ts)), [f"{t:.2f}" for t in Ts])
    plt.xlabel("C")
    plt.ylabel("τ (target retained-class AUC)")
    plt.title(r"Required noise $\sigma^{\ast}$ heatmap (log scale)")
    plt.tight_layout()
    plt.show()

# ===========
# Main sweep
# ===========
def main():
    # --- data & tf-idf ---
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    # pick one class to unlearn for comparability across C
    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    # C grid and target AUCs  (KEEP EXACTLY AS SPECIFIED)
    C_list = [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    taus   = [0.51, 0.60, 0.70, 0.80, 0.90]

    retained_mask_test = (y_test != class_to_unlearn)

    # To store per-τ line plots
    results = {tau: [] for tau in taus}
    # To build heatmaps (rows = τ, cols = C)
    util_grid  = np.zeros((len(taus), len(C_list)))
    sigma_grid = np.zeros((len(taus), len(C_list)))

    for jC, C in enumerate(C_list):
        print("\n" + "="*72)
        print(f"[C={C}] training baseline LR & unlearning (Hessian + zero row)")

        # Baseline LR
        lr_orig = train_logreg(X_train, y_train, C=C, seed=42)

        # === Hessian unlearning on TARGET (no randomized relabel) ===
        lr_un = unlearn_class_via_hessian_lr(
            lr_model=lr_orig, X_train=X_train, y_train=y_train,
            class_to_unlearn=class_to_unlearn, C=C,
            cg_tol=1e-3, cg_max_iter=300
        )

        # Release weights: ensure removed class row is zeroed
        theta_base = lr_un.coef_.copy()
        if class_to_unlearn < theta_base.shape[0]:
            theta_base[class_to_unlearn, :] = 0.0

        # fixed attacker (pooled multi-shadow; no unlearning inside)
        attack_clf = train_shadow_attack_lr(vectorizer, train_docs, y_train, C=C, S=20)

        # iterate over target AUCs
        for itau, tau in enumerate(taus):
            cal = calibrate_noise_for_tau_noft_unbounded(
                theta_base=theta_base,
                vectorizer=vectorizer,
                train_docs=train_docs, train_labels=y_train,
                test_docs=test_docs,  test_labels=y_test,
                attack_clf=attack_clf,
                retained_mask_test=retained_mask_test,
                tau=tau, auc_tol=1e-3,
                max_grow_steps=60, bisection_steps=40,
                trials=5, seed=0
            )

            util = eval_utility_under_weight_noise(
                theta_base, vectorizer, test_docs, y_test, class_to_unlearn,
                sigma=cal["sigma"], trials=5, seed_base=1,
                key=("C_sigma", float(C), float(np.round(cal["sigma"], 12)))
            )

            reg_strength = 1.0 / C
            print(f"[τ={tau:.2f}, C={C:>5}] 1/C={reg_strength:.5f}  σ*={cal['sigma']:.12g}  "
                  f"AUC={cal['auc']:.3f}  Acc_wo={util:.4f}  boundary={cal['hit_boundary']}")

            results[tau].append({
                "C": C,
                "reg": reg_strength,
                "sigma": cal["sigma"],
                "auc": cal["auc"],
                "acc_wo": util,
                "boundary": cal["hit_boundary"],
            })
            # fill heatmap cells
            util_grid[itau, jC]  = util
            sigma_grid[itau, jC] = cal["sigma"]

    # Per-τ line plots (same as before)
    for tau in taus:
        rows = results[tau]
        rows = sorted(rows, key=lambda r: C_list.index(r["C"]))
        Cs     = [r["C"] for r in rows]
        accs   = [r["acc_wo"] for r in rows]
        sigmas = [max(r["sigma"], 1e-12) for r in rows]
        plot_utility_and_noise_vs_C(Cs, accs, sigmas, tau=tau)

    # Heatmaps across (τ x C)
    plot_heatmaps(C_list, taus, util_grid, sigma_grid)

    # Optional tables
    for tau in taus:
        rows = sorted(results[tau], key=lambda r: C_list.index(r["C"]))
        print(f"\n=== Results for τ={tau:.2f} (retained-class AUC target) ===")
        print("  C         1/C (reg)    σ* (noise)          AUC_hit   Acc_wo   boundary")
        for r in rows:
            print(f"  {str(r['C']):>6}   {r['reg']:>10.6f}   {r['sigma']:<16.12g}   {r['auc']:.3f}    {r['acc_wo']:.4f}   {str(r['boundary']):>8}")

if __name__ == "__main__":
    main()