import numpy as np
import random
import scipy.sparse as sp
from scipy.special import softmax
from scipy.sparse.linalg import LinearOperator, cg
from scipy import stats
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# =========================
# Data & base model utils
# =========================
def prepare_data(test_size: float = 0.2, random_state: int = 42):
    data = fetch_20newsgroups(subset="all")
    return train_test_split(
        data.data, data.target,
        test_size=test_size, random_state=random_state, stratify=data.target
    )

def build_tfidf(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1,3),
        max_df=0.85, min_df=5, sublinear_tf=True, norm="l2"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

def train_ls_svm(X_train, y_train, C: float = 10.0):
    classes = np.unique(y_train)
    Y = np.zeros((X_train.shape[0], len(classes)))
    for ci, cls in enumerate(classes):
        Y[y_train == cls, ci] = 1.0
    ridge = Ridge(alpha=1.0/C, fit_intercept=False, solver="auto")
    ridge.fit(X_train, Y)
    return ridge.coef_  # (K,d)

# =========================
# Influence downdate
# =========================
def influence_removal_ls_svm(X_train, y_train, theta_orig,
                             removal_indices: np.ndarray, C: float = 10.0):
    n_classes, n_features = theta_orig.shape
    grad_sum = np.zeros_like(theta_orig)
    for idx in removal_indices:
        x = X_train[idx].toarray().ravel()
        one_hot = np.zeros(n_classes); one_hot[y_train[idx]] = 1
        scores = theta_orig @ x
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

# =========================
# Attack for σ* calibration
# =========================
def build_attack_features(P: np.ndarray, labels: np.ndarray):
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    true_p  = P[np.arange(len(labels)), labels].reshape(-1,1)
    ce_loss = -np.log(true_p + 1e-12)
    top2 = np.sort(P, axis=1)[:, -2:]
    gap  = (top2[:,1] - top2[:,0]).reshape(-1,1)
    return np.hstack([P, entropy, ce_loss, gap])

def train_shadow_attack(vectorizer, train_docs, train_labels, C: float = 1.0):
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
    keep_tr = np.ones(len(train_labels), dtype=bool)
    keep_te = retained_mask_test
    for _ in range(trials):
        noise = rng.normal(0.0, sigma, size=theta_base.shape)
        theta_noisy = theta_base + noise
        Ptr = softmax((Xtr @ theta_noisy.T)[keep_tr], axis=1)
        Pte = softmax((Xte @ theta_noisy.T)[keep_te], axis=1)
        A_tr = build_attack_features(Ptr, train_labels[keep_tr])
        A_te = build_attack_features(Pte, test_labels[keep_te])
        X_att = np.vstack([A_tr, A_te])
        y_att = np.concatenate([np.ones(A_tr.shape[0]), np.zeros(A_te.shape[0])])
        aucs.append(roc_auc_score(y_att, attack_clf.predict_proba(X_att)[:,1]))
    return float(np.mean(aucs))

def calibrate_noise_for_tau_noft_unbounded(theta_base, vectorizer,
                                           train_docs, train_labels,
                                           test_docs,  test_labels,
                                           attack_clf,
                                           retained_mask_test,
                                           tau, auc_tol=1e-3,
                                           max_grow_steps=60, bisection_steps=40,
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
        if auc_high <= tau: break
        sigma_low, auc_low = sigma_high, auc_high
        sigma_high *= 2.0
    else:
        return {"sigma": sigma_high, "auc": auc_high, "hit_boundary": True}
    left, right = sigma_low, sigma_high
    best_sigma, best_auc = sigma_high, auc_high
    for _ in range(bisection_steps):
        mid = 0.5*(left+right)
        auc_mid = eval_retained_auc_under_weight_noise(theta_base, vectorizer,
                                                       train_docs, train_labels,
                                                       test_docs,  test_labels,
                                                       attack_clf, sigma=mid,
                                                       retained_mask_test=retained_mask_test,
                                                       trials=trials, seed=seed)
        if auc_mid > tau: left = mid
        else: right = mid; best_sigma, best_auc = mid, auc_mid
        if abs(auc_mid - tau) <= auc_tol:
            best_sigma, best_auc = mid, auc_mid
            break
    return {"sigma": best_sigma, "auc": best_auc, "hit_boundary": False}

# =========================
# True-vs-largest-incorrect margin
# =========================
def true_vs_comp_margin(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    true = logits[np.arange(logits.shape[0]), y_true]
    L = logits.copy()
    L[np.arange(L.shape[0]), y_true] = -np.inf
    comp = L.max(axis=1)
    return true - comp

# =========================
# Pooled STANDARDIZED modality at (C, τ)
# =========================
def pooled_standardized_modality(theta_base, vectorizer, test_docs, test_labels,
                                 class_to_unlearn, sigma, n_draws=400, batch=50, rng_seed=0):
    """
    1) For retained test points, sample margins under N(0, sigma^2) weight noise.
    2) z-score per point; pool all z's.
    3) Return global modality/normality diagnostics + helper stats.
    """
    rng = np.random.RandomState(rng_seed)
    X = vectorizer.transform(test_docs)
    keep = (test_labels != class_to_unlearn)
    y = test_labels[keep]
    Xk = X[keep]

    N = Xk.shape[0]
    K, d = theta_base.shape
    if sigma <= 0:
        # tiny floor so tests run
        theta_scale = np.linalg.norm(theta_base) / np.sqrt(theta_base.size)
        sigma = max(1e-12, 1e-3 * theta_scale)

    margins = np.zeros((N, n_draws), dtype=np.float32)

    done = 0
    while done < n_draws:
        b = min(batch, n_draws - done)
        noise = rng.normal(0.0, sigma, size=(b, K, d))
        for i in range(b):
            W = theta_base + noise[i]
            logits = Xk @ W.T
            margins[:, done+i] = true_vs_comp_margin(logits, y)
        done += b

    # Per-point standardization, then pool
    mu = margins.mean(axis=1, keepdims=True)
    sd = margins.std(axis=1, keepdims=True) + 1e-12
    Z = (margins - mu) / sd
    z_all = Z.ravel()

    # 1) Normality tests on pooled standardized z
    k2, p_normal = stats.normaltest(z_all)
    try:
        ad_stat, crit_vals, _ = stats.anderson(z_all, dist='norm')
        ad_reject = ad_stat > crit_vals[-1]  # reject at 1%
    except Exception:
        ad_reject = False

    # 2) GMM model selection on pooled standardized z
    xcol = z_all.reshape(-1, 1)
    best_k, best_bic = None, np.inf
    bics = []
    for k in [1, 2, 3]:
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        g.fit(xcol)
        b = g.bic(xcol)
        bics.append(b)
        if b < best_bic:
            best_bic = b
            best_k = k

    # 3) KDE + peak counting on pooled standardized z
    kde = stats.gaussian_kde(z_all)
    grid = np.linspace(np.percentile(z_all, 0.5), np.percentile(z_all, 99.5), 2048)
    dens = kde(grid)
    # Find local maxima; require some prominence to avoid tiny ripples
    peaks, _ = find_peaks(dens, prominence=(dens.max()*0.02))
    n_peaks = int(len(peaks))

    # 4) Bimodality coefficient (Pearson’s): BC = (γ^2 + 1) / (κ + 3)
    # where κ is Pearson kurtosis (not excess). scipy returns excess kurtosis.
    skew = stats.skew(z_all, bias=False)
    kurt_ex = stats.kurtosis(z_all, fisher=True, bias=False)
    kurt = kurt_ex + 3.0
    BC = (skew**2 + 1.0) / (kurt + 1e-12)

    # Global flip-rate (nice to see alongside)
    flip_rate = (margins < 0).mean()

    return {
        "p_normal": float(p_normal),
        "AD_reject_1pct": bool(ad_reject),
        "GMM_best_k": int(best_k),
        "GMM_BICs": [float(x) for x in bics],
        "KDE_peak_count": n_peaks,
        "BC": float(BC),
        "flip_rate": float(flip_rate),
        "N_points": int(N),
        "n_draws": int(n_draws)
    }

# =========================
# Driver across (C, τ)
# =========================
def main():
    # data
    train_docs, test_docs, y_train, y_test = prepare_data()
    vectorizer, X_train, X_test = build_tfidf(train_docs, test_docs)

    random.seed(7)
    class_to_unlearn = random.randint(0, int(y_train.max()))
    print(f"[experiment] class_to_unlearn = {class_to_unlearn}")

    C_list = [0.01, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    taus   = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]

    retained_mask_test = (y_test != class_to_unlearn)

    # storage for heatmaps
    bestk = np.zeros((len(C_list), len(taus)))
    peaks = np.zeros((len(C_list), len(taus)))
    pnorm = np.zeros((len(C_list), len(taus)))
    sigma_star = np.zeros((len(C_list), len(taus)))
    fliprate = np.zeros((len(C_list), len(taus)))

    for ic, C in enumerate(C_list):
        print("\n" + "="*72)
        print(f"[C={C}] baseline + downdate + base release")
        theta_orig = train_ls_svm(X_train, y_train, C)
        rem_idx = np.where(y_train == class_to_unlearn)[0]
        theta_un = influence_removal_ls_svm(X_train, y_train, theta_orig, rem_idx, C)
        theta_base = theta_un.copy()
        theta_base[class_to_unlearn, :] = 0.0

        attack_clf = train_shadow_attack(vectorizer, train_docs, y_train, C)

        for it, tau in enumerate(taus):
            cal = calibrate_noise_for_tau_noft_unbounded(
                theta_base, vectorizer,
                train_docs, y_train,
                test_docs,  y_test,
                attack_clf,
                retained_mask_test,
                tau=tau, auc_tol=1e-3,
                max_grow_steps=60, bisection_steps=40,
                trials=5, seed=0
            )
            sig = float(cal["sigma"])
            sigma_star[ic, it] = sig

            res = pooled_standardized_modality(
                theta_base, vectorizer, test_docs, y_test,
                class_to_unlearn, sigma=sig,
                n_draws=600, batch=60, rng_seed=0
            )
            bestk[ic, it]  = res["GMM_best_k"]
            peaks[ic, it]  = res["KDE_peak_count"]
            pnorm[ic, it]  = res["p_normal"]
            fliprate[ic,it] = res["flip_rate"]

            print(f"(C={C:>5}, τ={tau:.2f}) σ*={sig:.4g} | GMM-k*={int(bestk[ic,it])} | "
                  f"peaks={int(peaks[ic,it])} | pNorm={res['p_normal']:.3g} | flip={res['flip_rate']:.3f}")

    # ---- plotting helpers ----
    def heat(Z, title, xt, yt, vmin=None, vmax=None):
        plt.figure(figsize=(8,6))
        plt.imshow(Z, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xticks(np.arange(len(xt)), [f"{t:.2f}" for t in xt], rotation=45)
        plt.yticks(np.arange(len(yt)), [str(c) for c in yt])
        plt.xlabel("τ"); plt.ylabel("C"); plt.title(title)
        plt.tight_layout(); plt.show()

    heat(bestk,   "GMM best component count k* (pooled standardized)", taus, C_list, vmin=1, vmax=3)
    heat(peaks,   "KDE peak count (pooled standardized)",              taus, C_list, vmin=1, vmax=3)
    heat(pnorm,   "D’Agostino K² p-value (pooled standardized)",        taus, C_list)
    heat(fliprate,"Mean flip-rate (true−max(other))",                   taus, C_list)
    heat(sigma_star, "σ* (calibrated noise)",                           taus, C_list)

if __name__ == "__main__":
    main()