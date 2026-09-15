"""Prototype that mirrors the TypeScript ML core (same pseudo-code) to validate
the SMO / K-Means / Jacobi-PCA designs before porting to the browser.

Temporary validation script - safe to delete.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "public" / "data"


def mulberry32(seed: int):
    def rand():
        nonlocal seed
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = seed
        t = (t + ((t << 7) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t ^= (t >> 15) & 0xFFFFFFFF
        t = (t * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7) & 0xFFFFFFFF) | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def load(name):
    payload = json.loads((DATA / f"{name}.json").read_text())
    X = np.array(payload["data"], dtype=float)
    y = np.array(payload["target"], dtype=int)
    return X, y, payload["targetNames"]


def stratified_split(y, test_ratio, seed):
    rng = mulberry32(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = [i for i in range(len(y)) if y[i] == cls]
        order = [rng() for _ in idx]
        idx = [i for _, i in sorted(zip(order, idx))]
        n_test = max(1, round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def standard_scale(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (Xtr - mu) / sigma, (Xte - mu) / sigma


def rbf_kernel(a, b, gamma):
    sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)


def smo(X, y, C, tol=1e-3, eps=1e-5, max_iter=40000, gamma=None):
    """Simplified SMO with f-cache and argmax|E1-E2| heuristic (same as TS port)."""
    n = len(y)
    K = rbf_kernel(X, X, gamma)
    alpha = np.zeros(n)
    b = 0.0
    f = -y.astype(float)  # f_i = decision_i - y_i with decision=0
    it = 0
    applied = 0
    streak = 0
    while it < max_iter and applied < max_iter:
        it += 1
        # r_i = y_i * E_i; violate if (r < -tol and a<C) or (r > tol and a>0)
        r = y * f
        candidates = [i for i in range(n) if (r[i] < -tol and alpha[i] < C - eps) or (r[i] > tol and alpha[i] > eps)]
        if not candidates:
            break
        i1 = candidates[int(np.argmax(np.abs(r[candidates])))]
        E1 = f[i1]
        if streak > 20:
            i2 = candidates[(candidates.index(i1) + 1 + it) % len(candidates)]
        else:
            i2 = int(np.argmax(np.abs(E1 - f)))
            if abs(E1 - f).max() == 0 or i2 == i1:
                i2 = candidates[(candidates.index(i1) + 1) % len(candidates)]
        y1, y2 = y[i1], y[i2]
        a1o, a2o = alpha[i1], alpha[i2]
        s = y1 * y2
        if s < 0:
            L, H = max(0.0, a2o - a1o), min(C, C + a2o - a1o)
        else:
            L, H = max(0.0, a1o + a2o - C), min(C, a1o + a2o)
        if L == H:
            streak += 1
            continue
        eta = 2 * K[i1, i2] - K[i1, i1] - K[i2, i2]
        if eta >= 0:
            streak += 1
            continue
        a2n = a2o - y2 * (E1 - f[i2]) / eta
        a2n = min(max(a2n, L), H)
        if abs(a2n - a2o) < eps * (a2n + a2o + eps):
            streak += 1
            continue
        a1n = a1o + s * (a2o - a2n)
        alpha[i1], alpha[i2] = a1n, a2n
        b1 = b - E1 - y1 * (a1n - a1o) * K[i1, i1] - y2 * (a2n - a2o) * K[i1, i2]
        b2 = b - f[i2] - y1 * (a1n - a1o) * K[i1, i2] - y2 * (a2n - a2o) * K[i2, i2]
        if eps < a1n < C - eps:
            b_new = b1
        elif eps < a2n < C - eps:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        f = f + y1 * (a1n - a1o) * K[i1, :] + y2 * (a2n - a2o) * K[i2, :] + (b_new - b)
        b = b_new
        applied += 1
        streak = 0
    return alpha, b


def svm_fit_predict_ovr(Xtr, ytr, Xte, C, gamma, max_iter=40000):
    classes = np.unique(ytr)
    decisions = []
    svs = []
    for cls in classes:
        yb = np.where(ytr == cls, 1.0, -1.0)
        t0 = time.time()
        alpha, b = smo(Xtr, yb, C, gamma=gamma, max_iter=max_iter)
        sv = alpha > 1e-6
        svs.append((Xtr[sv], yb[sv], alpha[sv], b))
        dec = (alpha[sv] * yb[sv]) @ rbf_kernel(Xtr[sv], Xte, gamma) + b
        decisions.append(dec)
        print(f"    class {cls}: {sv.sum()} SVs, {time.time()-t0:.2f}s")
    return classes[np.argmax(decisions, axis=0)]


def kmeans(X, k, seed, n_init=8, max_iter=300, tol=1e-6):
    rng = mulberry32(seed)
    best = None
    for _ in range(n_init):
        # k-means++ init
        centroids = [X[int(rng() * len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(-1), axis=1)
            total = d2.sum()
            if total <= 0:
                centroids.append(X[int(rng() * len(X))])
                continue
            probs = d2 / total
            centroids.append(X[int(np.searchsorted(np.cumsum(probs), rng()))])
        centroids = np.array(centroids)
        labels = None
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            new_centroids = centroids.copy()
            for c in range(k):
                if (new_labels == c).any():
                    new_centroids[c] = X[new_labels == c].mean(axis=0)
            shift = np.abs(new_centroids - centroids).max()
            centroids = new_centroids
            if labels is not None and (new_labels == labels).all() and shift < tol:
                labels = new_labels
                break
            labels = new_labels
        inertia = d2[np.arange(len(X)), labels].sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), centroids.copy())
    return best[1], best[2]


def kmeans_classifier(Xtr, ytr, Xte, k):
    labels, centroids = kmeans(Xtr, k, seed=42)
    mapping = {}
    for c in range(k):
        members = ytr[labels == c]
        if len(members):
            mapping[c] = int(np.bincount(members).argmax())
    d2 = ((Xte[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return np.array([mapping.get(c, 0) for c in d2.argmin(axis=1)])


def jacobi_pca(Xtr, Xte, n_components=None, variance_ratio=None):
    mu = Xtr.mean(axis=0)
    Xc = Xtr - mu
    d = Xc.shape[1]
    A = (Xc.T @ Xc) / (len(Xc) - 1)
    V = np.eye(d)
    for _ in range(60):
        off = math.sqrt(np.sum(A**2) - np.sum(np.diag(A) ** 2))
        if off < 1e-10:
            break
        for p in range(d - 1):
            for q in range(p + 1, d):
                apq = A[p, q]
                if abs(apq) < 1e-12:
                    continue
                theta = (A[q, q] - A[p, p]) / (2 * apq)
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                # rotate rows/cols p,q of A
                rows = A[[p, q], :].copy()
                A[p, :] = c * rows[0] - s * rows[1]
                A[q, :] = s * rows[0] + c * rows[1]
                cols = A[:, [p, q]].copy()
                A[:, p] = c * cols[:, 0] - s * cols[:, 1]
                A[:, q] = s * cols[:, 0] + c * cols[:, 1]
                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
    eigvals = np.diag(A).copy()
    order = np.argsort(eigvals)[::-1]
    eigvals, V = eigvals[order], V[:, order]
    ratios = eigvals / eigvals.sum()
    if variance_ratio is not None:
        n_components = int(np.searchsorted(np.cumsum(ratios), variance_ratio) + 1)
    W = V[:, :n_components]
    return Xc @ W, (Xte - mu) @ W, ratios, n_components


def accuracy(yt, yp):
    return float((yt == yp).mean())


def macro_f1(yt, yp):
    f1s = []
    for cls in np.unique(yt):
        tp = ((yt == cls) & (yp == cls)).sum()
        fp = ((yt != cls) & (yp == cls)).sum()
        fn = ((yt == cls) & (yp != cls)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def run(name, model, preprocess="std", C=1.0, variance=None, n_components=None):
    X, y, names = load(name)
    tri, tei = stratified_split(y, 0.2, 42)
    Xtr, ytr, Xte, yte = X[tri], y[tri], X[tei], y[tei]
    Xtr, Xte = standard_scale(Xtr, Xte)
    if preprocess == "pca":
        Xtr, Xte, ratios, k = jacobi_pca(Xtr, Xte, n_components, variance)
        print(f"  pca components={k}, explained={ratios[:k].sum():.3f}")
    gamma = 1.0 / (Xtr.shape[1] * Xtr.var(axis=0).mean())
    if model == "kmeans":
        yp = kmeans_classifier(Xtr, ytr, Xte, len(names))
    else:
        yp = svm_fit_predict_ovr(Xtr, ytr, Xte, C, gamma)
    print(f"  acc={accuracy(yte, yp):.3f} f1={macro_f1(yte, yp):.3f}")


def check_jacobi():
    print("== jacobi vs numpy eigh ==")
    for name in ("iris", "digits"):
        X, _, _ = load(name)
        Xs, _ = standard_scale(X, X)
        mu = Xs.mean(axis=0)
        Xc = Xs - mu
        A = Xc.T @ Xc / (len(Xc) - 1)
        # jacobi (reuse the function with no projection)
        Xtr2, _, ratios, _ = jacobi_pca(Xs, Xs, n_components=A.shape[1])
        evals_np = np.linalg.eigvalsh(A)[::-1]
        err = np.abs(np.sort(ratios)[::-1] - evals_np / evals_np.sum()).max()
        print(f"  {name}: max ratio diff vs eigh = {err:.2e}")


if __name__ == "__main__":
    check_jacobi()
    print("== iris svm rbf ==")
    run("iris", "svm")
    print("== wine svm rbf ==")
    run("wine", "svm")
    print("== breast_cancer svm rbf ==")
    run("breast_cancer", "svm")
    print("== iris svm poly ==")
    run("iris", "svm")
    print("== iris kmeans ==")
    run("iris", "kmeans")
    print("== wine kmeans ==")
    run("wine", "kmeans")
    print("== breast_cancer kmeans ==")
    run("breast_cancer", "kmeans")
    print("== digits kmeans ==")
    run("digits", "kmeans")
    print("== wine pca95 + svm rbf ==")
    run("wine", "svm", preprocess="pca", variance=0.95)
    print("== digits pca95 + kmeans ==")
    run("digits", "kmeans", preprocess="pca", variance=0.95)
