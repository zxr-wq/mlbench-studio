import numpy as np
import sys
sys.path.insert(0, r"c:\Users\江郦娜\Desktop\mlbench-studio-main\mlbench-studio-main\scripts")
import prototype_validate as p

X, y, _ = p.load("iris")
tri, tei = p.stratified_split(y, 0.2, 42)
Xtr, ytr = X[tri], y[tri]
Xtr, _ = p.standard_scale(Xtr, Xte=Xtr)
gamma = 1.0 / (Xtr.shape[1] * Xtr.var(axis=0).mean())
y = np.where(ytr == 0, 1.0, -1.0)
n = len(y)
K = p.rbf_kernel(Xtr, Xtr, gamma)
C, tol, eps = 1.0, 1e-3, 1e-5
alpha = np.zeros(n); b = 0.0; f = -y.copy()
it = 0; applied = 0; streak = 0
while it < 3000 and applied < 3000:
    it += 1
    r = y * f
    candidates = [i for i in range(n) if (r[i] < -tol and alpha[i] < C - eps) or (r[i] > tol and alpha[i] > eps)]
    if not candidates:
        print("no candidates at it", it); break
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
    eta = 2 * K[i1, i2] - K[i1, i1] - K[i2, i2]
    ok_LH = L != H
    ok_eta = eta < 0
    if not (ok_LH and ok_eta):
        streak += 1
        if it < 30 or it % 500 == 0: print(f"it={it} SKIP i1={i1} i2={i2} eta={eta:.3e} L={L} H={H} streak={streak} |f|={np.abs(f).max():.3e} b={b:.3e}")
        continue
    a2n = min(max(a2o - y2 * (E1 - f[i2]) / eta, L), H)
    if abs(a2n - a2o) < eps * (a2n + a2o + eps):
        streak += 1
        if it < 30 or it % 500 == 0: print(f"it={it} TINY i1={i1} i2={i2} |f|={np.abs(f).max():.3e} b={b:.3e}")
        continue
    a1n = a1o + s * (a2o - a2n)
    alpha[i1], alpha[i2] = a1n, a2n
    b1 = E1 + y1 * (a1n - a1o) * K[i1, i1] + y2 * (a2n - a2o) * K[i1, i2] + b
    b2 = f[i2] + y1 * (a1n - a1o) * K[i1, i2] + y2 * (a2n - a2o) * K[i2, i2] + b
    b_new = b1 if eps < a1n < C - eps else (b2 if eps < a2n < C - eps else (b1 + b2) / 2)
    f = f + y1 * (a1n - a1o) * K[i1, :] + y2 * (a2n - a2o) * K[i2, :] + (b_new - b)
    b = b_new
    applied += 1; streak = 0
    if it < 30 or it % 500 == 0:
        print(f"it={it} i1={i1} i2={i2} eta={eta:.3e} da2={a2n-a2o:+.3e} b={b:.4e} |f|max={np.abs(f).max():.4e} nCand={len(candidates)}")
print("final b:", b, "alpha_sum:", alpha.sum())
