"""results/exp*.csv를 읽어 그래프(results/figures/*.png)와 요약표(results/summary.md)를 만든다.

사용법: python src/analyze.py
추가로 수렴 곡선(반복별 상대 잔차)과 BLAS 처리량(GEMM 대 GEMV)을 직접 측정해 그린다.
"""
import csv                                             # CSV 읽기
import os                                              # 경로
import sys                                             # import 경로
import time                                            # 처리량 측정
from collections import defaultdict                    # 그룹 묶기

import matplotlib                                      # 그래프
matplotlib.use("Agg")                                  # 화면 없이 저장
import matplotlib.pyplot as plt                        # 그리기
import numpy as np                                     # 계산

sys.path.insert(0, os.path.dirname(__file__))          # src/ import
from data import make_data                             # noqa: E402
from methods import power_iter_L                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 저장소 루트
RES = os.path.join(ROOT, "results")                    # 결과 폴더
FIG = os.path.join(RES, "figures")                     # 그림 폴더
os.makedirs(FIG, exist_ok=True)
COLORS = {"normal_eq": "#2a6fdb", "gd": "#e07b00", "cg": "#1a9e5c"}  # 방법별 색
LABELS = {"normal_eq": "Normal eq.", "gd": "GD", "cg": "CG"}         # 범례 이름
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load(exp):                                         # CSV → dict 리스트 (첫 줄 주석은 건너뜀)
    with open(os.path.join(RES, f"exp{exp}.csv")) as f:
        env = f.readline().strip("# \n")               # 환경 정보 줄
        rows = list(csv.DictReader(f))
    for r in rows:                                     # 문자열 → 숫자
        for k in ("n", "d", "seed", "iters"):
            r[k] = int(r[k])
        for k in ("kappa", "eta_scale", "time_s", "rel_residual"):
            r[k] = float(r[k])
        r["peak_mem_bytes"] = float(r["peak_mem_bytes"]) if r["peak_mem_bytes"] else None
    return rows, env


def agg(rows, key):                                    # (method, key값)별 평균·표준편차
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r[key])].append(r)
    out = {}
    for (m, x), rs in g.items():
        t = np.array([r["time_s"] for r in rs])        # 시드별 시간
        it = np.array([r["iters"] for r in rs])        # 시드별 반복 수
        mem = [r["peak_mem_bytes"] for r in rs if r["peak_mem_bytes"] is not None]
        res = max(r["rel_residual"] for r in rs)       # 최악 잔차
        out[(m, x)] = dict(t=t.mean(), ts=t.std(), it=it.mean(), its=it.std(),
                           mem=mem[0] if mem else None, res=res, k=len(rs))
    return out


def slope(x, y):                                       # 로그-로그 최소제곱 기울기
    return np.polyfit(np.log10(x), np.log10(y), 1)[0]


def fmt_t(t, s):                                       # 시간 표기 (ms)
    return f"{t*1e3:.2f} ± {s*1e3:.2f}"


md = []                                                # summary.md 내용

# ------------------------------------------------------------ 실험 1: 차원 d
rows, env = load(1)
md.append(f"환경: `{env}`\n")
A1 = agg(rows, "d")
ds = sorted({r["d"] for r in rows})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for m in ("normal_eq", "gd", "cg"):
    t = [A1[(m, d)]["t"] for d in ds]
    s = [A1[(m, d)]["ts"] for d in ds]
    ax[0].errorbar(ds, t, yerr=s, marker="o", color=COLORS[m], label=LABELS[m], capsize=3)
    ax[1].plot(ds, [A1[(m, d)]["mem"] / 1e6 for d in ds], marker="o", color=COLORS[m], label=LABELS[m])
ds_a = np.array(ds, float)
ax[1].plot(ds_a, 8 * ds_a**2 / 1e6, "k--", lw=1, label="theory 8d² bytes")
ax[0].set(xscale="log", yscale="log", xlabel="d (features)", ylabel="time [s]",
          title="Exp1: time vs d (n=10000, κ=100)")
ax[1].set(xscale="log", yscale="log", xlabel="d (features)", ylabel="peak extra memory [MB]",
          title="Exp1: tracemalloc peak vs d")
ax[0].legend(); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp1_d.png"), dpi=200); plt.close(fig)

md.append("### 실험 1 (n=10000, κ=100)\n")
md.append("| d | 정규방정식 시간 [ms] | GD 시간 [ms] | GD 반복 | CG 시간 [ms] | CG 반복 | 메모리 NE / GD / CG [KB] |")
md.append("|---|---|---|---|---|---|---|")
for d in ds:
    a, b, c = A1[("normal_eq", d)], A1[("gd", d)], A1[("cg", d)]
    md.append(f"| {d} | {fmt_t(a['t'], a['ts'])} | {fmt_t(b['t'], b['ts'])} | {b['it']:.0f} | "
              f"{fmt_t(c['t'], c['ts'])} | {c['it']:.0f} | "
              f"{a['mem']/1e3:.0f} / {b['mem']/1e3:.0f} / {c['mem']/1e3:.0f} |")
md.append("")
for m in ("normal_eq", "gd", "cg"):                    # 시간의 d 기울기 (큰 d 구간)
    big = [d for d in ds if d >= 200]
    md.append(f"- {LABELS[m]} 시간 기울기 (d ≥ 200): {slope(big, [A1[(m, d)]['t'] for d in big]):.2f}")
md.append(f"- 정규방정식 메모리 기울기: {slope(ds, [A1[('normal_eq', d)]['mem'] for d in ds]):.2f}, "
          f"CG 메모리 기울기: {slope(ds, [A1[('cg', d)]['mem'] for d in ds]):.2f}\n")

# ------------------------------------------------------------ 실험 5 (확장): 실제 교차점
rows5, _ = load(5)
A5 = agg(rows5, "d")
ds5 = sorted({r["d"] for r in rows5})
allD = sorted(set(ds) | set(ds5))                      # 실험 1 + 5의 d
def pick(m, d):                                        # d=2000은 실험 1 값 사용
    return A1[(m, d)] if (m, d) in A1 else A5[(m, d)]
fig, ax = plt.subplots(figsize=(6, 3.9))
for m in ("normal_eq", "cg"):
    ax.errorbar(allD, [pick(m, d)["t"] for d in allD], yerr=[pick(m, d)["ts"] for d in allD],
                marker="o", color=COLORS[m], label=LABELS[m], capsize=3)
ax.plot(ds, [A1[("gd", d)]["t"] for d in ds], marker="o", color=COLORS["gd"], label="GD (Exp1)")
ax.axvline(168, color="gray", ls=":", lw=1)
ax.text(180, 0.8, "flop theory\nd≈2K≈168", fontsize=8, color="gray", transform=ax.get_xaxis_transform())
ax.set(xscale="log", yscale="log", xlabel="d (features)", ylabel="time [s]",
       title="Exp1+5: crossover (n=10000, κ=100)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_crossover.png"), dpi=200); plt.close(fig)
md.append("### 실험 5 (확장, n=10000, κ=100, 시드 0~2)\n")
md.append("| d | 정규방정식 [ms] | CG [ms] | CG 반복 | NE/CG 시간비 | 메모리 NE / CG [MB] |")
md.append("|---|---|---|---|---|---|")
for d in ds5:
    a, c = A5[("normal_eq", d)], A5[("cg", d)]
    md.append(f"| {d} | {fmt_t(a['t'], a['ts'])} | {fmt_t(c['t'], c['ts'])} | {c['it']:.0f} | "
              f"{a['t']/c['t']:.2f} | {a['mem']/1e6:.1f} / {c['mem']/1e6:.2f} |")
r = [(d, pick("normal_eq", d)["t"] / pick("cg", d)["t"]) for d in allD]
for (d0, r0), (d1, r1) in zip(r, r[1:]):               # 시간비가 1을 넘는 구간에서 로그 보간
    if r0 < 1 <= r1:
        dx = 10 ** (np.log10(d0) + (np.log10(d1) - np.log10(d0)) * (0 - np.log10(r0)) / (np.log10(r1) - np.log10(r0)))
        md.append(f"\n- 정규방정식/CG 시간비가 1이 되는 d (로그 보간): 약 {dx:.0f}")
md.append("")

# ------------------------------------------------------------ 실험 2: 조건수 κ
rows, _ = load(2)
A2 = agg(rows, "kappa")
ks = sorted({r["kappa"] for r in rows})
eps = 1e-8
k_a = np.array(ks)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for m in ("gd", "cg"):
    ax[0].errorbar(ks, [A2[(m, k)]["it"] for k in ks], yerr=[A2[(m, k)]["its"] for k in ks],
                   marker="o", color=COLORS[m], label=LABELS[m], capsize=3)
ax[0].plot(k_a, k_a * np.log(1 / eps), "--", color=COLORS["gd"], lw=1, label="κ ln(1/ε)")
ax[0].plot(k_a, k_a * np.log(1 / (eps * k_a)), ":", color=COLORS["gd"], lw=1.5, label="κ ln(1/(εκ)) (residual-corrected)")
ax[0].plot(k_a, np.sqrt(k_a) / 2 * np.log(2 / eps), "--", color=COLORS["cg"], lw=1, label="(√κ/2) ln(2/ε)")
ax[0].axhline(200, color="gray", ls=":", lw=1, label="d = 200")
ax[0].set(xscale="log", yscale="log", xlabel="κ", ylabel="iterations",
          title="Exp2: iterations vs κ (n=5000, d=200)")
ax[0].legend(fontsize=8)
for m in ("normal_eq", "gd", "cg"):
    ax[1].errorbar(ks, [A2[(m, k)]["t"] for k in ks], yerr=[A2[(m, k)]["ts"] for k in ks],
                   marker="o", color=COLORS[m], label=LABELS[m], capsize=3)
ax[1].set(xscale="log", yscale="log", xlabel="κ", ylabel="time [s]", title="Exp2: time vs κ")
ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp2_kappa.png"), dpi=200); plt.close(fig)

md.append("### 실험 2 (n=5000, d=200)\n")
md.append("| κ | 정규방정식 [ms] | 잔차 | GD 반복 | 이론 κln(1/ε) | 보정 κln(1/(εκ)) | GD [ms] | CG 반복 | 이론 (√κ/2)ln(2/ε) | CG [ms] | CG 잔차 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for k in ks:
    a, b, c = A2[("normal_eq", k)], A2[("gd", k)], A2[("cg", k)]
    md.append(f"| {k:.0e} | {fmt_t(a['t'], a['ts'])} | {a['res']:.1e} | {b['it']:.0f} ± {b['its']:.0f} | "
              f"{k*np.log(1/eps):.0f} | {k*np.log(1/(eps*k)):.0f} | {fmt_t(b['t'], b['ts'])} | {c['it']:.0f} ± {c['its']:.0f} | "
              f"{np.sqrt(k)/2*np.log(2/eps):.0f} | {fmt_t(c['t'], c['ts'])} | {c['res']:.1e} |")
md.append("")
md.append(f"- GD 반복 수 기울기 (전체): {slope(ks, [A2[('gd', k)]['it'] for k in ks]):.3f}")
md.append(f"- CG 반복 수 기울기 (전체): {slope(ks, [A2[('cg', k)]['it'] for k in ks]):.3f}, "
          f"(κ ≤ 1e2): {slope(ks[:2], [A2[('cg', k)]['it'] for k in ks[:2]]):.3f}, "
          f"(κ ≤ 1e3): {slope(ks[:3], [A2[('cg', k)]['it'] for k in ks[:3]]):.3f}\n")

# ------------------------------------------------------------ 실험 3: 데이터 수 n
rows, _ = load(3)
A3 = agg(rows, "n")
ns = sorted({r["n"] for r in rows})
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in ("normal_eq", "gd", "cg"):
    ax.errorbar(ns, [A3[(m, n)]["t"] for n in ns], yerr=[A3[(m, n)]["ts"] for n in ns],
                marker="o", color=COLORS[m], label=LABELS[m], capsize=3)
ax.set(xscale="log", yscale="log", xlabel="n (samples)", ylabel="time [s]",
       title="Exp3: time vs n (d=200, κ=100)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_n.png"), dpi=200); plt.close(fig)

md.append("### 실험 3 (d=200, κ=100)\n")
md.append("| n | 정규방정식 [ms] | GD [ms] | GD 반복 | CG [ms] | CG 반복 |")
md.append("|---|---|---|---|---|---|")
for n in ns:
    a, b, c = A3[("normal_eq", n)], A3[("gd", n)], A3[("cg", n)]
    md.append(f"| {n} | {fmt_t(a['t'], a['ts'])} | {fmt_t(b['t'], b['ts'])} | {b['it']:.0f} | "
              f"{fmt_t(c['t'], c['ts'])} | {c['it']:.0f} |")
md.append("")
for m in ("normal_eq", "gd", "cg"):
    md.append(f"- {LABELS[m]} 시간의 n 기울기 (전체): {slope(ns, [A3[(m, n)]['t'] for n in ns]):.2f}, "
              f"(n ≥ 10000): {slope(ns[2:], [A3[(m, n)]['t'] for n in ns[2:]]):.2f}")
for m in ("gd", "cg"):
    per = [A3[(m, n)]["t"] / A3[(m, n)]["it"] for n in ns]
    md.append(f"- {LABELS[m]} 반복 1번 시간 [µs]: " + ", ".join(f"{p*1e6:.0f}" for p in per))
md.append("")

# ------------------------------------------------------------ 실험 4: 학습률
rows, _ = load(4)
A4 = agg(rows, "eta_scale")
es = sorted({r["eta_scale"] for r in rows})
md.append("### 실험 4 (GD, n=5000, d=200, κ=1e3)\n")
md.append("| η | 반복 수 | 시간 [ms] | 최종 상대 잔차 (최악) | 이론 수렴 비율 max|1−ηλ|/... |")
md.append("|---|---|---|---|---|")
for e in es:
    a = A4[("gd", e)]
    rho = max(abs(1 - e / 1e3), abs(1 - e))            # λ/L ∈ [1/κ, 1] → |1 − η λ| 최댓값
    md.append(f"| {e}/L | {a['it']:.0f} ± {a['its']:.0f} | {fmt_t(a['t'], a['ts'])} | {a['res']:.1e} | {rho:.4f} |")
md.append("")

# 수렴 곡선: 시드 0, 실험 4 조건에서 반복별 상대 잔차
X, y = make_data(5000, 200, 1e3, seed=0)
b = X.T @ y
bn = np.linalg.norm(b)
L = power_iter_L(X)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for e, col in zip(es, ["#7aa6e8", "#e07b00", "#b5179e", "#d62828"]):
    th, hist = np.zeros(200), []
    for k in range(40000):
        r = b - X.T @ (X @ th)
        hist.append(np.linalg.norm(r) / bn)
        if hist[-1] <= 1e-8 or hist[-1] > 1e6:
            break
        th = th + (e / L) * r
    ax[0].plot(hist, color=col, label=f"η = {e}/L")
ax[0].set(yscale="log", xlabel="iteration", ylabel="relative residual",
          title="GD convergence (κ=1e3)", ylim=(1e-9, 1e6))
ax[0].legend()
# CG 곡선
th = np.zeros(200); r = b.copy(); p = r.copy(); rs = r @ r; hist = [1.0]
for k in range(2000):
    Ap = X.T @ (X @ p); al = rs / (p @ Ap); th += al * p; r -= al * Ap
    rn = r @ r; hist.append(np.sqrt(rn) / bn)
    if hist[-1] <= 1e-8:
        break
    p = r + rn / rs * p; rs = rn
ax[1].plot(hist, color=COLORS["cg"], label="CG")
ax[1].axvline(200, color="gray", ls=":", label="d = 200")
ax[1].set(yscale="log", xlabel="iteration", ylabel="relative residual", title="CG convergence (κ=1e3)")
ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp4_convergence.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------ BLAS 처리량: GEMM 대 GEMV
md.append("### BLAS 처리량 (n=10000)\n")
md.append("| d | XᵀX (GEMM) GFLOP/s | Xᵀ(Xp) (GEMV 2번) GFLOP/s | 비율 |")
md.append("|---|---|---|---|")
thr = []
for d in [50, 100, 200, 500, 1000, 2000]:
    X = np.random.default_rng(0).standard_normal((10000, d))
    p = np.ones(d)
    t0 = time.perf_counter(); [X.T @ X for _ in range(3)]; tg = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter(); [X.T @ (X @ p) for _ in range(50)]; tv = (time.perf_counter() - t0) / 50
    gemm = 2 * 10000 * d * d / tg / 1e9                # XᵀX: 2nd² flop
    gemv = 4 * 10000 * d / tv / 1e9                    # Xᵀ(Xp): 4nd flop
    thr.append((d, gemm, gemv))
    md.append(f"| {d} | {gemm:.0f} | {gemv:.1f} | {gemm/gemv:.0f}× |")
md.append("")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot([t[0] for t in thr], [t[1] for t in thr], "o-", color=COLORS["normal_eq"], label="XᵀX (matrix-matrix)")
ax.plot([t[0] for t in thr], [t[2] for t in thr], "o-", color=COLORS["cg"], label="Xᵀ(Xp) (matrix-vector)")
ax.set(xscale="log", yscale="log", xlabel="d", ylabel="GFLOP/s", title="BLAS throughput (n=10000)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "blas_throughput.png"), dpi=200); plt.close(fig)

with open(os.path.join(RES, "summary.md"), "w") as f:
    f.write("# 실험 결과 요약 (src/analyze.py 자동 생성)\n\n" + "\n".join(md) + "\n")
print("\n".join(md))
