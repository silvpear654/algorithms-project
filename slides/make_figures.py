"""슬라이드 5·6장 그림: 길쭉한 2차 손실 등고선 위 GD 지그재그 대 CG 2단계 도착."""
import os                                              # 경로
import matplotlib                                      # 그래프
matplotlib.use("Agg")                                  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt                        # 그리기
import numpy as np                                     # 행렬 연산

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")  # 저장 폴더
os.makedirs(OUT, exist_ok=True)

# 2차원 예시: g(θ) = ½ θᵀAθ − bᵀθ, κ(A) = 10, 회전시켜 축과 어긋나게
c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)            # 30도 회전
R = np.array([[c, -s], [s, c]])                        # 회전행렬
A = R @ np.diag([10.0, 1.0]) @ R.T                     # 고윳값 10, 1 → κ = 10
theta_star = np.array([1.0, -0.5])                     # 최적해
b = A @ theta_star                                     # Aθ* = b
theta0 = np.array([-2.5, 2.0])                         # 공통 시작점


def gd_path(eta, steps):                               # GD 경로
    th, path = theta0.copy(), [theta0.copy()]
    for _ in range(steps):
        th = th - eta * (A @ th - b)                   # θ ← θ − η∇g
        path.append(th.copy())
    return np.array(path)


def cg_path():                                         # CG 경로 (2차원 → 2번 만에 도착)
    th = theta0.copy()
    r = b - A @ th                                     # r₀
    p = r.copy()                                       # p₀
    path = [th.copy()]
    for _ in range(2):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)                     # αₖ
        th = th + alpha * p                            # θ 갱신
        r_new = r - alpha * Ap                         # r 갱신
        p = r_new + (r_new @ r_new) / (r @ r) * p      # 새 켤레 방향
        r = r_new
        path.append(th.copy())
    return np.array(path)


L = 10.0                                               # λmax
g1, g2 = np.meshgrid(np.linspace(-3.5, 3.5, 300), np.linspace(-3.5, 3.5, 300))
G = np.stack([g1, g2], -1) - theta_star                # θ − θ*
Z = 0.5 * np.einsum("...i,ij,...j->...", G, A, G)      # 등고선 값 ½eᵀAe


def panel(ax, path, color, title):
    ax.contour(g1, g2, Z, levels=np.geomspace(0.02, 60, 18), colors="0.75", linewidths=0.7)
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, ms=3, lw=1.2)
    ax.plot(*theta_star, "*", color="red", ms=12)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


fig, ax = plt.subplots(figsize=(4, 4.2))               # 슬라이드 5장: GD 단독
panel(ax, gd_path(1.8 / L, 25), "tab:blue", "GD (κ = 10, η = 1.8/L), 25 steps")
fig.savefig(os.path.join(OUT, "gd_zigzag.png"), dpi=250, bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))       # 슬라이드 6장: GD 대 CG
panel(axes[0], gd_path(1.8 / L, 25), "tab:blue", "GD: zigzag (25 steps)")
panel(axes[1], cg_path(), "tab:orange", "CG: d = 2 steps")
fig.savefig(os.path.join(OUT, "gd_vs_cg.png"), dpi=250, bbox_inches="tight")
print("saved →", OUT)
