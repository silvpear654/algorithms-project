"""보조 확인: X가 희소 행렬이면 CG가 정규방정식을 더 일찍 이기는가.

본 실험(밀집 행렬)과 같은 정지 기준(상대 잔차 ≤ 1e-8)을 쓴다. scipy가 필요하다 (pip install scipy).
사용법: python src/sparse_check.py
"""
import time                                            # 시간 측정
import numpy as np                                     # 밀집 연산
import scipy.sparse as sp                              # 희소 행렬


def cg(matvec, b, tol=1e-8, max_iter=100000):          # methods.py의 CG와 같은 식, A p만 함수로 받음
    theta = np.zeros_like(b); r = b.copy(); p = r.copy()
    rs = r @ r; bn = np.linalg.norm(b)
    for k in range(1, max_iter + 1):
        Ap = matvec(p)                                 # A p = Xᵀ(X p)
        a = rs / (p @ Ap); theta += a * p; r -= a * Ap
        rn = r @ r
        if np.sqrt(rn) / bn <= tol:
            return theta, k
        p = r + rn / rs * p; rs = rn
    return theta, max_iter


def best_of(fn, reps=3):                               # 3번 중 최소 시간 (잡음 줄이기)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), out


rng = np.random.default_rng(0)                         # 시드 고정
n, density = 20000, 0.005                              # 행 20000개, 원소의 0.5%만 0이 아님
for scale in [False, True]:                            # True: 열 크기를 1~10배로 벌려 κ를 수백으로 키움
  print(f"\nn={n}, density={density}, 열 스케일링={'1~10배 (κ 증가)' if scale else '없음'}")
  print("| d | nnz | κ(XᵀX) | NE(희소 XᵀX→밀집 solve) [ms] | CG 희소 [ms] | CG 반복 | CG 밀집 X [ms] | 상대 잔차 NE / CG |")
  print("|---|---|---|---|---|---|---|---|")
  for d in [500, 1000, 2000, 4000]:
      Xs = sp.random(n, d, density=density, format="csr", random_state=rng,
                     data_rvs=rng.standard_normal)       # 희소 X (CSR)
      if scale:                                      # 열 j에 √(10^(2j/(d-1))) 곱하기 → 열 노름 1~10배
          Xs = (Xs @ sp.diags(np.sqrt(np.logspace(0, 2, d)))).tocsr()
      Xd = Xs.toarray()                                  # 같은 X의 밀집 버전
      y = rng.standard_normal(n)
      b = Xs.T @ y
      XsT = Xs.T.tocsr()                                 # Xᵀ도 CSR로 미리 변환

      def ne():                                          # 정규방정식: XᵀX는 희소 곱, 풀이는 밀집 LU
          A = (XsT @ Xs).toarray()
          return np.linalg.solve(A, XsT @ y)
      t_ne, th_ne = best_of(ne)
      t_cs, (th_cs, k) = best_of(lambda: cg(lambda p: XsT @ (Xs @ p), XsT @ y))
      t_cd, _ = best_of(lambda: cg(lambda p: Xd.T @ (Xd @ p), Xd.T @ y))
      ev = np.linalg.eigvalsh((XsT @ Xs).toarray())     # 조건수 확인 (밀집 실험의 κ=100과 비교용)
      kap = ev[-1] / ev[0]
      res = lambda th: np.linalg.norm(b - XsT @ (Xs @ th)) / np.linalg.norm(b)
      print(f"| {d} | {Xs.nnz} | {kap:.1f} | {t_ne*1e3:.1f} | {t_cs*1e3:.1f} | {k} | {t_cd*1e3:.1f} | "
            f"{res(th_ne):.1e} / {res(th_cs):.1e} |", flush=True)
