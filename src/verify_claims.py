"""보고서에서 '추정'으로 남겼던 주장들을 직접 확인하는 스크립트.

사용법: python src/verify_claims.py [all|sme|lapack|gd|determinism]
결과는 화면에 출력하고 results/verify_claims.txt에도 저장한다.
"""
import os                                              # 환경 변수, 경로
import subprocess                                      # 단일 스레드 측정용 하위 프로세스
import sys                                             # 인자, 실행 파일 경로
import textwrap                                        # 하위 프로세스 코드

import numpy as np                                     # 행렬 연산

sys.path.insert(0, os.path.dirname(__file__))          # src/ import
from data import make_data                             # noqa: E402
from methods import conjugate_gradient, gradient_descent, power_iter_L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = []                                               # 저장할 출력


def log(s=""):                                         # 출력 + 기록
    print(s, flush=True)
    OUT.append(s)


def run_py(code, env_extra=None):                      # 새 파이썬 프로세스에서 코드 실행 → stdout
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


# ------------------------------------------------------------ 1. GEMM이 행렬 유닛(SME)을 쓰는가
def check_sme():
    log("## 1. GEMM 단일 스레드 처리량 (NEON 이론 최대치와 비교)")
    code = """
    import time, numpy as np
    X = np.random.default_rng(0).standard_normal((10000, 1000))
    X.T @ X                                            # 워밍업
    t0 = time.perf_counter(); [X.T @ X for _ in range(5)]; t = (time.perf_counter() - t0) / 5
    print(f"{2 * 10000 * 1000**2 / t / 1e9:.1f}")     # GFLOP/s
    """
    for th in ["1", "4", "10"]:                        # BLAS 스레드 수 제한 (Accelerate: VECLIB_MAXIMUM_THREADS)
        g = run_py(code, {"VECLIB_MAXIMUM_THREADS": th})
        log(f"- VECLIB_MAXIMUM_THREADS={th:>2}: XᵀX (n=10000, d=1000) {g} GFLOP/s")
    # NEON FP64 한 코어 최대: 128-bit FMA 파이프 4개 × 2 double × 2 flop = 16 flop/cycle
    log("- 비교: NEON FP64 한 코어 이론 최대 = 16 flop/cycle × 클럭(≤ 4.6 GHz 가정) ≈ 74 GFLOP/s")
    log("")


# ------------------------------------------------------------ 2. tracemalloc이 LAPACK 작업 공간을 잡는가
def check_lapack():
    log("## 2. np.linalg.solve: tracemalloc 최대값 대 실제 RSS 증가")
    code = """
    import resource, tracemalloc, numpy as np
    d = 4000
    A = np.random.default_rng(0).standard_normal((d, d)); A = A @ A.T + d * np.eye(d)
    b = np.ones(d)
    np.linalg.solve(A[:10, :10], b[:10])               # 라이브러리 초기화
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # macOS: bytes
    tracemalloc.start()
    x = np.linalg.solve(A, b)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(peak, rss1 - rss0, 8 * d * d)
    """
    peak, drss, d2 = map(int, run_py(code).split())
    log(f"- d=4000, 행렬 A 하나 = 8d² = {d2/1e6:.0f} MB")
    log(f"- tracemalloc 최대값: {peak/1e6:.2f} MB")
    log(f"- 최대 RSS 증가:      {drss/1e6:.0f} MB  (= {drss/d2:.2f} × 8d²)")
    log("")


# ------------------------------------------------------------ 3. GD 반복 수: 스펙트럼으로 정확히 예측
def check_gd():
    log("## 3. GD 반복 수: 실측 대 스펙트럼 정확 계산 대 보정식 κ ln(1/(εκ))")
    log("| κ | 실측 (시드 0) | 정확 계산 | 보정식 | ln(‖b‖/(λ_min|c_min|)) | ln κ |")
    log("|---|---|---|---|---|---|")
    eps = 1e-8
    for kappa in [1e1, 1e2, 1e3, 1e4]:
        X, y = make_data(5000, 200, kappa, seed=0)
        _, k_meas = gradient_descent(X, y, max_iter=3 * 10**6)   # 실측
        A = X.T @ X
        lam, Q = np.linalg.eigh(A)                     # A = Q diag(λ) Qᵀ
        b = X.T @ y
        c = Q.T @ b                                    # b의 고유 성분 (= λ_i · θ*_i 성분)
        eta = 1.0 / power_iter_L(X)                    # GD와 같은 학습률
        # θ₀ = 0이면 r_k = b − Aθ_k, 고유 성분은 c_i (1 − ηλ_i)^k
        rho = 1 - eta * lam
        k = np.arange(0, 3 * 10**6, dtype=float)
        lo, hi = 0, int(3 * kappa * np.log(1 / eps)) + 10   # 이분 탐색으로 처음 ‖r_k‖ ≤ ε‖b‖ 되는 k
        target = eps * np.linalg.norm(b)
        while lo < hi:
            mid = (lo + hi) // 2
            if np.linalg.norm(c * rho ** mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        k_exact = lo + 1                               # 함수는 k=1부터 세고, 검사 후 갱신하므로 +1
        corr = kappa * np.log(1 / (eps * kappa))
        i = np.argmin(lam)
        log(f"| {kappa:.0e} | {k_meas} | {k_exact} | {corr:.0f} | "
            f"{np.log(np.linalg.norm(b) / abs(c[i])):.2f} | {np.log(kappa):.2f} |")
    log("")


# ------------------------------------------------------------ 4. 같은 시드면 반복 수가 정확히 같은가
def check_determinism():
    log("## 4. 결정성: 같은 입력으로 여러 번 실행")
    X, y = make_data(5000, 200, 1e4, seed=0)
    res = [conjugate_gradient(X, y) for _ in range(10)]
    same_cg = all(np.array_equal(r[0], res[0][0]) for r in res)
    log(f"- CG 10회 (κ=1e4): 반복 수 {sorted({r[1] for r in res})}, θ 비트 단위 동일: {same_cg}")
    res = [gradient_descent(X, y) for _ in range(3)]
    same_gd = all(np.array_equal(r[0], res[0][0]) for r in res)
    log(f"- GD 3회 (κ=1e4): 반복 수 {sorted({r[1] for r in res})}, θ 비트 단위 동일: {same_gd}")
    code = """
    import sys; sys.path.insert(0, "src")
    from data import make_data; from methods import conjugate_gradient
    X, y = make_data(5000, 200, 1e4, seed=0); print(conjugate_gradient(X, y)[1])
    """
    for th in ["1", "10"]:                             # 스레드 수가 다르면 덧셈 순서가 달라질 수 있음
        it = run_py(code, {"VECLIB_MAXIMUM_THREADS": th})
        log(f"- CG, VECLIB_MAXIMUM_THREADS={th:>2}: 반복 수 {it}")
    log("")


if __name__ == "__main__":
    os.chdir(ROOT)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    checks = dict(sme=check_sme, lapack=check_lapack, gd=check_gd, determinism=check_determinism)
    for name, fn in checks.items():
        if which in ("all", name):
            fn()
    with open(os.path.join(ROOT, "results", "verify_claims.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
