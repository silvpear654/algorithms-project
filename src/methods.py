"""최소제곱 선형회귀의 세 풀이: 정규방정식, 경사하강법(GD), 켤레기울기법(CG).

모든 반복법은 같은 정지 기준(상대 잔차 ‖Xᵀy − XᵀXθ‖/‖Xᵀy‖ ≤ tol)과 같은 시작점 θ₀ = 0을 쓴다.
반환값은 (theta, iters) 튜플이다.
"""
import numpy as np                                     # 행렬 연산 라이브러리


def rel_residual(X, y, theta):                         # 공통 품질 지표: 상대 잔차
    b = X.T @ y                                        # b = Xᵀy
    r = b - X.T @ (X @ theta)                          # r = Xᵀy − XᵀXθ (XᵀX를 만들지 않음)
    return np.linalg.norm(r) / np.linalg.norm(b)       # ‖r‖ / ‖Xᵀy‖


# ---------------------------------------------------------------- ① 정규방정식
def normal_eq(X, y):                                   # 직접법
    A = X.T @ X                                        # d×d, 약 n·d² 연산, 메모리 d²
    b = X.T @ y                                        # 길이 d, 약 n·d 연산
    theta = np.linalg.solve(A, b)                      # LU 분해로 풀기, 약 (2/3)d³
    return theta, 1                                    # 반복 없음 → 1로 기록


# ---------------------------------------------------------------- ② 경사하강법
def power_iter_L(X, iters=100, seed=0):                # λmax(XᵀX) 추정 (eig의 d³ 피하기)
    v = np.random.default_rng(seed).standard_normal(X.shape[1])  # 임의 시작 벡터
    v /= np.linalg.norm(v)                             # 단위 벡터로
    for _ in range(iters):                             # 최대 고유벡터 방향으로 수렴
        w = X.T @ (X @ v)                              # w = XᵀX v, 약 2nd
        v = w / np.linalg.norm(w)                      # 정규화
    return v @ (X.T @ (X @ v))                         # 레일리 몫 vᵀAv ≈ λmax


def gradient_descent(X, y, tol=1e-8, max_iter=10**6, eta_scale=1.0):
    """eta = eta_scale / L. eta_scale < 2 이면 수렴, > 2 이면 발산 (이론)."""
    d = X.shape[1]                                     # 특성 수
    L = power_iter_L(X)                                # L = λmax(XᵀX)
    eta = eta_scale / L                                # 학습률 (θ ← θ + η r, r = Xᵀy − XᵀXθ)
    b = X.T @ y                                        # Xᵀy
    b_norm = np.linalg.norm(b)                         # ‖Xᵀy‖
    theta = np.zeros(d)                                # θ₀ = 0
    for k in range(1, max_iter + 1):                   # 반복
        r = b - X.T @ (X @ theta)                      # r = −n∇f(θ), 약 2nd
        res = np.linalg.norm(r) / b_norm               # 상대 잔차
        if res <= tol:                                 # 정지 기준
            return theta, k                            # 해와 반복 수
        if not np.isfinite(res) or res > 1e12:         # 발산 감지 (η > 2/L 실험용)
            return theta, k                            # 발산한 상태 그대로 반환 → 잔차로 판별
        theta = theta + eta * r                        # θ ← θ + η·r
    return theta, max_iter                             # 최대 반복 도달


# ---------------------------------------------------------------- ③ 켤레기울기법
def conjugate_gradient(X, y, tol=1e-8, max_iter=None):  # A = XᵀX를 만들지 않는 CG (CGNR)
    d = X.shape[1]                                     # 특성 수
    max_iter = max_iter or 10 * d                      # 이론상 d번이지만 반올림 오차 대비 여유
    b = X.T @ y                                        # b = Xᵀy
    b_norm = np.linalg.norm(b)                         # ‖b‖
    theta = np.zeros(d)                                # θ₀ = 0
    r = b.copy()                                       # r₀ = b − A·0 = b
    p = r.copy()                                       # p₀ = r₀
    rs_old = r @ r                                     # r₀ᵀr₀
    for k in range(1, max_iter + 1):                   # 반복
        Ap = X.T @ (X @ p)                             # A p = Xᵀ(Xp), 약 2nd
        alpha = rs_old / (p @ Ap)                      # αₖ = rₖᵀrₖ / pₖᵀApₖ
        theta = theta + alpha * p                      # θₖ₊₁ = θₖ + αₖ pₖ
        r = r - alpha * Ap                             # rₖ₊₁ = rₖ − αₖ Apₖ
        rs_new = r @ r                                 # rₖ₊₁ᵀrₖ₊₁
        if np.sqrt(rs_new) / b_norm <= tol:            # GD와 같은 정지 기준
            return theta, k                            # 해와 반복 수
        p = r + (rs_new / rs_old) * p                  # pₖ₊₁ = rₖ₊₁ + βₖ pₖ
        rs_old = rs_new                                # 다음 반복 준비
    return theta, max_iter                             # 최대 반복 도달


METHODS = {                                            # 이름 → 함수 (benchmark에서 사용)
    "normal_eq": normal_eq,
    "gd": gradient_descent,
    "cg": conjugate_gradient,
}


if __name__ == "__main__":                             # 정확도 검증: 세 방법이 같은 해를 내는지
    from data import make_data                         # 같은 폴더의 데이터 생성 함수
    X, y = make_data(3000, 100, 100, seed=0)           # n=3000, d=100, κ=100
    ref = np.linalg.lstsq(X, y, rcond=None)[0]         # 기준해 (SVD 기반)
    for name, fn in METHODS.items():                   # 세 방법 차례로
        theta, k = fn(X, y)                            # 풀기
        err = np.linalg.norm(theta - ref) / np.linalg.norm(ref)  # 기준해와의 상대 오차
        print(f"{name:10s} iters={k:6d}  rel_res={rel_residual(X, y, theta):.2e}  "
              f"|θ-θ_ref|/|θ_ref|={err:.2e}")          # 세 방법 모두 rel_res ≤ 1e-8
