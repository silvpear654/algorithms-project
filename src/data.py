"""조건수 κ를 정확히 조절한 최소제곱 회귀 데이터를 만든다."""
import numpy as np                                     # 행렬 연산 라이브러리


def make_data(n, d, kappa, seed, noise=0.01):          # X: n×d, κ(XᵀX) = kappa
    rng = np.random.default_rng(seed)                  # 시드 고정 → 같은 시드면 같은 데이터
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))   # n×d, 열들이 정규직교 (UᵀU = I)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))   # d×d 직교행렬 (VᵀV = I)
    s = np.sqrt(np.logspace(0, np.log10(kappa), d))    # 특이값 1 ~ √kappa → XᵀX 고윳값 1 ~ kappa
    X = (U * s) @ V.T                                  # X = U·diag(s)·Vᵀ → XᵀX = V·diag(s²)·Vᵀ
    theta_true = rng.standard_normal(d)                # 진짜 가중치
    y = X @ theta_true + noise * rng.standard_normal(n)  # 잡음 추가 → 잔차가 0이 아닌 일반적 상황
    return X, y                                        # 입력 반환


if __name__ == "__main__":                             # 직접 실행하면 κ가 맞는지 확인
    X, y = make_data(2000, 50, 1e3, seed=0)            # 작은 예시
    ev = np.linalg.eigvalsh(X.T @ X)                   # XᵀX 고윳값 (오름차순)
    print(f"kappa = {ev[-1] / ev[0]:.4f}")             # 출력: kappa ≈ 1000.0000
