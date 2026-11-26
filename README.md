# 최소제곱 선형회귀를 푸는 세 가지 방법 비교

정규방정식(직접법) · 경사하강법(GD) · 켤레기울기법(CG)이
문제 크기(n, d)와 조건수 κ에 따라 **시간, 반복 수, 추가 메모리, 정확도**가 어떻게 달라지는지
같은 언어·같은 환경에서 실측하고 이론과 비교한다. (고급알고리즘 개인 프로젝트 2026)

> 문제: 데이터 n개, 특성 d개인 최소제곱 선형회귀의 최적 가중치를 구하는 세 방식은
> 문제 크기와 조건수에 따라 시간, 반복 횟수, 메모리가 어떻게 달라지는가?

## 구조

```
lsq-solvers/
├── src/
│   ├── data.py        # κ(XᵀX)를 정확히 조절한 데이터 생성 (X = U·diag(s)·Vᵀ)
│   ├── methods.py     # 세 방법 + 상대 잔차 (직접 실행하면 정확도 검증)
│   ├── benchmark.py   # 실험 1~5 → results/exp*.csv
│   ├── analyze.py     # 그래프 results/figures/*.png, 요약 results/summary.md
│   ├── verify_claims.py  # 보고서 5장 보충 확인 → results/verify_claims.txt
│   └── sparse_check.py   # 희소 행렬 보조 확인 (scipy) → results/sparse_check.txt
├── results/           # 측정 결과 CSV, 실행 로그, 그래프, 요약
├── report/            # 최종 보고서 (report.html → build_pdf.sh → report.pdf)
├── slides/            # 중간 제출 슬라이드 PDF + 그림 생성 스크립트
├── requirements.txt
└── README.md
```

## 실행

```bash
git clone https://github.com/silvpear654/algorithms-project.git && cd algorithms-project
pip install -r requirements.txt

python src/data.py                          # κ가 맞게 만들어지는지 확인 → kappa = 1000.0000
python src/methods.py                       # 세 방법이 같은 해를 내는지 확인 (상대 잔차 ≤ 1e-8)
python src/benchmark.py --exp all --quick   # 작은 크기로 전체 흐름 확인 (1초 내외)
python src/benchmark.py --exp all           # 본 실험 1~4 (M5에서 4분 1초, 실험 2의 GD가 대부분)
python src/benchmark.py --exp 5 --seeds 0 1 2   # 확장 실험: d = 2000~8000 (2분 44초, 최대 메모리 7.0 GB)
python src/benchmark.py --exp 2 --seeds 0   # 실험 하나, 시드 하나만
python src/analyze.py                       # 그래프 + 요약표 + BLAS 처리량 측정
python src/verify_claims.py                 # SME 처리량, LAPACK 작업 공간, GD 스펙트럼 계산, 결정성 확인
pip install scipy && python src/sparse_check.py   # 희소 행렬 보조 확인
bash report/build_pdf.sh                    # 보고서 PDF (Google Chrome 필요)
python slides/make_figures.py               # 슬라이드 5·6장 등고선 그림
```

## 실험 설계

공통: 정지 기준 상대 잔차 ‖Xᵀy − XᵀXθ‖/‖Xᵀy‖ ≤ 1e-8, θ₀ = 0, 시드 0~4 (평균 ± 표준편차).

| 실험 | 고정 | 변화 | 확인할 것 |
|---|---|---|---|
| 1. 차원 증가 | n=10000, κ=100 | d = 50, 100, 200, 500, 1000, 2000 | 시간 교차점, 메모리 d² 대 d |
| 2. 조건수 증가 | n=5000, d=200 | κ = 1e1 ~ 1e5 | 반복 수 로그-로그 기울기 (GD≈1, CG≈0.5) |
| 3. 데이터 증가 | d=200, κ=100 | n = 1000 ~ 100000 | 시간이 n에 선형인지 |
| 4. 학습률 | n=5000, d=200, κ=1e3 | η = 0.5/L, 1/L, 1.9/L, 2.1/L (GD만) | 발산 경계 2/L |
| 5. 확장 | n=10000, κ=100 | d = 2000, 3000, 4000, 6000, 8000 (GD 제외, 시드 0~2) | 실제 시간 교차점 |

반복 상한: CG 10d, GD 1e6 (실험 2는 3e6).

## 주요 결과

- CG가 정규방정식을 이기는 실제 교차점은 **d ≈ 4700** (n=10000, κ=100). 연산량 이론(d ≈ 2K ≈ 170)보다 약 28배 뒤 →
  XᵀX(GEMM, 약 1 TFLOP/s)와 Xᵀ(Xp)(GEMV, 약 20 GFLOP/s)의 처리량 차이 약 50배로 설명된다.
- 반복 수의 κ 기울기: GD 0.91 (보정 이론 κ ln(1/(εκ))와 86~89% 일치), CG 0.37. κ ≥ 1e4에서 CG는 반올림 오차로 d번을 넘는다 (κ=1e5에서 822번).
- 정규방정식 추가 메모리는 XᵀX(8d², tracemalloc으로 측정) + LAPACK의 LU 복사본(8d², tracemalloc에 안 잡혀 RSS로 확인) ≈ 16d². 반복법은 O(n + d).
- 희소 X(밀도 0.5%, κ 150~350)에서는 교차점이 d = 1000~2000으로 앞당겨진다 (`src/sparse_check.py`).
- GD는 η < 2/L에서 수렴, 2.1/L에서 발산.

자세한 내용은 `report/report.pdf`, 전체 수치는 `results/summary.md`.

## 측정 방법

- **시간**: `time.perf_counter`. 데이터 생성 시간은 제외.
- **반복 수**: 각 함수가 반환하는 k (정규방정식은 1).
- **추가 메모리**: `tracemalloc` 최대값 (LAPACK 내부 작업 공간은 잡히지 않음 — 보고서 5.5절). tracemalloc은 실행을 느리게 하므로 시간 측정과 **따로**, 시드 0에서 한 번만 잰다.
  반복법의 최대 메모리는 반복 수와 무관하므로 메모리 측정 실행은 50회로 자른다.
- **정확도**: 상대 잔차. 셋 모두 1e-8 이하여야 공정한 비교로 본다.
- GD의 L = λmax(XᵀX)는 거듭제곱법(100회)으로 추정하며, 이 시간도 GD 시간에 포함된다.
- 각 CSV 첫 줄에 Python·NumPy 버전과 기기 정보를 기록한다.

## 환경

Python 3.14.4, NumPy 2.4.5 (BLAS: Apple Accelerate, GEMM은 SME 행렬 유닛 사용 확인), macOS 26.7, Apple M5 (성능 코어 4 + 효율 코어 6, 성능 코어 L2 16 MiB, 32 GB). 희소 확인만 SciPy 1.18.1 추가.

## AI 사용 및 출처

- 주제 정리, 수식 유도, 코드 뼈대 작성에 Claude(claude.ai, Claude Code)를 사용했다. 질문과 답변 요약은 최종 보고서에 기록한다.
- CG 갱신식은 표준 알고리즘(Hestenes–Stiefel)을 따랐고, A = XᵀX를 만들지 않도록 Ap = Xᵀ(Xp)로 계산했다.
