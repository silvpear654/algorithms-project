"""실험 1~4를 실행해 results/*.csv로 저장한다.

사용법:
    python src/benchmark.py --exp all            # 전체 (오래 걸림: 실험 2의 GD가 가장 김)
    python src/benchmark.py --exp 1 --quick      # 작은 크기로 빠르게 확인
    python src/benchmark.py --exp 2 --seeds 0 1  # 시드 지정
    python src/benchmark.py --exp 5 --seeds 0 1 2  # 확장: d를 n 가까이 키워 교차점 찾기

측정 방식:
- 시간과 메모리는 따로 잰다. tracemalloc은 할당마다 기록하느라 실행을 느리게 하므로,
  시간 측정 실행에서는 끄고, 메모리는 별도 실행(시드 0 한 번)에서만 잰다.
- 반복법의 최대 추가 메모리는 반복 횟수와 무관하므로 메모리 실행은 50회로 자른다.
- 데이터 생성 시간은 측정에서 제외한다.
"""
import argparse                                        # 명령줄 인자
import csv                                             # 결과 저장
import os                                              # 경로
import platform                                        # 환경 정보 기록
import sys                                             # 파이썬 버전
import time                                            # 실행 시간 측정
import tracemalloc                                     # 추가 메모리 측정

import numpy as np                                     # 행렬 연산

sys.path.insert(0, os.path.dirname(__file__))          # src/를 import 경로에 추가
from data import make_data                             # noqa: E402  데이터 생성
from methods import METHODS, rel_residual              # noqa: E402  세 방법 + 품질 지표

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 저장소 루트
RESULTS = os.path.join(ROOT, "results")                # 결과 폴더
TOL = 1e-8                                             # 공통 정지 기준

# 실험 설정 (슬라이드 8장과 동일)
FULL = {
    1: dict(n=[10_000], d=[50, 100, 200, 500, 1000, 2000], kappa=[100]),
    2: dict(n=[5_000], d=[200], kappa=[1e1, 1e2, 1e3, 1e4, 1e5]),
    3: dict(n=[1_000, 3_000, 10_000, 30_000, 100_000], d=[200], kappa=[100]),
    4: dict(n=[5_000], d=[200], kappa=[1e3], eta=[0.5, 1.0, 1.9, 2.1]),
    5: dict(n=[10_000], d=[2000, 3000, 4000, 6000, 8000], kappa=[100]),  # 확장: 실제 교차점 찾기
}
QUICK = {                                              # --quick: 동작 확인용 작은 설정
    1: dict(n=[2_000], d=[20, 50, 100, 200], kappa=[100]),
    2: dict(n=[1_000], d=[50], kappa=[1e1, 1e2, 1e3]),
    3: dict(n=[500, 1_000, 2_000, 4_000], d=[50], kappa=[100]),
    4: dict(n=[1_000], d=[50], kappa=[1e2], eta=[0.5, 1.0, 1.9, 2.1]),
    5: dict(n=[2_000], d=[200, 400], kappa=[100]),
}
GD_MAX_ITER = {1: 10**6, 2: 3 * 10**6, 3: 10**6, 4: 10**6, 5: 10**6}  # 실험 2는 κ=1e5에서 약 1.8e6번 필요


def call(method, X, y, exp, eta_scale=1.0, mem_run=False):  # 방법별 인자를 맞춰 호출
    if method == "normal_eq":                          # 직접법은 추가 인자 없음
        return METHODS[method](X, y)
    if method == "gd":                                 # GD: 학습률 배율과 반복 상한
        max_iter = 50 if mem_run else GD_MAX_ITER[exp]
        return METHODS[method](X, y, tol=TOL, max_iter=max_iter, eta_scale=eta_scale)
    max_iter = 50 if mem_run else None                 # CG: 기본 상한 10d
    return METHODS[method](X, y, tol=TOL, max_iter=max_iter)


def time_run(method, X, y, exp, eta_scale):            # 시간 측정 (tracemalloc 끔)
    t0 = time.perf_counter()                           # 시작 시각
    theta, iters = call(method, X, y, exp, eta_scale)  # 실행
    elapsed = time.perf_counter() - t0                 # 걸린 시간 [s]
    return theta, iters, elapsed


def mem_run(method, X, y, exp, eta_scale):             # 최대 추가 메모리 측정
    tracemalloc.start()                                # 추적 시작 (X, y는 이미 할당됨 → 제외)
    call(method, X, y, exp, eta_scale, mem_run=True)   # 실행
    _, peak = tracemalloc.get_traced_memory()          # 실행 중 최대 추가 할당 [bytes]
    tracemalloc.stop()                                 # 추적 종료
    return peak


def run(exp, seeds, quick):                            # 실험 하나 실행
    cfg = (QUICK if quick else FULL)[exp]              # 설정 선택
    etas = cfg.get("eta", [1.0])                       # 실험 4만 η를 바꿈
    methods = {4: ["gd"], 5: ["normal_eq", "cg"]}.get(exp, list(METHODS))  # 실험 4는 GD만, 5는 GD 제외
    rows = []                                          # CSV 행 모음
    for n in cfg["n"]:
        for d in cfg["d"]:
            for kappa in cfg["kappa"]:
                for seed in seeds:
                    X, y = make_data(n, d, kappa, seed)  # 같은 입력을 세 방법에 똑같이
                    for m in methods:
                        for eta in etas:
                            theta, iters, sec = time_run(m, X, y, exp, eta)
                            res = rel_residual(X, y, theta)  # 품질 확인 (≤ 1e-8이어야 공정)
                            peak = mem_run(m, X, y, exp, eta) if seed == seeds[0] else ""
                            row = dict(exp=exp, method=m, n=n, d=d, kappa=kappa, eta_scale=eta,
                                       seed=seed, time_s=sec, iters=iters,
                                       peak_mem_bytes=peak, rel_residual=res)
                            rows.append(row)
                            print(f"[exp{exp}] {m:9s} n={n:6d} d={d:5d} κ={kappa:8.0e} "
                                  f"η={eta}/L seed={seed} t={sec:9.4f}s it={iters:8d} "
                                  f"res={res:.1e} mem={peak}", flush=True)
    return rows


def save(rows, exp, quick):                            # CSV 저장 (환경 정보는 헤더 주석으로)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"exp{exp}{'_quick' if quick else ''}.csv")
    with open(path, "w", newline="") as f:
        f.write(f"# python={platform.python_version()} numpy={np.__version__} "
                f"machine={platform.machine()} os={platform.platform()}\n")  # 같은 환경 기록
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="1~5 또는 all (all은 1~4, 5는 확장 실험)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true", help="작은 크기로 빠르게 확인")
    args = ap.parse_args()
    exps = [1, 2, 3, 4] if args.exp == "all" else [int(args.exp)]
    for e in exps:
        save(run(e, args.seeds, args.quick), e, args.quick)


if __name__ == "__main__":
    main()
