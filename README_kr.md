# Game-Theoretic Racing Toy

[English](README.md) | **한국어**

**1대1 자율주행 레이싱을 여섯 가지 방법으로 풀어보기: 최적제어, 예측 후 계획, 게임이론의 세 가지 해 개념, self-play 딥 강화학습.**

레이싱 문제를 어떻게 *정의*하느냐가 어떻게 풀 수 있는지를 결정합니다. 이 저장소는 대표적인 문제 정의들을 하나의 작고 빠른 시뮬레이터에 나란히 놓고, 누가 추월하고, 누가 막고, 누가 부딪히는지 직접 비교합니다.

<p align="center">
  <img src="figures/gif_overtake_nash.gif" width="48%" alt="Nash 공격자가 예측 기반 방어자를 추월">
  <img src="figures/gif_block_stackelberg.gif" width="48%" alt="Stackelberg 방어자가 Nash 공격자를 막음">
</p>

> 세미나용으로 만든 **토이**입니다. 일반적인 결론이 아니라 경향을 보여줍니다. [한계](#한계)를 참고하세요.

## 여섯 가지 방법

| 방법 | 문제 정의 | 상대를 어떻게 취급하나 |
|---|---|---|
| Solo MPC | 단일 에이전트 최적제어 | 무시 |
| Predict+MPC | 예측 후 계획 | 지금 라인과 속도를 유지한다고 가정 |
| Nash (IBR) | 동적 게임, Nash 균형 | 합리적 최적화자. 반복 best response |
| Stackelberg | 동적 게임, 리더–팔로워 | 내가 먼저 정하면 거기에 best response 한다고 가정 |
| Minimax (LP) | 제로섬 게임, security strategy | 최악의 적. 선형계획(LP)으로 풂 |
| Deep RL | Markov game, PPO self-play | 자기 자신·과거 버전과 대결하며 암묵적으로 학습 |

문제 정의별 입력, 출력, 해법은 [docs/problem_formulation_kr.md](docs/problem_formulation_kr.md)에 정리되어 있습니다.

## 실험 설정

- **트랙**: 397 m 오벌 (직선 120 m × 2, 반경 25 m 코너), Frenet 점질량 모델, `dt = 0.1 s`
- **방어자**(car 0)는 6–12 m 앞에서 출발. **공격자**(car 1)는 뒤에서 출발하고 최고속도가 2.5 m/s 높음
- **상호작용**: 충돌하면 두 차 모두 감속하고 뒤차는 앞차 뒤로 밀림. 바로 뒤 같은 라인이면 슬립스트림으로 +2 m/s
- **공통 행동 집합**: 2초짜리 모션 프리미티브 10개 (목표 횡위치 ∈ {−4, −2, 0, 2, 4} m × 속도 모드 ∈ {100%, 85%}), 0.2초마다 재계획
- **효용** (2초 horizon): `U_i = (Δs_i − Δs_j) + 0.2·Δs_i − 30·충돌`. 계획 방법들은 모든 프리미티브 조합을 시뮬레이션해 10 × 10 효용 행렬을 만들고, 각자의 규칙을 적용
- **대결**: 방어자 6 × 공격자 6 × 16경주 = 576경주, 경주당 35초 (약 2바퀴)

## 결과

![방어 성공률과 경주당 충돌 횟수](figures/fig_heatmaps.png)

| 방법 | 방어 성공 | 추월 성공* | 충돌/경주 |
|---|---|---|---|
| Solo MPC | 0.99 | 0.04 | 3.3 |
| Predict+MPC | 0.85 | 0.09 | 2.1 |
| Nash (IBR) | 0.93 | **0.14** | 2.5 |
| Stackelberg | 0.94 | 0.09 | **6.0** |
| Minimax (LP) | **0.50** | 0.00 | 1.9 |
| Deep RL (self-play) | **1.00** | 0.00 | **1.2** |

\* Minimax를 제외한 방어자 5개 상대 평균. Minimax 방어자는 거의 모두에게 자리를 내줘서 평균을 부풀립니다. 충돌은 공격·방어 평균입니다.

![경주 다섯 장면: 시간에 따른 간격과 횡위치](figures/fig_cases.png)

**토이가 보여주는 것**

1. **상대 모델이 곧 성능이다.** 단방향 예측(Predict+MPC)은 양방향 상호작용을 고려하는 Nash 공격자에게 가장 많이 추월당했습니다.
2. **해 개념이 곧 드라이버의 성격이다.** Stackelberg는 공격적으로 막으면서 접촉이 잦고, Minimax는 비관적이라 길을 비켜 주며, Nash는 그 중간입니다.
3. **가정이 어긋나면 위험하다.** 두 차가 모두 Stackelberg면 서로 자기가 리더라고 믿어서 경주당 충돌이 17.5회까지 늘었습니다.
4. **Self-play RL은 연습 상대의 분포를 배운다.** 정책은 "안쪽 라인 지키기"라는 단순한 수비법을 스스로 찾아 방어는 완벽하고 충돌도 가장 적었지만, 추월은 한 번도 배우지 못했습니다. Self-play가 약 30 iteration 만에 "막는 쪽이 이기는" 균형에 수렴했습니다.

<p align="center"><img src="figures/gif_rl_defense.gif" width="60%" alt="RL 방어자가 안쪽 라인을 지킴"></p>

## 빠른 시작

```bash
git clone https://github.com/jeongsang-ryu/game-theoretic-racing-toy.git
cd game-theoretic-racing-toy
pip install -r requirements.txt

make test         # 단위 테스트 (~15초)
make tournament   # 포함된 RL 모델로 576경주 -> results/results.json
make figures      # figures/ 다시 생성
make train        # self-play 정책 재학습 -> models/ppo_selfplay.pkl
```

참고 시간 (CPU 2코어): 학습 1,000 iteration ≈ 10분, 대결 ≈ 9분.
스크립트는 인자를 받습니다. 예: `python scripts/run_tournament.py --races 32 --workers 8`, `python scripts/train_rl.py --iters 2000 --seed 1`

## 저장소 구조

```
racing/
  sim.py         트랙, 동역학, 충돌, 슬립스트림, 10x10 조합 롤아웃
  agents.py      Solo, Predict, Nash (IBR), Stackelberg, Minimax (scipy HiGHS로 LP)
  rl.py          self-play + 스냅샷 리그 PPO (JAX)
  race.py        두 에이전트의 경주 1회
  tournament.py  라운드로빈
  viz.py         히트맵, 경주 장면 그림, GIF
scripts/         train_rl.py, run_tournament.py, make_figures.py
models/          ppo_selfplay.pkl (학습된 정책)
results/         results.json (위 대결 결과), train_log.txt
figures/         모든 그림과 GIF
docs/            problem_formulation.md, problem_formulation_kr.md
tests/           pytest
```

## 한계

- Frenet 점질량 모델, 이산 모션 프리미티브 10개뿐
- 모든 계획 방법이 정확한 동역학과 상대의 효용을 안다고 가정
- 2초 계획 horizon은 근시안적
- RL: 시드 1개, CPU 약 10분 학습. 결과가 연습 상대 풀에 크게 좌우됨
- 조합당 16경주라 칸별 추정치의 잡음이 큼 (비율 기준 ±0.1–0.2)

## 다음 단계

- 연속 동적 게임: iLQGames, ALGAMES
- 모르는 상대 목적 추정: 미분 가능한 게임 풀이기
- 규칙 인지 계획 (불법 블로킹 페널티)
- 공격을 배우기 위한 RL 리그와 커리큘럼 (Gran Turismo Sophy 방식)
- 하이브리드: 학습된 예측 + 게임이론 플래너

## 참고문헌

- A. Liniger, J. Lygeros, "A Noncooperative Game Approach to Autonomous Racing," *IEEE TCST*, 2020.
- M. Wang et al., "Game-Theoretic Planning for Self-Driving Cars in Multivehicle Competitive Scenarios," *IEEE T-RO*, 2021.
- D. Fridovich-Keil et al., "Efficient Iterative Linear-Quadratic Approximations for Nonlinear Multi-Player General-Sum Differential Games," *ICRA*, 2020.
- S. Le Cleac'h, M. Schwager, Z. Manchester, "ALGAMES: A Fast Solver for Constrained Dynamic Games," *RSS*, 2020.
- P. Wurman et al., "Outracing champion Gran Turismo drivers with deep reinforcement learning," *Nature*, 2022.
- X. Liu, L. Peters, J. Alonso-Mora, "Learning to Play Trajectory Games Against Opponents with Unknown Objectives," *RA-L*, 2023.
- S. Boyd, L. Vandenberghe, *Convex Optimization*, Ch. 4–5.

## 라이선스

MIT
