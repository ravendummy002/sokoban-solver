# Sokoban Solver

A push-optimal Sokoban solver, written for make.fun's
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty ($100 in USDC).

## How it works

Each level is a hallway with rooms off it, one door each, and a box pushed out
of a room into the hallway can never come back — so every room is a problem of
its own, and the level's push-optimal count P\* is the sum of the rooms'
optima. The solver:

1. **Splits the level into rooms** — the hallway's squares are dead (no push
   from one reaches a goal), so cutting them apart separates the rooms.
2. **Solves each room push-optimally with A\* over pushes** — a state is the
   boxes plus the player's reachable region, the estimate is the cheapest
   matching of boxes to goals over walking distances, and dead squares plus
   frozen 2×2 blocks prune positions no push can save.
3. **Walks the player from room to room** between solutions. Walking costs no
   pushes, so the rooms' optima add up to P\*.

A greedy best-first search backs up the optimal search if a room ever exhausts
its work budget, and every answer is replayed before it is written. The solver
is fully deterministic — work-based budgets only, no clock, no randomness —
so two runs write identical bytes.

## Run it

```sh
node main.ts
```

Reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt` (moves in LURD, an upper letter marking a push).
Locally, `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` stand in for `/input`
and `/output`:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Results

- Practice set: **1,000,000 / 1,000,000** — all 5 instances solved in exactly
  P\* pushes (13, 16, 29, 47, 60), in under a second.
- House baseline (plain-astar 1.0.0) on the same set: 200,000.
- 45 further sets drawn fresh from the competition's generator: 1,000,000 on
  every one, worst runtime under 2 seconds of the 60-second budget.

Node 22, standard library only. MIT licensed.
