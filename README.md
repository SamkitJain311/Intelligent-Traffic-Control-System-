# Intelligent Traffic Signal Control (Hill Climbing)

A Python + Pygame simulation of one four-way intersection. It compares a **fixed-timing**
signal (30 s N–S / 30 s E–W) with a signal whose green split is chosen by the
**Hill Climbing** search algorithm, using identical vehicle arrivals. This is a search-based
AI optimisation project, not a trained machine-learning model.

![AI mode](screenshots/2_ai_hill_climbing.png)

## Install and run

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python main.py
```

No display? The logic can be run without the GUI:

```bash
python evaluate.py            # the 4 required test scenarios, prints a table + saves results/evaluation.csv
python evaluate.py --seeds 10 # repeat on 10 different random arrival schedules
python test_simulation.py     # automated acceptance checks (15 tests)
```

## Using the program

| Control | What it does |
| --- | --- |
| **Start / Pause**, `SPACE` | start, pause or restart the run |
| **Reset**, `R` | restart the same scenario from t = 0 |
| **Signal mode** (`M`) | *Fixed 30/30* or *AI: Hill Climbing* |
| **N–S / E–W traffic** | Low / Medium / High for each road (4 / 9 / 14 cars per minute per approach) |
| **New scenario**, `N` | new random arrival schedule (new seed) |
| **Sim speed** | 1x to 32x (a run is 600 simulated seconds) |
| **Compare modes**, `C` | runs Fixed and AI on identical arrivals for the 4 required scenarios + your current setting; shows a side-by-side table; *Save CSV* exports it |
| `P` | save a screenshot to `screenshots/` |

The right-hand panel shows the current phase and timer, average waiting time, vehicles served and
queue remaining. The **Hill Climbing panel** shows every step of the search (current split,
neighbours tried, move or stop) and a bar chart of the waiting time W for all nine valid splits.

**Suggested demo (2 minutes):** set N–S = High, E–W = Low → keep *Fixed* and press Start at 8x
(watch the N–S queue grow) → press Reset, choose *AI: Hill Climbing* (the search log appears and
the plan becomes 45/15) → Start again and compare the queue and average wait → press **Compare modes**.

## How it works

| File | Responsibility |
| --- | --- |
| `main.py` | Pygame window, controls, drawing, comparison screen |
| `traffic.py` | rules/constants, `Scenario` (arrival schedule), `Vehicle`, `Simulation` (signal phases, car movement, metrics) |
| `optimizer.py` | waiting-time evaluation, neighbour generation, **Hill Climbing**, Fixed-vs-AI report |
| `evaluate.py` | command-line version of the comparison (extra) |
| `test_simulation.py` | automated checks of the acceptance checklist (extra) |

### Rules implemented (from the specification)

* Two phases: N–S green then E–W green; opposite directions share a phase; conflicting greens never occur.
* 60 s green budget per cycle; each green 10–50 s in 5 s steps and the pair sums to 60 s.
* 3 s yellow after each green, *in addition* to the 60 s (so one cycle is 66 s).
* Straight-through traffic only, one lane per direction. Same vehicle speed, spawn and queue rules in every run.
* The fixed baseline is always (30, 30).

### Hill Climbing

A candidate is `(NS_green, EW_green)`; the objective is **W = total waiting time of all vehicles** (minimise).

```
current = (30, 30);  score = W(current)               # one full simulation
repeat up to 10 times:
    neighbours = [(NS+5, EW-5), (NS-5, EW+5)]  (only if valid)
    evaluate each neighbour = simulate the SAME arrivals, get W
    best = neighbour with the lowest W
    if W(best) < score: current = best  else: stop     # no improving neighbour
return current                                          # used by AI mode
```

Evaluations are cached, so the search runs at most 9 simulations. Because Hill Climbing is a local
search it is not guaranteed to find the global best; the program also evaluates all 9 splits
**for reporting only** (the "Brute-force best" column) so you can see whether it found the optimum.

### Metrics (all computed by the program)

* **Waiting time** – a vehicle waits while it is before the stop line and slower than 0.5 m/s; arrivals
  that cannot yet enter the road also count as waiting.
* **Average waiting time** – total waiting time ÷ vehicles that have arrived (includes vehicles still queued at the end).
* **Vehicles served** – vehicles that have driven through the junction box.
* **Queue remaining** – vehicles that have arrived but not yet crossed the stop line.
* **Improvement %** = (fixed avg − AI avg) ÷ fixed avg × 100 (not computed if the fixed average is 0).

## Measured results

Actual output of `python evaluate.py` (seed 42, 600 s each, identical arrivals for both modes):

| Scenario | Fixed avg wait | Fixed served / queue | AI plan | AI avg wait | AI served / queue | Improvement |
| --- | --- | --- | --- | --- | --- | --- |
| Low traffic, both roads | 7.1 s | 151 / 2 | 30/30 | 7.1 s | 151 / 2 | 0.0 % |
| Heavy N–S, light E–W | 26.2 s | 327 / 39 | 45/15 | 6.4 s | 351 / 15 | 75.3 % |
| Light N–S, heavy E–W | 17.3 s | 331 / 25 | 15/45 | 4.7 s | 347 / 8 | 73.1 % |
| Heavy traffic, both roads | 25.8 s | 507 / 62 | 30/30 | 25.8 s | 507 / 62 | 0.0 % |

Averaged over seeds 1–10 (`python evaluate.py --seeds 10`): mean improvement was +3.0 % (low/low),
+74.6 % (heavy N–S), +72.4 % (heavy E–W) and +0.0 % (heavy/heavy); Hill Climbing matched the brute-force
best split in 10/10 seeds for every scenario.

How to read this: the AI helps when demand is **unbalanced**, because the fixed 30/30 split wastes green
on the quiet road. When both roads are equally busy, 30/30 is already (close to) the best split, so Hill
Climbing correctly stays there and there is no improvement.

## Limitations (be upfront about these in the evaluation)

* As the specification requires, the split is optimised on the **same** arrival schedule it is then measured on,
  so the improvement is "in-sample". It is a fixed plan for the whole run; the signal does not react live to cars.
  Try *New scenario* / other seeds to see results vary.
* Hill Climbing can stop at a local optimum on other traffic patterns. In every run tested here it matched the
  brute-force best split, but that is a measured outcome, not a guarantee.
* The vehicle model is deliberately simple (one lane per direction, car-following with a reaction time, no turns,
  pedestrians or emergency vehicles). Results describe this simulation, not real roads.
* Traffic is drawn left-hand (India). Set `LEFT_HAND_TRAFFIC = False` in `main.py` for right-hand traffic; the
  simulation results do not change.

## Customising

* Traffic volumes: `LEVELS` in `traffic.py`.  Run length: `SIM_DURATION`.  Vehicle behaviour: constants in `traffic.py`.
* Iteration limit of the search: `MAX_ITERATIONS` in `optimizer.py`.

## Short explanation for the evaluation

"This project simulates a four-way traffic intersection. The baseline uses fixed green-light timings. The AI
version uses Hill Climbing to test nearby valid splits of the green-time budget and select the split that gives
the lowest vehicle waiting time in the simulated scenario. We compare both approaches using average waiting time,
vehicles served and the remaining queue. It is a search-based AI optimization project, not a trained
machine-learning model."
