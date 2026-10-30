"""
traffic.py - Simulation engine for the Intelligent Traffic Signal Control project.

Contains everything that is NOT drawing and NOT search:
  * the fixed rules from the project specification (green budget, yellow, ...)
  * Scenario  : a reproducible vehicle-arrival schedule
  * Vehicle   : one car on a single straight-through lane
  * Simulation: the intersection, its two-phase signal and the car dynamics

The SAME Simulation class is used by
  * the Pygame window (main.py)            -> drawn live, one tick at a time
  * the Hill Climbing optimizer (optimizer.py) -> run "headless" at full speed
so the numbers shown on screen and the numbers the optimizer minimises are
produced by identical code.
"""

import math
import random
from collections import deque

# --------------------------------------------------------------------------
# Fixed rules from the specification (section 3)
# --------------------------------------------------------------------------
TOTAL_GREEN = 60        # seconds of green shared by the two phases per cycle
MIN_GREEN = 10          # allowed green duration range ...
MAX_GREEN = 50
GREEN_STEP = 5          # ... in steps of 5 seconds
YELLOW = 3              # seconds, extra transition time (not in the 60 s)
FIXED_PLAN = (30, 30)   # baseline: (NS_green, EW_green)
SIM_DURATION = 600      # simulated seconds per run (10 minutes)

# Time is advanced in integer ticks so that phase timing never drifts.
TICKS_PER_SEC = 10
DT = 1.0 / TICKS_PER_SEC

DIRECTIONS = ("N", "S", "E", "W")   # the approach a car comes FROM
NS_DIRS = ("N", "S")
EW_DIRS = ("E", "W")

# --------------------------------------------------------------------------
# Traffic levels: vehicles per minute on EACH approach of a road
# --------------------------------------------------------------------------
LEVELS = {"Low": 4, "Medium": 9, "High": 14}


# --------------------------------------------------------------------------
# Geometry (metres, measured along a lane from where cars appear)
# --------------------------------------------------------------------------
HALF = 100.0                 # centre of the junction is at s = HALF
LANE_W = 4.0
LANE_LEN = 2 * HALF          # lane runs from one screen edge to the other
BOX_START = HALF - LANE_W    # near edge of the junction box
BOX_END = HALF + LANE_W      # far edge of the junction box
STOP_LINE = BOX_START - 2.0  # front bumper must stop before this

# --------------------------------------------------------------------------
# Vehicle behaviour (identical for fixed and AI runs)
# --------------------------------------------------------------------------
VEH_LEN = 5.0
V_MAX = 10.0         # m/s  (36 km/h)
ACCEL = 2.0          # m/s^2
B_COMF = 3.5         # comfortable braking used for "can I stop?" decisions
B_MAX = 7.0          # hard physical braking limit
TAU = 1.5            # driver reaction / desired time headway (s)
MIN_GAP = 2.5        # bumper-to-bumper gap when stopped (m)
WAIT_SPEED = 0.5     # a car slower than this (m/s) before the stop line is "waiting"


def valid_plan(plan):
    """A plan is (NS_green, EW_green); both in 10..50, step 5, summing to 60."""
    ns, ew = plan
    return (
        ns + ew == TOTAL_GREEN
        and MIN_GREEN <= ns <= MAX_GREEN
        and MIN_GREEN <= ew <= MAX_GREEN
        and ns % GREEN_STEP == 0
        and ew % GREEN_STEP == 0
    )


# --------------------------------------------------------------------------
# Scenario: reproducible arrival schedule
# --------------------------------------------------------------------------
def generate_arrivals(rate_per_min, seed, direction, duration):
    """Poisson arrivals for one approach, returned as a sorted list of ticks."""
    rng = random.Random("%s-%s" % (seed, direction))   # same seed -> same stream
    rate = rate_per_min / 60.0
    t, ticks = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            break
        ticks.append(int(t * TICKS_PER_SEC))
    return ticks


class Scenario:
    """
    A traffic scenario = traffic level of each road + random seed + duration.
    The arrival times are generated ONCE here, so every simulation built from
    the same Scenario sees exactly the same vehicles arriving at the same
    moments - this is what makes the Fixed-vs-AI comparison fair.
    """

    def __init__(self, ns_level="Medium", ew_level="Medium", seed=42,
                 duration=SIM_DURATION):
        self.ns_level = ns_level
        self.ew_level = ew_level
        self.seed = seed
        self.duration = duration
        self.arrivals = {}
        for d in DIRECTIONS:
            level = ns_level if d in NS_DIRS else ew_level
            self.arrivals[d] = generate_arrivals(LEVELS[level], seed, d, duration)

    @property
    def key(self):
        return (self.ns_level, self.ew_level, self.seed, self.duration)

    @property
    def total_vehicles(self):
        return sum(len(v) for v in self.arrivals.values())

    def describe(self):
        return "NS %s / EW %s (seed %d)" % (self.ns_level, self.ew_level, self.seed)


# --------------------------------------------------------------------------
# Vehicle
# --------------------------------------------------------------------------
class Vehicle:
    __slots__ = ("direction", "s", "v", "committed", "done", "color_id")

    def __init__(self, direction, v, color_id):
        self.direction = direction
        self.s = 0.0            # position of the FRONT bumper along the lane
        self.v = v              # speed (m/s)
        self.committed = False  # decided to run through the yellow
        self.done = False       # already counted as "served"
        self.color_id = color_id


def safe_speed(gap, v_lead):
    """
    Largest speed from which a driver can still stop without hitting an
    obstacle `gap` metres ahead that is moving at `v_lead` (Gipps-style).
    For a stop line use v_lead = 0.
    """
    if gap <= 0.0:
        return 0.0
    bt = B_COMF * TAU
    return -bt + math.sqrt(bt * bt + v_lead * v_lead + 2.0 * B_COMF * gap)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
# Signal phase machine:  0 NS green -> 1 NS yellow -> 2 EW green -> 3 EW yellow
PHASE_NAMES = ("NS GREEN", "NS YELLOW", "EW GREEN", "EW YELLOW")


class Simulation:
    def __init__(self, scenario, plan=FIXED_PLAN):
        if not valid_plan(plan):
            raise ValueError("invalid signal plan %r" % (plan,))
        self.scenario = scenario
        self.plan = tuple(plan)
        self.end_tick = scenario.duration * TICKS_PER_SEC

        self.tick = 0
        self.phase = 0
        self.phase_tick = 0
        self._phase_len = (plan[0] * TICKS_PER_SEC, YELLOW * TICKS_PER_SEC,
                           plan[1] * TICKS_PER_SEC, YELLOW * TICKS_PER_SEC)

        self.lanes = {d: [] for d in DIRECTIONS}        # front car first
        self.backlog = {d: deque() for d in DIRECTIONS}  # arrived, no room to enter yet
        self._arr_idx = {d: 0 for d in DIRECTIONS}

        # metrics
        self.arrived = 0
        self.served = 0
        self.wait_ticks = 0          # total vehicle-ticks spent waiting
        self.red_violations = 0      # cars that crossed the stop line on red (should be 0)
        self.hard_clamps = 0         # emergency position corrections (should be ~0)
        self._car_counter = 0

    # ---------------- signal ----------------
    def signal_state(self, direction):
        """'green' | 'yellow' | 'red' for one approach at the current tick."""
        p = self.phase
        if direction in NS_DIRS:
            return "green" if p == 0 else "yellow" if p == 1 else "red"
        return "green" if p == 2 else "yellow" if p == 3 else "red"

    def signal_states(self):
        return {d: self.signal_state(d) for d in DIRECTIONS}

    @property
    def phase_name(self):
        return PHASE_NAMES[self.phase]

    @property
    def phase_time_left(self):
        return (self._phase_len[self.phase] - self.phase_tick) * DT

    @property
    def phase_duration(self):
        return self._phase_len[self.phase] * DT

    @property
    def time(self):
        return self.tick * DT

    @property
    def finished(self):
        return self.tick >= self.end_tick

    # ---------------- one time step ----------------
    def step(self):
        if self.tick >= self.end_tick:
            return
        t = self.tick
        states = self.signal_states()

        for d in DIRECTIONS:
            # 1. new arrivals join the backlog of their approach
            arr = self.scenario.arrivals[d]
            i = self._arr_idx[d]
            while i < len(arr) and arr[i] <= t:
                self.backlog[d].append(arr[i])
                self.arrived += 1
                i += 1
            self._arr_idx[d] = i

            lane = self.lanes[d]

            # 2. let the oldest waiting arrival enter if there is room
            if self.backlog[d]:
                v0 = V_MAX
                room = True
                if lane:
                    last = lane[-1]
                    free = last.s - VEH_LEN - MIN_GAP
                    if free < 0.5:
                        room = False
                    else:
                        v0 = min(V_MAX, safe_speed(free, last.v))
                if room:
                    self.backlog[d].popleft()
                    self._car_counter += 1
                    lane.append(Vehicle(d, v0, self._car_counter))
                    # (cars are appended at the back of the list = rear of queue)

            # 3. move every car on this approach, front to back
            sig = states[d]
            leader = None
            for v in lane:
                target = V_MAX
                if leader is not None:
                    gap = leader.s - VEH_LEN - v.s - MIN_GAP
                    sp = safe_speed(gap, leader.v)
                    if sp < target:
                        target = sp

                before_line = v.s < STOP_LINE
                if before_line and not v.committed and sig != "green":
                    dist = STOP_LINE - v.s
                    stop = True
                    if sig == "yellow" and dist < v.v * v.v / (2.0 * B_COMF):
                        # cannot stop comfortably -> drive through the yellow
                        v.committed = True
                        stop = False
                    if stop:
                        sp = safe_speed(dist, 0.0)
                        if sp < target:
                            target = sp

                if target >= v.v:
                    nv = v.v + ACCEL * DT
                    if nv > target:
                        nv = target
                else:
                    nv = v.v - B_MAX * DT
                    if nv < target:
                        nv = target
                    if nv < 0.0:
                        nv = 0.0

                ns = v.s + nv * DT
                if leader is not None:                       # never overlap the car ahead
                    limit = leader.s - VEH_LEN - 0.3
                    if ns > limit:
                        ns = max(v.s, limit)
                        nv = min(nv, (ns - v.s) / DT)
                        self.hard_clamps += 1

                if before_line and ns >= STOP_LINE and sig == "red" and not v.committed:
                    self.red_violations += 1

                v.s = ns
                v.v = nv

                if not v.done and ns >= BOX_END:
                    v.done = True
                    self.served += 1

                # waiting = hasn't reached the stop line and is (almost) standing still
                if ns < STOP_LINE and nv < WAIT_SPEED:
                    self.wait_ticks += 1

                leader = v

            # 4. cars whose rear has left the lane disappear
            while lane and lane[0].s - VEH_LEN > LANE_LEN:
                lane.pop(0)

            # vehicles that have arrived but cannot enter yet are also waiting
            self.wait_ticks += len(self.backlog[d])

        # 5. advance the signal
        self.tick += 1
        self.phase_tick += 1
        if self.phase_tick >= self._phase_len[self.phase]:
            self.phase = (self.phase + 1) % 4
            self.phase_tick = 0

    def run(self):
        """Run to the end of the scenario (used by the optimizer / comparison)."""
        while self.tick < self.end_tick:
            self.step()
        return self.metrics()

    # ---------------- metrics ----------------
    def queue_length(self, direction=None):
        """Vehicles that have arrived but not yet crossed the stop line."""
        dirs = DIRECTIONS if direction is None else (direction,)
        n = 0
        for d in dirs:
            n += len(self.backlog[d])
            for v in self.lanes[d]:
                if v.s < STOP_LINE:
                    n += 1
        return n

    def metrics(self):
        total = self.wait_ticks * DT
        return {
            "plan": self.plan,
            "total_wait": total,
            "avg_wait": total / self.arrived if self.arrived else 0.0,
            "served": self.served,
            "queue": self.queue_length(),
            "arrived": self.arrived,
            "time": self.time,
        }
