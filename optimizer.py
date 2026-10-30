"""
optimizer.py - Hill Climbing search for the traffic-signal green-time split.

A candidate solution is a plan  (NS_green, EW_green)  e.g. (30, 30).
Objective (to MINIMISE):  W = total waiting time of all vehicles, in seconds.
(If you prefer a maximisation-style fitness, use  fitness = -W.)

Hill Climbing (steepest-ascent variant), exactly as in the project spec:
    1. start at (30, 30)
    2. generate the valid neighbours: shift 5 s between the two directions
       e.g. (35, 25) and (25, 35)
    3. evaluate every neighbour by running the traffic simulation on the
       SAME arrival scenario
    4. move to the neighbour with the lowest W *if* it beats the current W
    5. repeat until no neighbour improves, or the iteration limit is reached

Hill Climbing is a LOCAL search: it can stop at a local optimum and is not
guaranteed to find the global best split.  `exhaustive_landscape` evaluates
all 9 valid splits, but only so the result can be reported honestly -
the AI mode itself never uses it.
"""

from traffic import (Simulation, valid_plan, FIXED_PLAN, GREEN_STEP,
                     MIN_GREEN, MAX_GREEN, TOTAL_GREEN)

MAX_ITERATIONS = 10
EPS = 1e-9


class Cancelled(Exception):
    """Raised (via should_stop) to abort a search that is no longer wanted."""


# --------------------------------------------------------------------------
# Objective function
# --------------------------------------------------------------------------
class Evaluator:
    """
    Evaluates a plan on one scenario = runs the simulation and returns W.
    Results are cached, so the number of *simulations actually run* is
    `len(cache)`.
    """

    def __init__(self, scenario, should_stop=None):
        self.scenario = scenario
        self.cache = {}                 # plan -> metrics dict
        self.should_stop = should_stop

    def metrics(self, plan):
        if plan not in self.cache:
            if self.should_stop and self.should_stop():
                raise Cancelled()
            self.cache[plan] = Simulation(self.scenario, plan).run()
        return self.cache[plan]

    def waiting_time(self, plan):
        """Total waiting time W (seconds) - the value Hill Climbing minimises."""
        return self.metrics(plan)["total_wait"]


# --------------------------------------------------------------------------
# Neighbourhood
# --------------------------------------------------------------------------
def all_plans():
    """Every valid plan: (10,50), (15,45), ... (50,10)."""
    return [(ns, TOTAL_GREEN - ns)
            for ns in range(MIN_GREEN, MAX_GREEN + 1, GREEN_STEP)]


def neighbours(plan):
    """Valid plans reachable by moving 5 s of green from one direction to the other."""
    ns, ew = plan
    candidates = [(ns + GREEN_STEP, ew - GREEN_STEP),    # more to North-South
                  (ns - GREEN_STEP, ew + GREEN_STEP)]    # more to East-West
    return [c for c in candidates if valid_plan(c)]


# --------------------------------------------------------------------------
# Hill Climbing
# --------------------------------------------------------------------------
class HillClimbResult:
    def __init__(self):
        self.start_plan = None
        self.start_score = None
        self.best_plan = None
        self.best_score = None
        self.steps = []             # one dict per iteration (see hill_climb)
        self.stop_reason = ""
        self.simulations = 0        # distinct simulations run

    @property
    def improved(self):
        return self.best_plan != self.start_plan


def hill_climb(evaluator, start=FIXED_PLAN, max_iter=MAX_ITERATIONS, on_event=None):
    """
    Run Hill Climbing and return a HillClimbResult.

    on_event(kind, **info) is an optional callback used by the GUI to show the
    search live.  kinds:
        'current'   plan, score, iteration
        'neighbour' plan, score, iteration
        'move'      frm, to, score, iteration
        'stop'      reason, plan, score
    """
    def emit(kind, **info):
        if on_event:
            on_event(kind, **info)

    if not valid_plan(start):
        raise ValueError("invalid start plan %r" % (start,))

    res = HillClimbResult()
    current = start
    current_score = evaluator.waiting_time(current)
    res.start_plan, res.start_score = current, current_score

    for iteration in range(1, max_iter + 1):
        emit("current", plan=current, score=current_score, iteration=iteration)
        step = {"iteration": iteration, "current": current,
                "current_score": current_score, "neighbours": [], "move": None}

        best_nb, best_nb_score = None, None
        for nb in neighbours(current):
            score = evaluator.waiting_time(nb)
            step["neighbours"].append((nb, score))
            emit("neighbour", plan=nb, score=score, iteration=iteration)
            if best_nb is None or score < best_nb_score - EPS:
                best_nb, best_nb_score = nb, score

        if best_nb is not None and best_nb_score < current_score - EPS:
            step["move"] = best_nb
            res.steps.append(step)
            emit("move", frm=current, to=best_nb, score=best_nb_score,
                 iteration=iteration)
            current, current_score = best_nb, best_nb_score
        else:
            res.steps.append(step)
            res.stop_reason = "no neighbour improves the current plan"
            break
    else:
        res.stop_reason = "iteration limit (%d) reached" % max_iter

    res.best_plan, res.best_score = current, current_score
    res.simulations = len(evaluator.cache)
    emit("stop", reason=res.stop_reason, plan=current, score=current_score)
    return res


# --------------------------------------------------------------------------
# Reference only: brute force over the 9 valid plans
# --------------------------------------------------------------------------
def exhaustive_landscape(evaluator):
    """{plan: W} for every valid plan. Used ONLY to report how good HC's result is."""
    return {plan: evaluator.waiting_time(plan) for plan in all_plans()}


def improvement_percent(fixed_avg, ai_avg):
    """((fixed - AI) / fixed) * 100, or None when the fixed average wait is 0."""
    if fixed_avg <= 0:
        return None
    return (fixed_avg - ai_avg) / fixed_avg * 100.0


# --------------------------------------------------------------------------
# Fixed-vs-AI comparison (shared by the GUI and evaluate.py)
# --------------------------------------------------------------------------
# The four test scenarios required by the specification (section 5).
TEST_SCENARIOS = [
    ("Low traffic, both roads", "Low", "Low"),
    ("Heavy N-S, light E-W", "High", "Low"),
    ("Light N-S, heavy E-W", "Low", "High"),
    ("Heavy traffic, both roads", "High", "High"),
]


def build_report(scenario, evaluator, hc_result):
    """
    Compare Fixed (30/30) with the Hill-Climbing plan on ONE scenario.
    Every number comes from a real simulation run on identical arrivals.
    """
    landscape = exhaustive_landscape(evaluator)       # reference only
    fixed = evaluator.metrics(FIXED_PLAN)
    ai = evaluator.metrics(hc_result.best_plan)
    global_best = min(landscape, key=landscape.get)
    return {
        "scenario": scenario,
        "hc": hc_result,
        "fixed": fixed,
        "ai": ai,
        "ai_plan": hc_result.best_plan,
        "improvement": improvement_percent(fixed["avg_wait"], ai["avg_wait"]),
        "landscape": landscape,
        "global_best": global_best,
        "found_global": landscape[hc_result.best_plan] <= landscape[global_best] + EPS,
        "hc_simulations": hc_result.simulations,
    }


def compare_modes(scenario, on_event=None):
    """Run Hill Climbing on a scenario and return the full Fixed-vs-AI report."""
    ev = Evaluator(scenario)
    hc = hill_climb(ev, on_event=on_event)
    return build_report(scenario, ev, hc)
