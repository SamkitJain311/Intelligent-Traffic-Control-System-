"""
test_simulation.py - automated checks for the minimum acceptance checklist.

Run:  python test_simulation.py        (no display needed, takes ~15 s)
"""

import unittest

import optimizer as O
import traffic as T


def heavy_ns():
    return T.Scenario("High", "Low", 42)


class RuleTests(unittest.TestCase):
    def test_plan_validation(self):
        self.assertTrue(T.valid_plan((30, 30)))
        self.assertTrue(T.valid_plan((10, 50)))
        self.assertFalse(T.valid_plan((5, 55)))      # below 10 s
        self.assertFalse(T.valid_plan((55, 5)))      # above 50 s
        self.assertFalse(T.valid_plan((32, 28)))     # not a 5 s step
        self.assertFalse(T.valid_plan((30, 35)))     # does not sum to 60

    def test_fixed_baseline_is_30_30(self):
        self.assertEqual(T.FIXED_PLAN, (30, 30))

    def test_signal_never_conflicts_and_timings_are_exact(self):
        plan = (35, 25)
        sim = T.Simulation(T.Scenario("Medium", "Medium", 7), plan)
        durations, last_phase, count = {}, 0, 0
        while not sim.finished:
            st = sim.signal_states()
            ns_go = st["N"] != "red" or st["S"] != "red"
            ew_go = st["E"] != "red" or st["W"] != "red"
            self.assertFalse(ns_go and ew_go, "conflicting signals at tick %d" % sim.tick)
            self.assertEqual(st["N"], st["S"])
            self.assertEqual(st["E"], st["W"])
            if sim.phase != last_phase:
                durations.setdefault(last_phase, set()).add(count)
                last_phase, count = sim.phase, 0
            count += 1
            sim.step()
        self.assertEqual(durations[0], {35 * T.TICKS_PER_SEC})
        self.assertEqual(durations[1], {3 * T.TICKS_PER_SEC})
        self.assertEqual(durations[2], {25 * T.TICKS_PER_SEC})
        self.assertEqual(durations[3], {3 * T.TICKS_PER_SEC})


class VehicleTests(unittest.TestCase):
    def test_cars_only_cross_on_green_or_yellow_and_never_overlap(self):
        for plan in (T.FIXED_PLAN, (45, 15)):
            sim = T.Simulation(heavy_ns(), plan)
            prev = {}
            while not sim.finished:
                states = sim.signal_states()
                sim.step()
                for d in T.DIRECTIONS:
                    lane = sim.lanes[d]
                    for i, v in enumerate(lane):
                        was = prev.get(id(v))
                        if was is not None and was < T.STOP_LINE <= v.s:
                            self.assertNotEqual(states[d], "red",
                                                "car crossed on red (%s, tick %d)" % (d, sim.tick))
                        prev[id(v)] = v.s
                        if i > 0:
                            leader = lane[i - 1]
                            self.assertLessEqual(v.s, leader.s - T.VEH_LEN + 1e-6, "cars overlap")
            self.assertEqual(sim.red_violations, 0)

    def test_red_light_actually_holds_cars(self):
        """While an approach is red, cars really queue (stand still) before the stop line."""
        sim = T.Simulation(T.Scenario("Medium", "Medium", 3), T.FIXED_PLAN)
        stood_still_on_red = False
        while not sim.finished:
            sim.step()
            if sim.signal_state("E") == "red":
                for v in sim.lanes["E"]:
                    if v.s < T.STOP_LINE and v.v < T.WAIT_SPEED:
                        stood_still_on_red = True
        self.assertTrue(stood_still_on_red)


class FairnessTests(unittest.TestCase):
    def test_same_arrivals_for_every_plan(self):
        sc = heavy_ns()
        a = T.Simulation(sc, (30, 30)).run()
        b = T.Simulation(sc, (45, 15)).run()
        self.assertEqual(a["arrived"], b["arrived"])
        self.assertEqual(a["arrived"], sc.total_vehicles)
        self.assertEqual(a["time"], b["time"])

    def test_deterministic(self):
        r1 = T.Simulation(T.Scenario("High", "High", 5), (30, 30)).run()
        r2 = T.Simulation(T.Scenario("High", "High", 5), (30, 30)).run()
        self.assertEqual(r1, r2)

    def test_stepping_live_equals_headless_run(self):
        sc = T.Scenario("Medium", "High", 11)
        live = T.Simulation(sc, (25, 35))
        while not live.finished:
            live.step()                       # what the GUI does, tick by tick
        self.assertEqual(live.metrics(), T.Simulation(sc, (25, 35)).run())

    def test_traffic_levels_change_arrivals(self):
        low = T.Scenario("Low", "Low", 1).total_vehicles
        high = T.Scenario("High", "High", 1).total_vehicles
        self.assertGreater(high, 2 * low)


class HillClimbingTests(unittest.TestCase):
    def test_neighbours(self):
        self.assertEqual(sorted(O.neighbours((30, 30))), [(25, 35), (35, 25)])
        self.assertEqual(O.neighbours((50, 10)), [(45, 15)])     # only one valid neighbour
        self.assertEqual(O.neighbours((10, 50)), [(15, 45)])
        for p in O.all_plans():
            for n in O.neighbours(p):
                self.assertTrue(T.valid_plan(n))

    def test_hill_climbing_evaluates_neighbours_and_moves_uphill(self):
        sc = heavy_ns()
        ev = O.Evaluator(sc)
        res = O.hill_climb(ev)
        self.assertEqual(res.start_plan, (30, 30))
        self.assertTrue(T.valid_plan(res.best_plan))
        self.assertNotEqual(res.best_plan, (30, 30), "heavy N-S traffic should shift green to N-S")
        self.assertGreater(res.best_plan[0], 30)
        self.assertLess(res.best_score, res.start_score)
        first = res.steps[0]
        self.assertEqual(sorted(p for p, _ in first["neighbours"]), [(25, 35), (35, 25)])
        # every accepted move strictly improved the objective
        score = res.start_score
        for st in res.steps:
            if st["move"] is not None:
                nb_score = dict(st["neighbours"])[st["move"]]
                self.assertLess(nb_score, score)
                score = nb_score
        # the reported best is a genuine simulated result
        self.assertAlmostEqual(res.best_score, T.Simulation(sc, res.best_plan).run()["total_wait"])
        self.assertLessEqual(res.simulations, len(O.all_plans()))

    def test_stops_when_no_neighbour_improves(self):
        sc = T.Scenario("High", "High", 42)
        res = O.hill_climb(O.Evaluator(sc))
        self.assertLessEqual(res.best_score, res.start_score)
        self.assertEqual(res.stop_reason, "no neighbour improves the current plan")

    def test_iteration_limit(self):
        res = O.hill_climb(O.Evaluator(heavy_ns()), max_iter=1)
        self.assertEqual(len(res.steps), 1)
        self.assertIn("iteration limit", res.stop_reason)

    def test_all_four_required_scenarios(self):
        for name, ns, ew in O.TEST_SCENARIOS:
            rep = O.compare_modes(T.Scenario(ns, ew, 42))
            self.assertEqual(rep["fixed"]["plan"], (30, 30), name)
            self.assertTrue(T.valid_plan(rep["ai_plan"]), name)
            self.assertLessEqual(rep["ai"]["total_wait"], rep["fixed"]["total_wait"] + 1e-9, name)
            self.assertEqual(rep["fixed"]["arrived"], rep["ai"]["arrived"], name)

    def test_improvement_percent(self):
        self.assertAlmostEqual(O.improvement_percent(20.0, 5.0), 75.0)
        self.assertIsNone(O.improvement_percent(0.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
