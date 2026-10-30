#!/usr/bin/env python3
"""
main.py - Pygame window, controls, simulation loop and display.

Intelligent Traffic Signal Control
    Fixed timing (30 s / 30 s)  versus  Hill-Climbing optimised timing.

Run:   python main.py

Keys:  SPACE start/pause   R reset   M switch mode   C comparison
       N new random scenario   P save screenshot   ESC close / quit
"""

import csv
import os
import random
import sys
import threading
import time

import pygame

import optimizer as O
import traffic as T

# --------------------------------------------------------------------------
# Window / drawing constants
# --------------------------------------------------------------------------
WIN_W, WIN_H = 1280, 720
SIM_W = 720                     # the intersection occupies the left 720x720
CX = CY = SIM_W / 2
SCALE = 4.8                     # pixels per metre (view shows the central +/-75 m)
LEFT_HAND_TRAFFIC = True        # India drives on the left. Set False for right-hand.
CAR_W = 2.4                     # drawn car width (m)

PX = 736                        # right-hand panel
PW = WIN_W - PX - 16

C_GRASS = (66, 122, 80)
C_ROAD = (56, 58, 64)
C_PANEL = (22, 26, 34)
C_CARD = (33, 39, 50)
C_TEXT = (232, 236, 242)
C_MUTED = (142, 152, 168)
C_ACCENT = (64, 150, 225)
C_GOOD = (74, 190, 122)
C_BAD = (232, 88, 88)
C_AMBER = (238, 184, 60)
C_BTN = (48, 56, 72)
C_BTN_HOVER = (62, 72, 92)
CAR_COLORS = [(231, 76, 60), (52, 152, 219), (241, 196, 15), (155, 89, 182),
              (236, 240, 241), (230, 126, 34), (26, 188, 156), (189, 195, 199)]

HEADING = {"N": (0, 1), "S": (0, -1), "E": (-1, 0), "W": (1, 0)}   # screen y points down
SIGNAL_COLOR = {"red": (235, 64, 52), "yellow": (250, 200, 40), "green": (60, 220, 110)}
SPEEDS = [1, 2, 4, 8, 16, 32]
MAX_STEPS_PER_FRAME = 600


def lane_xy(direction, s, lateral=0.0):
    """Screen position of lane coordinate `s` (metres). `lateral` is metres to the driver's left."""
    hx, hy = HEADING[direction]
    nx, ny = hy, -hx                                 # driver's left-hand normal
    off = (T.LANE_W / 2.0) * (1 if LEFT_HAND_TRAFFIC else -1) + lateral
    x = CX + ((s - T.HALF) * hx + off * nx) * SCALE
    y = CY + ((s - T.HALF) * hy + off * ny) * SCALE
    return x, y


def fmt_plan(plan):
    return "%d/%d" % plan


# --------------------------------------------------------------------------
# Tiny button widget
# --------------------------------------------------------------------------
class Button:
    def __init__(self, rect, label, on_click, active=None, enabled=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.active = active
        self.enabled = enabled

    def is_enabled(self):
        return self.enabled() if self.enabled else True

    def draw(self, surf, font, mouse):
        on = self.active() if self.active else False
        ok = self.is_enabled()
        if not ok:
            col = (38, 43, 54)
        elif on:
            col = C_ACCENT
        elif self.rect.collidepoint(mouse):
            col = C_BTN_HOVER
        else:
            col = C_BTN
        pygame.draw.rect(surf, col, self.rect, border_radius=7)
        label = self.label() if callable(self.label) else self.label
        txt = font.render(label, True, C_TEXT if ok else C_MUTED)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def click(self, pos):
        if self.rect.collidepoint(pos) and self.is_enabled():
            self.on_click()
            return True
        return False


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class App:
    def __init__(self, headless=False):
        pygame.display.init()
        pygame.font.init()          # (no audio needed, so no sound-device errors)
        pygame.display.set_caption("Intelligent Traffic Signal Control - Fixed vs Hill Climbing")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        sans = "dejavusans,segoeui,helveticaneue,arial,helvetica"
        mono = "dejavusansmono,consolas,menlo,couriernew,monospace"
        self.f_title = pygame.font.SysFont(sans, 22, bold=True)
        self.f_big = pygame.font.SysFont(sans, 26, bold=True)
        self.f_md = pygame.font.SysFont(sans, 15)
        self.f_md_b = pygame.font.SysFont(sans, 15, bold=True)
        self.f_sm = pygame.font.SysFont(sans, 12)
        self.f_mono = pygame.font.SysFont(mono, 12)
        self.f_mono_b = pygame.font.SysFont(mono, 12, bold=True)

        # ----- state -----
        self.mode = "fixed"                       # "fixed" | "ai"
        self.levels = {"NS": "Medium", "EW": "Medium"}
        self.seed = 42
        self.speed_idx = 2                        # 4x
        self.running = False
        self.acc = 0.0                            # left-over sim time (for smooth drawing)
        self.sim = None
        self.scenario = None

        self.lock = threading.Lock()
        self.opt_cache = {}                       # scenario.key -> entry dict (see optimise())
        self.cmp = {"open": False, "busy": False, "rows": [], "message": ""}
        self.toast = ("", 0.0)

        self.build_buttons()
        self.new_scenario()
        self.quit = False

    # ------------------------------------------------------------------
    # Scenario / simulation management
    # ------------------------------------------------------------------
    def new_scenario(self):
        self.scenario = T.Scenario(self.levels["NS"], self.levels["EW"], self.seed)
        self.rebuild()

    def rebuild(self):
        """(Re)create the simulation for the current scenario and mode."""
        self.running = False
        self.acc = 0.0
        self.sim = None
        if self.mode == "fixed":
            self.sim = T.Simulation(self.scenario, T.FIXED_PLAN)
        else:
            self.ensure_optimising(self.scenario)    # sim is created when ready

    def current_entry(self):
        return self.opt_cache.get(self.scenario.key)

    def ensure_optimising(self, scenario):
        if scenario.key not in self.opt_cache:
            threading.Thread(target=self.optimise, args=(scenario,), daemon=True).start()
            time.sleep(0.001)

    def optimise(self, scenario):
        """
        Run Hill Climbing (+ the reference report) for a scenario. Blocking, meant for
        a worker thread. Results are cached so each scenario is optimised only once.
        """
        with self.lock:
            entry = self.opt_cache.get(scenario.key)
            owner = entry is None
            if owner:
                entry = {"events": [], "result": None, "report": None, "done": False,
                         "error": None}
                self.opt_cache[scenario.key] = entry
        if not owner:
            while not entry["done"]:
                time.sleep(0.02)
            return entry
        try:
            ev = O.Evaluator(scenario)
            res = O.hill_climb(ev, on_event=lambda k, **i: entry["events"].append((k, i)))
            entry["result"] = res                    # the GUI can start the AI run now
            entry["report"] = O.build_report(scenario, ev, res)
        except Exception as exc:                     # pragma: no cover
            entry["error"] = str(exc)
        finally:
            entry["done"] = True
        return entry

    def update_ai_sim(self):
        if self.mode == "ai" and self.sim is None:
            entry = self.current_entry()
            if entry and entry["result"] is not None:
                self.sim = T.Simulation(self.scenario, entry["result"].best_plan)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def toggle_run(self):
        if self.sim is None:
            return
        if self.sim.finished:
            self.rebuild()
            self.update_ai_sim()
            if self.sim is None:
                return
        self.running = not self.running

    def reset(self):
        self.rebuild()

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.rebuild()

    def set_level(self, road, level):
        if self.levels[road] != level:
            self.levels[road] = level
            self.new_scenario()

    def new_seed(self):
        self.seed = random.randint(1, 9999)
        self.new_scenario()

    def set_speed(self, idx):
        self.speed_idx = idx

    def screenshot(self):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "screenshot_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
        pygame.image.save(self.screen, path)
        self.toast = ("Saved " + os.path.relpath(path), time.time() + 3)

    # ---- comparison -----------------------------------------------------
    def open_compare(self):
        self.cmp["open"] = True
        if not self.cmp["rows"] and not self.cmp["busy"]:
            self.run_compare()

    def run_compare(self):
        if self.cmp["busy"]:
            return
        scenarios = [(name, T.Scenario(ns, ew, self.seed)) for name, ns, ew in O.TEST_SCENARIOS]
        cur = "Current setting: NS %s / EW %s" % (self.levels["NS"], self.levels["EW"])
        scenarios.append((cur, T.Scenario(self.levels["NS"], self.levels["EW"], self.seed)))
        rows = []
        self.cmp.update(busy=True, rows=rows, message="Starting...", seed=self.seed)

        def work():
            try:
                for i, (name, sc) in enumerate(scenarios, 1):
                    self.cmp["message"] = "Running %d/%d: %s ..." % (i, len(scenarios), name)
                    entry = self.optimise(sc)
                    if entry["report"] is not None:
                        rows.append((name, entry["report"]))
                self.cmp["message"] = "Done - all values measured by the simulation."
            finally:
                self.cmp["busy"] = False

        threading.Thread(target=work, daemon=True).start()

    def save_csv(self):
        rows = self.cmp["rows"]
        if not rows:
            return
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "comparison_results.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "ns_level", "ew_level", "seed", "duration_s",
                        "fixed_plan", "fixed_avg_wait_s", "fixed_served", "fixed_queue",
                        "ai_plan", "ai_avg_wait_s", "ai_served", "ai_queue",
                        "improvement_pct", "bruteforce_best_plan", "hc_simulations"])
            for name, r in rows:
                sc, f, a = r["scenario"], r["fixed"], r["ai"]
                imp = "" if r["improvement"] is None else "%.1f" % r["improvement"]
                w.writerow([name, sc.ns_level, sc.ew_level, sc.seed, sc.duration,
                            fmt_plan(f["plan"]), "%.2f" % f["avg_wait"], f["served"], f["queue"],
                            fmt_plan(a["plan"]), "%.2f" % a["avg_wait"], a["served"], a["queue"],
                            imp, fmt_plan(r["global_best"]), r["hc_simulations"]])
        self.toast = ("Saved " + os.path.relpath(path), time.time() + 3)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_buttons(self):
        B = Button
        b = self.buttons = []
        b.append(B((PX, 62, 118, 34),
                   lambda: "Pause" if self.running else
                   ("Restart" if self.sim and self.sim.finished else "Start"),
                   self.toggle_run, active=lambda: self.running,
                   enabled=lambda: self.sim is not None))
        b.append(B((PX + 126, 62, 86, 34), "Reset", self.reset))
        b.append(B((PX + 220, 62, 138, 34), "New scenario", self.new_seed))
        b.append(B((PX + 366, 62, 162, 34), "Compare modes", self.open_compare))

        b.append(B((PX + 110, 106, 150, 32), "Fixed  30/30",
                   lambda: self.set_mode("fixed"), active=lambda: self.mode == "fixed"))
        b.append(B((PX + 268, 106, 210, 32), "AI: Hill Climbing",
                   lambda: self.set_mode("ai"), active=lambda: self.mode == "ai"))

        for row, (road, y) in enumerate((("NS", 148), ("EW", 186))):
            for i, lvl in enumerate(("Low", "Medium", "High")):
                b.append(B((PX + 110 + i * 98, y, 90, 30), lvl,
                           (lambda r=road, l=lvl: self.set_level(r, l)),
                           active=(lambda r=road, l=lvl: self.levels[r] == l)))

        for i, sp in enumerate(SPEEDS):
            b.append(B((PX + 110 + i * 62, 226, 56, 30), "%dx" % sp,
                       (lambda k=i: self.set_speed(k)), active=(lambda k=i: self.speed_idx == k)))

        # comparison overlay buttons
        c = self.cmp_buttons = []
        c.append(B((70, 640, 190, 38), "Run tests again", self.run_compare,
                   enabled=lambda: not self.cmp["busy"]))
        c.append(B((270, 640, 170, 38), "Save CSV", self.save_csv,
                   enabled=lambda: bool(self.cmp["rows"])))
        c.append(B((1090, 640, 120, 38), "Close", lambda: self.cmp.update(open=False)))

    # ------------------------------------------------------------------
    # Events / update
    # ------------------------------------------------------------------
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.quit = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if self.cmp["open"]:
                        self.cmp["open"] = False
                    else:
                        self.quit = True
                elif ev.key == pygame.K_p:
                    self.screenshot()
                elif self.cmp["open"]:
                    pass
                elif ev.key == pygame.K_SPACE:
                    self.toggle_run()
                elif ev.key == pygame.K_r:
                    self.reset()
                elif ev.key == pygame.K_m:
                    self.set_mode("ai" if self.mode == "fixed" else "fixed")
                elif ev.key == pygame.K_n:
                    self.new_seed()
                elif ev.key == pygame.K_c:
                    self.open_compare()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                group = self.cmp_buttons if self.cmp["open"] else self.buttons
                for btn in group:
                    if btn.click(ev.pos):
                        break

    def update(self, dt_real):
        self.update_ai_sim()
        sim = self.sim
        if sim is None or not self.running:
            return
        if sim.finished:
            self.running = False
            return
        self.acc += dt_real * SPEEDS[self.speed_idx]
        steps = 0
        while self.acc >= T.DT and steps < MAX_STEPS_PER_FRAME and not sim.finished:
            sim.step()
            self.acc -= T.DT
            steps += 1
        if steps >= MAX_STEPS_PER_FRAME:
            self.acc = 0.0
        if sim.finished:
            self.running = False
            self.acc = 0.0

    # ------------------------------------------------------------------
    # Drawing: intersection
    # ------------------------------------------------------------------
    def draw_world(self):
        scr = self.screen
        pygame.draw.rect(scr, C_GRASS, (0, 0, SIM_W, WIN_H))
        rw = T.LANE_W * 2 * SCALE
        pygame.draw.rect(scr, C_ROAD, (0, CY - rw / 2, SIM_W, rw))
        pygame.draw.rect(scr, C_ROAD, (CX - rw / 2, 0, rw, SIM_W))

        # road edges and dashed centre lines (outside the junction box)
        edge = (210, 212, 216)
        half = rw / 2
        for sgn in (-1, 1):
            for a, b in ((0, CX - half), (CX + half, SIM_W)):
                pygame.draw.line(scr, edge, (a, CY + sgn * half), (b, CY + sgn * half), 2)
                pygame.draw.line(scr, edge, (CX + sgn * half, a), (CX + sgn * half, b), 2)
        dash = 14
        for a, b in ((0, CX - half), (CX + half, SIM_W)):
            x = a
            while x < b:
                pygame.draw.line(scr, C_AMBER, (x, CY), (min(x + dash, b), CY), 2)
                pygame.draw.line(scr, C_AMBER, (CX, x), (CX, min(x + dash, b)), 2)
                x += dash * 2

        sim = self.sim
        states = sim.signal_states() if sim else {d: "red" for d in T.DIRECTIONS}

        # stop lines coloured by the signal, plus signal heads
        for d in T.DIRECTIONS:
            p1 = lane_xy(d, T.STOP_LINE, -T.LANE_W / 2)
            p2 = lane_xy(d, T.STOP_LINE, +T.LANE_W / 2)
            pygame.draw.line(scr, SIGNAL_COLOR[states[d]], p1, p2, 6)
            self.draw_signal_head(d, states[d])

        # cars (interpolated by the left-over time so motion is smooth)
        lead = self.acc if self.running else 0.0
        if sim:
            for d in T.DIRECTIONS:
                for v in sim.lanes[d]:
                    waiting = v.s < T.STOP_LINE and v.v < T.WAIT_SPEED
                    self.draw_car(d, v.s + v.v * lead, CAR_COLORS[v.color_id % len(CAR_COLORS)],
                                  waiting)

        # queue labels (placed beside the road, never on top of the lanes)
        sign = 1 if LEFT_HAND_TRAFFIC else -1
        for d in T.DIRECTIONS:
            q = sim.queue_length(d) if sim else 0
            x, y = lane_xy(d, 40, 6 * sign)
            txt = self.f_sm.render("%s approach: %d waiting" % (d, q), True, C_TEXT)
            box = txt.get_rect().inflate(14, 8)
            if HEADING[d][0] == 0:                       # vertical road: put label to the side
                if x > CX:
                    box.midleft = (CX + rw / 2 + 10, y)
                else:
                    box.midright = (CX - rw / 2 - 10, y)
            else:
                box.center = (x, y)
            pygame.draw.rect(scr, (20, 24, 30), box, border_radius=5)
            scr.blit(txt, txt.get_rect(center=box.center))

    def draw_signal_head(self, d, state):
        scr = self.screen
        sign = 1 if LEFT_HAND_TRAFFIC else -1
        x, y = lane_xy(d, T.STOP_LINE - 3, 4.2 * sign)
        body = pygame.Rect(0, 0, 18, 46)
        body.center = (x, y)
        pygame.draw.rect(scr, (14, 14, 16), body, border_radius=5)
        for i, name in enumerate(("red", "yellow", "green")):
            on = state == name
            col = SIGNAL_COLOR[name] if on else (52, 52, 56)
            c = (body.centerx, body.top + 9 + i * 14)
            pygame.draw.circle(scr, col, c, 5)
            if on:
                pygame.draw.circle(scr, (255, 255, 255), c, 5, 1)
        if self.sim and state in ("green", "yellow"):
            secs = int(self.sim.phase_time_left + 0.999)
            t = self.f_sm.render(str(secs), True, SIGNAL_COLOR[state])
            scr.blit(t, t.get_rect(midtop=(body.centerx, body.bottom + 2)))

    def draw_car(self, d, s_front, color, waiting):
        hx, hy = HEADING[d]
        nx, ny = hy, -hx
        cx, cy = lane_xy(d, s_front - T.VEH_LEN / 2)
        L = T.VEH_LEN * SCALE / 2
        Wd = CAR_W * SCALE / 2

        def pt(a, l):
            return (cx + hx * a + nx * l, cy + hy * a + ny * l)

        body = [pt(L, Wd), pt(L, -Wd), pt(-L, -Wd), pt(-L, Wd)]
        pygame.draw.polygon(self.screen, color, body)
        pygame.draw.polygon(self.screen, (20, 24, 30),
                            [pt(L * 0.55, Wd * 0.8), pt(L * 0.55, -Wd * 0.8),
                             pt(L * 0.15, -Wd * 0.8), pt(L * 0.15, Wd * 0.8)])
        if waiting:   # brake lights
            pygame.draw.polygon(self.screen, (255, 40, 40),
                                [pt(-L, Wd), pt(-L, -Wd), pt(-L * 0.8, -Wd), pt(-L * 0.8, Wd)])

    def draw_sim_overlays(self):
        scr = self.screen
        sim = self.sim
        entry = self.current_entry()
        if self.mode == "fixed":
            head = "FIXED TIMING   NS %d s / EW %d s" % T.FIXED_PLAN
            col = C_AMBER
        elif sim:
            head = "AI (HILL CLIMBING)   NS %d s / EW %d s" % sim.plan
            col = C_GOOD
        else:
            head = "AI (HILL CLIMBING)   optimising..."
            col = C_GOOD
        t = self.f_md_b.render(head, True, col)
        r = t.get_rect(topleft=(12, 10)).inflate(16, 10)
        pygame.draw.rect(scr, (16, 20, 26), r, border_radius=8)
        scr.blit(t, t.get_rect(center=r.center))

        # progress bar
        if sim:
            frac = min(1.0, sim.tick / float(sim.end_tick))
            pygame.draw.rect(scr, (16, 20, 26), (12, WIN_H - 26, SIM_W - 24, 14), border_radius=7)
            pygame.draw.rect(scr, C_ACCENT, (12, WIN_H - 26, (SIM_W - 24) * frac, 14), border_radius=7)
            t = self.f_sm.render("%d / %d s" % (sim.time, sim.scenario.duration), True, C_TEXT)
            scr.blit(t, t.get_rect(center=(SIM_W / 2, WIN_H - 19)))

        msg = None
        if self.mode == "ai" and sim is None:
            msg = "Hill Climbing is searching..."
        elif sim and sim.finished:
            msg = "Run complete - press Restart"
        elif sim and not self.running:
            msg = "Press SPACE or Start" if sim.tick == 0 else "Paused"
        if msg:
            t = self.f_md_b.render(msg, True, C_TEXT)
            r = t.get_rect(topleft=(12, 50)).inflate(24, 12)
            r.topleft = (12, 50)
            pygame.draw.rect(scr, (16, 20, 26), r, border_radius=10)
            pygame.draw.rect(scr, C_ACCENT, r, 1, border_radius=10)
            scr.blit(t, t.get_rect(center=r.center))

    # ------------------------------------------------------------------
    # Drawing: right-hand panel
    # ------------------------------------------------------------------
    def text(self, s, pos, font=None, color=C_TEXT):
        t = (font or self.f_md).render(s, True, color)
        self.screen.blit(t, pos)
        return t.get_width()

    def draw_panel(self):
        scr = self.screen
        pygame.draw.rect(scr, C_PANEL, (SIM_W, 0, WIN_W - SIM_W, WIN_H))
        self.text("Intelligent Traffic Signal Control", (PX, 10), self.f_title)
        self.text("Fixed timing  vs  Hill-Climbing optimised timing", (PX, 38), self.f_sm, C_MUTED)

        mouse = pygame.mouse.get_pos()
        self.text("Signal mode", (PX, 114), self.f_md_b, C_MUTED)
        self.text("N-S traffic", (PX, 155), self.f_md_b, C_MUTED)
        self.text("E-W traffic", (PX, 193), self.f_md_b, C_MUTED)
        self.text("Sim speed", (PX, 233), self.f_md_b, C_MUTED)
        for b in self.buttons:
            b.draw(scr, self.f_md, mouse)

        self.draw_stats()
        self.draw_hc_panel()

    def draw_stats(self):
        scr = self.screen
        box = pygame.Rect(PX, 268, PW, 124)
        pygame.draw.rect(scr, C_CARD, box, border_radius=10)
        sim = self.sim
        x1, x2 = PX + 14, PX + 280
        y = 278
        self.text("SIGNAL", (x1, y), self.f_sm, C_MUTED)
        self.text("RESULTS (measured)", (x2, y), self.f_sm, C_MUTED)
        if sim:
            ph = sim.phase_name
            pc = SIGNAL_COLOR["green"] if "GREEN" in ph else SIGNAL_COLOR["yellow"]
            self.text("Phase: ", (x1, y + 20))
            w = self.text(ph, (x1 + 52, y + 20), self.f_md_b, pc)
            self.text("%.1f s left of %.0f s" % (sim.phase_time_left, sim.phase_duration),
                      (x1, y + 42), self.f_md)
            self.text("Plan: NS %d s | EW %d s | +%d s yellow" % (sim.plan + (T.YELLOW,)),
                      (x1, y + 64), self.f_md)
            self.text("Time: %d / %d s    Seed %d" % (sim.time, sim.scenario.duration,
                                                      sim.scenario.seed), (x1, y + 86), self.f_md)
            m = sim.metrics()
            self.text("Avg wait: ", (x2, y + 18), self.f_md)
            self.text("%.1f s" % m["avg_wait"], (x2 + 74, y + 12), self.f_big, C_TEXT)
            self.text("Vehicles served: %d of %d arrived" % (m["served"], m["arrived"]),
                      (x2, y + 48), self.f_md)
            self.text("Queue remaining: %d  (NS %d | EW %d)" % (
                m["queue"], sim.queue_length("N") + sim.queue_length("S"),
                sim.queue_length("E") + sim.queue_length("W")), (x2, y + 68), self.f_md)
            self.text("Total waiting W: %.0f s" % m["total_wait"], (x2, y + 88), self.f_md,
                      C_MUTED)
        else:
            self.text("Waiting for Hill Climbing result...", (x1, y + 24), self.f_md, C_AMBER)

    def hc_structure(self, entry):
        """Turn the raw event stream into per-iteration records for display."""
        iters, final = [], None
        for kind, info in list(entry["events"]):
            if kind == "current":
                iters.append({"i": info["iteration"], "cur": info["plan"],
                              "score": info["score"], "nbrs": [], "move": None})
            elif kind == "neighbour":
                iters[-1]["nbrs"].append((info["plan"], info["score"]))
            elif kind == "move":
                iters[-1]["move"] = info["to"]
            elif kind == "stop":
                final = info
        return iters, final

    def draw_hc_panel(self):
        scr = self.screen
        box = pygame.Rect(PX, 400, PW, WIN_H - 400 - 10)
        pygame.draw.rect(scr, C_CARD, box, border_radius=10)
        self.text("HILL CLIMBING  -  search for the best (NS, EW) green split", (PX + 14, 408),
                  self.f_md_b, C_ACCENT)
        entry = self.current_entry()
        if entry is None:
            self.text("Select  'AI: Hill Climbing'  to run the search on this scenario.",
                      (PX + 14, 440), self.f_md, C_MUTED)
            self.text("It simulates neighbouring splits (+/-5 s) on the SAME arrivals and",
                      (PX + 14, 462), self.f_sm, C_MUTED)
            self.text("climbs to the one with the lowest total waiting time W.",
                      (PX + 14, 478), self.f_sm, C_MUTED)
            return
        iters, final = self.hc_structure(entry)
        report = entry["report"]
        self.draw_landscape(pygame.Rect(PX + 12, 428, PW - 24, 114), iters, final, report)

        # text log
        lines = []
        for it in iters:
            nb = "  ".join("%s=%s" % (fmt_plan(p), format(s, ",.0f")) for p, s in it["nbrs"])
            lines.append(("#%d %s W=%s | try %s" % (it["i"], fmt_plan(it["cur"]),
                                                    format(it["score"], ",.0f"), nb), C_TEXT))
            if it["move"] is not None:
                lines.append(("   -> better: move to %s" % fmt_plan(it["move"]), C_GOOD))
            elif entry["done"] or final:
                lines.append(("   -> no neighbour is better: STOP", C_AMBER))
            else:
                lines.append(("   evaluating ...", C_MUTED))
        if final:
            lines.append(("RESULT %s  (W = %s s) - %s" % (
                fmt_plan(final["plan"]), format(final["score"], ",.0f"), final["reason"]),
                C_GOOD))
        if report:
            lines.append(("Simulations run by HC: %d.  Brute-force best of all 9: %s -> %s" % (
                report["hc_simulations"], fmt_plan(report["global_best"]),
                "HC found it" if report["found_global"] else "HC stopped at a local optimum"),
                C_MUTED))
        elif not final:
            lines.append(("searching ...", C_MUTED))

        cap = 12
        y = 548
        for txt, col in lines[-cap:]:
            self.text(txt, (PX + 14, y), self.f_mono, col)
            y += 14

    def draw_landscape(self, rect, iters, final, report):
        """Bar chart: W for each of the 9 valid splits. Blue = evaluated by HC."""
        scr = self.screen
        plans = O.all_plans()
        visited = set()
        for it in iters:
            visited.add(it["cur"])
            for p, _ in it["nbrs"]:
                visited.add(p)
        scores = {}
        for it in iters:
            scores[it["cur"]] = it["score"]
            for p, s in it["nbrs"]:
                scores[p] = s
        landscape = report["landscape"] if report else {}
        allv = dict(landscape)
        allv.update(scores)
        if not allv:
            return
        vmax = max(allv.values()) or 1.0
        chart = pygame.Rect(rect.x, rect.y + 30, rect.w, rect.h - 46)
        n = len(plans)
        slot = chart.w / n
        best = final["plan"] if (final and report) else None
        for i, p in enumerate(plans):
            val = allv.get(p)
            bx = chart.x + i * slot + 6
            bw = slot - 12
            if val is not None:
                h = max(2, int(chart.h * (val / vmax) ** 0.5))
                col = C_ACCENT if p in visited else (82, 92, 110)
                if best == p:
                    col = C_GOOD
                r = pygame.Rect(bx, chart.bottom - h, bw, h)
                pygame.draw.rect(scr, col, r, border_radius=3)
                if p == T.FIXED_PLAN:
                    pygame.draw.rect(scr, C_AMBER, r, 2, border_radius=3)
                lab = self.f_sm.render(format(val / 1000.0, ".1f") + "k" if val >= 1000
                                       else "%d" % val, True, C_MUTED)
                scr.blit(lab, lab.get_rect(midbottom=(bx + bw / 2, r.top - 1)))
            nt = self.f_sm.render(fmt_plan(p), True, C_TEXT if p in visited else C_MUTED)
            scr.blit(nt, nt.get_rect(midtop=(bx + bw / 2, chart.bottom + 2)))
        cap = self.f_sm.render("W per split (sqrt scale) | blue: tested by HC | "
                               "green: chosen | amber: fixed", True, C_MUTED)
        scr.blit(cap, (rect.x, rect.y))

    # ------------------------------------------------------------------
    # Drawing: comparison overlay
    # ------------------------------------------------------------------
    def draw_compare(self):
        scr = self.screen
        shade = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        shade.fill((8, 10, 14, 215))
        scr.blit(shade, (0, 0))
        card = pygame.Rect(40, 30, WIN_W - 80, WIN_H - 60)
        pygame.draw.rect(scr, C_CARD, card, border_radius=14)
        self.text("Fixed vs AI (Hill Climbing) - same arrival schedule for both modes",
                  (70, 46), self.f_title)
        self.text("Each row: identical vehicle arrivals, %d s simulated, seed %s. "
                  "All numbers are produced by the simulation runs." % (
                      T.SIM_DURATION, self.cmp.get("seed", self.seed)),
                  (70, 78), self.f_sm, C_MUTED)

        cols = [("Scenario", 70, 280), ("Fixed avg wait", 350, 110), ("served", 462, 62),
                ("queue", 526, 54), ("AI plan", 596, 70), ("AI avg wait", 670, 100),
                ("served", 772, 62), ("queue", 836, 54), ("Improvement", 906, 110),
                ("Brute-force best", 1030, 120)]
        y = 108
        pygame.draw.rect(scr, (44, 52, 66), (60, y - 6, WIN_W - 120, 30), border_radius=6)
        for name, x, w in cols:
            self.text(name, (x, y), self.f_sm, C_MUTED)

        y = 146
        for name, r in self.cmp["rows"]:
            f, a = r["fixed"], r["ai"]
            self.text(name, (70, y), self.f_md)
            self.text("%.1f s" % f["avg_wait"], (350, y), self.f_md)
            self.text(str(f["served"]), (462, y))
            self.text(str(f["queue"]), (526, y))
            changed = r["ai_plan"] != T.FIXED_PLAN
            self.text(fmt_plan(r["ai_plan"]), (596, y), self.f_md_b, C_GOOD if changed else C_AMBER)
            self.text("%.1f s" % a["avg_wait"], (670, y), self.f_md)
            self.text(str(a["served"]), (772, y))
            self.text(str(a["queue"]), (836, y))
            imp = r["improvement"]
            if imp is None:
                self.text("n/a (fixed wait = 0)", (906, y), self.f_sm, C_MUTED)
            else:
                col = C_GOOD if imp > 0.05 else C_MUTED
                self.text("%+.1f %%" % imp, (906, y), self.f_md_b, col)
            self.text("%s  %s" % (fmt_plan(r["global_best"]),
                                  "(HC found it)" if r["found_global"] else "(HC: local opt.)"),
                      (1030, y), self.f_sm, C_MUTED)
            y += 38
            pygame.draw.line(scr, (52, 60, 76), (70, y - 8), (WIN_W - 70, y - 8), 1)

        y = max(y + 6, 360)
        self.text(self.cmp["message"], (70, y), self.f_md_b,
                  C_AMBER if self.cmp["busy"] else C_GOOD)
        notes = [
            "Average wait = total waiting time of every arrived vehicle / vehicles arrived (a car",
            "counts as waiting while it is before the stop line and slower than 0.5 m/s).",
            "Improvement % = (fixed avg - AI avg) / fixed avg x 100.   Served = cars through the junction.",
            "Hill Climbing is a local search, so it is not guaranteed to find the global best split;",
            "'Brute-force best' (all 9 splits) is shown only to report how well it did. The AI never uses it.",
            "Note: the plan is optimised on the same arrival schedule it is then measured on, as the",
            "project specification asks - try 'New scenario' or other seeds to see results vary.",
        ]
        y += 34
        for n in notes:
            self.text(n, (70, y), self.f_sm, C_MUTED)
            y += 18

        mouse = pygame.mouse.get_pos()
        for b in self.cmp_buttons:
            b.draw(scr, self.f_md, mouse)

    # ------------------------------------------------------------------
    def draw(self):
        self.draw_world()
        self.draw_sim_overlays()
        self.draw_panel()
        hint = self.f_sm.render("SPACE start/pause   R reset   M mode   N new scenario   "
                                "C compare   P screenshot", True, C_MUTED)
        self.screen.blit(hint, (12, WIN_H - 48))
        if self.cmp["open"]:
            self.draw_compare()
        if self.toast[1] > time.time():
            t = self.f_md_b.render(self.toast[0], True, C_TEXT)
            r = t.get_rect(center=(WIN_W // 2, WIN_H - 40)).inflate(24, 12)
            pygame.draw.rect(self.screen, (16, 20, 26), r, border_radius=8)
            pygame.draw.rect(self.screen, C_GOOD, r, 1, border_radius=8)
            self.screen.blit(t, t.get_rect(center=r.center))
        pygame.display.flip()

    def run(self):
        sys.setswitchinterval(0.002)          # keep the UI responsive while HC runs
        while not self.quit:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(min(dt, 0.1))
            self.draw()
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
