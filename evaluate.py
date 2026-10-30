#!/usr/bin/env python3
"""
evaluate.py - run the Fixed-vs-AI comparison WITHOUT the GUI (no display needed).

    python evaluate.py                 # the 4 required scenarios, seed 42
    python evaluate.py --seed 7        # a different random arrival schedule
    python evaluate.py --seeds 10      # repeat for seeds 1..10 and show the mean
    python evaluate.py --csv results/my_run.csv

Every number printed is measured by running the simulation; nothing is typed in.
"""

import argparse
import csv
import os

import optimizer as O
import traffic as T


def run_table(seed):
    rows = []
    for name, ns, ew in O.TEST_SCENARIOS:
        rows.append((name, O.compare_modes(T.Scenario(ns, ew, seed))))
    return rows


def fmt_imp(x):
    return "   n/a" if x is None else "%+5.1f%%" % x


def print_table(rows, seed):
    print("\nSeed %d | %d s simulated | identical arrivals for Fixed and AI" % (seed, T.SIM_DURATION))
    head = "%-27s | %-22s | %-30s | %7s | %s" % (
        "Scenario", "FIXED 30/30", "AI (Hill Climbing)", "Improve", "Best of all 9")
    sub = "%-27s | %7s %6s %6s  | %-5s %7s %6s %6s   | %7s | " % (
        "", "avgW(s)", "served", "queue", "plan", "avgW(s)", "served", "queue", "")
    print(head)
    print(sub)
    print("-" * len(head))
    for name, r in rows:
        f, a = r["fixed"], r["ai"]
        print("%-27s | %7.1f %6d %6d  | %-5s %7.1f %6d %6d   | %s | %s %s" % (
            name, f["avg_wait"], f["served"], f["queue"],
            "%d/%d" % r["ai_plan"], a["avg_wait"], a["served"], a["queue"],
            fmt_imp(r["improvement"]), "%d/%d" % r["global_best"],
            "(HC found it)" if r["found_global"] else "(HC stuck at local optimum)"))


def write_csv(path, all_rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "scenario", "fixed_plan", "fixed_avg_wait_s", "fixed_served",
                    "fixed_queue", "ai_plan", "ai_avg_wait_s", "ai_served", "ai_queue",
                    "improvement_pct", "bruteforce_best_plan", "hc_simulations"])
        for seed, rows in all_rows:
            for name, r in rows:
                f, a = r["fixed"], r["ai"]
                imp = "" if r["improvement"] is None else "%.2f" % r["improvement"]
                w.writerow([seed, name, "%d/%d" % f["plan"], "%.2f" % f["avg_wait"], f["served"],
                            f["queue"], "%d/%d" % a["plan"], "%.2f" % a["avg_wait"], a["served"],
                            a["queue"], imp, "%d/%d" % r["global_best"], r["hc_simulations"]])
    print("\nSaved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=0, help="run seeds 1..N and print the mean improvement")
    ap.add_argument("--csv", default=os.path.join("results", "evaluation.csv"))
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1)) if args.seeds else [args.seed]
    all_rows = []
    for sd in seeds:
        rows = run_table(sd)
        all_rows.append((sd, rows))
        print_table(rows, sd)

    if len(seeds) > 1:
        print("\nMean over %d seeds" % len(seeds))
        for i, (name, _, _) in enumerate(O.TEST_SCENARIOS):
            fixed = [rows[i][1]["fixed"]["avg_wait"] for _, rows in all_rows]
            ai = [rows[i][1]["ai"]["avg_wait"] for _, rows in all_rows]
            imps = [rows[i][1]["improvement"] for _, rows in all_rows
                    if rows[i][1]["improvement"] is not None]
            glob = sum(1 for _, rows in all_rows if rows[i][1]["found_global"])
            print("  %-27s fixed %.1f s  ->  AI %.1f s   mean improvement %s   "
                  "HC matched brute-force best in %d/%d seeds" % (
                      name, sum(fixed) / len(fixed), sum(ai) / len(ai),
                      fmt_imp(sum(imps) / len(imps) if imps else None), glob, len(seeds)))
    write_csv(args.csv, all_rows)


if __name__ == "__main__":
    main()
