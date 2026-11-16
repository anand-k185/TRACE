#!/usr/bin/env python3
"""
TRACE - turn test_all.py results into a seen-vs-unseen comparison for a talk.

Groups every test drive by how the model has met that track:
  Trained on    lab_00 .. lab_14   (the model learned from these)
  Development   lab_15             (never trained on, but used to judge and tune the model)
  Brand new     lab_16 .. lab_20   (never used for anything)

Prints a summary table and writes robustness_table.csv, robustness_drives.csv and robustness.png.

  python3 compare_results.py                      # reads test_results.json
  python3 compare_results.py runA.json runB.json  # or several result files
"""
import csv, json, sys
import numpy as np

GROUPS = ["Trained on", "Development", "Brand new"]
COLOR = {"Trained on": "#2a78d6", "Development": "#eb6834", "Brand new": "#1baf7a"}
MARK = {"Trained on": "o", "Development": "s", "Brand new": "D"}
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def group_of(name):
    try:
        n = int(name.split("_")[1])
    except Exception:
        return None
    if not name.startswith("lab_"):
        return None
    return "Trained on" if n <= 14 else "Development" if n == 15 else "Brand new"


def main():
    files = sys.argv[1:] or ["test_results.json"]
    rows = []
    for f in files:
        rows += json.load(open(f))
    drives, failed = [], 0
    for r in rows:
        g = group_of(r.get("map", ""))
        if g is None:
            continue
        if "error" in r or r.get("distance_m", 0) < 1:
            failed += 1
            continue
        dist = r["distance_m"]
        drives.append({
            "group": g, "map": r["map"], "direction": r["direction"],
            "distance_m": dist, "laps": r["laps"],
            "contacts": r["duct_contacts"],
            "contacts_per_100m": 100.0 * r["duct_contacts"] / dist,
            "mean_dev_cm": 100 * r["centerline_dev_mean_m"],
            "p95_dev_cm": 100 * r["centerline_dev_p95_m"],
            "closest_cm": 100 * r["closest_approach_m"],
            "stuck_pct": 100.0 * r["stuck_s"] / max(r["duration_s"], 1e-6),
            "clean": int(r["duct_contacts"] == 0 and r["stuck_s"] == 0),
        })
    if not drives:
        sys.exit("no usable drives found")

    # ---------------- table
    summ = []
    for g in GROUPS:
        d = [x for x in drives if x["group"] == g]
        if not d:
            continue
        tot = sum(x["distance_m"] for x in d)
        summ.append({
            "group": g, "tracks": len({x["map"] for x in d}), "drives": len(d),
            "distance_m": round(tot),
            "clean_drives": f"{sum(x['clean'] for x in d)}/{len(d)}",
            "contacts_per_100m": round(100.0 * sum(x["contacts"] for x in d) / tot, 2),
            "mean_dev_cm": round(float(np.mean([x["mean_dev_cm"] for x in d])), 1),
            "p95_dev_cm": round(float(np.mean([x["p95_dev_cm"] for x in d])), 1),
            "closest_cm_worst": round(min(x["closest_cm"] for x in d), 1),
            "stuck_pct": round(float(np.mean([x["stuck_pct"] for x in d])), 1),
        })
    cols = list(summ[0].keys())
    print("  ".join(f"{c:>17s}" for c in cols))
    for s in summ:
        print("  ".join(f"{str(s[c]):>17s}" for c in cols))
    if failed:
        print(f"\n({failed} drive(s) had no usable recording and are left out)")
    print("\nclean drive = no duct contact and never stuck. contacts_per_100m pools all drives in the group;")
    print("the deviation columns are the average over drives. With few tracks per group, treat small gaps as noise.")
    for name, data, keys in [("robustness_table.csv", summ, cols), ("robustness_drives.csv", drives, list(drives[0].keys()))]:
        with open(name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)

    # ---------------- figure: one panel per measure, one dot per drive
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("mean_dev_cm", "Average distance from centerline", "cm", "lower is better"),
              ("p95_dev_cm", "Worst-case drift (95th percentile)", "cm", "lower is better"),
              ("contacts_per_100m", "Duct contacts per 100 m", "contacts", "lower is better"),
              ("closest_cm", "Closest approach to a duct", "cm", "higher is better")]
    present = [g for g in GROUPS if any(x["group"] == g for x in drives)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), facecolor=SURF)
    rng = np.random.default_rng(0)
    for ax, (key, title, unit, hint) in zip(axes, panels):
        ax.set_facecolor(SURF)
        top = max(x[key] for x in drives)
        for i, g in enumerate(present):
            v = np.array([x[key] for x in drives if x["group"] == g])
            jit = (rng.random(len(v)) - 0.5) * 0.36 if len(v) > 1 else np.zeros(1)
            ax.scatter(i + jit, v, s=64, marker=MARK[g], color=COLOR[g], edgecolor=SURF, linewidth=1.5, zorder=3)
            m = v.mean()
            if key == "contacts_per_100m":            # pooled over the group's distance, as in the table
                gd = [x for x in drives if x["group"] == g]
                m = 100.0 * sum(x["contacts"] for x in gd) / sum(x["distance_m"] for x in gd)
            ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=2, zorder=4, solid_capstyle="round")
            ax.annotate(f"{m:.1f}" if key != "contacts_per_100m" else f"{m:.2f}", (i + 0.34, m),
                        va="center", ha="left", fontsize=10, color=INK)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{g}\n({sum(1 for x in drives if x['group'] == g)} drives)" for g in present],
                           fontsize=9.5, color=INK)
        ax.set_xlim(-0.6, len(present) - 0.25)
        ax.set_ylim(-0.04 * top if top > 0 else -0.04, top * 1.18 if top > 0 else 1)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=22, fontweight="bold")
        ax.text(0, 1.035, f"{unit}, {hint}", transform=ax.transAxes, fontsize=9.5, color=INK2)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
        ax.tick_params(axis="y", colors=INK2, labelsize=9, length=0)
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.text(0.008, 0.012, "Each mark is one test drive; the black bar is the group average.",
             fontsize=9.5, color=INK2)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig("robustness.png", dpi=160, facecolor=SURF)
    print("\nsaved robustness.png, robustness_table.csv, robustness_drives.csv")


if __name__ == "__main__":
    main()
