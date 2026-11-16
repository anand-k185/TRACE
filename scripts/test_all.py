#!/usr/bin/env python3
"""
TRACE - test the trained model on several tracks, unattended.

For every map and direction: start the sim (no GUI), wait for the car, record the drive, let the
model drive for --duration simulated seconds, shut down, then score the drive with eval_drive.py.
Prints one results table at the end and saves it to test_results.json / test_results.csv.

  cd ~/trace
  python3 test_all.py --ckpt trace_vit_bev.pt                       # lab_15 .. lab_20, both directions
  python3 test_all.py --ckpt trace_vit_bev.pt --maps lab_16 lab_17  # just these
  python3 test_all.py --ckpt trace_vit_bev.pt --dry-run

Needs record_all.py, gz_collect.py, model_drive.py, trace_bev.py, eval_drive.py, autodrive.py here.
"""
import argparse, csv, glob, json, math, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from record_all import cleanup, spawn_yaws, wait_for_car, stop

UNSEEN = ["lab_15", "lab_16", "lab_17", "lab_18", "lab_19", "lab_20"]     # never used for training


def one_test(a, mp, x, y, yaw, tag):
    log = lambda n: os.path.join(a.logs, f"{mp}_{tag}_{n}.log")
    cleanup()
    before = set(glob.glob(os.path.join(a.out, mp, "ep_*")))
    launch = (f"ros2 launch f1tenth_gym_gazebo sim_launch.py map_name:={mp} yaw:={yaw:.4f} "
              f"enable_lidar3d:=false gui:=false rviz:=false "
              f"camera_image_width:={a.width} camera_image_height:={a.height}")
    sim = subprocess.Popen(launch, shell=True, stdout=open(log("sim"), "w"), stderr=subprocess.STDOUT,
                           start_new_session=True)
    rec = drv = None
    try:
        if not wait_for_car(a, log("sim"), x, y, yaw, log("spawn")):
            return {"error": "sim did not come up"}
        time.sleep(3)
        rec = subprocess.Popen([sys.executable, os.path.join(HERE, "gz_collect.py"), "--map", mp,
                                "--out", a.out, "--hz", str(a.hz)],
                               stdout=open(log("rec"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(4)
        cmd = [sys.executable, os.path.join(HERE, "model_drive.py"), "--ckpt", a.ckpt,
               "--speed", str(a.speed), "--duration", str(a.duration)]
        if a.lookahead != 1.4:                     # only pass options that differ from model_drive's defaults,
            cmd += ["--lookahead", str(a.lookahead)]   # so an older model_drive.py still works
        if a.steer_gain != 1.0:
            cmd += ["--steer-gain", str(a.steer_gain)]
        if a.smooth != 0.0:
            cmd += ["--smooth", str(a.smooth)]
        t_drive = time.time()
        drv = subprocess.Popen(cmd, stdout=open(log("drive"), "w"), stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            drv.wait(timeout=a.run_timeout)
        except subprocess.TimeoutExpired:
            pass                                   # still score whatever was recorded
        if time.time() - t_drive < 30:             # the model should drive for minutes, not seconds
            tail = open(log("drive"), errors="ignore").read().strip().splitlines()[-3:]
            return {"error": "model_drive.py stopped right away: " + " | ".join(tail)}
    finally:
        for p in (drv, rec):
            if p is not None:
                stop(p)
        stop(sim, wait=20)
        cleanup()
    new = sorted(e for e in set(glob.glob(os.path.join(a.out, mp, "ep_*"))) - before
                 if os.path.exists(os.path.join(e, "meta.json")))
    if not new:
        return {"error": "no frames recorded (see drive log)"}
    ep = new[-1]
    r = subprocess.run([sys.executable, os.path.join(HERE, "eval_drive.py"), "--episode", ep, "--map", mp],
                       capture_output=True, text=True)
    try:
        rep = json.loads(r.stdout[r.stdout.index("{"): r.stdout.index("}") + 1])
    except Exception:
        return {"error": "scoring failed: " + (r.stderr.strip().splitlines() or ["?"])[-1], "episode": ep}
    rep["episode"] = ep
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maps", nargs="+", default=UNSEEN)
    ap.add_argument("--directions", choices=["both", "fwd", "rev"], default="both")
    ap.add_argument("--duration", type=float, default=300, help="simulated seconds per drive")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--lookahead", type=float, default=1.4)
    ap.add_argument("--steer-gain", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--out", default="test_drives/")
    ap.add_argument("--hz", type=float, default=5)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--sim-timeout", type=float, default=420)
    ap.add_argument("--run-timeout", type=float, default=3600, help="max wall-clock seconds for one drive")
    ap.add_argument("--logs", default="test_logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.ckpt):
        sys.exit(f"checkpoint not found: {a.ckpt}")
    os.makedirs(a.logs, exist_ok=True)
    yaws = spawn_yaws()
    jobs = []
    for mp in a.maps:
        if mp not in yaws:
            print(f"!! {mp}: no spawn entry, skipping")
            continue
        x, y, yaw = yaws[mp]
        if a.directions in ("both", "fwd"):
            jobs.append((mp, x, y, yaw, "fwd"))
        if a.directions in ("both", "rev"):
            jobs.append((mp, x, y, yaw + math.pi, "rev"))
    print(f"{len(jobs)} test drives: " + ", ".join(f"{j[0]}/{j[4]}" for j in jobs))
    if a.dry_run:
        return
    rows = []
    try:
        for k, (mp, x, y, yaw, tag) in enumerate(jobs):
            t0 = time.time()
            print(f"[{k + 1}/{len(jobs)}] {mp} {tag} ...", flush=True)
            rep = one_test(a, mp, x, y, yaw, tag)
            rep.update(map=mp, direction=tag)
            rows.append(rep)
            if "error" in rep:
                print(f"    FAILED: {rep['error']}", flush=True)
            else:
                print(f"    {rep['laps']} laps, {rep['duct_contacts']} contacts, mean dev "
                      f"{rep['centerline_dev_mean_m']} m, stuck {rep['stuck_s']} s  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            json.dump(rows, open("test_results.json", "w"), indent=1)
    except KeyboardInterrupt:
        print("\ninterrupted, cleaning up")
        cleanup()

    cols = ["map", "direction", "laps", "distance_m", "duct_contacts", "closest_approach_m",
            "centerline_dev_mean_m", "centerline_dev_p95_m", "stuck_s"]
    with open("test_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["error"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n" + "  ".join(f"{c[:14]:>14s}" for c in cols))
    for r in rows:
        if "error" in r:
            print(f"{r['map']:>14s}  {r['direction']:>14s}  FAILED - {r['error']}")
        else:
            print("  ".join(f"{str(r.get(c, '')):>14s}" for c in cols))
    ok = [r for r in rows if "error" not in r]
    if ok:
        clean = sum(1 for r in ok if r["duct_contacts"] == 0 and r["stuck_s"] == 0)
        print(f"\n{clean}/{len(ok)} drives with no duct contact and never stuck. "
              f"Saved test_results.csv and test_results.json; each episode folder has drive_eval.png.")


if __name__ == "__main__":
    main()
