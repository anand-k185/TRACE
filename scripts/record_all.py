#!/usr/bin/env python3
"""
TRACE - record many maps unattended.

For every map and both directions it: starts the sim (no GUI), waits until the car has spawned,
starts the recorder, lets the weaving autopilot drive N laps, then shuts everything down and
moves on. A run that fails (sim didn't come up, no frames) is retried once, then skipped.

  cd ~/trace
  python3 record_all.py                         # default map list, 2 laps each way
  python3 record_all.py --maps lab_03 lab_04    # just these
  python3 record_all.py --dry-run               # print what it would do

Run it from a normal terminal where `ros2` works. Logs go to record_logs/.
"""
import argparse, glob, math, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAPS = ["lab_00", "lab_01", "lab_02", "lab_05", "lab_10",            # already used (now with weave)
                "lab_03", "lab_04", "lab_06", "lab_07", "lab_08",            # new
                "lab_09", "lab_11", "lab_12", "lab_13", "lab_14"]            # lab_15 stays the test map
KILL = ["gzserver", "gzclient", "robot_state_publisher", "dynamics_plant", "map_publisher",
        "sim_launch.py", "spawn_entity", "gz_collect.py", "autodrive.py"]


def pkill(sig, pattern):
    subprocess.run(["pkill", sig, "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cleanup():
    for k in KILL:
        pkill("-INT", k)
    time.sleep(3)
    for k in KILL:
        pkill("-9", k)
    time.sleep(2)


def spawn_yaws():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("f1tenth_gym_gazebo"), "maps", "generated_safe_maps_spawns.yaml")
    except Exception:
        p = os.path.expanduser("~/sim_ws/src/f1tenth_gym_gazebo/maps/generated_safe_maps_spawns.yaml")
    import yaml
    out = {}
    for line in open(p):
        if "{" in line and ":" in line:
            name, rest = line.split(":", 1)
            try:
                d = yaml.safe_load(rest)
                out[name.strip()] = (float(d["x"]), float(d["y"]), float(d["yaw"]))
            except Exception:
                pass
    return out


def wait_for(logfile, text, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if text in open(logfile, errors="ignore").read():
                return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    return False


def wait_for_car(a, logfile, x, y, yaw, spawn_log):
    """Wait until the car is in Gazebo. If the launch's own spawner gives up (it only waits 30 s for
    Gazebo, which is often too short on WSL), spawn the car ourselves with a long timeout."""
    t0, respawned = time.time(), False
    while time.time() - t0 < a.sim_timeout:
        try:
            txt = open(logfile, errors="ignore").read()
        except FileNotFoundError:
            txt = ""
        if "Gazebo write path OK" in txt:
            return True
        if not respawned and ("Spawn service failed" in txt or "spawn_entity.py-3]: process has died" in txt):
            respawned = True
            print("    launch spawner timed out; spawning the car directly ...", flush=True)
            subprocess.run(["ros2", "run", "gazebo_ros", "spawn_entity.py", "-entity", "f1tenth",
                            "-topic", "ego_robot_description", "-x", f"{x}", "-y", f"{y}", "-z", "0.0",
                            "-Y", f"{yaw:.4f}", "-timeout", "180"],
                           stdout=open(spawn_log, "w"), stderr=subprocess.STDOUT, timeout=240)
        time.sleep(2)
    return False


def stop(proc, wait=15):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=wait)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass


def one_run(a, mp, x, y, yaw, tag, seed):
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
            return "sim did not come up (car never spawned)"
        time.sleep(3)
        rec = subprocess.Popen([sys.executable, os.path.join(HERE, "gz_collect.py"), "--map", mp,
                                "--out", a.out, "--hz", str(a.hz)],
                               stdout=open(log("rec"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(4)
        drv = subprocess.Popen([sys.executable, os.path.join(HERE, "autodrive.py"), "--map", mp,
                                "--laps", str(a.laps), "--weave", str(a.weave), "--speed", str(a.speed),
                                "--seed", str(seed)],
                               stdout=open(log("drive"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            drv.wait(timeout=a.run_timeout)
        except subprocess.TimeoutExpired:
            return "autopilot did not finish in time"
        if "laps done" not in open(log("drive"), errors="ignore").read():
            return "autopilot stopped early (see drive log)"
    finally:
        for p in (drv, rec):
            if p is not None:
                stop(p)
        stop(sim, wait=20)
        cleanup()
    new = [e for e in set(glob.glob(os.path.join(a.out, mp, "ep_*"))) - before
           if os.path.exists(os.path.join(e, "meta.json"))]
    if not new:
        return "recorder saved no frames"
    n = sum(1 for _ in open(os.path.join(new[0], "frames.jsonl")))
    if n < 100:
        return f"only {n} frames recorded"
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    ap.add_argument("--out", default="data/")
    ap.add_argument("--laps", type=float, default=2)
    ap.add_argument("--weave", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--hz", type=float, default=5)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--directions", choices=["both", "fwd", "rev"], default="both")
    ap.add_argument("--sim-timeout", type=float, default=420, help="seconds to wait for the car to spawn")
    ap.add_argument("--run-timeout", type=float, default=2400, help="max seconds for one drive")
    ap.add_argument("--logs", default="record_logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
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
    print(f"{len(jobs)} runs: " + ", ".join(f"{j[0]}/{j[4]}" for j in jobs))
    if a.dry_run:
        return
    results, t_all = [], time.time()
    try:
        for k, (mp, x, y, yaw, tag) in enumerate(jobs):
            for attempt in (1, 2):
                t0 = time.time()
                print(f"[{k + 1}/{len(jobs)}] {mp} {tag} (attempt {attempt}) ...", flush=True)
                r = one_run(a, mp, x, y, yaw, tag, seed=k * 2 + attempt)
                if isinstance(r, int):
                    print(f"    ok: {r} frames in {(time.time() - t0) / 60:.1f} min", flush=True)
                    break
                print(f"    FAILED: {r}", flush=True)
            results.append((mp, tag, r))
    except KeyboardInterrupt:
        print("\ninterrupted, cleaning up")
        cleanup()
    print(f"\n==== summary ({(time.time() - t_all) / 60:.0f} min) ====")
    for mp, tag, r in results:
        print(f"  {mp:8s} {tag}: " + (f"{r} frames" if isinstance(r, int) else f"FAILED - {r}"))
    print(f"total frames: {sum(r for _, _, r in results if isinstance(r, int))}")


if __name__ == "__main__":
    main()
