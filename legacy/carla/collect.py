"""
TRACE - CARLA data collector.

Drives an ego car with CARLA's BehaviorAgent (the "expert"), and records at a fixed rate:
  - front RGB camera image
  - ego pose (world), speed, control
  - camera world->camera matrix at the moment of capture
  - the route planner's high-level command (LANEFOLLOW, LEFT, RIGHT, STRAIGHT,
    CHANGELANELEFT, CHANGELANERIGHT)  <- needed later for the "take the exit" step
  - whether steering noise was being injected (for recovery data)

Labels (future waypoints in pixels) are NOT made here; run make_labels.py afterwards.

Usage (CARLA server already running, e.g. ./CarlaUE4.sh -RenderOffScreen):
  python collect.py --carla-root ~/CARLA_0.9.16 --town Town04 --episodes 20 --frames 2000 --out data/
"""
import argparse, json, math, os, random, sys, queue, time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--carla-root", required=True, help="CARLA install dir (for PythonAPI/carla/agents)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--town", default="Town04")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--frames", type=int, default=2000, help="frames per episode")
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=450)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--cam-pitch", type=float, default=-10.0, help="degrees, negative = tilt down")
    p.add_argument("--n-traffic", type=int, default=40)
    p.add_argument("--noise-prob", type=float, default=0.01, help="chance per frame to start a noise burst")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weather", default=None, help="fix one weather preset, e.g. ClearNoon (default: random)")
    p.add_argument("--out", default="data")
    return p.parse_args()


WEATHERS = [
    "ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset",
    "CloudySunset", "WetSunset", "SoftRainSunset", "ClearNight", "WetCloudyNight",
]


def main():
    a = parse_args()
    sys.path.append(os.path.join(a.carla_root, "PythonAPI", "carla"))
    import carla
    from agents.navigation.behavior_agent import BehaviorAgent

    random.seed(a.seed)
    client = carla.Client(a.host, a.port)
    client.set_timeout(30.0)
    world = client.load_world(a.town)

    # Synchronous mode: the sim only advances when we tick -> every image lines up with a pose.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / a.hz
    world.apply_settings(settings)
    tm = client.get_trafficmanager(a.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)

    bp_lib = world.get_blueprint_library()
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    for ep in range(a.episodes):
        actors = []
        ep_dir = os.path.join(a.out, a.town, f"ep_{ep:04d}")
        os.makedirs(os.path.join(ep_dir, "rgb"), exist_ok=True)
        try:
            # ---- weather ----
            wname = a.weather or random.choice(WEATHERS)
            world.set_weather(getattr(carla.WeatherParameters, wname))

            # ---- ego vehicle ----
            ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
            ego_bp.set_attribute("role_name", "hero")
            random.shuffle(spawn_points)
            ego = None
            for sp in spawn_points:
                ego = world.try_spawn_actor(ego_bp, sp)
                if ego:
                    break
            actors.append(ego)

            # ---- front camera ----
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(a.width))
            cam_bp.set_attribute("image_size_y", str(a.height))
            cam_bp.set_attribute("fov", str(a.fov))
            # roof-ish, tilted down so the nearest waypoint (~5 m ahead) stays in frame
            cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=a.cam_pitch))
            cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
            actors.append(cam)
            img_q = queue.Queue()
            cam.listen(img_q.put)

            # ---- collision sensor (end episode on crash) ----
            col_bp = bp_lib.find("sensor.other.collision")
            col = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
            actors.append(col)
            crashed = {"v": False}
            col.listen(lambda e: crashed.__setitem__("v", True))

            # ---- background traffic ----
            veh_bps = bp_lib.filter("vehicle.*")
            for sp in spawn_points[1 : a.n_traffic + 1]:
                v = world.try_spawn_actor(random.choice(veh_bps), sp)
                if v:
                    v.set_autopilot(True, a.tm_port)
                    actors.append(v)

            # ---- expert driver ----
            agent = BehaviorAgent(ego, behavior="normal")
            agent.set_destination(random.choice(spawn_points).location)

            # meta: intrinsics + camera mount, saved once per episode
            f = a.width / (2.0 * math.tan(math.radians(a.fov) / 2.0))
            meta = {
                "town": a.town, "weather": wname, "hz": a.hz,
                "width": a.width, "height": a.height, "fov": a.fov,
                "K": [[f, 0, a.width / 2.0], [0, f, a.height / 2.0], [0, 0, 1]],
                "cam_mount": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": a.cam_pitch},
            }
            with open(os.path.join(ep_dir, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2)

            # let the world settle
            for _ in range(int(a.hz)):
                world.tick()
            while not img_q.empty():
                img_q.get()

            noise_left, noise_steer = 0, 0.0
            log = open(os.path.join(ep_dir, "frames.jsonl"), "w")
            for i in range(a.frames):
                if agent.done():
                    agent.set_destination(random.choice(spawn_points).location)

                control = agent.run_step()
                road_option = agent.get_local_planner().target_road_option
                road_option = road_option.name if road_option is not None else "VOID"

                # steering-noise bursts -> the car drifts, then the expert recovers
                if noise_left == 0 and random.random() < a.noise_prob:
                    noise_left = random.randint(int(0.5 * a.hz), int(1.5 * a.hz))
                    noise_steer = random.uniform(-0.25, 0.25)
                noisy = noise_left > 0
                if noisy:
                    control.steer = max(-1.0, min(1.0, control.steer + noise_steer))
                    noise_left -= 1
                ego.apply_control(control)

                frame_id = world.tick()
                img = img_q.get(timeout=5.0)
                while img.frame < frame_id:  # make sure image matches this tick
                    img = img_q.get(timeout=5.0)

                tf = ego.get_transform()
                vel = ego.get_velocity()
                img.save_to_disk(os.path.join(ep_dir, "rgb", f"{i:06d}.jpg"))
                rec = {
                    "i": i, "frame": frame_id, "t": i / a.hz,
                    "ego": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                            "yaw": tf.rotation.yaw, "pitch": tf.rotation.pitch, "roll": tf.rotation.roll},
                    "ego_world2veh": [list(r) for r in tf.get_inverse_matrix()],
                    "cam_world2cam": [list(r) for r in img.transform.get_inverse_matrix()],
                    "speed": math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2),
                    "control": {"steer": control.steer, "throttle": control.throttle, "brake": control.brake},
                    "command": road_option,
                    "noise": noisy,
                    "crashed": crashed["v"],
                }
                log.write(json.dumps(rec) + "\n")
                if crashed["v"]:
                    print(f"ep {ep}: collision at frame {i}, ending episode")
                    break
            log.close()
            print(f"ep {ep} done ({wname}) -> {ep_dir}")
        finally:
            for act in actors:
                if act is not None and act.is_alive:
                    if hasattr(act, "stop"):
                        try:
                            act.stop()
                        except Exception:
                            pass
                    act.destroy()
            world.tick()

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
