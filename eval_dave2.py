import glob
import sys
import os

carla_root = r"C:/Users/Kangh/Downloads/newCARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist"

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                carla_root,
                "carla-*%d.%d-%s.egg" % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    "win-amd64"
                )
            )
        )[0]
    )
except IndexError:
    raise RuntimeError("CARLA egg not found. Check your path and Python version.")

import carla

import math
import argparse, time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import csv
from pathlib import Path
from Network_Baseline import Dave2Regression

# ---- EXACT match to Dataset_Baseline.py ----
def rgb_to_yuv(img_tensor: torch.Tensor) -> torch.Tensor:
    R, G, B = img_tensor[0], img_tensor[1], img_tensor[2]
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.14713*R - 0.28886*G + 0.436*B + 0.5
    V =  0.615*R - 0.51499*G - 0.10001*B + 0.5
    return torch.stack([Y, U.clamp(0,1), V.clamp(0,1)], dim=0)

to_tensor = transforms.ToTensor()
resize = transforms.Resize((66, 200), interpolation=transforms.InterpolationMode.BILINEAR)
center_crop = transforms.CenterCrop((160, 320))

def preprocess(rgb_np: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(rgb_np).convert("RGB")
    pil = center_crop(pil)   # matches Dataset_Baseline default
    pil = resize(pil)        # (66,200)
    x = to_tensor(pil)       # RGB [0,1], (3,66,200)
    x = rgb_to_yuv(x)        # YUV [0,1]
    x = x * 2.0 - 1.0        # [-1,1]
    return x.unsqueeze(0)    # (1,3,66,200)

class EventCounter:
    def __init__(self):
        self.collision = False
        self.lane_invasions = 0

def attach_collision_sensor(world, vehicle, counter):
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    def _on_collision(ev):
        counter.collision = True
    sensor.listen(_on_collision)
    return sensor

def attach_lane_invasion_sensor(world, vehicle, counter):
    bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    def _on_invasion(ev):
        counter.lane_invasions += 1
    sensor.listen(_on_invasion)
    return sensor


def speed_mps(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def run_episode(world, vehicle, camera, latest, model, device, fps,
                max_seconds=60.0, brake_thresh=0.9,
                stuck_speed=0.2, stuck_seconds=2.0,
                steer_gain=1.0, throttle_gain=1.0, brake_gain=1.0,
                follow_fn=None, debug=False):

    counter = EventCounter()
    col_s = attach_collision_sensor(world, vehicle, counter)
    lane_s = attach_lane_invasion_sensor(world, vehicle, counter)

    # warmup
    for _ in range(10):
        world.tick()
    

    max_steps = int(max_seconds * fps)
    stuck_steps_needed = int(stuck_seconds * fps)
    stuck_steps = 0

    total_dist = 0.0
    speeds = []

    prev_loc = vehicle.get_location()
    start_frame = world.get_snapshot().frame

    try:
        for t in range(max_steps):
            rgb = latest["rgb"]
            if rgb is None:
                world.tick()
                if follow_fn: follow_fn(vehicle)
                continue

            x = preprocess(rgb).to(device)

            with torch.no_grad():
                pred = model(x)[0]  # (3,) = [steer, thr, brk] (your convention)

            steer = float((pred[0] * steer_gain).clamp(-1.0, 1.0).item())
            thr   = float((pred[1] * throttle_gain).clamp(0.0, 1.0).item())

            # IMPORTANT: pred[2] is ALREADY sigmoided if your network does sigmoid inside forward()
            brk_prob = float((pred[2] * brake_gain).clamp(0.0, 1.0).item())

            if brk_prob > brake_thresh:
                brk = 1.0
                thr = 0.0
            else:
                brk = 0.0

            vehicle.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk))

            world.tick()
            #print("proceeding")
            if follow_fn: follow_fn(vehicle)

            # metrics update after physics step
            loc = vehicle.get_location()
            total_dist += dist(prev_loc, loc)
            prev_loc = loc

            spd = speed_mps(vehicle)
            speeds.append(spd)

            if spd < stuck_speed:
                stuck_steps += 1
            else:
                stuck_steps = 0

            if counter.collision:
                break
            if stuck_steps >= stuck_steps_needed:
                break

            if debug and (t % 50 == 0):
                print(f"[t={t}] dist={total_dist:.1f} speed={spd:.2f} coll={counter.collision} lane={counter.lane_invasions}")

        end_frame = world.get_snapshot().frame
        steps = max(1, end_frame - start_frame)
        duration_s = steps / float(fps)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        stuck = (stuck_steps >= stuck_steps_needed)

        return {
            "distance_m": total_dist,
            "duration_s": duration_s,
            "avg_speed_mps": avg_speed,
            "collision": int(counter.collision),
            "lane_invasions": int(counter.lane_invasions),
            "stuck": int(stuck),
        }

    finally:
        for s in [col_s, lane_s]:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass


def evaluate(model, device, client, fps, spawn_indices, repeats=3, out_csv="eval_results.csv"):
    world = client.get_world()

    # sync mode (important)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", str(1.0 / fps))

    results = []
    out_csv = Path(out_csv)

    try:
        for rep in range(repeats):
            for sp_idx in spawn_indices:
                # spawn
                spawn = world.get_map().get_spawn_points()[sp_idx]
                vehicle = world.try_spawn_actor(vehicle_bp, spawn)
                if vehicle is None:
                    # skip if blocked
                    results.append({"spawn_idx": sp_idx, "rep": rep, "spawn_failed": 1})
                    continue

                vehicle.set_autopilot(False)
                vehicle.set_simulate_physics(True)

                camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.7)), attach_to=vehicle)
                latest = {"rgb": None}

                def on_image(image):
                    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
                    latest["rgb"] = arr[:, :, :3][:, :, ::-1].copy()
                camera.listen(on_image)

                # optional spectator follow (can disable for eval speed)
                spectator = world.get_spectator()
                def follow(vehicle, distance=7.0, height=3.0, pitch=-15.0):
                    tf = vehicle.get_transform()
                    fwd = tf.get_forward_vector()
                    loc = tf.location - fwd * distance + carla.Location(z=height)
                    rot = carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw, roll=0.0)
                    spectator.set_transform(carla.Transform(loc, rot))

                try:
                    ep = run_episode(
                        world=world, vehicle=vehicle, camera=camera, latest=latest,
                        model=model, device=device, fps=fps,
                        max_seconds=60.0,
                        brake_thresh=0.9,
                        follow_fn=follow,   # set to follow if you want visuals
                        debug=False
                    )
                    print("episode is done")
                    ep.update({"spawn_idx": sp_idx, "rep": rep, "spawn_failed": 0})
                    results.append(ep)

                finally:
                    try:
                        camera.stop(); camera.destroy()
                    except: pass
                    try:
                        vehicle.destroy()
                    except: pass

        # write CSV
        keys = sorted({k for r in results for k in r.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)

        print(f"[DONE] wrote {len(results)} rows to {out_csv}")

    finally:
        world.apply_settings(original)

def parse_spawns(s: str):
    # supports: "0,10,20" or "0 10 20"
    s = s.replace(",", " ").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--spawns", type=str, required=True,
                    help='Comma-separated spawn indices, e.g. "0,10,20"')
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out_csv", type=str, default="eval_results.csv")
    args = ap.parse_args()

    print("=== DAVE2 EVAL START ===", flush=True)
    print("model:", args.model_path, flush=True)
    print("fps:", args.fps, "repeats:", args.repeats, "out:", args.out_csv, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    model = Dave2Regression(out_dim=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    spawn_indices = parse_spawns(args.spawns)
    print("spawns:", spawn_indices, flush=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    evaluate(
        model=model,
        device=device,
        client=client,
        fps=args.fps,
        spawn_indices=spawn_indices,
        repeats=args.repeats,
        out_csv=args.out_csv
    )

if __name__ == "__main__":
    main()

