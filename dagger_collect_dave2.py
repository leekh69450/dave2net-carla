#!/usr/bin/env python3
import glob, sys, os, time, csv, argparse, threading
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

# --- CARLA egg path ---
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
import random

from Network_Baseline import Dave2Regression  # your model
from expert_pid import PIDExpertPolicy

# ---- same preprocessing as Dataset_Baseline ----
def rgb_to_yuv(img_tensor: torch.Tensor) -> torch.Tensor:
    R, G, B = img_tensor[0], img_tensor[1], img_tensor[2]
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.14713*R - 0.28886*G + 0.436*B + 0.5
    V =  0.615*R - 0.51499*G - 0.10001*B + 0.5
    return torch.stack([Y, U.clamp(0,1), V.clamp(0,1)], dim=0)

class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self.frame = -1
        self.rgb = None
        self.timestamp = None

    def update(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3]
        rgb = bgr[:, :, ::-1]
        with self._lock:
            self.frame = int(image.frame)
            self.rgb = rgb
            self.timestamp = float(image.timestamp)

    def get_if_frame(self, target_frame: int):
        with self._lock:
            if self.frame == target_frame and self.rgb is not None:
                return self.rgb.copy(), self.timestamp
        return None, None

def save_jpg(rgb: np.ndarray, path: Path, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(str(path), quality=quality, subsampling=0)

def preprocess(rgb_np: np.ndarray):
    # rgb_np: HxWx3 uint8 240x320
    pil = Image.fromarray(rgb_np).convert("RGB")
    pil = transforms.CenterCrop((160, 320))(pil)
    pil = transforms.Resize((66, 200), interpolation=transforms.InterpolationMode.BILINEAR)(pil)
    x = transforms.ToTensor()(pil)        # [0,1]
    x = rgb_to_yuv(x)                     # [0,1]
    x = x * 2.0 - 1.0                     # [-1,1]
    return x.unsqueeze(0)                 # (1,3,66,200)

def spawn_episode(world, blueprint_library, vehicle_bp, cam_bp, spawn_points, seed, spawn_idx=None):
    # pick spawn
    if spawn_idx is None:
        random.seed(seed)
        sp = random.choice(spawn_points)
    else:
        sp = spawn_points[int(spawn_idx) % len(spawn_points)]
    sp.location.z += 0.3  # avoid ground collision

    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle. Try a different spawn / seed.")

    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    fb = FrameBuffer()
    camera.listen(fb.update)

    vehicle.set_autopilot(False)
    return vehicle, camera, fb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=8000)
    p.add_argument("--town", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1)

    # β-DAGGER mixing: probability of using expert control to keep car safe
    p.add_argument("--beta", type=float, default=0.2)  # 0.2 = 20% expert takeover
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spawn_idx", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dave2Regression(out_dim=3).to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    if args.town and args.town not in world.get_map().name:
        world = client.load_world(args.town)

    original_settings = world.get_settings()

    dt = 1.0 / float(args.fps)
    expert_policy = PIDExpertPolicy(
        target_speed_mps=8.0,
        lookahead_m=6.0
    )
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if args.spawn_idx is None:
        spawn_point = spawn_points[args.seed % len(spawn_points)]
    else:
        spawn_point = spawn_points[int(args.spawn_idx) % len(spawn_points)]

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle. Try a different --spawn_idx.")
    spectator = world.get_spectator()

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    cam_bp.set_attribute("fov", "90")
    #cam_bp.set_attribute("sensor_tick", str(dt))
    cam_bp.set_attribute("sensor_tick", "0.0")

    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    fb = FrameBuffer()
    camera.listen(fb.update)

    # Ego vehicle will always be controlled via apply_control()
    # Expert labels come from PIDExpertPolicy (queryable)
    vehicle.set_autopilot(False)


    csv_path = out_dir / "labels.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["idx", "frame", "timestamp", "throttle", "steer", "brake"])

    actors = [camera, vehicle]
    print(f"[INFO] DAGGER collect -> {out_dir}")
    print(f"[INFO] model={args.model_path} steps={args.steps} fps={args.fps} beta={args.beta}")

    saved = 0
    episode = 0
    actors = []  # will hold current episode actors only

    try:
        # Warm up world once
        for _ in range(20):
            world.tick()
        print("Warmup ticks done", flush=True)

        while saved < args.steps:
            episode += 1
            print(f"[EP] starting episode {episode} (saved={saved}/{args.steps})")

            # Cleanup previous episode actors if any
            for a in actors:
                try: a.destroy()
                except Exception: pass
            actors = []

            # Spawn new episode (random spawn if spawn_idx is None)
            vehicle, camera, fb = spawn_episode(
                world, blueprint_library, vehicle_bp, cam_bp, spawn_points,
                seed=args.seed + episode,   # change seed per episode to vary spawns
                spawn_idx=args.spawn_idx
            )
            actors = [camera, vehicle]

            spectator = world.get_spectator()

            # reset expert
            expert_policy.reset()

            # kick-start
            vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.0, brake=0.0, hand_brake=False))
            for _ in range(10):
                world.tick()

            stuck_count = 0

            # run this episode up to some cap (so it doesn't run forever)
            for _ in range(2000):

                frame = world.tick()

                rgb, ts = None, None
                t0 = time.time()
                while time.time() - t0 < 0.12:
                    rgb, ts = fb.get_if_frame(frame)
                    if rgb is not None:
                        break
                    time.sleep(0.001)

                if rgb is None:
                    continue

                # expert label
                expert = expert_policy(vehicle, world, dt)

                # learner action
                with torch.no_grad():
                    x = preprocess(rgb).to(device)
                    pred = model(x)[0].detach().cpu().numpy()

                thr = float(np.clip(pred[0], 0.0, 1.0))
                steer = float(np.clip(pred[1], -1.0, 1.0))
                brk = float(np.clip(pred[2], 0.0, 1.0))
                learner_ctrl = carla.VehicleControl(throttle=thr, steer=steer, brake=brk)

                # mix
                use_expert = (np.random.rand() < args.beta)
                applied_ctrl = expert if use_expert else learner_ctrl
                applied_ctrl.hand_brake = False
                applied_ctrl.reverse = False
                vehicle.apply_control(applied_ctrl)

                # spectator (throttle it!)
                if frame % 5 == 0:
                    transform = vehicle.get_transform()
                    forward = transform.get_forward_vector()
                    spectator.set_transform(
                        carla.Transform(
                            transform.location - forward * 6.0 + carla.Location(z=3.0),
                            carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0),
                        )
                    )

                # save
                if (saved % args.save_every) == 0:
                    img_path = img_dir / f"frame_{saved:06d}.jpg"
                    save_jpg(rgb, img_path)
                    writer.writerow([saved, frame, ts,
                                    float(expert.throttle), float(expert.steer), float(expert.brake)])

                saved += 1
                if saved % 200 == 0:
                    print(f"[INFO] Saved {saved}/{args.steps}")

                if saved >= args.steps:
                    break

                # stuck detection
                v = vehicle.get_velocity()
                speed = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5
                if speed < 0.2:
                    stuck_count += 1
                else:
                    stuck_count = 0

                if stuck_count >= 30:
                    print(f"[RESET] Stuck for {stuck_count} ticks at frame={frame}. Respawning...")
                    break

                # debug print rarely
                if frame % 50 == 0:
                    print(f"[DAGGER] beta={args.beta:.2f} | "
                        f"source={'EXPERT' if use_expert else 'LEARNER'} | "
                        f"EXP(thr={expert.throttle:.2f}, st={expert.steer:+.2f}, br={expert.brake:.2f}) | "
                        f"LEARN(thr={learner_ctrl.throttle:.2f}, st={learner_ctrl.steer:+.2f}, br={learner_ctrl.brake:.2f})")

        print(f"[DONE] Saved total {saved}")

    finally:
        csv_f.close()
        for a in actors:
            try: a.destroy()
            except Exception: pass
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        world.apply_settings(original_settings)
        print("[INFO] Cleaned up and restored world settings.")


if __name__ == "__main__":
    main()
