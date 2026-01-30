#!/usr/bin/env python3
"""
Minimal CARLA Autopilot Expert Data Collector (DAVE-2 style)

Records ONLY:
  - Front RGB camera image
  - Corresponding expert control (steer, throttle, brake) from Traffic Manager autopilot

Output directory:
  <out_dir>/
    images/frame_000000.jpg
    images/frame_000001.jpg
    ...
    labels.csv   (idx, frame, timestamp, steer, throttle, brake)

Usage example:
  python autopilot_collect_dave2_min.py --town Town05 --steps 6000 --fps 20 --out_dir dataset_dave2_tm

Notes:
  - Requires CARLA server running.
  - Uses synchronous mode + synchronous Traffic Manager for clean frame alignment.
"""
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
import argparse
import csv
import math
import time
import threading
from pathlib import Path

import numpy as np
from PIL import Image

import carla


class FrameBuffer:
    """Thread-safe storage for the latest camera frame (RGB uint8)."""
    def __init__(self):
        self._lock = threading.Lock()
        self.frame = -1
        self.rgb = None
        self.timestamp = None

    def update(self, image: carla.Image):
        # CARLA provides BGRA in raw_data. Convert to RGB numpy.
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3]
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=8000)

    p.add_argument("--town", default=None, help="Optional: load a town (e.g., Town05).")
    p.add_argument("--out_dir", default="dataset_dave2_tm")
    p.add_argument("--steps", type=int, default=6000, help="How many ticks to record.")
    p.add_argument("--fps", type=int, default=10, help="Synchronous FPS.")
    p.add_argument("--save_every", type=int, default=1, help="Save every N frames (1 = save all).")

    p.add_argument("--img_w", type=int, default=320)
    p.add_argument("--img_h", type=int, default=240)
    p.add_argument("--fov", type=float, default=90.0)

    # Traffic Manager expert behavior knobs (optional)
    p.add_argument("--speed_diff", type=float, default=0.0,
                   help="TM percentage speed difference. +10 slower, -10 faster.")
    p.add_argument("--min_dist", type=float, default=2.5, help="TM following distance (m).")
    p.add_argument("--disable_lane_change", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spawn_idx", type=int, default=None)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.get_world()

    if args.town:
        # Only reload if different
        if args.town not in world.get_map().name:
            world = client.load_world(args.town)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on this map.")

    # Save original settings to restore on exit
    original_settings = world.get_settings()

    # --- Sync mode (world + traffic manager) ---
    dt = 1.0 / float(args.fps)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm.global_percentage_speed_difference(float(args.speed_diff))

    # --- Spawn ego vehicle ---
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

    if args.spawn_idx is None:
        spawn_point = spawn_points[args.seed % len(spawn_points)]
    else:
        spawn_point = spawn_points[int(args.spawn_idx) % len(spawn_points)]

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle. Try a different --spawn_idx.")
    spectator = world.get_spectator()
    # --- Attach RGB camera ---
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.img_w))
    cam_bp.set_attribute("image_size_y", str(args.img_h))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", str(dt))  # align with world tick

    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    fb = FrameBuffer()
    camera.listen(fb.update)

    # --- Enable TM autopilot (expert) ---
    vehicle.set_autopilot(True, args.tm_port)
    tm.distance_to_leading_vehicle(vehicle, float(args.min_dist))
    if args.disable_lane_change:
        tm.auto_lane_change(vehicle, False)

    # --- CSV logger (ONLY actions) ---
    csv_path = out_dir / "labels.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["idx", "frame", "timestamp", "steer", "throttle", "brake"])

    actors = [camera, vehicle]

    print(f"[INFO] Writing to: {out_dir}")
    print(f"[INFO] steps={args.steps}, fps={args.fps}, save_every={args.save_every}")
    print(f"[INFO] images -> {img_dir}")
    print(f"[INFO] labels -> {csv_path}")

    try:
        # Warm up sensor stream
        for _ in range(10):
            world.tick()
        print("warm up done")

        saved = 0
        for i in range(args.steps):
            frame = world.tick()
            # ALWAYS follow immediately after tick (even if RGB missing)
            transform = vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(x=1.5, z=1.3),
                    carla.Rotation(pitch=0.0, yaw=transform.rotation.yaw, roll=0.0)
                )
            )
            

            # Wait up to 1s for the camera callback with matching frame
            rgb, ts = None, None
            t0 = time.time()
            while time.time() - t0 < 1.0:
                rgb, ts = fb.get_if_frame(frame)
                if rgb is not None:
                    break
                time.sleep(0.001)

            if rgb is None:
                print(f"[WARN] Missed camera frame alignment at world frame={frame}")
                continue

            transform = vehicle.get_transform()

            spectator.set_transform(
                carla.Transform(
                    transform.location
                    + carla.Location(
                        x=1.5,   # forward into hood / windshield
                        z=1.3    # driver eye height
                    ),
                    carla.Rotation(
                        pitch=0.0,
                        yaw=transform.rotation.yaw,
                        roll=0.0
                    )
                )
            )

            if (i % args.save_every) != 0:
                continue

            # Expert label: control actually applied at this tick
            ctrl = vehicle.get_control()

            img_path = img_dir / f"frame_{saved:06d}.jpg"
            save_jpg(rgb, img_path)

            writer.writerow([
                saved, frame, ts,
                float(ctrl.steer), float(ctrl.throttle), float(ctrl.brake)
            ])

            saved += 1
            if saved % 200 == 0:
                print(f"[INFO] Saved {saved} samples (world frame={frame})")

        print(f"[DONE] Saved total {saved} samples.")

    finally:
        csv_f.close()

        # Clean up actors
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass

        # Restore settings
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        world.apply_settings(original_settings)
        print("[INFO] Restored world settings and destroyed actors.")


if __name__ == "__main__":
    main()
