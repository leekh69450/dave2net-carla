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


import argparse, time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

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


def main():
    print("=== Starting DAVE2 inference ===", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--throttle_gain", type=float, default=1.0)
    ap.add_argument("--steer_gain", type=float, default=1.0)
    ap.add_argument("--brake_gain", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = Dave2Regression(out_dim=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Sync mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    rng = np.random.default_rng()

    vehicle = None
    for idx in rng.permutation(len(spawn_points))[:30]:  # try up to 30 different spawns
        spawn = spawn_points[idx]
        vehicle = world.try_spawn_actor(vehicle_bp, spawn)
        if vehicle is not None:
            print(f"Vehicle spawned at spawn_points[{idx}]", flush=True)
            break

    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle after trying many spawn points. Try restarting CARLA or clearing actors.")

    print("Vehicle spawned", flush=True)
    spectator = world.get_spectator()

    def follow(vehicle, distance=7.0, height=3.0, pitch=-15.0):
        tf = vehicle.get_transform()
        forward = tf.get_forward_vector()
        loc = tf.location - forward * distance + carla.Location(z=height)
        rot = carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw, roll=0.0)
        spectator.set_transform(carla.Transform(loc, rot))

    vehicle.set_autopilot(False)
    vehicle.set_simulate_physics(True)
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))



    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", str(1.0 / args.fps))

    cam_tf = carla.Transform(carla.Location(x=1.5, z=1.7))  # common; adjust if your collector used different
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    latest = {"rgb": None}

    def on_image(image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA->RGB
        latest["rgb"] = rgb

    camera.listen(on_image)

    try:
        # Warmup ticks
        for _ in range(10):
            world.tick()
        print("warm up tick")
        
        #vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0, brake=0.0))
        for t in range(args.max_steps):
            # use last image to compute control
            
            rgb = latest["rgb"]
            if rgb is None:
                world.tick()
                follow(vehicle)
                continue

            x = preprocess(rgb).to(device)

            with torch.no_grad():
                pred = model(x)[0]  # (3,): [steer, thr, brk]
            if t % 50 == 0:
                print("raw pred:", pred.detach().cpu().numpy())


            steer = float((pred[0] * args.steer_gain).clamp(-1.0, 1.0).item())
            thr = float((pred[1] * args.throttle_gain).clamp(0.0, 1.0).item())
            #brk = float((pred[2] * args.brake_gain).clamp(0.0, 1.0).item())
            #brk = 0.0
            brk_prob = float(pred[2].item())  # safer than raw pred[2]

            if brk_prob > 0.9:
                brk = 1.0
                thr = 0.0
            else:
                brk = 0.0
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=thr,
                    steer=steer,
                    brake=brk,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                )
            )
            if t % 50 == 0:
                c = vehicle.get_control()
                print(
                    f"[t={t}] pred thr={thr:.3f} steer={steer:.3f} brk={brk:.3f} | "
                    f"actual thr={c.throttle:.3f} steer={c.steer:.3f} brk={c.brake:.3f}",
                    flush=True
                )
            world.tick()
            follow(vehicle)

            if t % 50 == 0:
                v = vehicle.get_velocity()
                speed = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5
                loc = vehicle.get_location()
                print(f"[t={t}] speed={speed:.3f} loc=({loc.x:.1f},{loc.y:.1f})", flush=True)


    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up.")

if __name__ == "__main__":
    main()
