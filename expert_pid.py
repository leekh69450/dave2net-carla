import math
import carla

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _vec2(x, y):
    return (float(x), float(y))

def _norm2(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1]) + 1e-8

def _dot2(a, b):
    return a[0]*b[0] + a[1]*b[1]

def _cross2(a, b):
    # z-component of 2D cross product (a.x*b.y - a.y*b.x)
    return a[0]*b[1] - a[1]*b[0]


class PID:
    def __init__(self, kp, ki, kd, i_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = abs(i_limit)
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt):
        if dt <= 0:
            dt = 1e-3
        self.integral += error * dt
        self.integral = _clamp(self.integral, -self.i_limit, self.i_limit)

        if self.prev_error is None:
            deriv = 0.0
        else:
            deriv = (error - self.prev_error) / dt

        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv


class PIDExpertPolicy:
    """
    Queryable expert: given current vehicle state, outputs VehicleControl.
    - Lateral control: steer PID on signed heading-to-waypoint angle
    - Longitudinal control: speed PID to target_speed
    """
    def __init__(
        self,
        target_speed_mps=8.0,     # ~28.8 km/h (safe for Town01)
        lookahead_m=6.0,          # waypoint lookahead distance
        steer_pid=(1.2, 0.00, 0.20),
        speed_pid=(0.6, 0.05, 0.00),
        max_throttle=0.6,
        max_brake=0.6,
    ):
        self.target_speed_mps = float(target_speed_mps)
        self.lookahead_m = float(lookahead_m)
        self.steer = PID(*steer_pid, i_limit=1.0)
        self.speed = PID(*speed_pid, i_limit=5.0)
        self.max_throttle = max_throttle
        self.max_brake = max_brake

    def reset(self):
        self.steer.reset()
        self.speed.reset()

    def __call__(self, vehicle, world, dt):
        # --- current kinematics ---
        v = vehicle.get_velocity()
        speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)  # m/s

        tf = vehicle.get_transform()
        loc = tf.location
        yaw = math.radians(tf.rotation.yaw)
        fwd = _vec2(math.cos(yaw), math.sin(yaw))

        # --- get a target waypoint ahead on the road ---
        m = world.get_map()
        wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            # Fallback: stop gently if no waypoint
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.4)

        # dynamic lookahead: slightly shorter at low speed, longer at high speed
        la = _clamp(self.lookahead_m + 0.5 * speed, 4.0, 12.0)
        nxt = wp.next(la)
        target_wp = nxt[0] if nxt else wp

        tgt = target_wp.transform.location
        to_tgt = _vec2(tgt.x - loc.x, tgt.y - loc.y)

        # --- lateral error: signed angle from forward vector to target direction ---
        to_tgt_n = (to_tgt[0] / _norm2(to_tgt), to_tgt[1] / _norm2(to_tgt))
        dot = _clamp(_dot2(fwd, to_tgt_n), -1.0, 1.0)
        ang = math.acos(dot)  # [0, pi]
        sign = 1.0 if _cross2(fwd, to_tgt_n) > 0 else -1.0
        heading_error = sign * ang  # signed radians

        # Convert to a nice scale for steering PID
        # (radians are small; PID gains above assume radians)
        steer_cmd = self.steer.step(heading_error, dt)
        steer_cmd = _clamp(steer_cmd, -1.0, 1.0)

        # --- longitudinal: target speed PID ---
        speed_error = self.target_speed_mps - speed
        accel_cmd = self.speed.step(speed_error, dt)

        # Map accel_cmd to throttle/brake (simple split)
        if accel_cmd >= 0:
            throttle = _clamp(accel_cmd, 0.0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(-accel_cmd, 0.0, self.max_brake)

        return carla.VehicleControl(throttle=throttle, steer=steer_cmd, brake=brake)
