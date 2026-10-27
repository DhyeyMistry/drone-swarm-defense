"""Tello EDU firmware and rotor model with ese651-style domain randomization.

A real Tello only takes ``rc a b c d``: velocity set-points in its heading frame plus a
yaw rate. The firmware turns those into attitude and motor commands, and we cannot tune
or even read its gains. This module simulates that firmware as a cascade and randomizes
every gain around a nominal value, so a policy trained against it has to cope with
whatever the real firmware does:

    action [vx, vy, vz, yaw_rate] in [-1, 1]   (heading frame, like ``rc``)
      -> velocity PI           (100 Hz)  -> desired acceleration -> thrust + tilt
      -> attitude P            (100 Hz)  -> desired body rates
      -> body-rate PID         (500 Hz)  -> desired moments        (ese651 _get_moment_from_ctbr)
      -> mixer + motor lag + saturation + rotor drag                 (ese651 _apply_action)

The rate loop, mixer, motor model and drag are ported from ``quadcopter_env.py`` in
Jirl-upenn/ese651_project, retuned for the Tello's heavier airframe and slower brushed
motors. The randomization ranges mirror that project's ``_twr_min``/``_kp_omega_rp_max``
block, with the outer-loop gains, motor time constant, mass and command latency added.

Everything is batched over N drones (for a multi-drone env, N = num_envs * drones).
Only torch is imported, so the controller can be exercised without Isaac Sim.

Nominal airframe values are estimates for the Tello EDU (87 g, 3" props, 8520 brushed
motors). Refine ``arm_length``, ``thrust_to_weight`` and ``tau_m`` against flight logs.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field

D2R = math.pi / 180.0


@dataclass
class TelloDynamicsCfg:
    """Nominal Tello firmware/airframe parameters and their randomization ranges."""

    # -- command interface (matches the SDK ``rc`` channels)
    max_speed_xy: float = 1.0
    """[m/s] horizontal speed at |action| = 1 (rc 100 in normal mode is roughly 1 m/s)."""
    max_speed_z: float = 1.0
    """[m/s] vertical speed at |action| = 1."""
    max_yaw_rate: float = 100.0 * D2R
    """[rad/s] yaw rate at |action| = 1."""
    max_tilt: float = 25.0 * D2R
    """[rad] firmware tilt limit in normal mode."""

    # -- loop rates, as decimations of the physics step
    outer_loop_decimation: int = 5
    """Velocity and attitude loops run every this many physics steps (100 Hz at 500 Hz sim)."""

    # -- airframe / rotor model (ese651 names)
    arm_length: float = 0.06
    """[m] centre to motor. Rotors sit at (+-r, +-r) with r = arm_length * sqrt(2) / 2."""
    k_eta: float = 4.0e-8
    """[N / (rad/s)^2] thrust coefficient per motor."""
    k_m: float = 8.0e-10
    """[Nm / (rad/s)^2] drag-torque coefficient per motor (k_m / k_eta = 0.02 m)."""
    tau_m: float = 0.02
    """[s] first-order motor time constant (brushed motors are slower than a Crazyflie's)."""
    thrust_to_weight: float = 2.0
    """Maximum collective thrust over weight. Sets the motor speed ceiling."""
    k_aero_xy: float = 1.5e-6
    """Rotor drag, ese651 form: F_drag_b = -sum(motor_speeds) * K_aero * v_b.
    At hover (sum ~ 9200 rad/s) this is ~0.16 m/s^2 of deceleration per m/s."""
    k_aero_z: float = 2.0e-6

    # -- velocity loop (firmware analogue)
    kp_vel_xy: float = 2.0
    ki_vel_xy: float = 0.4
    kp_vel_z: float = 3.0
    ki_vel_z: float = 3.0
    i_limit_vel: float = 1.0
    """[m/s * s] integrator clamp of the velocity loop."""

    # -- attitude loop
    kp_att_rp: float = 8.0
    """[1/s] roll/pitch attitude error -> body rate set-point."""

    # -- body-rate PID (ese651 structure; gains are angular acceleration per rad/s error)
    kp_omega_rp: float = 30.0
    ki_omega_rp: float = 20.0
    kd_omega_rp: float = 0.3
    i_limit_rp: float = 0.5
    kp_omega_y: float = 12.0
    ki_omega_y: float = 5.0
    kd_omega_y: float = 0.0
    i_limit_y: float = 1.0

    # -- domain randomization (per drone, sampled on reset)
    domain_randomization: bool = True
    twr_range: tuple[float, float] = (0.95, 1.05)
    k_aero_range: tuple[float, float] = (0.5, 2.0)
    kp_omega_range: tuple[float, float] = (0.85, 1.15)
    ki_omega_range: tuple[float, float] = (0.85, 1.15)
    kd_omega_range: tuple[float, float] = (0.7, 1.3)
    outer_gain_range: tuple[float, float] = (0.85, 1.15)
    """Applied independently to kp_vel, ki_vel and kp_att."""
    tau_m_range: tuple[float, float] = (0.8, 1.2)
    mass_range: tuple[float, float] = (0.95, 1.05)
    """True mass relative to the nominal mass the firmware assumes."""
    latency_steps: list[int] = field(default_factory=lambda: [0, 1, 2])
    """Policy steps of command delay, sampled uniformly from this list."""


def _uniform(lo_hi: tuple[float, float], nominal: float, n: int, device) -> torch.Tensor:
    lo, hi = lo_hi
    return nominal * (lo + (hi - lo) * torch.rand(n, device=device))


class TelloCascadeController:
    """Batched Tello firmware + rotor model.

    Per physics step, call :meth:`compute` with the body state. It returns the force and
    torque to apply to the rigid body, both in the body frame (``is_global=False``).
    Call :meth:`set_command` once per policy step with the new action.
    """

    def __init__(
        self,
        cfg: TelloDynamicsCfg,
        num_drones: int,
        nominal_mass: float,
        inertia: torch.Tensor,
        physics_dt: float,
        device: str,
        gravity: float = 9.81,
    ):
        self.cfg = cfg
        self.n = num_drones
        self.device = device
        self.dt = physics_dt
        self.g = gravity
        self.mass_nominal = nominal_mass
        self.weight_nominal = nominal_mass * gravity
        # the firmware's model of the airframe; the true mass is randomized separately
        self.inertia = inertia.to(device).reshape(3, 3).expand(num_drones, 3, 3).clone()

        # mixer (ese651): [T, Mx, My, Mz] = f_to_TM @ motor_forces
        r = cfg.arm_length * math.sqrt(2.0) / 2.0
        rotor_pos = torch.tensor([[r, r, 0.0], [r, -r, 0.0], [-r, -r, 0.0], [-r, r, 0.0]], device=device)
        rotor_dir = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
        z = torch.tensor([0.0, 0.0, 1.0], device=device)
        arms = torch.stack([torch.linalg.cross(rotor_pos[i], z)[:2] for i in range(4)], dim=1)
        self.f_to_TM = torch.cat(
            [torch.ones(1, 4, device=device), arms, (cfg.k_m / cfg.k_eta) * rotor_dir.view(1, -1)], dim=0
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # per-drone parameters (filled by randomize())
        self.twr = torch.full((num_drones,), cfg.thrust_to_weight, device=device)
        self.motor_speed_max = torch.zeros(num_drones, device=device)
        self.K_aero = torch.zeros(num_drones, 3, device=device)
        self.kp_omega = torch.zeros(num_drones, 3, device=device)
        self.ki_omega = torch.zeros(num_drones, 3, device=device)
        self.kd_omega = torch.zeros(num_drones, 3, device=device)
        self.kp_vel = torch.zeros(num_drones, 3, device=device)
        self.ki_vel = torch.zeros(num_drones, 3, device=device)
        self.kp_att = torch.zeros(num_drones, device=device)
        self.tau_m = torch.zeros(num_drones, 1, device=device)
        self.mass_scale = torch.ones(num_drones, device=device)
        self.latency = torch.zeros(num_drones, dtype=torch.long, device=device)

        # state
        max_lat = max(cfg.latency_steps) if cfg.latency_steps else 0
        self._cmd_history = torch.zeros(num_drones, max_lat + 1, 4, device=device)
        self._cmd = torch.zeros(num_drones, 4, device=device)
        self._vel_int = torch.zeros(num_drones, 3, device=device)
        self._omega_err_integral = torch.zeros(num_drones, 3, device=device)
        self._previous_omega_meas = torch.zeros(num_drones, 3, device=device)
        self._omega_des = torch.zeros(num_drones, 3, device=device)
        self._thrust_des = torch.zeros(num_drones, device=device)
        self.motor_speeds = torch.zeros(num_drones, 4, device=device)
        self._motor_speeds_des = torch.zeros(num_drones, 4, device=device)
        self._step = 0

        self.randomize(torch.arange(num_drones, device=device))

    # -- parameters ---------------------------------------------------------------

    def randomize(self, ids: torch.Tensor) -> None:
        """Sample per-drone parameters. With randomization off, write the nominal values."""
        cfg = self.cfg
        n = len(ids)
        dev = self.device
        if cfg.domain_randomization:
            # identical in spirit to ese651's _twr_min/_twr_max ... _kd_omega_y_max block
            self.twr[ids] = _uniform(cfg.twr_range, cfg.thrust_to_weight, n, dev)
            self.K_aero[ids, 0] = _uniform(cfg.k_aero_range, cfg.k_aero_xy, n, dev)
            self.K_aero[ids, 1] = self.K_aero[ids, 0]
            self.K_aero[ids, 2] = _uniform(cfg.k_aero_range, cfg.k_aero_z, n, dev)
            kp_rp = _uniform(cfg.kp_omega_range, cfg.kp_omega_rp, n, dev)
            ki_rp = _uniform(cfg.ki_omega_range, cfg.ki_omega_rp, n, dev)
            kd_rp = _uniform(cfg.kd_omega_range, cfg.kd_omega_rp, n, dev)
            self.kp_omega[ids] = torch.stack([kp_rp, kp_rp, _uniform(cfg.kp_omega_range, cfg.kp_omega_y, n, dev)], -1)
            self.ki_omega[ids] = torch.stack([ki_rp, ki_rp, _uniform(cfg.ki_omega_range, cfg.ki_omega_y, n, dev)], -1)
            self.kd_omega[ids] = torch.stack([kd_rp, kd_rp, _uniform(cfg.kd_omega_range, cfg.kd_omega_y, n, dev)], -1)
            kp_xy = _uniform(cfg.outer_gain_range, cfg.kp_vel_xy, n, dev)
            ki_xy = _uniform(cfg.outer_gain_range, cfg.ki_vel_xy, n, dev)
            self.kp_vel[ids] = torch.stack([kp_xy, kp_xy, _uniform(cfg.outer_gain_range, cfg.kp_vel_z, n, dev)], -1)
            self.ki_vel[ids] = torch.stack([ki_xy, ki_xy, _uniform(cfg.outer_gain_range, cfg.ki_vel_z, n, dev)], -1)
            self.kp_att[ids] = _uniform(cfg.outer_gain_range, cfg.kp_att_rp, n, dev)
            self.tau_m[ids, 0] = _uniform(cfg.tau_m_range, cfg.tau_m, n, dev)
            self.mass_scale[ids] = _uniform(cfg.mass_range, 1.0, n, dev)
            choices = torch.tensor(cfg.latency_steps or [0], device=dev)
            self.latency[ids] = choices[torch.randint(len(choices), (n,), device=dev)]
        else:
            self.twr[ids] = cfg.thrust_to_weight
            self.K_aero[ids] = torch.tensor([cfg.k_aero_xy, cfg.k_aero_xy, cfg.k_aero_z], device=dev)
            self.kp_omega[ids] = torch.tensor([cfg.kp_omega_rp, cfg.kp_omega_rp, cfg.kp_omega_y], device=dev)
            self.ki_omega[ids] = torch.tensor([cfg.ki_omega_rp, cfg.ki_omega_rp, cfg.ki_omega_y], device=dev)
            self.kd_omega[ids] = torch.tensor([cfg.kd_omega_rp, cfg.kd_omega_rp, cfg.kd_omega_y], device=dev)
            self.kp_vel[ids] = torch.tensor([cfg.kp_vel_xy, cfg.kp_vel_xy, cfg.kp_vel_z], device=dev)
            self.ki_vel[ids] = torch.tensor([cfg.ki_vel_xy, cfg.ki_vel_xy, cfg.ki_vel_z], device=dev)
            self.kp_att[ids] = cfg.kp_att_rp
            self.tau_m[ids] = cfg.tau_m
            self.mass_scale[ids] = 1.0
            self.latency[ids] = 0
        # TWR is a property of the motors: it sets the per-motor speed ceiling
        self.motor_speed_max[ids] = torch.sqrt(self.twr[ids] * self.weight_nominal / (4.0 * self.cfg.k_eta))

    def reset(self, ids: torch.Tensor, randomize: bool = True) -> None:
        """Clear controller state for the given drones and spin motors up to hover."""
        if randomize:
            self.randomize(ids)
        self._cmd_history[ids] = 0.0
        self._cmd[ids] = 0.0
        self._vel_int[ids] = 0.0
        self._omega_err_integral[ids] = 0.0
        self._previous_omega_meas[ids] = 0.0
        self._omega_des[ids] = 0.0
        self._thrust_des[ids] = self.weight_nominal
        hover = math.sqrt(self.weight_nominal / (4.0 * self.cfg.k_eta))
        self.motor_speeds[ids] = hover
        self._motor_speeds_des[ids] = hover

    # -- command ------------------------------------------------------------------

    def set_command(self, actions: torch.Tensor) -> None:
        """Latch a new policy action, applying the per-drone command latency."""
        self._cmd_history = torch.roll(self._cmd_history, shifts=1, dims=1)
        self._cmd_history[:, 0] = actions.clamp(-1.0, 1.0)
        idx = self.latency.view(-1, 1, 1).expand(-1, 1, 4)
        self._cmd = torch.gather(self._cmd_history, 1, idx).squeeze(1)

    def to_rc(self, actions: torch.Tensor) -> torch.Tensor:
        """Integer ``rc a b c d`` channels for the real drone: a=left, b=fwd, c=up, d=yaw.

        The SDK's yaw channel is clockwise-positive, while the action (like the sim) is
        counter-clockwise-positive about +z, so it is negated here.
        """
        a = actions.clamp(-1.0, 1.0)
        rc = torch.stack([-a[:, 1], a[:, 0], a[:, 2], -a[:, 3]], dim=-1)
        return (rc * 100.0).round()

    # -- loops --------------------------------------------------------------------

    def _outer_loop(self, lin_vel_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Velocity PI and attitude P: set the collective thrust and body-rate set-points."""
        cfg = self.cfg
        dt = self.dt * cfg.outer_loop_decimation
        R = _quat_to_rot(quat_w)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        # rc channels are in the heading frame: rotate forward/left into the world
        vx = self._cmd[:, 0] * cfg.max_speed_xy
        vy = self._cmd[:, 1] * cfg.max_speed_xy
        v_des = torch.stack([cy * vx - sy * vy, sy * vx + cy * vy, self._cmd[:, 2] * cfg.max_speed_z], dim=-1)

        err = v_des - lin_vel_w
        self._vel_int = (self._vel_int + err * dt).clamp(-cfg.i_limit_vel, cfg.i_limit_vel)
        acc = self.kp_vel * err + self.ki_vel * self._vel_int
        acc[:, 2] = acc[:, 2].clamp(-0.5 * self.g, 0.8 * self.g)
        # keep the thrust vector inside the tilt cone
        acc_h_max = (self.g + acc[:, 2]) * math.tan(cfg.max_tilt)
        acc_h = torch.linalg.norm(acc[:, :2], dim=-1)
        scale = torch.clamp(acc_h_max / torch.clamp(acc_h, min=1e-6), max=1.0)
        acc[:, :2] = acc[:, :2] * scale.unsqueeze(-1)

        f_des = self.mass_nominal * (acc + torch.tensor([0.0, 0.0, self.g], device=self.device))
        z_b = R[:, :, 2]
        self._thrust_des = torch.sum(f_des * z_b, dim=-1).clamp(min=0.0)

        # desired attitude: thrust axis along f_des, heading kept at the current yaw
        z_d = f_des / torch.linalg.norm(f_des, dim=-1, keepdim=True).clamp(min=1e-6)
        x_c = torch.stack([cy, sy, torch.zeros_like(cy)], dim=-1)
        y_d = torch.linalg.cross(z_d, x_c)
        y_d = y_d / torch.linalg.norm(y_d, dim=-1, keepdim=True).clamp(min=1e-6)
        x_d = torch.linalg.cross(y_d, z_d)
        R_d = torch.stack([x_d, y_d, z_d], dim=-1)
        # geometric attitude error e_R = 0.5 * vee(R_d^T R - R^T R_d)
        E = torch.bmm(R_d.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_d)
        e_R = 0.5 * torch.stack([E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]], dim=-1)
        self._omega_des[:, 0] = -self.kp_att * e_R[:, 0]
        self._omega_des[:, 1] = -self.kp_att * e_R[:, 1]
        self._omega_des[:, 2] = self._cmd[:, 3] * cfg.max_yaw_rate

    def _get_moment_from_rates(self, omega_meas: torch.Tensor) -> torch.Tensor:
        """Body-rate PID -> moments. Port of ese651 ``_get_moment_from_ctbr``."""
        cfg = self.cfg
        omega_err = self._omega_des - omega_meas
        self._omega_err_integral += omega_err * self.dt
        limits = torch.tensor([cfg.i_limit_rp, cfg.i_limit_rp, cfg.i_limit_y], device=self.device)
        self._omega_err_integral = torch.clamp(self._omega_err_integral, min=-limits, max=limits)

        self._previous_omega_meas = torch.where(
            torch.abs(self._previous_omega_meas) < 0.0001, omega_meas, self._previous_omega_meas
        )
        omega_meas_dot = (omega_meas - self._previous_omega_meas) / self.dt
        self._previous_omega_meas = omega_meas.clone()

        omega_dot = (
            self.kp_omega * omega_err + self.ki_omega * self._omega_err_integral - self.kd_omega * omega_meas_dot
        )
        return torch.bmm(self.inertia, omega_dot.unsqueeze(2)).squeeze(2)

    def compute(
        self, lin_vel_w: torch.Tensor, quat_w: torch.Tensor, ang_vel_b: torch.Tensor, lin_vel_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one physics step. Returns (force_b, torque_b), each (N, 3).

        Args:
            lin_vel_w: linear velocity, world frame.
            quat_w: orientation (w, x, y, z), world frame.
            ang_vel_b: angular velocity, body frame.
            lin_vel_b: linear velocity, body frame (for rotor drag).
        """
        if self._step % self.cfg.outer_loop_decimation == 0:
            self._outer_loop(lin_vel_w, quat_w)
        self._step += 1

        wrench_des = torch.cat(
            [self._thrust_des.unsqueeze(-1), self._get_moment_from_rates(ang_vel_b)], dim=-1
        )
        # ese651 _compute_motor_speeds
        f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        speed_sq = f_des / self.cfg.k_eta
        speeds_des = torch.sign(speed_sq) * torch.sqrt(torch.abs(speed_sq))
        self._motor_speeds_des = torch.minimum(speeds_des.clamp(min=0.0), self.motor_speed_max.unsqueeze(-1))

        # ese651 _apply_action: motor lag, saturation, thrust and rotor drag
        self.motor_speeds = self.motor_speeds + (self._motor_speeds_des - self.motor_speeds) / self.tau_m * self.dt
        self.motor_speeds = torch.minimum(self.motor_speeds.clamp(min=0.0), self.motor_speed_max.unsqueeze(-1))
        wrench = torch.matmul(self.cfg.k_eta * self.motor_speeds**2, self.f_to_TM.t())
        theta_dot = torch.sum(self.motor_speeds, dim=1, keepdim=True)
        force_b = -theta_dot * self.K_aero * lin_vel_b
        force_b[:, 2] += wrench[:, 0]
        return force_b, wrench[:, 1:]


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) -> rotation matrix, body-to-world."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )
