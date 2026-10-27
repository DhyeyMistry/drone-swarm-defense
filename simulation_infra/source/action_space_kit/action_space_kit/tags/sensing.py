"""What the drone's camera can localize from: which mapped tags are detectable, and the fix.

On the real Tello the only position source is the room's AprilTags (``core/positional.py``):
a frame with any mapped tag in it gives the camera position in the tag-10 frame, averaged
over every visible tag. Between sightings the flight engine dead-reckons on the state
packet. This module models the sighting half, batched in torch:

  detectable  tag centre in front of the camera, the whole tag inside the image, the
              printed face turned toward the camera (``max_view_angle``), its apparent
              edge at least ``min_side_px`` pixels; then it decodes at ``decode_rate``
              (``dropout``, falling with the view angle past ``view_full_angle``) unless
              motion blur loses the whole frame (mostly turning, ``blur_*``)
  fix         the true camera position plus PnP-style noise per tag (depth error grows
              as range^2 / (fx * size), lateral error as range / fx), averaged over the
              detectable tags like ``Positioner.locate``

Camera frame is OpenCV's: x right, y down, z forward. Only torch is imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TagSensorCfg:
    # intrinsics: defaults are config/camera/tello_3.yaml (960x720, fx ~ fy ~ 914.5 px)
    fx: float = 914.5
    fy: float = 914.4
    width: int = 960
    height: int = 720
    camera_offset: tuple[float, float, float] = (0.05, 0.0, 0.0)
    """[m] camera centre in the body frame (x forward, y left, z up): the Tello's nose."""
    camera_pitch: float = 0.0
    """[rad] camera tilt below the horizon. The Tello's camera is fixed, looking ahead."""

    min_side_px: float = 15.0
    """Smallest apparent tag edge that still decodes (tags.yaml gate is 12 px; blur costs more)."""
    max_view_angle: float = 62.0 * math.pi / 180.0
    """[rad] between the tag normal and the line to the camera: past this a tag never decodes."""
    view_full_angle: float = 10.0 * math.pi / 180.0
    """[rad] up to this view angle a tag decodes at the full rate (1 - dropout); from here to
    max_view_angle the rate falls linearly in cos(view angle) to zero. Fitted to 3,700 in-frame
    tag sightings in flights 030-034 and 040-043: detected 89% at 0-20 deg, 77% at 20-35,
    41% at 35-50, 16% at 50-60, 6% at 60-70 (the old flat 90% up to 70 deg overcounted)."""
    max_range: float = 8.0
    """[m] beyond this no tag is used (also caps how far the small tags are trusted)."""
    dropout: float = 0.1
    """Per tag, per frame probability that a detectable tag is missed anyway."""
    # Motion blur loses whole frames. Flights 030-034 (IMU yaw rate vs fresh fixes, tags in
    # view): turning dominates (fix on 95% of frames below 0.2 rad/s, 84% at 0.2-0.5, 53% at
    # 0.5-1, 56% above 1 rad/s); straight flight barely matters (89-100% up to > 2 m/s).
    blur_free_speed: float = 0.5
    """[m/s] below this linear speed frames are sharp."""
    blur_loss_per_mps: float = 0.05
    """Whole-frame loss probability per m/s of linear speed above blur_free_speed."""
    blur_free_rate: float = 0.2
    """[rad/s] below this yaw rate frames are sharp."""
    blur_loss_per_radps: float = 1.0
    """Whole-frame loss probability per rad/s of yaw rate above blur_free_rate."""
    blur_rot_max_loss: float = 0.45
    """Cap on the turning loss (the real loss plateaus at ~45% from 0.5 rad/s up)."""
    blur_max_loss: float = 0.5
    """Cap on the combined loss 1 - (1 - linear)(1 - turning)."""
    pixel_noise: float = 1.0
    """[px] corner noise that drives the PnP position error."""
    video_delay_steps: int = 4
    """Policy steps between a frame being captured and its fix reaching the controller
    (Tello H.264 stream + decode + detect: ~0.2 s at 20 Hz)."""


def camera_axes(body_rot: torch.Tensor, pitch: float) -> torch.Tensor:
    """(N, 3, 3) OpenCV camera axes in the world (columns x right, y down, z forward).

    ``body_rot`` is (N, 3, 3) body-to-world with body x forward, y left, z up.
    """
    c, s = math.cos(pitch), math.sin(pitch)
    # in the body frame: forward tilted down by pitch, right = -y, down completes the triad
    fwd = torch.tensor([c, 0.0, -s], device=body_rot.device)
    right = torch.tensor([0.0, -1.0, 0.0], device=body_rot.device)
    down = torch.linalg.cross(fwd, right)
    cam_in_body = torch.stack([right, down, fwd], dim=-1)  # (3, 3) columns
    return body_rot @ cam_in_body


def detectable(cfg: TagSensorCfg, cam_pos: torch.Tensor, cam_rot: torch.Tensor, tag_pos: torch.Tensor,
               tag_normal: torch.Tensor, tag_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Which tags the camera can decode, ignoring dropout.

    Args:
        cam_pos: (N, 3) world.  cam_rot: (N, 3, 3) from :func:`camera_axes`.
        tag_pos, tag_normal: (T, 3) world.  tag_size: (T,) black-square edge.
    Returns:
        mask (N, T) bool, range (N, T) metres.
    """
    d = tag_pos.unsqueeze(0) - cam_pos.unsqueeze(1)  # (N, T, 3) camera -> tag, world
    rng = torch.linalg.norm(d, dim=-1).clamp(min=1e-6)
    p = torch.einsum("nij,nti->ntj", cam_rot, d)  # in the camera frame
    z = p[..., 2]
    in_front = z > 0.1
    zc = z.clamp(min=0.1)
    # all four black-square corners must be inside the image (the decoder needs them)
    half = 0.5 * tag_size.unsqueeze(0)
    u = (p[..., 0].abs() + half) / zc * cfg.fx
    v = (p[..., 1].abs() + half) / zc * cfg.fy
    in_image = (u < 0.5 * cfg.width) & (v < 0.5 * cfg.height)
    # face toward the camera, within the decode angle
    cos_view = -(d * tag_normal.unsqueeze(0)).sum(-1) / rng
    facing = cos_view > math.cos(cfg.max_view_angle)
    # foreshortened edge in pixels: the short side shrinks with cos(view)
    side_px = cfg.fx * tag_size.unsqueeze(0) * cos_view.clamp(min=0.0).sqrt() / rng
    big_enough = side_px >= cfg.min_side_px
    mask = in_front & in_image & facing & big_enough & (rng <= cfg.max_range)
    return mask, rng


def blur_loss(cfg: TagSensorCfg, speed: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
    """Probability that motion blur loses a whole frame, from linear speed [m/s] and |yaw rate| [rad/s]."""
    lin = (cfg.blur_loss_per_mps * (speed - cfg.blur_free_speed)).clamp(0.0, 1.0)
    rot = (cfg.blur_loss_per_radps * (yaw_rate.abs() - cfg.blur_free_rate)).clamp(0.0, cfg.blur_rot_max_loss)
    return (1.0 - (1.0 - lin) * (1.0 - rot)).clamp(max=cfg.blur_max_loss)


def decode_rate(cfg: TagSensorCfg, cos_view: torch.Tensor) -> torch.Tensor:
    """Per-frame probability that a detectable tag decodes: 1 - dropout up to view_full_angle,
    then linear in cos(view angle) down to zero at max_view_angle."""
    c0, c1 = math.cos(cfg.view_full_angle), math.cos(cfg.max_view_angle)
    return (1.0 - cfg.dropout) * ((cos_view - c1) / (c0 - c1)).clamp(0.0, 1.0)


def sighting(cfg: TagSensorCfg, cam_pos: torch.Tensor, cam_rot: torch.Tensor, tag_pos: torch.Tensor,
             tag_normal: torch.Tensor, tag_size: torch.Tensor, noise: bool = True,
             speed: torch.Tensor | None = None, yaw_rate: torch.Tensor | None = None,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One camera frame: (fix (N, 3) world, any_tag (N,) bool, mask (N, T) bool).

    ``fix`` is only meaningful where ``any_tag``; elsewhere it is the true position.
    With ``noise`` each detectable tag decodes at :func:`decode_rate` (dropout and view angle),
    and ``speed`` / ``yaw_rate`` (N,) blur away whole frames (:func:`blur_loss`).
    """
    mask, rng = detectable(cfg, cam_pos, cam_rot, tag_pos, tag_normal, tag_size)
    if noise:
        d = tag_pos.unsqueeze(0) - cam_pos.unsqueeze(1)
        cos_view = -(d * tag_normal.unsqueeze(0)).sum(-1) / rng
        mask = mask & (torch.rand_like(rng) < decode_rate(cfg, cos_view))
    if noise and speed is not None and yaw_rate is not None:
        p_blur = blur_loss(cfg, speed, yaw_rate)
        mask = mask & (torch.rand_like(p_blur) >= p_blur).unsqueeze(-1)
    per_tag = cam_pos.unsqueeze(1).expand(-1, tag_pos.shape[0], -1)
    if noise:
        los = (cam_pos.unsqueeze(1) - tag_pos.unsqueeze(0)) / rng.unsqueeze(-1)
        sigma_depth = cfg.pixel_noise * rng**2 / (cfg.fx * tag_size.unsqueeze(0))
        sigma_lat = cfg.pixel_noise * rng / cfg.fx
        per_tag = (
            per_tag
            + los * (torch.randn_like(rng) * sigma_depth).unsqueeze(-1)
            + torch.randn_like(per_tag) * sigma_lat.unsqueeze(-1)
        )
    w = mask.float().unsqueeze(-1)
    count = w.sum(1)
    fix = torch.where(count > 0, (per_tag * w).sum(1) / count.clamp(min=1.0), cam_pos)
    return fix, mask.any(dim=1), mask
