"""Configuration for the DJI Tello EDU quadcopter.

The USD is converted from the BSD-3 licensed URDF in tello_ros (see THIRD_PARTY.md),
with the physical properties corrected to Tello EDU spec: 87 g, 98 x 92.5 x 41 mm body.

The converted asset is a single rigid body (no rotor joints): the real Tello exposes no
motor-level control, so modelling rotors would buy nothing. Thrust is applied by the
velocity controller in ``action_space_kit.control.tello_command``.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# Tello EDU physical properties
TELLO_MASS = 0.087  # [kg]
TELLO_SIZE = (0.098, 0.0925, 0.041)  # [m] body, without propellers

# Override with TELLO_USD_PATH to keep the kit portable across machines.
TELLO_USD_PATH = os.environ.get("TELLO_USD_PATH", "/mnt/data/isaac/assets/tello.usd")

TELLO_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TELLO_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
"""Configuration for the Tello EDU quadcopter as a single rigid body."""
