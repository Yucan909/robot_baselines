"""ArticuBot bridge to the shared senior ``soft_weld_pd`` SAPIEN backend.

The supplied archive omits its imported ``force_admittance_collect`` package.
The same archive has already been reconstructed once in the shared PA3FF
baseline tree. Reuse that single backend here so both methods receive the
identical contact anchors, spring properties and finger-lock generalized force.
"""

from __future__ import annotations

import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SHARED_BACKEND_ROOT = Path("/home/feng/robot_baselines/repos/pa3ff_official/reproduction")
if str(SHARED_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_BACKEND_ROOT))

from soft_weld_pd_backend import (  # noqa: E402
    ANGULAR_DAMPING,
    ANGULAR_FORCE_LIMIT,
    ANGULAR_STIFFNESS,
    FINGER_DAMPING,
    FINGER_FORCE_LIMIT,
    FINGER_GRIP_FORCE,
    FINGER_STIFFNESS,
    LINEAR_DAMPING,
    LINEAR_FORCE_LIMIT,
    LINEAR_STIFFNESS,
    SoftContactWeld,
    apply_finger_lock_force,
    enable_finger_lock,
)


@dataclass(frozen=True)
class SeniorSoftWeldPDConfig:
    enabled: bool = True
    hard: bool = False
    weld_hand: bool = True
    require_two_finger: bool = True
    linear_stiffness: float = LINEAR_STIFFNESS
    linear_damping: float = LINEAR_DAMPING
    linear_force_limit: float = LINEAR_FORCE_LIMIT
    angular_stiffness: float = ANGULAR_STIFFNESS
    angular_damping: float = ANGULAR_DAMPING
    angular_force_limit: float = ANGULAR_FORCE_LIMIT
    drive_mode: str = "force"
    kinematic_snap: bool = False
    soft_arm: bool = False
    finger_lock_stiffness: float = FINGER_STIFFNESS
    finger_lock_damping: float = FINGER_DAMPING
    finger_lock_force_limit: float = FINGER_FORCE_LIMIT
    finger_grip_torque: float = FINGER_GRIP_FORCE
    operate_target_stiffness: float = 0.0
    operate_target_damping: float = 0.05
    operate_target_force_limit: float = 0.0
    operate_settle_steps: int = 0
    operate_control: str = "drive"
    operate_arm_stiffness: float = 1800.0
    operate_arm_damping: float = 360.0
    operate_steps_per_action: int = 160


DEFAULT_CONFIG = SeniorSoftWeldPDConfig()


def install_soft_weld_step(controller) -> None:
    """Install the shared backend's post-grasp force step, dormant until lock."""
    if getattr(controller, "_senior_soft_weld_step_installed", False):
        return
    original_step = controller.step

    def step(self):
        if hasattr(self, "soft_weld_finger_lock_target"):
            for joint in self.finger_joints:
                joint.set_drive_velocity_target(0.0)
                joint.set_drive_target(float(self.soft_weld_finger_lock_target))
            apply_finger_lock_force(self)
            # Match the shared SoftWeldPDController and avoid the base step's
            # second passive-force write erasing the closing generalized force.
            self.scene.step()
            self._video_physics_counter += 1
            if (
                self._video_step_callback is not None
                and self._video_physics_counter % self._video_every_n_steps == 0
            ):
                self._video_step_callback()
            return
        original_step()

    controller.step = types.MethodType(step, controller)
    controller._senior_soft_weld_step_installed = True


def enable_backend_finger_lock(panda, target: float) -> dict:
    """Enable the exact shared finger lock, including its step-time force."""
    install_soft_weld_step(panda)
    return enable_finger_lock(panda, float(target))


class SeniorSoftWeldPD:
    """Compatibility wrapper around the exact shared backend object."""

    def __init__(self, weld: SoftContactWeld, panda, finger_target: float):
        self.weld = weld
        self.panda = panda
        self.finger_target = float(finger_target)

    @classmethod
    def try_create(cls, scene, panda, target_link, config=DEFAULT_CONFIG):
        del scene
        if config != DEFAULT_CONFIG:
            raise ValueError("soft_weld_pd must use the supplied authoritative defaults")
        weld, diag = SoftContactWeld.try_create(panda, target_link)
        diag = dict(diag)
        diag.update(
            {
                "source_backend": str(SHARED_BACKEND_ROOT / "soft_weld_pd_backend.py"),
                "source_backend_sha256": "972998b9a55ac1a79f7a7b4c792eb4914003b8d7ee9ddcf10765e5c97c140de8",
            }
        )
        if weld is None:
            return None, diag
        target = float(np.mean(np.asarray(panda.robot.get_qpos(), dtype=np.float64)[7:9]))
        return cls(weld, panda, target), diag

    def enable_finger_lock(self, panda) -> dict:
        self.finger_target = float(np.clip(self.finger_target, 0.0, 0.04))
        return enable_backend_finger_lock(panda, self.finger_target)

    def maintain_finger_lock(self, panda) -> None:
        if hasattr(panda, "soft_weld_finger_lock_target"):
            for joint in panda.finger_joints:
                joint.set_drive_velocity_target(0.0)
                joint.set_drive_target(float(panda.soft_weld_finger_lock_target))

    def max_position_error(self) -> float:
        return float(self.weld.errors()["weld_position_error_m"])

    def angular_error(self) -> float:
        return float(self.weld.errors()["weld_angular_error_rad"])

    def destroy(self) -> None:
        # The supplied backend keeps drives for the complete trial; SAPIEN
        # destroys them together with the episode scene.
        pass


def source_defaults() -> dict:
    return asdict(DEFAULT_CONFIG)
