import numpy as np

from robocasa.environments.kitchen.kitchen import *


class CountertopMugPickup(Kitchen):
    """Minimal pickup task: a single mug on a countertop in a full kitchen scene."""

    LIFT_SUCCESS_DELTA_Z = 0.08
    LIFT_HOLD_STEPS = 5
    DEFAULT_FIXED_LAYOUT_ID = 0
    DEFAULT_FIXED_STYLE_ID = 0
    DEFAULT_FIXED_COUNTER_NAME = "counter_right_main_group"
    DEFAULT_COUNTER_SIZE = (0.35, 0.35)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        self.randomize_scene = bool(kwargs.pop("randomize_scene", False))
        self.randomize_robot_pose = bool(kwargs.pop("randomize_robot_pose", False))
        self.robot_pose_jitter_xy = float(max(0.0, kwargs.pop("robot_pose_jitter_xy", 0.0)))
        self.robot_pose_outward_only = bool(kwargs.pop("robot_pose_outward_only", True))
        self.fixed_layout_id = int(kwargs.pop("fixed_layout_id", self.DEFAULT_FIXED_LAYOUT_ID))
        self.fixed_style_id = int(kwargs.pop("fixed_style_id", self.DEFAULT_FIXED_STYLE_ID))
        self.fixed_counter_name = str(kwargs.pop("fixed_counter_name", self.DEFAULT_FIXED_COUNTER_NAME))

        # Keep the scene deterministic unless scene randomization is explicitly enabled.
        if not self.randomize_scene:
            kwargs["layout_ids"] = [self.fixed_layout_id]
            kwargs["style_ids"] = [self.fixed_style_id]

        self._obj_init_z = None
        self._lift_hold_count = 0
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """Select a valid countertop region and place the robot near it."""
        super()._setup_kitchen_references()

        if self.randomize_scene:
            counter_kwargs = {"id": FixtureType.COUNTER, "size": self.DEFAULT_COUNTER_SIZE}
        else:
            counter_id = (
                self.fixed_counter_name
                if self.fixed_counter_name in self.fixtures
                else FixtureType.COUNTER
            )
            counter_kwargs = {"id": counter_id}
            if counter_id == FixtureType.COUNTER:
                counter_kwargs["size"] = self.DEFAULT_COUNTER_SIZE

        self.counter = self.register_fixture_ref("counter", counter_kwargs)
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        """Expose a fixed pickup instruction for prompt ablations."""
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the mug"
        return ep_meta

    def _get_obj_cfgs(self):
        """Spawn exactly one graspable mug on the selected countertop."""
        return [
            dict(
                name="obj",
                obj_groups="mug",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=self.DEFAULT_COUNTER_SIZE,
                    pos=(0.0, 0.0),
                    rotation=(0.0, 0.0),
                ),
            )
        ]

    def compute_robot_base_placement_pose(self, ref_fixture, offset=None):
        if self.randomize_robot_pose and self.robot_pose_jitter_xy > 0.0:
            x_jitter = float(
                self.rng.uniform(
                    low=-self.robot_pose_jitter_xy,
                    high=self.robot_pose_jitter_xy,
                )
            )
            if self.robot_pose_outward_only:
                # Keep y-offset outward from the counter to avoid invalid poses inside cabinetry.
                y_jitter = float(self.rng.uniform(low=-self.robot_pose_jitter_xy, high=0.0))
            else:
                y_jitter = float(
                    self.rng.uniform(
                        low=-self.robot_pose_jitter_xy,
                        high=self.robot_pose_jitter_xy,
                    )
                )
            if offset is None:
                offset = [0.0, 0.0]
            else:
                offset = list(offset)
            offset[0] += x_jitter
            offset[1] += y_jitter
        return super().compute_robot_base_placement_pose(ref_fixture=ref_fixture, offset=offset)

    def _reset_internal(self):
        super()._reset_internal()
        self._obj_init_z = float(self.sim.data.body_xpos[self.obj_body_id["obj"]][2])
        self._lift_hold_count = 0

    def _check_success(self):
        """Succeed only after the mug stays lifted off the counter for several steps."""
        if self._obj_init_z is None:
            return False

        obj_z = float(self.sim.data.body_xpos[self.obj_body_id["obj"]][2])
        lifted = obj_z > (self._obj_init_z + self.LIFT_SUCCESS_DELTA_Z)
        on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)

        if lifted and not on_counter:
            self._lift_hold_count += 1
        else:
            self._lift_hold_count = 0

        return self._lift_hold_count >= self.LIFT_HOLD_STEPS
