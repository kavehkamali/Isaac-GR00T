import numpy as np

from robocasa.environments.kitchen.kitchen import *


class CountertopMugPickup(Kitchen):
    """Minimal pickup task: a single mug on a countertop in a full kitchen scene."""

    LIFT_SUCCESS_DELTA_Z = 0.08
    LIFT_HOLD_STEPS = 5

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        self._obj_init_z = None
        self._lift_hold_count = 0
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """Select a valid countertop region and place the robot near it."""
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.35, 0.35))
        )
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
                    size=(0.35, 0.35),
                    pos=(0.0, 0.0),
                    rotation=(0.0, 2 * np.pi),
                ),
            )
        ]

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
