import numpy as np

from robocasa.environments.kitchen.kitchen import *


class _CountertopObjectPickup(Kitchen):
    """Minimal pickup task: a single graspable object on a countertop."""

    OBJECT_GROUP = None
    INSTRUCTION = None
    LIFT_SUCCESS_DELTA_Z = 0.08
    LIFT_HOLD_STEPS = 5
    DEFAULT_FIXED_LAYOUT_ID = 0
    DEFAULT_FIXED_STYLE_ID = 0
    DEFAULT_FIXED_COUNTER_NAME = "counter_right_main_group"
    DEFAULT_COUNTER_SIZE = (0.35, 0.35)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        self.randomize_scene = bool(kwargs.pop("randomize_scene", False))
        self.fixed_layout_id = int(kwargs.pop("fixed_layout_id", self.DEFAULT_FIXED_LAYOUT_ID))
        self.fixed_style_id = int(kwargs.pop("fixed_style_id", self.DEFAULT_FIXED_STYLE_ID))
        self.fixed_counter_name = str(kwargs.pop("fixed_counter_name", self.DEFAULT_FIXED_COUNTER_NAME))

        # Keep the scene deterministic unless scene randomization is explicitly enabled.
        if not self.randomize_scene and "layout_and_style_ids" not in kwargs:
            kwargs["layout_ids"] = [self.fixed_layout_id]
            kwargs["style_ids"] = [self.fixed_style_id]

        self._obj_init_z = None
        self._lift_hold_count = 0
        super().__init__(*args, **kwargs)

    @property
    def counter_size(self):
        return self.DEFAULT_COUNTER_SIZE

    def _setup_kitchen_references(self):
        """Select a valid countertop region and place the robot near it."""
        super()._setup_kitchen_references()

        if self.randomize_scene:
            counter_kwargs = {"id": FixtureType.COUNTER, "size": self.counter_size}
        else:
            counter_id = (
                self.fixed_counter_name
                if self.fixed_counter_name in self.fixtures
                else FixtureType.COUNTER
            )
            counter_kwargs = {"id": counter_id}
            if counter_id == FixtureType.COUNTER:
                counter_kwargs["size"] = self.counter_size

        self.counter = self.register_fixture_ref("counter", counter_kwargs)
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        """Expose a fixed pickup instruction for prompt ablations."""
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.INSTRUCTION
        return ep_meta

    def _get_obj_cfgs(self):
        """Spawn exactly one graspable object on the selected countertop."""
        return [
            dict(
                name="obj",
                obj_groups=self.OBJECT_GROUP,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=self.counter_size,
                    pos=(0.0, 0.0),
                    rotation=(0.0, 0.0),
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


class CountertopMugPickup(_CountertopObjectPickup):
    """Minimal pickup task: a single mug on a countertop in a full kitchen scene."""

    OBJECT_GROUP = "mug"
    INSTRUCTION = "pick up the mug"


class CountertopPanPickup(_CountertopObjectPickup):
    """Minimal pickup task: a single pan on a countertop in a full kitchen scene."""

    OBJECT_GROUP = "pan"
    INSTRUCTION = "pick up the pan"
    DEFAULT_COUNTER_SIZE = (0.45, 0.45)
