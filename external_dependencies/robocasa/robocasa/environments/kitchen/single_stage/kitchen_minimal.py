import numpy as np

from robocasa.environments.kitchen.kitchen import *


class _CountertopObjectPickup(Kitchen):
    """Minimal pickup task: a single graspable object on a countertop."""

    OBJECT_GROUP = None
    INSTRUCTION = None
    OBJECT_ROTATION_RANGE = (0.0, 0.0)
    LIFT_SUCCESS_DELTA_Z = 0.08
    LIFT_HOLD_STEPS = 5
    REQUIRE_OFF_COUNTER = True
    DEFAULT_FIXED_LAYOUT_ID = 0
    DEFAULT_FIXED_STYLE_ID = 0
    DEFAULT_FIXED_COUNTER_NAME = "counter_right_main_group"
    DEFAULT_COUNTER_SIZE = (0.35, 0.35)
    PREFERRED_COUNTER_HINTS = (
        "counter_right_main_group",
        "counter_corner_right_main_group",
        "counter_corner_main_group",
        "counter_right_group",
        "counter_corner_right_group",
    )
    SECONDARY_COUNTER_HINTS = (
        "counter_main_main_group",
        "counter_main_group",
    )

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
            counter_name = self.select_counter_name_for_random_scene()
            if counter_name is not None:
                counter_kwargs = {"id": counter_name}
            else:
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

    def select_counter_name_for_random_scene(self):
        # Prefer counters that keep the robot in front of work surfaces and away from
        # island / left-side layouts that often produce unreachable starts.
        candidates = [name for (name, fxtr) in self.fixtures.items() if isinstance(fxtr, Counter)]
        if len(candidates) == 0:
            return None

        valid = []
        for name in candidates:
            fixture = self.fixtures.get(name, None)
            if fixture is None:
                continue
            try:
                fixture.sample_reset_region(env=self)
            except Exception:
                continue
            valid.append(name)

        if len(valid) == 0:
            return None

        safe_valid = []
        for name in valid:
            lname = name.lower()
            if "island" in lname or "left" in lname or "front" in lname:
                continue
            safe_valid.append(name)

        ranked_source = safe_valid if len(safe_valid) > 0 else valid
        preferred = []
        secondary = []
        other = []
        for name in ranked_source:
            lname = name.lower()
            if any(hint in lname for hint in self.PREFERRED_COUNTER_HINTS):
                preferred.append(name)
            elif any(hint in lname for hint in self.SECONDARY_COUNTER_HINTS):
                secondary.append(name)
            else:
                other.append(name)

        pool = preferred if len(preferred) > 0 else (secondary if len(secondary) > 0 else other)
        if len(pool) == 0:
            pool = ranked_source
        return str(self.rng.choice(pool))

    def get_ep_meta(self):
        """Expose a fixed pickup instruction for prompt ablations."""
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.INSTRUCTION
        return ep_meta

    def _get_obj_cfgs(self):
        """Spawn exactly one graspable object on the selected countertop."""
        placement = self.get_object_placement()
        return [
            dict(
                name="obj",
                obj_groups=self.OBJECT_GROUP,
                graspable=True,
                placement=placement,
            )
        ]

    def get_object_placement(self):
        return dict(
            fixture=self.counter,
            size=self.counter_size,
            pos=(0.0, 0.0),
            rotation=self.OBJECT_ROTATION_RANGE,
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._obj_init_z = float(self.sim.data.body_xpos[self.obj_body_id["obj"]][2])
        self._lift_hold_count = 0

    def _check_success(self):
        """Succeed only after the object stays lifted for several steps."""
        if self._obj_init_z is None:
            return False

        obj_z = float(self.sim.data.body_xpos[self.obj_body_id["obj"]][2])
        lifted = obj_z > (self._obj_init_z + self.LIFT_SUCCESS_DELTA_Z)
        on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        lift_ok = lifted and ((not self.REQUIRE_OFF_COUNTER) or (not on_counter))

        if lift_ok:
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
    OBJECT_ROTATION_RANGE = (-np.pi / 4.0, np.pi / 4.0)
    DEFAULT_COUNTER_SIZE = (0.45, 0.45)
    PAN_POSITION_JITTER_X = 0.45
    PAN_POSITION_Y = 0.0
    PAN_INNER_PLACEMENT_SIZE = (0.30, 0.24)

    def get_object_placement(self):
        # Randomize pan placement location on the counter every scene reset.
        pan_pos = (
            float(self.rng.uniform(-self.PAN_POSITION_JITTER_X, self.PAN_POSITION_JITTER_X)),
            float(self.PAN_POSITION_Y),
        )
        return dict(
            fixture=self.counter,
            size=self.PAN_INNER_PLACEMENT_SIZE,
            pos=pan_pos,
            rotation=self.OBJECT_ROTATION_RANGE,
            # Pan handle can extend outside the region; do not force full boundary containment.
            ensure_object_boundary_in_range=False,
        )
