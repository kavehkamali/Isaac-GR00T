from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


robocasa_vr_config = {
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["sim_state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["sim_action"],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


tag = EmbodimentTag.NEW_EMBODIMENT
if tag.value in MODALITY_CONFIGS:
    MODALITY_CONFIGS[tag.value] = robocasa_vr_config
else:
    register_modality_config(robocasa_vr_config, embodiment_tag=tag)
