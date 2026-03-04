# RoboCasa Evaluation Benchmark

[RoboCasa](https://robocasa.ai/) is a large-scale simulation framework for training generally capable robots to perform everyday tasks, featuring realistic kitchen environments with over 2,500 3D assets and 100 diverse manipulation tasks. This evaluation benchmark uses RoboCasa with the Panda robot equipped with an Omron gripper to test household manipulation tasks including operating kitchen appliances, pick-and-place operations, and interacting with doors, drawers, and various objects.

---

# RoboCasa evaluation benchmark result
Checkpoint: [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)

| Task | Success rate |
| ---- | ------------ |
| `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` | 31.0% |
| `robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env` | 63.5% |
| `robocasa_panda_omron/CoffeePressButton_PandaOmron_Env` | 98.5% |
| `robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env` | 81.5% |
| `robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env` | 39.0% |
| `robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env` | 96.0% |
| `robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env` | 88.5% |
| `robocasa_panda_omron/OpenDrawer_PandaOmron_Env` | 81.1% |
| `robocasa_panda_omron/CloseDrawer_PandaOmron_Env` | 100.0% |
| `robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env` | 91.5% |
| `robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env` | 96.0% |
| `robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env` | 47.5% |
| `robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env` | 41.0% |
| `robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env` | 46.0% |
| `robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env` | 50.0% |
| `robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env` | 19.0% |
| `robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env` | 24.5% |
| `robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env` | 63.2% |
| `robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env` | 54.5% |
| `robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env` | 89.0% |
| `robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env` | 93.5% |
| `robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env` | 87.0% |
| `robocasa_panda_omron/TurnOnStove_PandaOmron_Env` | 76.5% |
| `robocasa_panda_omron/TurnOffStove_PandaOmron_Env` | 31.0% |
| **Average** | 66.22% |

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 5
```

# Convert RoboCasa-VR demos to GR00T LeRobot format

Use this converter when your trajectories were collected in the RoboCasa-VR repository (`demo.hdf5` files).

```bash
python scripts/robocasa/convert_robocasa_vr_to_lerobot.py \
    --input /path/to/robocasa_vr_data \
    --output /path/to/robocasa_vr_lerobot \
    --robot-type PandaOmron \
    --fallback-task "pick up the mug" \
    --overwrite
```

Notes:
- `--input` can be a single `demo.hdf5` or a directory containing many `demo.hdf5` files.
- Output follows GR00T-flavored LeRobot v2 structure (`meta/` + chunked parquet in `data/`).
- Runtime dependencies: `h5py`, `pandas`, `pyarrow`, `numpy`.


# Fine-tune on RoboCasa-VR CountertopMugPickup

Run in this exact order:

1. Convert RoboCasa-VR demos:
```bash
python scripts/robocasa/convert_robocasa_vr_to_lerobot.py \
    --input /path/to/robocasa_vr_data \
    --output /path/to/robocasa_vr_lerobot \
    --robot-type PandaOmron \
    --fallback-task "pick up the mug" \
    --overwrite
```

2. Launch fine-tuning from the converted dataset:
```bash
bash examples/robocasa/finetune_countertop_mug_vr.sh \
    /path/to/robocasa_vr_lerobot \
    /tmp/robocasa_vr_countertop_mug_finetune
```

Optional env overrides for step 2:
- `NUM_GPUS` (default `1`)
- `BASE_MODEL_PATH` (default `nvidia/GR00T-N1.6-3B`)
- `MAX_STEPS` (default `10000`)
- `GLOBAL_BATCH_SIZE` (default `64`)
- `USE_WANDB=1` to enable wandb logging

Notes:
- This fine-tuning path uses `NEW_EMBODIMENT` with `sim_state` and `sim_action` from the RoboCasa-VR converter.
- RoboCasa-VR demos are commonly state/action only; the processor now supports that path without requiring recorded videos.
