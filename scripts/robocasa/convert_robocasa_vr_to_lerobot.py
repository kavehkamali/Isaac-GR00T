#!/usr/bin/env python3
"""Convert RoboCasa-VR demo.hdf5 files to GR00T-flavored LeRobot v2 format.

Input format expected from RoboCasa-VR collect_demos:
- one or more demo.hdf5 files
- each file contains group: data/demo_<N> with datasets:
  - states: (T, state_dim)
  - actions: (T, action_dim)
- optional group attrs / demo attrs with env metadata and ep_meta JSON

Output format:
- <output>/meta/{info.json,episodes.jsonl,tasks.jsonl,modality.json,stats.json,relative_stats.json}
- <output>/data/chunk-XXX/episode_XXXXXX.parquet

Notes:
- This converter writes state-only data (no videos).
- By default, modality keys are single blocks: "sim_state" and "sim_action".
- If mixed state/action dimensions are present in input files, the converter
  automatically keeps the most common (state_dim, action_dim) pair unless
  --state-dim and --action-dim are explicitly provided.
- Requires: h5py, pandas, pyarrow, numpy
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


@dataclass
class EpisodeRecord:
    episode_index: int
    task_text: str
    env_name: str
    states: np.ndarray
    actions: np.ndarray


def _load_runtime_deps():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Missing dependency 'h5py'. Install with: pip install h5py") from exc

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing dependency 'pandas' (and parquet backend). Install with: pip install pandas pyarrow"
        ) from exc

    return h5py, pd


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        return json.loads(_to_text(raw))
    except Exception:
        return {}


def _discover_demo_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "demo.hdf5":
            raise ValueError(f"Input file must be demo.hdf5, got: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    matches = sorted(input_path.rglob("demo.hdf5"))
    if not matches:
        raise FileNotFoundError(f"No demo.hdf5 files found under: {input_path}")
    return matches


def _demo_key_sort(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        suffix = name.split("demo_", 1)[1]
        if suffix.isdigit():
            return (int(suffix), name)
    return (10**9, name)


def _extract_task_text(ep_meta_attr: Any, fallback_task: str) -> str:
    ep_meta = _safe_json_loads(ep_meta_attr)
    lang = ep_meta.get("lang")
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return fallback_task


def _read_episodes_from_hdf5(
    demo_files: list[Path],
    fallback_task: str,
    h5py,
) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []
    episode_idx = 0

    for h5_path in demo_files:
        with h5py.File(h5_path, "r") as h5f:
            data_grp = h5f.get("data")
            if data_grp is None:
                continue

            env_name = _to_text(data_grp.attrs.get("env", "CountertopMugPickup"))
            demo_keys = sorted(list(data_grp.keys()), key=_demo_key_sort)

            for demo_key in demo_keys:
                demo_grp = data_grp.get(demo_key)
                if demo_grp is None:
                    continue
                if "states" not in demo_grp or "actions" not in demo_grp:
                    continue

                states = np.asarray(demo_grp["states"], dtype=np.float32)
                actions = np.asarray(demo_grp["actions"], dtype=np.float32)
                if states.ndim != 2 or actions.ndim != 2:
                    continue

                length = int(min(states.shape[0], actions.shape[0]))
                if length <= 0:
                    continue

                states = states[:length]
                actions = actions[:length]
                task_text = _extract_task_text(demo_grp.attrs.get("ep_meta"), fallback_task)

                episodes.append(
                    EpisodeRecord(
                        episode_index=episode_idx,
                        task_text=task_text,
                        env_name=env_name,
                        states=states,
                        actions=actions,
                    )
                )
                episode_idx += 1

    if not episodes:
        raise RuntimeError("No valid episodes were found in input demo.hdf5 files.")

    return episodes


def _choose_target_dims(
    episodes: list[EpisodeRecord],
    state_dim_override: int | None,
    action_dim_override: int | None,
) -> tuple[tuple[int, int], Counter[tuple[int, int]]]:
    dim_counts: Counter[tuple[int, int]] = Counter(
        (int(ep.states.shape[1]), int(ep.actions.shape[1])) for ep in episodes
    )

    if (state_dim_override is None) != (action_dim_override is None):
        raise ValueError("Use --state-dim and --action-dim together, or omit both.")

    if state_dim_override is not None and action_dim_override is not None:
        target = (int(state_dim_override), int(action_dim_override))
        if target not in dim_counts:
            available = ", ".join(
                f"({s},{a})x{c}" for (s, a), c in sorted(dim_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            raise ValueError(
                f"Requested dimensions {target} not found in input episodes. Available: {available}"
            )
        return target, dim_counts

    # Default: pick the most common dimensions.
    target = sorted(dim_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return target, dim_counts


def _compute_stats(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for stats, got shape={arr.shape}")

    return {
        "mean": np.mean(arr, axis=0).astype(np.float64).tolist(),
        "std": np.std(arr, axis=0).astype(np.float64).tolist(),
        "min": np.min(arr, axis=0).astype(np.float64).tolist(),
        "max": np.max(arr, axis=0).astype(np.float64).tolist(),
        "q01": np.percentile(arr, 1, axis=0).astype(np.float64).tolist(),
        "q99": np.percentile(arr, 99, axis=0).astype(np.float64).tolist(),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(args: argparse.Namespace) -> None:
    h5py, pd = _load_runtime_deps()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    demo_files = _discover_demo_files(input_path)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    raw_episodes = _read_episodes_from_hdf5(
        demo_files=demo_files,
        fallback_task=args.fallback_task,
        h5py=h5py,
    )

    target_dims, dim_counts = _choose_target_dims(
        raw_episodes,
        state_dim_override=args.state_dim,
        action_dim_override=args.action_dim,
    )
    state_dim, action_dim = target_dims

    filtered_episodes = [
        ep for ep in raw_episodes if (int(ep.states.shape[1]), int(ep.actions.shape[1])) == target_dims
    ]
    skipped = len(raw_episodes) - len(filtered_episodes)

    if skipped > 0:
        histogram = ", ".join(
            f"({s},{a})x{c}" for (s, a), c in sorted(dim_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        print(f"Detected mixed dimensions across episodes: {histogram}")
        print(
            f"Keeping dimensions (state_dim={state_dim}, action_dim={action_dim}); "
            f"skipped {skipped} / {len(raw_episodes)} episodes"
        )

    if not filtered_episodes:
        raise RuntimeError("No episodes remain after dimension filtering.")

    # Reindex episodes densely after filtering so chunk paths are contiguous.
    episodes: list[EpisodeRecord] = [
        EpisodeRecord(
            episode_index=i,
            task_text=ep.task_text,
            env_name=ep.env_name,
            states=ep.states,
            actions=ep.actions,
        )
        for i, ep in enumerate(filtered_episodes)
    ]

    # Build task vocabulary.
    valid_label = args.valid_label
    task_to_idx: dict[str, int] = {}

    def register_task(task: str) -> int:
        if task not in task_to_idx:
            task_to_idx[task] = len(task_to_idx)
        return task_to_idx[task]

    # Register episode tasks first; add validity label after.
    for ep in episodes:
        register_task(ep.task_text)
    valid_task_idx = register_task(valid_label)

    # Aggregate arrays for stats.
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_relative_actions: list[np.ndarray] = []

    episodes_meta: list[dict[str, Any]] = []
    total_frames = 0
    global_index = 0

    for ep in episodes:
        task_idx = task_to_idx[ep.task_text]
        length = int(ep.states.shape[0])

        episode_chunk = ep.episode_index // int(args.chunk_size)
        parquet_path = (
            output_root
            / "data"
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{ep.episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        timestamps = (np.arange(length, dtype=np.float64) / float(args.fps)).reshape(-1, 1)
        rewards = np.zeros((length, 1), dtype=np.float64)
        rewards[-1, 0] = 1.0
        next_done = np.zeros((length,), dtype=bool)
        next_done[-1] = True

        frame_df = pd.DataFrame(
            {
                "observation.state": [row for row in ep.states],
                "action": [row for row in ep.actions],
                "timestamp": timestamps[:, 0],
                "annotation.human.action.task_description": np.full(length, task_idx, dtype=np.int64),
                "task_index": np.full(length, task_idx, dtype=np.int64),
                "annotation.human.validity": np.full(length, valid_task_idx, dtype=np.int64),
                "episode_index": np.full(length, ep.episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "next.reward": rewards[:, 0],
                "next.done": next_done,
            }
        )
        frame_df.to_parquet(parquet_path, index=False)

        if length > 1:
            all_relative_actions.append(np.diff(ep.actions, axis=0))

        all_states.append(ep.states)
        all_actions.append(ep.actions)
        all_timestamps.append(timestamps)
        all_rewards.append(rewards)

        episodes_meta.append(
            {
                "episode_index": ep.episode_index,
                "tasks": [ep.task_text, valid_label],
                "length": length,
            }
        )

        global_index += length
        total_frames += length

    states_arr = np.concatenate(all_states, axis=0)
    actions_arr = np.concatenate(all_actions, axis=0)
    timestamps_arr = np.concatenate(all_timestamps, axis=0)
    rewards_arr = np.concatenate(all_rewards, axis=0)

    if all_relative_actions:
        rel_actions_arr = np.concatenate(all_relative_actions, axis=0)
    else:
        rel_actions_arr = np.zeros((1, action_dim), dtype=np.float32)

    tasks_rows = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(task_to_idx.items(), key=lambda kv: kv[1])
    ]

    chunk_count = max(1, (len(episodes) + int(args.chunk_size) - 1) // int(args.chunk_size))
    info = {
        "codebase_version": "v2.0",
        "robot_type": args.robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_to_idx),
        "total_videos": 0,
        "total_chunks": chunk_count - 1,
        "chunks_size": int(args.chunk_size),
        "fps": float(args.fps),
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"state_{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [f"action_{i}" for i in range(action_dim)],
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.validity": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
    }

    modality = {
        "state": {
            "sim_state": {"start": 0, "end": state_dim},
        },
        "action": {
            "sim_action": {"start": 0, "end": action_dim},
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }

    stats = {
        "observation.state": _compute_stats(states_arr),
        "action": _compute_stats(actions_arr),
        "timestamp": _compute_stats(timestamps_arr),
        "next.reward": _compute_stats(rewards_arr),
    }

    relative_stats = {
        "sim_action": _compute_stats(rel_actions_arr),
    }

    meta_dir = output_root / "meta"
    _write_json(meta_dir / "info.json", info)
    _write_json(meta_dir / "modality.json", modality)
    _write_json(meta_dir / "stats.json", stats)
    _write_json(meta_dir / "relative_stats.json", relative_stats)
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)

    print(f"Converted {len(episodes)} episodes from {len(demo_files)} demo.hdf5 file(s)")
    print(f"Output dataset: {output_root}")
    print(f"state_dim={state_dim}, action_dim={action_dim}, total_frames={total_frames}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to RoboCasa-VR demo.hdf5 file or directory containing demo.hdf5 files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for GR00T LeRobot dataset",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frame rate used to synthesize timestamp column",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of episodes per chunk directory",
    )
    parser.add_argument(
        "--robot-type",
        default="PandaOmron",
        help="robot_type field in meta/info.json",
    )
    parser.add_argument(
        "--fallback-task",
        default="pick up the mug",
        help="Task text used when ep_meta.lang is missing",
    )
    parser.add_argument(
        "--valid-label",
        default="valid",
        help="Label text used for annotation.human.validity",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Optional: keep only episodes with this state dimension (must be used with --action-dim)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Optional: keep only episodes with this action dimension (must be used with --state-dim)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
