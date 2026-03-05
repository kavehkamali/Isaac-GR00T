#!/usr/bin/env python3
"""Convert RoboCasa-VR demo.hdf5 files to GR00T-compatible LeRobot v2 data.

This converter targets the PandaOmron RoboCasa schema expected by GR00T's
`robocasa_panda_omron` modality:

State order (16):
- end_effector_position_relative (3)
- end_effector_rotation_relative (4)   # quaternion xyzw
- gripper_qpos (2)
- base_position (3)
- base_rotation (4)                    # quaternion xyzw

Action order (12):
- end_effector_position (3)
- end_effector_rotation (3)
- gripper_close (1)                    # 0/1
- base_motion (4)
- control_mode (1)                     # 0/1
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


STATE_LAYOUT: list[tuple[str, int]] = [
    ("end_effector_position_relative", 3),
    ("end_effector_rotation_relative", 4),
    ("gripper_qpos", 2),
    ("base_position", 3),
    ("base_rotation", 4),
]

ACTION_LAYOUT: list[tuple[str, int]] = [
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
    ("base_motion", 4),
    ("control_mode", 1),
]

STATE_DIM = sum(width for _, width in STATE_LAYOUT)
ACTION_DIM = sum(width for _, width in ACTION_LAYOUT)

_ASSET_SEARCH_ROOTS: list[Path] | None = None
_ASSET_PATH_CACHE: dict[str, Path | None] = {}


@dataclass
class EpisodeRecord:
    episode_index: int
    task_text: str
    env_name: str
    states: np.ndarray
    actions: np.ndarray
    model_xml: str | None = None


@dataclass
class EpisodeProjection:
    model: Any
    data: Any
    qpos_count: int
    qvel_count: int
    gripper_qpos_idx: list[int]
    base_site_id: int
    eef_site_id: int
    eef_body_id: int


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

    try:
        import mujoco  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Missing dependency 'mujoco'. Install with: pip install mujoco") from exc

    return h5py, pd, mujoco


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
                model_xml_attr = demo_grp.attrs.get("model_file")
                model_xml = _to_text(model_xml_attr) if model_xml_attr is not None else None

                episodes.append(
                    EpisodeRecord(
                        episode_index=episode_idx,
                        task_text=task_text,
                        env_name=env_name,
                        states=states,
                        actions=actions,
                        model_xml=model_xml,
                    )
                )
                episode_idx += 1

    if not episodes:
        raise RuntimeError("No valid episodes were found in input demo.hdf5 files.")

    return episodes


def _compute_stats(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for stats, got shape={arr.shape}")

    arr64 = arr.astype(np.float64)
    return {
        "mean": np.mean(arr64, axis=0).astype(np.float64).tolist(),
        "std": np.std(arr64, axis=0).astype(np.float64).tolist(),
        "min": np.min(arr64, axis=0).astype(np.float64).tolist(),
        "max": np.max(arr64, axis=0).astype(np.float64).tolist(),
        "q01": np.percentile(arr64, 1, axis=0).astype(np.float64).tolist(),
        "q99": np.percentile(arr64, 99, axis=0).astype(np.float64).tolist(),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _quat_xyzw_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), ww - xx + yy - zz, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    m = mat
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quat = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm > 0:
        quat = quat / norm
    return quat


def _mj_name2id(model: Any, mujoco: Any, obj_type: Any, name: str) -> int:
    idx = int(mujoco.mj_name2id(model, obj_type, name))
    return idx


def _mj_id2name(model: Any, mujoco: Any, obj_type: Any, idx: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, idx)
    return "" if name is None else str(name)


def _joint_qpos_width_from_mj_type(mj_type: int, mujoco: Any) -> int:
    if mj_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if mj_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _get_asset_search_roots() -> list[Path]:
    global _ASSET_SEARCH_ROOTS
    if _ASSET_SEARCH_ROOTS is not None:
        return _ASSET_SEARCH_ROOTS

    repo_root = Path(__file__).resolve().parents[2]
    roots = [
        repo_root,
        repo_root / "external_dependencies",
        repo_root / "external_dependencies" / "robocasa",
        repo_root
        / "gr00t"
        / "eval"
        / "sim"
        / "robocasa"
        / "robocasa_uv"
        / ".venv"
        / "lib"
        / "python3.10"
        / "site-packages",
        Path("/home/kaveh/projects/API/RoboCasa-VR"),
    ]
    _ASSET_SEARCH_ROOTS = [r for r in roots if r.exists()]
    return _ASSET_SEARCH_ROOTS


def _collect_asset_refs(model_xml: str) -> tuple[set[str], dict[str, str]]:
    refs: set[str] = set()
    kind_by_ref: dict[str, str] = {}
    try:
        root = ET.fromstring(model_xml)
    except Exception:
        return refs, kind_by_ref

    for tag, kind in (("mesh", "mesh"), ("texture", "texture"), ("include", "include"), ("hfield", "hfield")):
        for elem in root.iter(tag):
            file_attr = elem.attrib.get("file")
            if file_attr:
                refs.add(file_attr)
                kind_by_ref[file_attr] = kind
    return refs, kind_by_ref


def _resolve_asset_path(ref: str) -> Path | None:
    if ref in _ASSET_PATH_CACHE:
        return _ASSET_PATH_CACHE[ref]

    p = Path(ref)
    if p.is_file():
        resolved = p.resolve()
        _ASSET_PATH_CACHE[ref] = resolved
        return resolved

    ref_norm = ref.replace("\\", "/")
    parts = [part for part in ref_norm.split("/") if part not in ("", ".")]
    roots = _get_asset_search_roots()

    # Try longest-to-shortest suffix matches under known roots.
    for root in roots:
        for start in range(max(0, len(parts) - 8), len(parts)):
            suffix = Path(*parts[start:])
            candidate = root / suffix
            if candidate.is_file():
                resolved = candidate.resolve()
                _ASSET_PATH_CACHE[ref] = resolved
                return resolved

    # Common package-relative fallbacks.
    for root in roots:
        for pkg in ("robosuite", "robocasa"):
            for start in range(max(0, len(parts) - 8), len(parts)):
                suffix = Path(*parts[start:])
                candidate = root / pkg / suffix
                if candidate.is_file():
                    resolved = candidate.resolve()
                    _ASSET_PATH_CACHE[ref] = resolved
                    return resolved

    # Last resort: basename search.
    basename = Path(ref_norm).name
    if basename:
        for root in roots:
            try:
                candidate = next(root.rglob(basename))
                if candidate.is_file():
                    resolved = candidate.resolve()
                    _ASSET_PATH_CACHE[ref] = resolved
                    return resolved
            except StopIteration:
                pass
            except Exception:
                pass

    _ASSET_PATH_CACHE[ref] = None
    return None


def _rewrite_xml_with_resolved_asset_paths(model_xml: str) -> str:
    try:
        root = ET.fromstring(model_xml)
    except Exception:
        return model_xml

    # Keep only the robot subtree to avoid scene-specific asset / inertia issues.
    robot_body = root.find(".//body[@name='robot0_base']")
    worldbody = root.find("worldbody")
    if robot_body is not None:
        new_worldbody = ET.Element("worldbody")
        new_worldbody.append(copy.deepcopy(robot_body))
        if worldbody is not None:
            root.remove(worldbody)
        root.append(new_worldbody)
        # Drop includes once robot body is inlined, otherwise extra scene bodies
        # can be pulled back in during MuJoCo parsing.
        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag == "include":
                    parent.remove(child)

    # Keep only assets required by the remaining robot geoms.
    asset = root.find("asset")
    if asset is not None:
        required_mesh: set[str] = set()
        required_material: set[str] = set()
        required_texture: set[str] = set()
        required_hfield: set[str] = set()

        for geom in root.iter("geom"):
            mesh_name = geom.attrib.get("mesh")
            if mesh_name:
                required_mesh.add(mesh_name)
            material_name = geom.attrib.get("material")
            if material_name:
                required_material.add(material_name)
            hfield_name = geom.attrib.get("hfield")
            if hfield_name:
                required_hfield.add(hfield_name)
            texture_name = geom.attrib.get("texture")
            if texture_name:
                required_texture.add(texture_name)

        material_by_name: dict[str, ET.Element] = {}
        for elem in asset:
            if elem.tag == "material":
                name = elem.attrib.get("name")
                if name:
                    material_by_name[name] = elem

        changed = True
        while changed:
            changed = False
            for mat_name in list(required_material):
                mat = material_by_name.get(mat_name)
                if mat is None:
                    continue
                tex_name = mat.attrib.get("texture")
                if tex_name and tex_name not in required_texture:
                    required_texture.add(tex_name)
                    changed = True

        for elem in list(asset):
            tag = elem.tag
            name = elem.attrib.get("name", "")
            keep = True
            if tag == "mesh":
                keep = name in required_mesh
            elif tag == "material":
                keep = name in required_material
            elif tag == "texture":
                keep = name in required_texture
            elif tag == "hfield":
                keep = name in required_hfield
            if not keep:
                asset.remove(elem)

    # Remove dynamics / constraint blocks not needed for forward kinematics replay.
    for tag in ("actuator", "sensor", "tendon", "equality", "contact", "keyframe"):
        for elem in list(root.findall(tag)):
            root.remove(elem)

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    # Kinematics-only compile path for robust offline replay.
    compiler.set("discardvisual", "true")
    compiler.set("inertiafromgeom", "auto")

    for tag in ("mesh", "texture", "include", "hfield"):
        for elem in root.iter(tag):
            file_attr = elem.attrib.get("file")
            if not file_attr:
                continue
            resolved = _resolve_asset_path(file_attr)
            if resolved is None:
                continue
            elem.set("file", str(resolved))

    return ET.tostring(root, encoding="unicode")


def _build_projection(model_xml: str, episode_index: int, mujoco: Any) -> EpisodeProjection:
    try:
        patched_xml = _rewrite_xml_with_resolved_asset_paths(model_xml)
        model = mujoco.MjModel.from_xml_string(patched_xml)
    except Exception as exc:
        raise RuntimeError(f"Episode {episode_index}: failed to parse model_file XML with MuJoCo ({exc})") from exc

    data = mujoco.MjData(model)

    base_site_id = _mj_name2id(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "mobilebase0_center")
    if base_site_id < 0:
        base_candidates = [
            i
            for i in range(int(model.nsite))
            if "mobilebase0" in _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, i)
            and _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, i).endswith("center")
        ]
        if not base_candidates:
            raise RuntimeError(f"Episode {episode_index}: could not find base center site in model")
        base_site_id = int(base_candidates[0])

    eef_site_id = _mj_name2id(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
    if eef_site_id < 0:
        site_candidates = [
            i
            for i in range(int(model.nsite))
            if "gripper0_right" in _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, i)
            and "grip_site" in _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, i)
        ]
        if not site_candidates:
            raise RuntimeError(f"Episode {episode_index}: could not find right-arm end-effector site")
        eef_site_id = int(site_candidates[0])

    eef_body_id = _mj_name2id(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_eef")
    if eef_body_id < 0:
        body_candidates = [
            i
            for i in range(int(model.nbody))
            if "gripper0_right" in _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, i)
            and _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, i).endswith("eef")
        ]
        if not body_candidates:
            raise RuntimeError(f"Episode {episode_index}: could not find right-arm end-effector body")
        eef_body_id = int(body_candidates[0])

    gripper_qpos_idx: list[int] = []
    candidate_joint_ids: list[int] = []
    for joint_id in range(int(model.njnt)):
        jname = _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if jname.startswith("gripper0_right_finger_joint"):
            candidate_joint_ids.append(joint_id)

    if not candidate_joint_ids:
        for joint_id in range(int(model.njnt)):
            jname = _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if jname.startswith("gripper0_right"):
                candidate_joint_ids.append(joint_id)

    if not candidate_joint_ids:
        raise RuntimeError(f"Episode {episode_index}: could not locate gripper joints in model")

    candidate_joint_ids = sorted(candidate_joint_ids, key=lambda j: _mj_id2name(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, j))
    for joint_id in candidate_joint_ids:
        adr = int(model.jnt_qposadr[joint_id])
        width = _joint_qpos_width_from_mj_type(int(model.jnt_type[joint_id]), mujoco)
        gripper_qpos_idx.extend(list(range(adr, adr + width)))

    if len(gripper_qpos_idx) < 2:
        raise RuntimeError(
            f"Episode {episode_index}: expected at least 2 gripper qpos indices, got {len(gripper_qpos_idx)}"
        )

    if len(gripper_qpos_idx) > 2:
        gripper_qpos_idx = gripper_qpos_idx[:2]

    return EpisodeProjection(
        model=model,
        data=data,
        qpos_count=int(model.nq),
        qvel_count=int(model.nv),
        gripper_qpos_idx=gripper_qpos_idx,
        base_site_id=int(base_site_id),
        eef_site_id=int(eef_site_id),
        eef_body_id=int(eef_body_id),
    )


def _project_actions_to_semantic(actions: np.ndarray, episode_index: int) -> np.ndarray:
    if actions.ndim != 2:
        raise RuntimeError(f"Episode {episode_index}: actions must be 2D, got {actions.shape}")
    if actions.shape[1] < 12:
        raise RuntimeError(f"Episode {episode_index}: expected action dim >= 12, got {actions.shape[1]}")

    out = np.zeros((actions.shape[0], ACTION_DIM), dtype=np.float32)
    out[:, 0:3] = actions[:, 0:3]
    out[:, 3:6] = actions[:, 3:6]
    out[:, 6:7] = (actions[:, 6:7] >= 0.0).astype(np.float32)
    out[:, 7:11] = actions[:, 7:11]
    out[:, 11:12] = (actions[:, 11:12] >= 0.0).astype(np.float32)
    return out


def _project_episode_to_semantic(ep: EpisodeRecord, mujoco: Any) -> EpisodeRecord:
    if not ep.model_xml:
        raise RuntimeError(f"Episode {ep.episode_index}: model_file missing")

    proj = _build_projection(ep.model_xml, episode_index=ep.episode_index, mujoco=mujoco)

    required_state_dim = 1 + proj.qpos_count + proj.qvel_count
    if ep.states.shape[1] < required_state_dim:
        raise RuntimeError(
            f"Episode {ep.episode_index}: state dim too small ({ep.states.shape[1]}) for model nq/nv ({required_state_dim})"
        )

    out_states = np.zeros((ep.states.shape[0], STATE_DIM), dtype=np.float32)
    out_actions = _project_actions_to_semantic(ep.actions, episode_index=ep.episode_index)

    for t in range(ep.states.shape[0]):
        row = ep.states[t]
        qpos = row[1 : 1 + proj.qpos_count]
        qvel = row[1 + proj.qpos_count : 1 + proj.qpos_count + proj.qvel_count]

        proj.data.qpos[:] = qpos
        proj.data.qvel[:] = qvel
        if proj.data.act.shape[0] > 0:
            proj.data.act[:] = 0.0
        mujoco.mj_forward(proj.model, proj.data)

        base_pos = np.asarray(proj.data.site_xpos[proj.base_site_id], dtype=np.float64)
        base_mat = np.asarray(proj.data.site_xmat[proj.base_site_id], dtype=np.float64).reshape(3, 3)
        base_quat = _mat_to_quat_xyzw(base_mat)

        eef_pos = np.asarray(proj.data.site_xpos[proj.eef_site_id], dtype=np.float64)
        eef_quat_xyzw = _wxyz_to_xyzw(np.asarray(proj.data.xquat[proj.eef_body_id], dtype=np.float64))
        eef_mat = _quat_xyzw_to_mat(eef_quat_xyzw)

        t_wa = np.eye(4, dtype=np.float64)
        t_wa[:3, :3] = base_mat
        t_wa[:3, 3] = base_pos

        t_wb = np.eye(4, dtype=np.float64)
        t_wb[:3, :3] = eef_mat
        t_wb[:3, 3] = eef_pos

        t_ab = np.linalg.inv(t_wa) @ t_wb
        rel_pos = t_ab[:3, 3]
        rel_quat = _mat_to_quat_xyzw(t_ab[:3, :3])
        gripper_qpos = qpos[proj.gripper_qpos_idx]

        out_states[t] = np.concatenate(
            [
                rel_pos.astype(np.float32),
                rel_quat.astype(np.float32),
                gripper_qpos.astype(np.float32),
                base_pos.astype(np.float32),
                base_quat.astype(np.float32),
            ],
            axis=0,
        )

    return EpisodeRecord(
        episode_index=ep.episode_index,
        task_text=ep.task_text,
        env_name=ep.env_name,
        states=out_states,
        actions=out_actions,
        model_xml=ep.model_xml,
    )


def _project_all_episodes(episodes: list[EpisodeRecord], mujoco: Any) -> tuple[list[EpisodeRecord], list[str]]:
    converted: list[EpisodeRecord] = []
    skipped: list[str] = []

    for ep in episodes:
        try:
            converted.append(_project_episode_to_semantic(ep, mujoco=mujoco))
        except Exception as exc:
            skipped.append(f"episode={ep.episode_index}: {exc}")

    return converted, skipped


def _build_modality() -> dict[str, Any]:
    state_modality: dict[str, Any] = {}
    cursor = 0
    for key, width in STATE_LAYOUT:
        state_modality[key] = {
            "original_key": "observation.state",
            "start": cursor,
            "end": cursor + width,
        }
        cursor += width

    action_modality: dict[str, Any] = {}
    cursor = 0
    for key, width in ACTION_LAYOUT:
        action_modality[key] = {
            "original_key": "action",
            "start": cursor,
            "end": cursor + width,
        }
        cursor += width

    return {
        "state": state_modality,
        "action": action_modality,
        "annotation": {
            "human.action.task_description": {
                "original_key": "annotation.human.action.task_description",
            },
            "human.validity": {
                "original_key": "annotation.human.validity",
            },
        },
    }


def convert(args: argparse.Namespace) -> None:
    h5py, pd, mujoco = _load_runtime_deps()

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

    if args.state_view != "semantic":
        print(
            f"Warning: --state-view={args.state_view} is ignored. "
            "This converter always exports semantic PandaOmron state for GR00T."
        )

    if args.state_dim is not None or args.action_dim is not None:
        print("Warning: --state-dim/--action-dim are ignored in semantic export mode.")

    converted, skipped = _project_all_episodes(raw_episodes, mujoco=mujoco)
    if not converted:
        if skipped:
            print("All episodes failed semantic projection. First errors:")
            for msg in skipped[:20]:
                print(f"  - {msg}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more")
        raise RuntimeError("All episodes failed semantic projection.")

    for i, ep in enumerate(converted):
        ep.episode_index = i

    valid_label = args.valid_label
    task_to_idx: dict[str, int] = {}

    def register_task(task: str) -> int:
        if task not in task_to_idx:
            task_to_idx[task] = len(task_to_idx)
        return task_to_idx[task]

    for ep in converted:
        register_task(ep.task_text)
    valid_task_idx = register_task(valid_label)

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_task_index: list[np.ndarray] = []
    all_episode_index: list[np.ndarray] = []
    all_global_index: list[np.ndarray] = []
    all_validity: list[np.ndarray] = []
    all_next_done: list[np.ndarray] = []
    all_relative_actions: list[np.ndarray] = []

    episodes_meta: list[dict[str, Any]] = []
    total_frames = 0
    global_index = 0

    for ep in converted:
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
        next_done = np.zeros((length, 1), dtype=np.float64)
        next_done[-1, 0] = 1.0

        episode_index_arr = np.full((length, 1), ep.episode_index, dtype=np.int64)
        task_index_arr = np.full((length, 1), task_idx, dtype=np.int64)
        validity_arr = np.full((length, 1), valid_task_idx, dtype=np.int64)
        global_index_arr = np.arange(global_index, global_index + length, dtype=np.int64).reshape(-1, 1)

        frame_df = pd.DataFrame(
            {
                "observation.state": [row.astype(np.float32) for row in ep.states],
                "action": [row.astype(np.float32) for row in ep.actions],
                "timestamp": timestamps[:, 0],
                "annotation.human.action.task_description": task_index_arr[:, 0],
                "task_index": task_index_arr[:, 0],
                "annotation.human.validity": validity_arr[:, 0],
                "episode_index": episode_index_arr[:, 0],
                "index": global_index_arr[:, 0],
                "next.reward": rewards[:, 0],
                "next.done": next_done[:, 0].astype(bool),
            }
        )
        frame_df.to_parquet(parquet_path, index=False)

        if length > 1:
            all_relative_actions.append(np.diff(ep.actions, axis=0))

        all_states.append(ep.states)
        all_actions.append(ep.actions)
        all_timestamps.append(timestamps)
        all_rewards.append(rewards)
        all_task_index.append(task_index_arr.astype(np.float64))
        all_episode_index.append(episode_index_arr.astype(np.float64))
        all_global_index.append(global_index_arr.astype(np.float64))
        all_validity.append(validity_arr.astype(np.float64))
        all_next_done.append(next_done)

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
    task_index_arr = np.concatenate(all_task_index, axis=0)
    episode_index_arr = np.concatenate(all_episode_index, axis=0)
    global_index_arr = np.concatenate(all_global_index, axis=0)
    validity_arr = np.concatenate(all_validity, axis=0)
    next_done_arr = np.concatenate(all_next_done, axis=0)

    if all_relative_actions:
        rel_actions_arr = np.concatenate(all_relative_actions, axis=0)
    else:
        rel_actions_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)

    tasks_rows = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(task_to_idx.items(), key=lambda kv: kv[1])
    ]

    chunk_count = max(1, (len(converted) + int(args.chunk_size) - 1) // int(args.chunk_size))
    info = {
        "codebase_version": "v2.0",
        "robot_type": args.robot_type,
        "total_episodes": len(converted),
        "total_frames": total_frames,
        "total_tasks": len(task_to_idx),
        "total_videos": 0,
        "total_chunks": chunk_count,
        "chunks_size": int(args.chunk_size),
        "fps": float(args.fps),
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {
                "dtype": "object",
                "shape": [STATE_DIM],
            },
            "action": {
                "dtype": "object",
                "shape": [ACTION_DIM],
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

    modality = _build_modality()

    stats = {
        "observation.state": _compute_stats(states_arr),
        "action": _compute_stats(actions_arr),
        "timestamp": _compute_stats(timestamps_arr),
        "next.reward": _compute_stats(rewards_arr),
        "next.done": _compute_stats(next_done_arr),
        "task_index": _compute_stats(task_index_arr),
        "episode_index": _compute_stats(episode_index_arr),
        "index": _compute_stats(global_index_arr),
        "annotation.human.action.task_description": _compute_stats(task_index_arr),
        "annotation.human.validity": _compute_stats(validity_arr),
    }

    relative_stats = {
        "action": _compute_stats(rel_actions_arr),
    }

    meta_dir = output_root / "meta"
    _write_json(meta_dir / "info.json", info)
    _write_json(meta_dir / "modality.json", modality)
    _write_json(meta_dir / "stats.json", stats)
    _write_json(meta_dir / "relative_stats.json", relative_stats)
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)

    print(f"Converted {len(converted)} episodes from {len(demo_files)} demo.hdf5 file(s)")
    print(f"Output dataset: {output_root}")
    print(f"state_dim={STATE_DIM}, action_dim={ACTION_DIM}, total_frames={total_frames}")
    if skipped:
        print(f"Skipped {len(skipped)} episodes due to projection errors.")
        for msg in skipped[:20]:
            print(f"  - {msg}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


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
        "--state-view",
        choices=["semantic", "robot", "full"],
        default="semantic",
        help="Compatibility flag. This converter always exports semantic PandaOmron state.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Compatibility flag (ignored in semantic mode)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Compatibility flag (ignored in semantic mode)",
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
