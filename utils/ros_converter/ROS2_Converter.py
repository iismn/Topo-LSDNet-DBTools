#!/usr/bin/env python3
"""ROS2 bag → OpenLaneV2-style RGB export utility.

This script extracts camera images from a ROS2 bag (rosbag2) and writes them
into an OpenLaneV2-like directory hierarchy:

	<output_root>/<split>/<city>/<bag_name>/image/<camera_name>/<timestamp>.jpg

By default, ``output_root`` is ``/home/iismn/Workspace_Share/IEEE_CVF_CVPR/Inhouse``
and ``split`` is ``train`` while ``city`` is ``Yeouido``.

Usage:

	1. Edit the "User-editable configuration" block near the top of this file
	   (bag path, output root, split, etc.).
	2. Run ``python ROS2_Converter.py``.

The script already knows the Yeouido camera topic aliases; update the mapping
dict if you need different camera names.

Requirements:
	* ROS 2 Python stack (``rosbag2_py``, ``rclpy``, ``rosidl_runtime_py``)
	* OpenCV (``cv2``) and NumPy

The script generates a ``metadata.json`` manifest inside each exported bag
directory summarizing the processed frames per camera, which can be reused
when fusing with HD map data later on.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import rosbag2_py  # type: ignore
from rclpy.serialization import deserialize_message  # type: ignore
from rosidl_runtime_py.utilities import get_message  # type: ignore
import yaml

try:  # type: ignore[attr-defined]
	from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency safeguard
	Transformer = None  # type: ignore


DEFAULT_OUTPUT_ROOT = Path("/home/iismn/Workspace_Share/IEEE_CVF_CVPR/Inhouse_V7")
DEFAULT_BAG_ROOT = Path("/media/iismn/SSD A/Dataset/inhouse/Dataset/ROSBag_Info")
DATASET_ROOT = Path(__file__).resolve().parents[2]
IMAGE_RAW_TYPE = "sensor_msgs/msg/Image"
IMAGE_COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
SUPPORTED_IMAGE_TYPES = {IMAGE_RAW_TYPE, IMAGE_COMPRESSED_TYPE}
CALIBRATION_FILE = Path("/media/iismn/SSD A/Dataset/inhouse/Dataset/Vehicle_Info/calib.yaml")
NGII_UTILS_DIR = DATASET_ROOT / "utils" / "ros_converter"
NGII_HDMAP_ROOT = DATASET_ROOT / "HDMap_Info"

# City별 SHP 경로 매핑
NGII_CITY_SHP_MAP = {
    "yeouido": NGII_HDMAP_ROOT / "Yeouido",
    "sangam": NGII_HDMAP_ROOT / "Sangam",
}

def get_ngii_shp_base_for_city(city_name: Optional[str]) -> Path:
    """city_name에 따라 적절한 NGII SHP 경로 반환"""
    if city_name:
        city_lower = city_name.lower()
        if city_lower in NGII_CITY_SHP_MAP:
            return NGII_CITY_SHP_MAP[city_lower]
    # 기본값: 첫 번째 존재하는 경로 또는 Yeouido
    for path in NGII_CITY_SHP_MAP.values():
        if path.exists():
            return path
    return NGII_HDMAP_ROOT / "Yeouido"

DEFAULT_NGII_ENABLE_ANNOTATIONS = True
DEFAULT_TRAJECTORY_ENABLE = True
DEFAULT_MAX_FRAMES_PER_CAMERA: Optional[int] = None
DEFAULT_TRAJECTORY_HORIZON_DISTANCE_M = 50.0
DEFAULT_TRAJECTORY_SAMPLE_RATE_HZ = 10.0
DEFAULT_POSE_SOURCE_EPSG = os.environ.get("ROS2_CONVERTER_POSE_EPSG", "EPSG:32652")
DEFAULT_RECTIFY_IMAGES = True
DEFAULT_STATIONARY_SKIP_ENABLE = True
DEFAULT_STATIONARY_SKIP_WINDOW_S = 3.0
DEFAULT_STATIONARY_MOVEMENT_THRESHOLD_M = 0.5
DEFAULT_NGII_VIEW_WIDTH = 50.0
DEFAULT_NGII_VIEW_LENGTH = 100.0
DEFAULT_NGII_FRONT_RATIO = 0.5
STAGING_SUBDIR_NAME = "__stage__"
CAMERA_SEQUENCE = [
	"UDP_GMSL_BL",
	"UDP_GMSL_BM",
	"UDP_GMSL_BR",
	"UDP_GMSL_FL",
	"UDP_GMSL_FM",
	"UDP_GMSL_FR",
]
ANCHOR_CAMERA = "UDP_GMSL_FM"
ODOM_TOPIC = "/NAVSIGHT/imu/odometry"
ODOM_MESSAGE_TYPE = "nav_msgs/msg/Odometry"


@dataclass
class ConverterConfig:
	bag_path: Path = DEFAULT_BAG_ROOT
	output_root: Path = DEFAULT_OUTPUT_ROOT
	split_name: str = "train"
	city_name: Optional[str] = None
	jpeg_quality: int = 100
	max_frames_per_camera: Optional[int] = DEFAULT_MAX_FRAMES_PER_CAMERA
	overwrite_existing: bool = True
	verbose_logging: bool = False
	calibration_file: Path = CALIBRATION_FILE
	enable_annotations: bool = DEFAULT_NGII_ENABLE_ANNOTATIONS
	ngii_utils_dir: Path = NGII_UTILS_DIR
	ngii_default_shp_base: Optional[Path] = None  # None이면 city_name 기반으로 자동 결정
	ngii_view_width: float = DEFAULT_NGII_VIEW_WIDTH
	ngii_view_length: float = DEFAULT_NGII_VIEW_LENGTH
	ngii_front_ratio: float = DEFAULT_NGII_FRONT_RATIO
	rectify_images: bool = DEFAULT_RECTIFY_IMAGES
	trajectory_enable: bool = DEFAULT_TRAJECTORY_ENABLE
	trajectory_horizon_distance_m: float = DEFAULT_TRAJECTORY_HORIZON_DISTANCE_M
	trajectory_sample_rate_hz: float = DEFAULT_TRAJECTORY_SAMPLE_RATE_HZ
	pose_source_epsg: str = DEFAULT_POSE_SOURCE_EPSG
	stationary_skip_enable: bool = DEFAULT_STATIONARY_SKIP_ENABLE
	stationary_skip_window_s: float = DEFAULT_STATIONARY_SKIP_WINDOW_S
	stationary_movement_threshold_m: float = DEFAULT_STATIONARY_MOVEMENT_THRESHOLD_M
	# Streaming converter knobs (safe defaults for legacy converter)
	stream_buffer_duration_ns: int = 400_000_000
	stream_sync_tolerance_ns: int = 120_000_000
	stream_allow_desync_fallback: bool = True
	stream_enable_progress_bar: bool = True
	stream_workers: Optional[int] = None
	json_only: bool = False


@dataclass(frozen=True)
class ExportStats:
	"""Aggregated statistics for a single bag export."""

	total_frames: int
	per_camera: Dict[str, int]
	topics_used: Dict[str, str]


@dataclass
class FrameCandidate:
	timestamp_ns: int
	camera_name: str
	topic: str
	staged_path: Optional[Path]
	message_type: str


@dataclass(frozen=True)
class SavedFrameRecord:
	timestamp_ns: int
	camera_name: str
	topic: str
	relative_path: str
	absolute_path: str


@dataclass(frozen=True)
class PoseRecord:
	timestamp_ns: int
	position: Tuple[float, float, float]
	quaternion: Tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraCalibration:
	name: str
	intrinsic: List[List[float]]
	distortion: List[float]
	rotation: List[List[float]]
	translation: List[float]
	distortion_model: str = "pinhole"


@dataclass(frozen=True)
class SyncedSample:
	timestamp_ns: int
	frames: Dict[str, FrameCandidate]



class _ColorFormatter(logging.Formatter):
	"""Custom formatter with bold yellow [INFO] tag"""
	YELLOW_BOLD = "\033[1;33m"
	RESET = "\033[0m"
	
	def format(self, record: logging.LogRecord) -> str:
		if record.levelno == logging.INFO:
			levelname = f"{self.YELLOW_BOLD}[INFO]{self.RESET}"
		else:
			levelname = f"[{record.levelname}]"
		return f"{levelname} {record.getMessage()}"


def _configure_logger(verbose: bool) -> logging.Logger:
	level = logging.DEBUG if verbose else logging.INFO
	logger = logging.getLogger("ros2_converter")
	logger.setLevel(level)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(_ColorFormatter())
		logger.addHandler(handler)
	logger.propagate = False
	return logger


def _discover_bag_directories(bag_path: Path) -> List[Path]:
	"""Return concrete rosbag2 directories starting at the provided path."""

	bag_path = bag_path.expanduser()
	if bag_path.is_file():
		# Allow pointing directly to the *.db3 file.
		bag_path = bag_path.parent

	def is_bag_directory(path: Path) -> bool:
		return path.is_dir() and (path / "metadata.yaml").is_file()

	if is_bag_directory(bag_path):
		return [bag_path]

	if not bag_path.exists():
		raise FileNotFoundError(f"Provided bag path does not exist: {bag_path}")

	candidates: List[Path] = [child for child in sorted(bag_path.iterdir()) if is_bag_directory(child)]
	if candidates:
		return candidates

	recursive_candidates = sorted({metadata.parent for metadata in bag_path.rglob("metadata.yaml")})
	if recursive_candidates:
		return recursive_candidates

	raise FileNotFoundError(
		f"No rosbag2 metadata.yaml found under {bag_path}. Provide a bag directory or parent folder."
	)


def _infer_city_name_from_bag_path(bag_path: Path) -> str:
	"""bag 경로에서 city name 추출.
	
	경로 구조: .../ROSBag_Info/{City}/{bag_name}/ 에서 City 추출.
	City는 NGII_CITY_SHP_MAP의 키와 매칭되어야 함 (yeouido, sangam 등).
	"""
	candidate = bag_path.expanduser()
	try:
		if candidate.is_file():
			candidate = candidate.parent
		# bag_dir의 부모 폴더가 city (e.g., .../Sangam/bag_name/ -> Sangam)
		if candidate.is_dir():
			parent_name = candidate.parent.name.lower()
			# City 폴더인지 확인 (NGII_CITY_SHP_MAP의 키와 매칭)
			if parent_name in NGII_CITY_SHP_MAP:
				return candidate.parent.name  # 원래 대소문자 유지
			# 현재 폴더명이 city인 경우도 체크
			current_name = candidate.name.lower()
			if current_name in NGII_CITY_SHP_MAP:
				return candidate.name
	except OSError:
		pass
	# Fallback: 경로에서 city 키워드 찾기
	for part in candidate.parts:
		if part.lower() in NGII_CITY_SHP_MAP:
			return part
	return "city"


def _sanitize_topic(topic: str) -> str:
	sanitized = topic.lstrip("/").replace("/", "_")
	return sanitized or "camera"


def _topic_to_camera_name(topic: str) -> str:
	stripped = topic.strip("/")
	if not stripped:
		return "camera"
	parts = stripped.split("/")
	return parts[0] if parts else "camera"


def _message_timestamp_ns(message: Any, fallback_ns: Optional[int] = None) -> int:
	header = getattr(message, "header", None)
	stamp = getattr(header, "stamp", None) if header is not None else None
	sec = getattr(stamp, "sec", None)
	nanosec = getattr(stamp, "nanosec", None)
	if sec is not None and nanosec is not None:
		try:
			return int(sec) * 1_000_000_000 + int(nanosec)
		except (TypeError, ValueError):
			pass
	if fallback_ns is not None:
		return int(fallback_ns)
	return 0


def _parse_float_list(segment: str) -> List[float]:
	head = segment.split("+-", 1)[0]
	if "[" not in head or "]" not in head:
		return []
	inside = head.split("[", 1)[-1].split("]", 1)[0]
	return [float(x) for x in inside.replace(",", " ").split() if x]


def _identity_matrix() -> List[List[float]]:
	return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _default_camera_calibration(name: str) -> CameraCalibration:
	return CameraCalibration(
		name=name,
		intrinsic=_identity_matrix(),
		distortion=[0.0, 0.0, 0.0, 0.0],
		rotation=_identity_matrix(),
		translation=[0.0, 0.0, 0.0],
		distortion_model="pinhole",
	)


def _intrinsic_matrix_from_params(params: Iterable[float]) -> List[List[float]]:
	values = list(params)
	fx, fy, cx, cy = (values + [0.0, 0.0, 0.0, 0.0])[:4]
	return [
		[fx, 0.0, cx],
		[0.0, fy, cy],
		[0.0, 0.0, 1.0],
	]


def _normalize_distortion(distortion: Iterable[float]) -> List[float]:
	values = list(distortion)
	return (values + [0.0, 0.0, 0.0, 0.0])[:4]


def _polyline_length(points: np.ndarray) -> float:
	if points.shape[0] < 2:
		return 0.0
	deltas = np.diff(points, axis=0)
	return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def _invert_se3(matrix: np.ndarray) -> np.ndarray:
	mat = np.asarray(matrix, dtype=np.float64)
	if mat.shape != (4, 4):
		raise ValueError(f"SE3 matrix must be 4x4, got {mat.shape}")
	rotation = mat[:3, :3]
	translation = mat[:3, 3]
	rot_inv = rotation.T
	trans_inv = -rot_inv @ translation
	result = np.eye(4, dtype=np.float64)
	result[:3, :3] = rot_inv
	result[:3, 3] = trans_inv
	return result


def _rotation_translation_from_se3(matrix: np.ndarray) -> Tuple[List[List[float]], List[float]]:
	mat = np.asarray(matrix, dtype=np.float64)
	if mat.shape != (4, 4):
		raise ValueError(f"SE3 matrix must be 4x4, got {mat.shape}")
	rotation = mat[:3, :3].tolist()
	translation = mat[:3, 3].tolist()
	return rotation, translation


CAM_INDEX_TO_NAME = {
	"cam0": "UDP_GMSL_BL",
	"cam1": "UDP_GMSL_BM",
	"cam2": "UDP_GMSL_BR",
	"cam3": "UDP_GMSL_FL",
	"cam4": "UDP_GMSL_FM",
	"cam5": "UDP_GMSL_FR",
}


def _camera_calibration_from_projection(name: str, projection: List[float], distortion: List[float]) -> CameraCalibration:
	intrinsic = _intrinsic_matrix_from_params(projection)
	dists = _normalize_distortion(distortion)
	return CameraCalibration(
		name=name,
		intrinsic=intrinsic,
		distortion=dists,
		rotation=_identity_matrix(),
		translation=[0.0, 0.0, 0.0],
		distortion_model="pinhole",
	)


FM_TO_EGO_TRANSLATION = np.array([1.5, 0.0, 1.7])
R_OPTICAL_TO_VEHICLE = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])


def _se3_from_rotation_translation(rotation: List[List[float]], translation: List[float]) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = np.array(rotation)
    mat[:3, 3] = np.array(translation)
    return mat


def _apply_navsim_transformation(calibs: Dict[str, CameraCalibration]) -> Dict[str, CameraCalibration]:
    fm_calib = calibs.get(ANCHOR_CAMERA)
    if not fm_calib:
        logging.getLogger("ros2_converter").warning(
            "Anchor camera %s not found in calibrations. Skipping re-centering.", ANCHOR_CAMERA
        )
        M_fm_inv = np.eye(4)
    else:
        M_fm = _se3_from_rotation_translation(fm_calib.rotation, fm_calib.translation)
        M_fm_inv = _invert_se3(M_fm)

    new_calibs = {}
    for name, calib in calibs.items():
        # Original Pose (relative to whatever cam0 was)
        M_cam = _se3_from_rotation_translation(calib.rotation, calib.translation)
        
        # Relative Pose (Cam in FM frame)
        M_rel = M_fm_inv @ M_cam
        
        R_rel = M_rel[:3, :3]
        T_rel = M_rel[:3, 3]

        # Transform translation (Position)
        # T_rel is in Optical Frame (X=Right, Y=Down, Z=Forward)
        # Convert to Vehicle Frame
        pos_vehicle_frame = R_OPTICAL_TO_VEHICLE @ T_rel
        
        # Add FM offset
        final_translation = FM_TO_EGO_TRANSLATION + pos_vehicle_frame
        
        # Transform rotation (Orientation)
        final_rotation = R_OPTICAL_TO_VEHICLE @ R_rel
        
        new_calibs[name] = CameraCalibration(
            name=calib.name,
            intrinsic=calib.intrinsic,
            distortion=calib.distortion,
            rotation=final_rotation.tolist(),
            translation=final_translation.tolist(),
            distortion_model=calib.distortion_model
        )
    return new_calibs


def load_camera_calibrations(calib_path: Path) -> Dict[str, CameraCalibration]:
	if not calib_path or not calib_path.is_file():
		return {}
	
	calibs: Dict[str, CameraCalibration] = {}
	suffix = calib_path.suffix.lower()
	if suffix in {".yaml", ".yml"}:
		try:
			calibs = _load_camera_calibrations_from_yaml(calib_path)
		except Exception as exc:  # pragma: no cover - defensive logging
			logging.getLogger("ros2_converter").warning(
				"Failed to parse calibration YAML %s (%s); falling back to legacy parser.",
				calib_path,
				exc,
			)
			calibs = _load_camera_calibrations_from_legacy(calib_path)
	else:
		calibs = _load_camera_calibrations_from_legacy(calib_path)

	return _apply_navsim_transformation(calibs)


def _load_camera_calibrations_from_legacy(calib_path: Path) -> Dict[str, CameraCalibration]:
	calibs: Dict[str, CameraCalibration] = {}
	current_cam: Optional[str] = None
	current_data: Dict[str, List[float]] = {}

	for line in calib_path.read_text(encoding="utf-8", errors="ignore").splitlines():
		stripped = line.strip()
		if not stripped:
			continue
		if stripped.startswith("cam") and "(/" in stripped:
			if current_cam and current_cam in CAM_INDEX_TO_NAME and {"projection", "distortion"}.issubset(current_data):
				alias = CAM_INDEX_TO_NAME[current_cam]
				calibs[alias] = _camera_calibration_from_projection(alias, current_data["projection"], current_data["distortion"])
			current_cam = stripped.split("(", 1)[0].strip()
			current_data = {}
			continue
		if current_cam and stripped.startswith("projection"):
			current_data["projection"] = _parse_float_list(stripped)
		elif current_cam and stripped.startswith("distortion"):
			current_data["distortion"] = _parse_float_list(stripped)

	if current_cam and current_cam in CAM_INDEX_TO_NAME and {"projection", "distortion"}.issubset(current_data):
		alias = CAM_INDEX_TO_NAME[current_cam]
		calibs[alias] = _camera_calibration_from_projection(alias, current_data["projection"], current_data["distortion"])

	return calibs


def _load_camera_calibrations_from_yaml(calib_path: Path) -> Dict[str, CameraCalibration]:
	content = yaml.safe_load(calib_path.read_text(encoding="utf-8"))
	if not isinstance(content, dict):
		return {}

	def cam_index(key: str) -> int:
		if key.startswith("cam") and key[3:].isdigit():
			return int(key[3:])
		return sys.maxsize

	ordered_keys = sorted((key for key in content.keys() if key.startswith("cam")), key=cam_index)
	calibs: Dict[str, CameraCalibration] = {}
	cumulative = np.eye(4)

	for key in ordered_keys:
		entry = content.get(key) or {}
		intrinsic = _intrinsic_matrix_from_params(entry.get("intrinsics", []))
		distortion = _normalize_distortion(entry.get("distortion_coeffs", []))
		distortion_model = str(entry.get("distortion_model", "pinhole") or "pinhole")
		alias = _topic_to_camera_name(str(entry.get("rostopic", "")))
		name = alias or key
		rotation = _identity_matrix()
		translation = [0.0, 0.0, 0.0]
		if key != "cam0":
			transform_list = entry.get("T_cn_cnm1")
			if transform_list is None:
				raise ValueError(f"Missing T_cn_cnm1 for {key} in {calib_path}")
			T_rel = np.asarray(transform_list, dtype=float)
			if T_rel.shape != (4, 4):
				raise ValueError(f"Invalid T_cn_cnm1 shape for {key}: {T_rel.shape}")
			cumulative = T_rel @ cumulative
			extrinsic_matrix = _invert_se3(cumulative)
			rotation, translation = _rotation_translation_from_se3(extrinsic_matrix)
		else:
			cumulative = np.eye(4)
		calibs[name] = CameraCalibration(
			name=name,
			intrinsic=intrinsic,
			distortion=distortion,
			rotation=rotation,
			translation=translation,
			distortion_model=distortion_model,
		)

	return calibs



def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
	norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
	if norm == 0.0:
		return _identity_matrix()
	qx /= norm
	qy /= norm
	qz /= norm
	qw /= norm
	return [
		[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
		[2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
		[2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
	]


def _quaternion_to_heading_deg(qx: float, qy: float, qz: float, qw: float) -> float:
	siny_cosp = 2.0 * (qw * qz + qx * qy)
	cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
	return math.degrees(math.atan2(siny_cosp, cosy_cosp))


class PoseLookup:
	def __init__(self, poses: List[PoseRecord]):
		self.poses = sorted(poses, key=lambda p: p.timestamp_ns)
		self.timestamps = [p.timestamp_ns for p in self.poses]

	def nearest(self, timestamp_ns: int) -> Optional[PoseRecord]:
		if not self.poses:
			return None
		idx = bisect_left(self.timestamps, timestamp_ns)
		candidates: List[PoseRecord] = []
		if idx < len(self.poses):
			candidates.append(self.poses[idx])
		if idx > 0:
			candidates.append(self.poses[idx - 1])
		return min(candidates, key=lambda pose: abs(pose.timestamp_ns - timestamp_ns)) if candidates else None

	def displacement_within_window(self, timestamp_ns: int, window_ns: int) -> Optional[float]:
		if not self.poses or window_ns <= 0:
			return None
		end_pose = self.nearest(timestamp_ns)
		if end_pose is None:
			return None
		start_ts = max(self.timestamps[0], timestamp_ns - window_ns)
		idx = bisect_left(self.timestamps, start_ts)
		if idx >= len(self.poses):
			idx = len(self.poses) - 1
		if self.poses[idx].timestamp_ns > end_pose.timestamp_ns and idx > 0:
			idx -= 1
		start_pose = self.poses[idx]
		if start_pose.timestamp_ns > end_pose.timestamp_ns:
			return None
		dx = end_pose.position[0] - start_pose.position[0]
		dy = end_pose.position[1] - start_pose.position[1]
		dz = end_pose.position[2] - start_pose.position[2]
		return math.sqrt(dx * dx + dy * dy + dz * dz)


def _image_encoding_to_numpy(encoding: str):
	encoding = encoding.lower()
	if encoding in {"bgr8", "rgb8", "mono8", "8uc1"}:
		dtype = np.uint8
	elif encoding in {"mono16", "16uc1"}:
		dtype = np.uint16
def _frame_from_raw_image(msg) -> np.ndarray:
	dtype, channels, color_format = _image_encoding_to_numpy(msg.encoding)
	expected_size = msg.height * msg.width * channels
	if len(msg.data) != expected_size * np.dtype(dtype).itemsize:
		raise ValueError(
			f"Unexpected image buffer size: got {len(msg.data)}, expected {expected_size} pixels"
		)

	frame = np.frombuffer(msg.data, dtype=dtype)
	frame = frame.reshape((msg.height, msg.width, channels)) if channels > 1 else frame.reshape((msg.height, msg.width))

	if color_format == "rgb":
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

	return frame


def _frame_from_compressed_image(msg) -> np.ndarray:
	buffer = np.frombuffer(msg.data, dtype=np.uint8)
	frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
	if frame is None:
		raise ValueError(f"Failed to decode compressed image with format '{msg.format}'")
	return frame


def _frame_from_message(msg, type_name: str) -> np.ndarray:
	if type_name == IMAGE_RAW_TYPE:
		return _frame_from_raw_image(msg)
	if type_name == IMAGE_COMPRESSED_TYPE:
		return _frame_from_compressed_image(msg)

	raise ValueError(f"Unsupported image message type: {type_name}")


def _build_rectify_maps(calibration: CameraCalibration, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
	width, height = size
	if width <= 0 or height <= 0:
		raise ValueError("Invalid image dimensions for rectification")
	K = np.asarray(calibration.intrinsic, dtype=np.float64)
	if K.shape != (3, 3):
		raise ValueError("Intrinsic matrix must be 3x3 for rectification")
	dist_coeffs = np.asarray(calibration.distortion, dtype=np.float64)
	model = (calibration.distortion_model or "pinhole").lower()
	if model in {"equidistant", "fisheye"}:
		if dist_coeffs.size == 0:
			raise ValueError("Fisheye rectification requires distortion coefficients")
		D = dist_coeffs.reshape(-1, 1)
		R = np.eye(3, dtype=np.float64)
		new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), R, balance=0.0)
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(
			K,
			D,
			R,
			new_K,
			(width, height),
			cv2.CV_16SC2,
		)
	else:
		R = np.eye(3, dtype=np.float64)
		map1, map2 = cv2.initUndistortRectifyMap(
			K,
			dist_coeffs,
			R,
			K,
			(width, height),
			cv2.CV_16SC2,
		)
	return map1, map2


def _open_reader(bag_dir: Path) -> rosbag2_py.SequentialReader:
	storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
	converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
	reader = rosbag2_py.SequentialReader()
	reader.open(storage_options, converter_options)
	return reader



def _select_image_topics(
	reader: rosbag2_py.SequentialReader,
	logger: logging.Logger,
) -> Tuple[Dict[str, str], Dict[str, type]]:
	topics_and_types = reader.get_all_topics_and_types()
	available = {entry.name: entry.type for entry in topics_and_types}

	camera_topics = {
		topic: type_name
		for topic, type_name in available.items()
		if type_name in SUPPORTED_IMAGE_TYPES
	}

	if not camera_topics:
		raise ValueError(
			"No supported image topics detected (expecting sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage)"
		)

	type_cache = {topic: get_message(type_name) for topic, type_name in camera_topics.items()}

	logger.info("Using %d image topic(s)", len(camera_topics))
	for topic in camera_topics:
		logger.debug(" • %s", topic)

	return camera_topics, type_cache


def _ensure_camera_dirs(
	output_root: Path,
	split: str,
	city: str,
	bag_name: str,
	camera_aliases: Dict[str, str],
	subdir: str = "image",
) -> Dict[str, Path]:
	base_dir = output_root / split / city / bag_name / subdir
	base_dir.mkdir(parents=True, exist_ok=True)

	directories: Dict[str, Path] = {}
	for topic, alias in camera_aliases.items():
		camera = alias or _sanitize_topic(topic)
		camera_dir = base_dir / camera
		camera_dir.mkdir(parents=True, exist_ok=True)
		directories[topic] = camera_dir

	return directories


def _sync_camera_frames(
	camera_frames: Dict[str, List[FrameCandidate]],
	anchor_camera: str,
	max_frames_per_camera: Optional[int],
) -> List[SyncedSample]:
	if anchor_camera not in camera_frames:
		return []
	sorted_frames = {
		camera: sorted(records, key=lambda r: r.timestamp_ns)
		for camera, records in camera_frames.items()
		if records
	}
	if anchor_camera not in sorted_frames or not sorted_frames[anchor_camera]:
		return []
	min_count = min(len(records) for records in sorted_frames.values())
	if max_frames_per_camera is not None:
		min_count = min(min_count, max_frames_per_camera)
	anchor_records = sorted_frames[anchor_camera][:min_count]
	timestamp_tables = {cam: [rec.timestamp_ns for rec in records] for cam, records in sorted_frames.items()}
	samples: List[SyncedSample] = []
	for anchor_record in anchor_records:
		frames_for_sample: Dict[str, FrameCandidate] = {anchor_camera: anchor_record}
		valid = True
		for cam, records in sorted_frames.items():
			if cam == anchor_camera:
				continue
			timestamps = timestamp_tables[cam]
			idx = bisect_left(timestamps, anchor_record.timestamp_ns)
			candidates: List[FrameCandidate] = []
			if idx < len(records):
				candidates.append(records[idx])
			if idx > 0:
				candidates.append(records[idx - 1])
			if not candidates:
				valid = False
				break
			best_record = min(candidates, key=lambda rec: abs(rec.timestamp_ns - anchor_record.timestamp_ns))
			frames_for_sample[cam] = best_record
		if valid and len(frames_for_sample) == len(sorted_frames):
			samples.append(SyncedSample(timestamp_ns=anchor_record.timestamp_ns, frames=frames_for_sample))
	return samples


def _build_sensor_entry(frame: SavedFrameRecord, calibration: CameraCalibration) -> Dict[str, object]:
	rel_path = frame.relative_path.replace("\\", "/")
	return {
		"timestamp_ns": frame.timestamp_ns,
		"image_path": rel_path,
		"image_path_relative": rel_path,
		"image_path_full": frame.absolute_path,
		"intrinsic": {"K": calibration.intrinsic, "distortion": calibration.distortion},
		"extrinsic": {"rotation": calibration.rotation, "translation": calibration.translation},
	}


def _normalize_annotation(annotation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
	base = {
		"lane_segment": [],
		"traffic_element": [],
		"area": [],
		"topology_lsls": [],
		"topology_lste": [],
	}
	if not annotation:
		return base
	for key in base:
		value = annotation.get(key)
		if isinstance(value, list):
			base[key] = value
	return base


def _structure_has_nonfinite(value: Any) -> bool:
	if isinstance(value, (np.floating, np.integer)):
		value = float(value)
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return not math.isfinite(float(value))
	if isinstance(value, (list, tuple)):
		return any(_structure_has_nonfinite(item) for item in value)
	if isinstance(value, dict):
		return any(_structure_has_nonfinite(item) for item in value.values())
	return False


def _filter_annotation_nonfinite(annotation: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
	removed = 0
	for key, value in annotation.items():
		if isinstance(value, list):
			filtered: List[Any] = []
			for item in value:
				if _structure_has_nonfinite(item):
					removed += 1
					continue
				filtered.append(item)
			annotation[key] = filtered
		elif _structure_has_nonfinite(value):
			removed += 1
			annotation[key] = []
	return annotation, removed


def _zero_lane_segment_z(annotation: Dict[str, Any]) -> Dict[str, Any]:
	lanes = annotation.get("lane_segment")
	if not isinstance(lanes, list):
		return annotation
	for idx, lane in enumerate(lanes):
		lanes[idx] = _zero_z_in_structure(lane)
	return annotation


def _zero_area_z(annotation: Dict[str, Any]) -> Dict[str, Any]:
	"""Zero out Z coordinates in area annotations (ped_crossing, etc.)."""
	areas = annotation.get("area")
	if not isinstance(areas, list):
		return annotation
	for idx, area in enumerate(areas):
		areas[idx] = _zero_z_in_structure(area)
	return annotation


def _zero_z_in_structure(value: Any) -> Any:
	if isinstance(value, dict):
		for key, child in value.items():
			value[key] = _zero_z_in_structure(child)
		return value
	if isinstance(value, list):
		if value and all(isinstance(item, (int, float)) for item in value):
			for index in range(2, len(value), 3):
				value[index] = 0.0
		else:
			for index, item in enumerate(value):
				value[index] = _zero_z_in_structure(item)
		return value
	if isinstance(value, tuple):
		mutable = list(value)
		mutable = _zero_z_in_structure(mutable)
		if isinstance(mutable, list):
			return tuple(mutable)
		return value
	return value



def _build_openlane_sample(
	bag_name: str,
	split: str,
	city: str,
	sample: SyncedSample,
	pose: PoseRecord,
	saved_frames: Dict[str, SavedFrameRecord],
	camera_calibrations: Dict[str, CameraCalibration],
	annotation: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
	sensors: Dict[str, object] = {}
	for camera_name in CAMERA_SEQUENCE:
		frame = saved_frames.get(camera_name)
		if not frame:
			continue
		calibration = camera_calibrations.get(camera_name, _default_camera_calibration(camera_name))
		sensors[camera_name] = _build_sensor_entry(frame, calibration)

	rotation_matrix = _quaternion_to_rotation_matrix(*pose.quaternion)
	heading_deg = _quaternion_to_heading_deg(*pose.quaternion)
	annotation_payload = _normalize_annotation(annotation)
	annotation_payload = _zero_lane_segment_z(annotation_payload)
	annotation_payload = _zero_area_z(annotation_payload)
	annotation_payload, _ = _filter_annotation_nonfinite(annotation_payload)
	return {
		"version": "OpenLaneV2_V2.0",
		"segment_id": bag_name,
		"meta_data": {
			"source": "ROS2_Inhouse",
			"source_id": bag_name,
			"city": city,
			"split": split,
			"ego_position_utm": list(pose.position),
			"ego_heading_deg": heading_deg,
		},
		"timestamp": sample.timestamp_ns,
		"sensor": sensors,
		"annotation": annotation_payload,
		"pose": {
			"rotation": rotation_matrix,
			"translation": list(pose.position),
		},
	}



def _position_to_latlon(position: Tuple[float, float, float], transformer: Optional[Any]) -> Optional[Tuple[float, float]]:
	if transformer is None:
		return None
	try:
		lon, lat = transformer.transform(position[0], position[1])  # type: ignore[call-arg]
		return float(lat), float(lon)
	except Exception:
		return None


def _build_future_trajectory(
	pose_lookup: PoseLookup,
	start_timestamp_ns: int,
	transformer: Optional[Any],
	*,
	horizon_m: float,
	sample_rate_hz: float,
) -> Tuple[np.ndarray, bool]:
	interval_ns = int(1e9 / sample_rate_hz)
	if interval_ns <= 0:
		return (
			np.empty((0, 2), dtype=np.float64),
			False,
		)
	if transformer is None or not pose_lookup.poses:
		return (
			np.empty((0, 2), dtype=np.float64),
			False,
		)
	start_pose = pose_lookup.nearest(start_timestamp_ns)
	if start_pose is None:
		return (
			np.empty((0, 2), dtype=np.float64),
			False,
		)
	start_latlon = _position_to_latlon(start_pose.position, transformer)
	if start_latlon is None:
		return (
			np.empty((0, 2), dtype=np.float64),
			False,
		)
	max_timestamp = pose_lookup.timestamps[-1]
	accumulated_distance = 0.0
	current_timestamp = start_timestamp_ns
	prev_position = start_pose.position
	latlon_samples: List[Tuple[float, float]] = [start_latlon]
	while accumulated_distance < horizon_m:
		current_timestamp += interval_ns
		if current_timestamp > max_timestamp:
			return (
				np.empty((0, 2), dtype=np.float64),
				False,
			)
		future_pose = pose_lookup.nearest(current_timestamp)
		if future_pose is None:
			return (
				np.empty((0, 2), dtype=np.float64),
				False,
			)
		curr_position = future_pose.position
		dx = curr_position[0] - prev_position[0]
		dy = curr_position[1] - prev_position[1]
		accumulated_distance += math.hypot(dx, dy)
		latlon = _position_to_latlon(curr_position, transformer)
		if latlon is None:
			return (
				np.empty((0, 2), dtype=np.float64),
				False,
			)
		latlon_samples.append(latlon)
		prev_position = curr_position
	if accumulated_distance < horizon_m or len(latlon_samples) < 2:
		return (
			np.empty((0, 2), dtype=np.float64),
			False,
		)
	return (
		np.asarray(latlon_samples, dtype=np.float64),
		True,
	)


class ROS2OpenLaneConverter:
	"""Class wrapper for ROS2 → OpenLane conversion."""

	def __init__(self, config: ConverterConfig):
		self.config = config
		if not self.config.city_name:
			self.config.city_name = _infer_city_name_from_bag_path(self.config.bag_path)
		self.logger = _configure_logger(config.verbose_logging)
		self.camera_calibrations = load_camera_calibrations(config.calibration_file)
		self.camera_sequence = list(CAMERA_SEQUENCE)
		self.anchor_camera = ANCHOR_CAMERA
		self._ngii_converter: Optional[Any] = None
		self._ngii_converter_initialized = False
		self._pose_transformer: Optional[Any] = None
		self._pose_transformer_initialized = False
		self._rectify_maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
		self._rectify_failures: set[str] = set()
		self._maybe_set_ngii_env()
		self._initialize_ngii_converter()

	def _get_ngii_shp_base(self) -> Path:
		"""city_name 기반으로 NGII SHP 경로 반환"""
		if self.config.ngii_default_shp_base is not None:
			return self.config.ngii_default_shp_base
		return get_ngii_shp_base_for_city(self.config.city_name)

	def _maybe_set_ngii_env(self) -> None:
		if not self.config.enable_annotations:
			return
		shp_base = self._get_ngii_shp_base()
		if shp_base.exists():
			os.environ["NGII_SHP_DIR"] = str(shp_base)

	def _initialize_ngii_converter(self) -> Optional[Any]:
		if self._ngii_converter_initialized:
			return self._ngii_converter
		self._ngii_converter_initialized = True
		if not self.config.enable_annotations:
			return None
		utils_dir = self.config.ngii_utils_dir
		if not utils_dir.exists():
			self.logger.warning("NGII utils directory not found at %s; annotations disabled.", utils_dir)
			return None
		if str(utils_dir) not in sys.path:
			sys.path.append(str(utils_dir))
		try:  # type: ignore[import]
			from ngii_openlane_converter import NGIIToOpenLaneConverter, set_quiet, _find_shp_in  # type: ignore
		except Exception as exc:  # pragma: no cover - import guard
			self.logger.warning("Failed to import NGII converter (%s); annotations disabled.", exc)
			return None
		try:
			set_quiet(True)  # type: ignore
		except Exception:
			pass
		try:
			# SHP 경로를 직접 지정하여 로드 (환경변수 캐시 문제 우회)
			shp_base = self._get_ngii_shp_base()
			shp_base_str = str(shp_base)
			
			link_shp = _find_shp_in(shp_base_str, "*a2_link*.shp")
			surfacemark_shp = _find_shp_in(shp_base_str, "*b3_surfacemark*.shp")
			surfacelinemark_shp = _find_shp_in(shp_base_str, "*b2_surfacelinemark*.shp")
			
			converter = NGIIToOpenLaneConverter(
				view_width=self.config.ngii_view_width,
				view_length=self.config.ngii_view_length,
				front_ratio=self.config.ngii_front_ratio,
			)
			converter.load_ngii_links(link_shp)
			converter.load_ngii_surfacemarks(surfacemark_shp)
			converter.load_ngii_surfacelinemarks(surfacelinemark_shp)
		except Exception as exc:
			self.logger.warning("Failed to load NGII shapefiles (%s); annotations disabled.", exc)
			return None
		self._ngii_converter = converter
		return self._ngii_converter

	def _get_pose_to_wgs_transformer(self) -> Optional[Any]:
		if not self.config.trajectory_enable:
			return None
		if self._pose_transformer_initialized:
			return self._pose_transformer
		self._pose_transformer_initialized = True
		if Transformer is None:
			self.logger.warning("pyproj is unavailable; trajectory export disabled.")
			return None
		try:
			self._pose_transformer = Transformer.from_crs(
				self.config.pose_source_epsg,
				"EPSG:4326",
				always_xy=True,
			)  # type: ignore[attr-defined]
		except Exception as exc:  # pragma: no cover - defensive logging
			self.logger.warning("Failed to configure pose transformer (%s); trajectory export disabled.", exc)
			self._pose_transformer = None
		return self._pose_transformer

	def _rectify_frame_if_enabled(
		self,
		frame: np.ndarray,
		calibration: Optional[CameraCalibration],
		camera_name: str,
	) -> np.ndarray:
		if not self.config.rectify_images or calibration is None:
			return frame
		if frame.ndim < 2:
			return frame
		height, width = frame.shape[:2]
		cache_key = f"{camera_name}:{width}x{height}"
		if cache_key in self._rectify_failures:
			return frame
		maps = self._rectify_maps.get(cache_key)
		if maps is None:
			try:
				maps = _build_rectify_maps(calibration, (width, height))
				self._rectify_maps[cache_key] = maps
			except Exception as exc:
				self.logger.debug("Rectification disabled for %s (%s)", camera_name, exc)
				self._rectify_failures.add(cache_key)
				return frame
		map1, map2 = maps
		return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

	def _build_ngii_annotation_payload(self, pose: PoseRecord) -> Optional[Dict[str, Any]]:
		converter = self._initialize_ngii_converter()
		if converter is None:
			return None
		heading_deg = _quaternion_to_heading_deg(*pose.quaternion)
		try:
			data = converter.convert_to_openlane_format(  # type: ignore[attr-defined]
				ego_x=pose.position[0],
				ego_y=pose.position[1],
				ego_heading_deg=heading_deg,
				segment_id="ngii_overlay",
				timestamp=pose.timestamp_ns,
			)
		except Exception as exc:
			self.logger.debug("NGII annotation generation failed at %d: %s", pose.timestamp_ns, exc)
			return None
		if not data:
			return None
		annotation = _normalize_annotation(data.get("annotation"))
		annotation = _zero_lane_segment_z(annotation)
		annotation, removed = _filter_annotation_nonfinite(annotation)
		if removed:
			self.logger.debug("Filtered %d invalid annotation entrie(s) at %d", removed, pose.timestamp_ns)
		return annotation

	def _save_trajectory_payload(
		self,
		traj_dir: Optional[Path],
		sample_timestamp_ns: int,
		pose: PoseRecord,
		transformer: Optional[Any],
		trajectory_latlon: np.ndarray,
	) -> None:
		if not self.config.trajectory_enable or traj_dir is None:
			return
		traj_dir.mkdir(parents=True, exist_ok=True)
		trajectory_path = traj_dir / f"{sample_timestamp_ns}.npy"
		np.save(trajectory_path, trajectory_latlon)
		latlon = _position_to_latlon(pose.position, transformer)
		if latlon is None:
			ego_latlon = np.array([np.nan, np.nan], dtype=np.float64)
		else:
			ego_latlon = np.array(latlon, dtype=np.float64)
		pose_path = traj_dir / f"{sample_timestamp_ns}_egopose.npy"
		np.save(pose_path, ego_latlon)

	def _stage_frame(
		self,
		stage_dir: Path,
		timestamp_ns: int,
		frame: np.ndarray,
	) -> Optional[Path]:
		stage_dir.mkdir(parents=True, exist_ok=True)
		stage_path = stage_dir / f"{timestamp_ns}.jpg"
		success = cv2.imwrite(
			str(stage_path),
			frame,
			[int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)],
		)
		if not success:
			self.logger.error("Failed to stage image %s", stage_path)
			return None
		return stage_path

	def _write_sample_frames(
		self,
		bag_name: str,
		sample: SyncedSample,
		camera_dirs: Dict[str, Path],
		manifest: List[Dict[str, Any]],
		per_camera_counts: Dict[str, int],
	) -> Dict[str, SavedFrameRecord]:
		saved: Dict[str, SavedFrameRecord] = {}
		sample_timestamp = sample.timestamp_ns
		for camera_name in CAMERA_SEQUENCE:
			frame = sample.frames.get(camera_name)
			if frame is None:
				continue
			camera_dir = camera_dirs.get(frame.topic)
			if camera_dir is None:
				self.logger.debug("No output directory for %s (topic %s)", camera_name, frame.topic)
				continue
			camera_dir.mkdir(parents=True, exist_ok=True)
			filename = f"{sample_timestamp}.jpg"
			destination = camera_dir / filename
			if not self.config.json_only:
				stage_path = frame.staged_path
				if stage_path is None or not stage_path.exists():
					self.logger.error("Staged image missing for %s at %s", camera_name, stage_path)
					continue
				try:
					shutil.copy2(stage_path, destination)
				except Exception as exc:
					self.logger.error("Failed to copy %s → %s (%s)", stage_path, destination, exc)
					continue
			rel_path = destination.relative_to(self.config.output_root).as_posix()
			absolute_path = str(destination)
			record = SavedFrameRecord(
				timestamp_ns=sample_timestamp,
				camera_name=camera_name,
				topic=frame.topic,
				relative_path=rel_path,
				absolute_path=absolute_path,
			)
			saved[camera_name] = record
			per_camera_counts[camera_name] = per_camera_counts.get(camera_name, 0) + 1
			manifest.append(
				{
					"timestamp_ns": sample_timestamp,
					"topic": frame.topic,
					"message_type": frame.message_type,
					"camera": camera_name,
					"relative_path": rel_path,
					"segment": bag_name,
				}
			)
		return saved

	def _generate_openlane_samples(
		self,
		bag_name: str,
		camera_frames: Dict[str, List[FrameCandidate]],
		odometry_records: List[PoseRecord],
		camera_dirs: Dict[str, Path],
		manifest: List[Dict[str, Any]],
		per_camera_counts: Dict[str, int],
	) -> int:
		if not camera_frames:
			self.logger.warning("No camera frames recorded; skipping JSON export")
			return 0
		if not odometry_records:
			self.logger.warning("No odometry records detected; skipping JSON export")
			return 0
		samples = _sync_camera_frames(
			camera_frames,
			self.anchor_camera,
			self.config.max_frames_per_camera,
		)
		if not samples:
			self.logger.warning("Unable to build synced samples (anchor=%s)", self.anchor_camera)
			return 0
		pose_lookup = PoseLookup(odometry_records)
		transformer = self._get_pose_to_wgs_transformer()
		stationary_window_ns = int(max(0.0, self.config.stationary_skip_window_s) * 1e9)
		stationary_threshold_m = max(0.0, self.config.stationary_movement_threshold_m)
		stationary_skip_enabled = self.config.stationary_skip_enable and stationary_window_ns > 0
		info_dir = self.config.output_root / self.config.split_name / self.config.city_name / bag_name / "info"
		info_dir.mkdir(parents=True, exist_ok=True)
		traj_dir: Optional[Path] = None
		if self.config.trajectory_enable:
			traj_dir = self.config.output_root / self.config.split_name / self.config.city_name / bag_name / "traj"
		generated = 0
		saved_frames_total = 0
		for sample in samples:
			pose = pose_lookup.nearest(sample.timestamp_ns)
			if pose is None:
				self.logger.warning("No odometry match for timestamp %d; skipping", sample.timestamp_ns)
				continue
			if stationary_skip_enabled:
				displacement = pose_lookup.displacement_within_window(sample.timestamp_ns, stationary_window_ns)
				if displacement is not None and displacement <= stationary_threshold_m:
					self.logger.debug(
						"Skipping %d: vehicle stationary for %.1fs (Δ=%.3fm)",
						sample.timestamp_ns,
						self.config.stationary_skip_window_s,
						displacement,
					)
					continue
			trajectory_latlon, has_future = _build_future_trajectory(
				pose_lookup,
				sample.timestamp_ns,
				transformer,
				horizon_m=self.config.trajectory_horizon_distance_m,
				sample_rate_hz=self.config.trajectory_sample_rate_hz,
			)
			if not has_future:
				self.logger.debug(
					"Skipping %d: insufficient future coverage for %.1fm horizon",
					sample.timestamp_ns,
					self.config.trajectory_horizon_distance_m,
				)
				continue
			saved_frames = self._write_sample_frames(
				bag_name,
				sample,
				camera_dirs,
				manifest,
				per_camera_counts,
			)
			if not saved_frames:
				continue
			annotation = self._build_ngii_annotation_payload(pose)
			payload = _build_openlane_sample(
				bag_name,
				self.config.split_name,
				self.config.city_name,
				sample,
				pose,
				saved_frames,
				self.camera_calibrations,
				annotation,
			)
			output_path = info_dir / f"{sample.timestamp_ns}-ls.json"
			output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
			self._save_trajectory_payload(
				traj_dir,
				sample.timestamp_ns,
				pose,
				transformer,
				trajectory_latlon,
			)
			generated += 1
			saved_frames_total += len(saved_frames)
		self.logger.info("OpenLane Samples : %d (폴더: %s)", generated, info_dir)
		return saved_frames_total
	def _reinitialize_ngii_for_city(self, city_name: str) -> None:
		"""city가 변경되면 NGII converter를 재초기화"""
		if not self.config.enable_annotations:
			return
		# 이전 city와 같으면 스킵
		if hasattr(self, '_current_ngii_city') and self._current_ngii_city == city_name.lower():
			return
		
		self._current_ngii_city = city_name.lower()
		
		# config.city_name도 업데이트 (중요: _get_ngii_shp_base()가 이 값을 사용)
		self.config.city_name = city_name
		
		shp_base = get_ngii_shp_base_for_city(city_name)
		
		if not shp_base.exists():
			self.logger.warning("NGII SHP directory not found for city %s: %s", city_name, shp_base)
			return
		
		# 환경변수 업데이트
		os.environ["NGII_SHP_DIR"] = str(shp_base)
		self.logger.info("NGII SHP : %s (%s)", shp_base, city_name)
		
		# NGII converter 재초기화
		self._ngii_converter = None
		self._ngii_converter_initialized = False
		self._initialize_ngii_converter()

	def export_bag(self, bag_dir: Path) -> ExportStats:
		bag_name = bag_dir.name
		
		# bag 경로에서 city 추출 및 NGII 재초기화
		bag_city = _infer_city_name_from_bag_path(bag_dir)
		if bag_city and bag_city.lower() in NGII_CITY_SHP_MAP:
			self._reinitialize_ngii_for_city(bag_city)
			# city_name도 bag 기반으로 업데이트
			self.config.city_name = bag_city
		
		target_dir = self.config.output_root / self.config.split_name / self.config.city_name / bag_name
		if target_dir.exists() and not self.config.overwrite_existing:
			raise FileExistsError(
				f"Target directory already exists: {target_dir}. Use --overwrite to replace it."
			)

		reader = _open_reader(bag_dir)
		camera_topics, type_cache = _select_image_topics(reader, self.logger)
		odom_msg_type = get_message(ODOM_MESSAGE_TYPE)

		camera_aliases = {topic: _topic_to_camera_name(topic) for topic in camera_topics}
		camera_dirs = _ensure_camera_dirs(
			self.config.output_root,
			self.config.split_name,
			self.config.city_name,
			bag_name,
			camera_aliases,
		)
		stage_dirs: Dict[str, Path] = {}
		stage_root: Optional[Path] = None
		if not self.config.json_only:
			stage_dirs = _ensure_camera_dirs(
				self.config.output_root,
				self.config.split_name,
				self.config.city_name,
				bag_name,
				camera_aliases,
				subdir=STAGING_SUBDIR_NAME,
			)
			stage_root = (
				self.config.output_root
				/ self.config.split_name
				/ self.config.city_name
				/ bag_name
				/ STAGING_SUBDIR_NAME
			)

		manifest: List[Dict[str, Any]] = []
		candidate_counts: Dict[str, int] = {alias: 0 for alias in camera_aliases.values()}
		cameras_capped: set[str] = set()
		camera_frames: Dict[str, List[FrameCandidate]] = defaultdict(list)
		odometry_records: List[PoseRecord] = []
		odometry_timestamps: List[int] = []
		latest_frame_timestamp: Optional[int] = None
		all_cameras_capped_logged = False

		def all_cameras_capped() -> bool:
			return (
				self.config.max_frames_per_camera is not None
				and len(cameras_capped) == len(camera_aliases)
			)

		def odom_has_coverage() -> bool:
			if latest_frame_timestamp is None or not odometry_records:
				return False
			if not self.config.trajectory_enable:
				return odometry_records[-1].timestamp_ns >= latest_frame_timestamp
			if not odometry_timestamps:
				return False
			idx = bisect_left(odometry_timestamps, latest_frame_timestamp)
			candidate_indices: List[int] = []
			if idx < len(odometry_records):
				candidate_indices.append(idx)
			if idx > 0:
				candidate_indices.append(idx - 1)
			if not candidate_indices:
				return False
			start_index = min(
				candidate_indices,
				key=lambda i: abs(odometry_timestamps[i] - latest_frame_timestamp),
			)
			prev_pose = odometry_records[start_index]
			if start_index == len(odometry_records) - 1:
				return False
			target_distance = self.config.trajectory_horizon_distance_m
			travelled = 0.0
			for future_pose in odometry_records[start_index + 1 :]:
				dx = future_pose.position[0] - prev_pose.position[0]
				dy = future_pose.position[1] - prev_pose.position[1]
				travelled += math.hypot(dx, dy)
				if travelled >= target_distance:
					return True
				prev_pose = future_pose
			return False

		self.logger.info("Export : %s → %s", bag_name, target_dir)

		while reader.has_next():
			topic, raw, timestamp = reader.read_next()
			if topic == ODOM_TOPIC:
				storage_timestamp_ns = int(timestamp)
				msg = deserialize_message(raw, odom_msg_type)
				msg_timestamp_ns = _message_timestamp_ns(msg, storage_timestamp_ns)
				pose = msg.pose.pose
				position = pose.position
				orientation = pose.orientation
				record = PoseRecord(
					timestamp_ns=msg_timestamp_ns,
					position=(float(position.x), float(position.y), float(position.z)),
					quaternion=(
						float(orientation.x),
						float(orientation.y),
						float(orientation.z),
						float(orientation.w),
					),
				)
				odometry_records.append(record)
				odometry_timestamps.append(record.timestamp_ns)
				if all_cameras_capped() and odom_has_coverage():
					self.logger.info(
						"Odometry Coverage : %d (early stop)",
						latest_frame_timestamp or 0,
					)
					break
				continue
			if topic not in camera_topics:
				continue

			storage_timestamp_ns = int(timestamp)
			msg_type = type_cache[topic]
			msg = deserialize_message(raw, msg_type)
			frame_timestamp_ns = _message_timestamp_ns(msg, storage_timestamp_ns)

			camera_name = camera_aliases[topic]
			stage_path: Optional[Path] = None
			if not self.config.json_only:
				try:
					frame = _frame_from_message(msg, camera_topics[topic])
				except ValueError as exc:
					self.logger.warning("Skipping frame at %d on %s (%s)", frame_timestamp_ns, topic, exc)
					continue

				calibration = self.camera_calibrations.get(camera_name)
				if calibration is not None:
					frame = self._rectify_frame_if_enabled(frame, calibration, camera_name)
			if (
				self.config.max_frames_per_camera is not None
				and candidate_counts[camera_name] >= self.config.max_frames_per_camera
			):
				if camera_name not in cameras_capped:
					cameras_capped.add(camera_name)
				if all_cameras_capped() and not all_cameras_capped_logged:
					self.logger.info(
						"Frame Cap : %d/cam × %d cams (odom만 계속)",
						self.config.max_frames_per_camera,
						len(camera_aliases),
					)
					all_cameras_capped_logged = True
				continue
			candidate_counts[camera_name] += 1
			if latest_frame_timestamp is None or frame_timestamp_ns > latest_frame_timestamp:
				latest_frame_timestamp = frame_timestamp_ns
			if not self.config.json_only:
				stage_dir = stage_dirs.get(topic)
				if stage_dir is None:
					self.logger.debug("No staging directory for %s (topic %s)", camera_name, topic)
					continue
				stage_path = self._stage_frame(stage_dir, frame_timestamp_ns, frame)
				if stage_path is None:
					continue
			camera_frames[camera_name].append(
				FrameCandidate(
					timestamp_ns=frame_timestamp_ns,
					camera_name=camera_name,
					topic=topic,
					staged_path=stage_path,
					message_type=camera_topics[topic],
				),
			)

			if (
				self.config.max_frames_per_camera is not None
				and candidate_counts[camera_name] >= self.config.max_frames_per_camera
			):
				cameras_capped.add(camera_name)
				if all_cameras_capped() and not all_cameras_capped_logged:
					self.logger.info(
						"Frame Cap : %d/cam × %d cams (odom만 계속)",
						self.config.max_frames_per_camera,
						len(camera_aliases),
					)
					all_cameras_capped_logged = True

		saved_per_camera_counts: Dict[str, int] = {alias: 0 for alias in camera_aliases.values()}
		total_saved_frames = 0
		try:
			total_saved_frames = self._generate_openlane_samples(
				bag_name,
				camera_frames,
				odometry_records,
				camera_dirs,
				manifest,
				saved_per_camera_counts,
			)
		finally:
			if stage_root and stage_root.exists():
				shutil.rmtree(stage_root, ignore_errors=True)

		metadata = {
			"bag_name": bag_name,
			"bag_path": str(bag_dir),
			"split": self.config.split_name,
			"city": self.config.city_name,
			"total_frames": total_saved_frames,
			"cameras": saved_per_camera_counts,
			"topics": camera_topics,
			"frames": manifest,
		}

		metadata_path = target_dir / "metadata.json"
		metadata_path.parent.mkdir(parents=True, exist_ok=True)
		metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

		self.logger.info(
			"Complete : %s (%d frames × %d cams)",
			bag_name,
			total_saved_frames,
			len(saved_per_camera_counts),
		)

		return ExportStats(
			total_frames=total_saved_frames,
			per_camera=saved_per_camera_counts,
			topics_used=camera_topics,
		)

	def run(self) -> int:
		bag_dirs = _discover_bag_directories(self.config.bag_path)
		if len(bag_dirs) > 1:
			self.logger.info("Bags Found : %d in %s", len(bag_dirs), self.config.bag_path)

		for bag_dir in bag_dirs:
			try:
				self.export_bag(bag_dir)
			except Exception as exc:  # pragma: no cover - runtime safeguard
				self.logger.error("Failed to export %s: %s", bag_dir, exc)
				return 1

		self.logger.info("Status : All bags processed successfully")
		return 0


def _parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="ROS2 bag → OpenLaneV2 RGB exporter")
	parser.add_argument(
		"--json-only",
		action="store_true",
		help="Skip writing image files and only generate JSON/info/trajectory outputs.",
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = _parse_arguments(argv)
	config = ConverterConfig(json_only=args.json_only)
	converter = ROS2OpenLaneConverter(config)
	return converter.run()


if __name__ == "__main__":
	sys.exit(main())

