#!/usr/bin/env python3
"""ROS2 bag → OpenLaneV2 exporter that streams frames without staging."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Deque, Dict, List, Optional
import os

import cv2  # type: ignore
import numpy as np
import yaml
from rclpy.serialization import deserialize_message  # type: ignore
from rosidl_runtime_py.utilities import get_message  # type: ignore
from tqdm import tqdm

# 현재 파일 기준 converter 패키지를 찾을 수 있도록 경로 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
	sys.path.append(str(current_dir))

from ros_converter.ROS2_Converter import (  # type: ignore[import]
	ANCHOR_CAMERA,
	CAMERA_SEQUENCE,
	ODOM_MESSAGE_TYPE,
	ODOM_TOPIC,
	CameraCalibration,
	ConverterConfig,
	ExportStats,
	PoseLookup,
	PoseRecord,
	SavedFrameRecord,
	ROS2OpenLaneConverter,
	_build_future_trajectory,
	_build_openlane_sample,
	_build_rectify_maps,
	_ensure_camera_dirs,
	_message_timestamp_ns,
	_frame_from_message,
	_open_reader,
	_select_image_topics,
	_topic_to_camera_name,
	_discover_bag_directories,
)


def _worker_process_bag(config: ConverterConfig, bag_dir: Path, position: int = 0) -> int:
	"""Worker function for processing a single bag in a separate process."""
	# Disable individual progress bars in worker processes to avoid console chaos
	# config.stream_enable_progress_bar = False
	
	# Adjust worker threads per process if needed to avoid explosion
	# If not explicitly set, we might want to reduce it, but let's trust the OS scheduler for now
	# or the user can set --workers explicitly.
	
	converter = StreamingROS2OpenLaneConverter(config, progress_position=position)
	try:
		stats = converter.export_bag(bag_dir)
		return stats.total_frames
	except Exception as exc:
		# Use print since logger might not be configured in worker
		print(f"[ERROR] Failed to export {bag_dir.name}: {exc}")
		return 0


@dataclass
class InlineFrame:
	"""JPEG-compressed frame buffered in memory until the sample is emitted."""

	timestamp_ns: int
	camera_name: str
	topic: str
	message_type: str
	jpeg_bytes: Optional[bytes] = None
	processing_future: Optional[Future] = None


@dataclass
class PendingSample:
	"""Multi-camera bundle aligned on the anchor timestamp."""

	timestamp_ns: int
	frames: Dict[str, InlineFrame]


class StreamingPoseLookup:
	"""Optimized PoseLookup for streaming that avoids O(N) copy/sort on every check."""

	def __init__(self, poses: List[PoseRecord], timestamps: List[int]):
		self.poses = poses
		self.timestamps = timestamps

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
		if not self.timestamps:
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
		return (dx * dx + dy * dy + dz * dz) ** 0.5


class StreamingROS2OpenLaneConverter(ROS2OpenLaneConverter):
	"""Converter that streams frames directly into their final OpenLane folders."""

	SYNC_TOLERANCE_NS = 50_000_000
	BUFFER_DURATION_NS = 200_000_000
	MIN_INFO_FILES = 20

	def __init__(self, config: ConverterConfig, progress_position: Optional[int] = None):
		super().__init__(config)
		self.progress_position = progress_position
		self.available_camera_names: List[str] = []
		self.sync_tolerance_ns = getattr(config, "stream_sync_tolerance_ns", self.SYNC_TOLERANCE_NS)
		self.buffer_duration_ns = getattr(config, "stream_buffer_duration_ns", self.BUFFER_DURATION_NS)
		self.allow_desync_fallback = getattr(config, "stream_allow_desync_fallback", True)
		self.progress_enabled = getattr(config, "stream_enable_progress_bar", True)
		
		workers = getattr(config, "stream_workers", None)
		if workers is None or workers < 1:
			workers = os.cpu_count() or 1
			
		# Image processing executor (Leaf tasks)
		self.executor = ThreadPoolExecutor(max_workers=workers)
		# Sample processing executor (Orchestrator tasks - waits for images)
		self.sample_executor = ThreadPoolExecutor(max_workers=workers)
		self._rectify_lock = threading.Lock()
		self._manifest_lock = threading.Lock()
		self._sample_futures: List[Future] = []
		
		self.logger.info("Initialized streaming converter with %d parallel workers", workers)

	def run(self) -> int:
		bag_dirs = _discover_bag_directories(self.config.bag_path)
		parallelism = getattr(self.config, "stream_bag_parallelism", 1)

		if parallelism <= 1:
			# Sequential processing (legacy behavior)
			if len(bag_dirs) > 1:
				self.logger.info("Found %d bag directories under %s", len(bag_dirs), self.config.bag_path)
			for bag_dir in bag_dirs:
				try:
					self.export_bag(bag_dir)
				except Exception as exc:
					self.logger.error("Failed to export %s: %s", bag_dir, exc)
					return 1
			self.logger.info("All bag(s) processed successfully.")
			return 0

		# Parallel processing
		self.logger.info("Found %d bag directories. Processing %d bags in parallel...", len(bag_dirs), parallelism)
		
		total_frames = 0
		failed_bags = 0
		
		with ProcessPoolExecutor(max_workers=parallelism) as executor:
			futures = {}
			for i, bag_dir in enumerate(bag_dirs):
				pos = (i % parallelism) + 1
				futures[executor.submit(_worker_process_bag, self.config, bag_dir, pos)] = bag_dir
			
			with tqdm(total=len(bag_dirs), desc="Total Progress", unit="bag", position=0) as pbar:
				for future in as_completed(futures):
					bag_dir = futures[future]
					try:
						frames = future.result()
						total_frames += frames
					except Exception as exc:
						self.logger.error("Bag %s generated an exception: %s", bag_dir.name, exc)
						failed_bags += 1
					pbar.update(1)
		
		self.logger.info("Parallel processing complete. Total frames: %d. Failed bags: %d", total_frames, failed_bags)
		return 1 if failed_bags > 0 else 0

	def _process_image_task(
		self,
		message,
		topic: str,
		camera_name: str,
		msg_type: str,
	) -> bytes:
		try:
			frame = _frame_from_message(message, msg_type)
		except ValueError as exc:
			self.logger.warning("Skipping %s (%s)", topic, exc)
			return b""

		calibration = self.camera_calibrations.get(camera_name)
		frame = self._rectify_frame_thread_safe(frame, calibration, camera_name)
		return self._encode_frame_to_jpeg(frame)

	def _rectify_frame_thread_safe(
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

		map1, map2 = None, None
		with self._rectify_lock:
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

	def export_bag(self, bag_dir: Path) -> ExportStats:
		bag_name = bag_dir.name
		# Infer city from bag's parent directory (e.g., Sangam, Yeouido)
		city_name = bag_dir.parent.name
		# Skip if parent is the root bag folder itself (e.g., ROSBag_Info)
		if city_name == self.config.bag_path.name:
			city_name = self.config.city_name or "unknown"
		
		# NGII converter 재초기화 (city 변경 시)
		self._reinitialize_ngii_for_city(city_name)
		self.config.city_name = city_name
		
		target_dir = (
			self.config.output_root
			/ self.config.split_name
			/ city_name
			/ bag_name
		)
		target_dir.mkdir(parents=True, exist_ok=True)
		
		# Store for use by helper methods
		self._current_city_name = city_name

		reader = _open_reader(bag_dir)
		camera_topics, type_cache = _select_image_topics(reader, self.logger)
		odom_msg_type = get_message(ODOM_MESSAGE_TYPE)
		progress = self._create_progress_bar(bag_dir, bag_name)

		camera_aliases = {topic: _topic_to_camera_name(topic) for topic in camera_topics}
		available_aliases = {alias for alias in camera_aliases.values() if alias}
		self.available_camera_names = [
			name for name in CAMERA_SEQUENCE if name in available_aliases
		]
		if not self.available_camera_names:
			raise RuntimeError("No camera topics match the configured CAMERA_SEQUENCE.")
		if self.anchor_camera not in self.available_camera_names:
			fallback = self.available_camera_names[0]
			self.logger.warning(
				"Anchor camera %s missing in bag; falling back to %s",
				self.anchor_camera,
				fallback,
			)
			self.anchor_camera = fallback

		camera_dirs_by_topic = _ensure_camera_dirs(
			self.config.output_root,
			self.config.split_name,
			city_name,
			bag_name,
			camera_aliases,
		)
		camera_dirs = {
			camera_aliases[topic]: path
			for topic, path in camera_dirs_by_topic.items()
			if camera_aliases[topic]
		}

		manifest: List[Dict[str, object]] = []
		per_camera_counts: Dict[str, int] = {name: 0 for name in self.available_camera_names}
		candidate_counts: Dict[str, int] = {name: 0 for name in self.available_camera_names}
		camera_buffers: Dict[str, Deque[InlineFrame]] = {
			name: deque() for name in self.available_camera_names
		}
		pending_samples: Deque[PendingSample] = deque()
		odometry_records: List[PoseRecord] = []
		odometry_timestamps: List[int] = []
		total_saved_frames = 0

		def cameras_saturated() -> bool:
			return (
				self.config.max_frames_per_camera is not None
				and all(
					count >= self.config.max_frames_per_camera
					for count in candidate_counts.values()
				)
			)

		try:
			while reader.has_next():
				topic, raw, timestamp = reader.read_next()
				if progress is not None:
					progress.update(1)
					# Show pending background tasks in progress bar
					pending_tasks = len(self._sample_futures)
					progress.set_postfix(pending=pending_tasks)
				
				storage_timestamp_ns = int(timestamp)
				if topic == ODOM_TOPIC:
					record = self._parse_odometry(raw, odom_msg_type, storage_timestamp_ns)
					odometry_records.append(record)
					odometry_timestamps.append(record.timestamp_ns)
					total_saved_frames += self._try_finalize_samples(
						bag_name,
						camera_dirs,
						manifest,
						per_camera_counts,
						pending_samples,
						odometry_records,
						odometry_timestamps,
						processing_complete=False,
					)
					continue

				if topic not in camera_topics:
					continue
				camera_name = camera_aliases.get(topic)
				if camera_name not in self.available_camera_names:
					continue
				if (
					self.config.max_frames_per_camera is not None
					and candidate_counts[camera_name] >= self.config.max_frames_per_camera
				):
					if cameras_saturated():
						break
					continue

				msg_type = type_cache[topic]
				message = deserialize_message(raw, msg_type)
				frame_timestamp_ns = _message_timestamp_ns(message, storage_timestamp_ns)
				
				inline = InlineFrame(
					timestamp_ns=frame_timestamp_ns,
					camera_name=camera_name,
					topic=topic,
					message_type=camera_topics[topic],
					jpeg_bytes=b""
				)

				if not self.config.json_only:
					future = self.executor.submit(
						self._process_image_task,
						message,
						topic,
						camera_name,
						camera_topics[topic],
					)
					inline.processing_future = future

				camera_buffers[camera_name].append(inline)
				self._prune_camera_buffer(camera_buffers[camera_name], frame_timestamp_ns)
				candidate_counts[camera_name] += 1

				if camera_name == self.anchor_camera:
					sample = self._attempt_build_sample(inline, camera_buffers)
					if sample is not None:
						pending_samples.append(sample)

				total_saved_frames += self._try_finalize_samples(
					bag_name,
					camera_dirs,
					manifest,
					per_camera_counts,
					pending_samples,
					odometry_records,
					odometry_timestamps,
					processing_complete=False,
				)
				if cameras_saturated():
					break
		finally:
			if progress is not None:
				progress.close()

		total_saved_frames += self._try_finalize_samples(
			bag_name,
			camera_dirs,
			manifest,
			per_camera_counts,
			pending_samples,
			odometry_records,
			odometry_timestamps,
			processing_complete=True,
		)

		# Wait for all sample processing to complete
		if self._sample_futures:
			for future in self._sample_futures:
				future.result()
		
		# Recalculate total frames from per_camera_counts
		total_saved_frames = sum(per_camera_counts.values()) // len(self.available_camera_names) if self.available_camera_names else 0

		metadata = {
			"bag_name": bag_name,
			"bag_path": str(bag_dir),
			"split": self.config.split_name,
			"city": city_name,
			"total_frames": total_saved_frames,
			"cameras": per_camera_counts,
			"topics": camera_topics,
			"frames": manifest,
		}
		metadata_path = target_dir / "metadata.json"
		metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

		info_dir = target_dir / "info"
		info_count = self._count_existing_info_files(info_dir)
		if info_count <= self.MIN_INFO_FILES:
			self.logger.warning(
				"Removing %s: only %d info files (≤%d threshold)",
				target_dir,
				info_count,
				self.MIN_INFO_FILES,
			)
			shutil.rmtree(target_dir, ignore_errors=True)
			zero_counts = {name: 0 for name in per_camera_counts}
			return ExportStats(
				total_frames=0,
				per_camera=zero_counts,
				topics_used=camera_topics,
			)

		return ExportStats(
			total_frames=total_saved_frames,
			per_camera=per_camera_counts,
			topics_used=camera_topics,
		)

	def _estimate_bag_message_count(self, bag_dir: Path) -> Optional[int]:
		metadata_path = bag_dir / "metadata.yaml"
		if not metadata_path.exists():
			return None
		try:
			content = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
		except Exception as exc:  # pragma: no cover - metadata optional
			self.logger.debug("Failed to parse %s (%s)", metadata_path, exc)
			return None
		info = content.get("rosbag2_bagfile_information") if isinstance(content, dict) else None
		if not isinstance(info, dict):
			return None
		message_count = info.get("message_count")
		if isinstance(message_count, int):
			return message_count
		try:
			return int(message_count)
		except (TypeError, ValueError):
			return None

	def _create_progress_bar(self, bag_dir: Path, bag_name: str):
		if not self.progress_enabled:
			return None
		total = self._estimate_bag_message_count(bag_dir)
		kwargs = {}
		if self.progress_position is not None:
			kwargs["position"] = self.progress_position
			
		return tqdm(
			total=total,
			desc=f"{bag_name}",
			unit="msg",
			leave=False,
			dynamic_ncols=True,
			**kwargs
		)

	def _parse_odometry(self, raw: bytes, odom_msg_type, storage_timestamp_ns: int) -> PoseRecord:
		message = deserialize_message(raw, odom_msg_type)
		timestamp_ns = _message_timestamp_ns(message, storage_timestamp_ns)
		pose = message.pose.pose
		position = pose.position
		orientation = pose.orientation
		return PoseRecord(
			timestamp_ns=int(timestamp_ns),
			position=(float(position.x), float(position.y), float(position.z)),
			quaternion=(
				float(orientation.x),
				float(orientation.y),
				float(orientation.z),
				float(orientation.w),
			),
		)

	def _encode_frame_to_jpeg(self, frame: np.ndarray) -> bytes:
		success, buffer = cv2.imencode(
			".jpg",
			frame,
			[int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)],
		)
		if not success:
			raise RuntimeError("Failed to encode frame as JPEG")
		return buffer.tobytes()

	def _prune_camera_buffer(self, buffer: Deque[InlineFrame], current_timestamp_ns: int) -> None:
		threshold = current_timestamp_ns - self.buffer_duration_ns
		while buffer and buffer[0].timestamp_ns < threshold:
			buffer.popleft()

	def _find_best_frame(
		self,
		buffer: Deque[InlineFrame],
		target_timestamp_ns: int,
	) -> Optional[InlineFrame]:
		best: Optional[InlineFrame] = None
		best_delta: Optional[int] = None
		for entry in buffer:
			delta = abs(entry.timestamp_ns - target_timestamp_ns)
			if best_delta is None or delta < best_delta:
				best = entry
				best_delta = delta
			if entry.timestamp_ns > target_timestamp_ns and best_delta is not None and delta > best_delta:
				break
		if best is None or best_delta is None:
			return None
		if best_delta <= self.sync_tolerance_ns:
			return best
		if self.allow_desync_fallback:
			delta_ms = best_delta / 1e6
			tolerance_ms = self.sync_tolerance_ns / 1e6
			self.logger.debug(
				"Using %s frame at %d despite Δ=%.2fms (>%.2fms)",
				best.camera_name,
				best.timestamp_ns,
				delta_ms,
				tolerance_ms,
			)
			return best
		return None

	def _attempt_build_sample(
		self,
		anchor_frame: InlineFrame,
		camera_buffers: Dict[str, Deque[InlineFrame]],
	) -> Optional[PendingSample]:
		frames: Dict[str, InlineFrame] = {}
		for camera_name in self.available_camera_names:
			if camera_name == anchor_frame.camera_name:
				frames[camera_name] = anchor_frame
				continue
			candidate = self._find_best_frame(camera_buffers[camera_name], anchor_frame.timestamp_ns)
			if candidate is None:
				return None
			frames[camera_name] = candidate
		return PendingSample(timestamp_ns=anchor_frame.timestamp_ns, frames=frames)

	def _try_finalize_samples(
		self,
		bag_name: str,
		camera_dirs: Dict[str, Path],
		manifest: List[Dict[str, object]],
		per_camera_counts: Dict[str, int],
		pending_samples: Deque[PendingSample],
		odometry_records: List[PoseRecord],
		odometry_timestamps: List[int],
		*,
		processing_complete: bool,
	) -> int:
		if not pending_samples or not odometry_records:
			return 0
		pose_lookup = StreamingPoseLookup(odometry_records, odometry_timestamps)
		transformer = self._get_pose_to_wgs_transformer()
		stationary_window_ns = int(max(0.0, self.config.stationary_skip_window_s) * 1e9)
		stationary_threshold_m = max(0.0, self.config.stationary_movement_threshold_m)
		stationary_enabled = self.config.stationary_skip_enable and stationary_window_ns > 0
		
		# Clean up finished futures to prevent memory leak
		self._sample_futures = [f for f in self._sample_futures if not f.done()]

		submitted_count = 0
		while pending_samples:
			sample = pending_samples[0]
			pose = pose_lookup.nearest(sample.timestamp_ns)
			if pose is None:
				if processing_complete:
					self.logger.debug("Dropping %d: no matching odometry", sample.timestamp_ns)
					pending_samples.popleft()
					continue
				break

			if stationary_enabled:
				displacement = pose_lookup.displacement_within_window(sample.timestamp_ns, stationary_window_ns)
				if displacement is None and not processing_complete:
					break
				if displacement is not None and displacement <= stationary_threshold_m:
					self.logger.debug(
						"Skipping %d: stationary (Δ=%.3fm)",
						sample.timestamp_ns,
						displacement,
					)
					pending_samples.popleft()
					continue

			trajectory_latlon = np.empty((0, 2), dtype=np.float64)
			has_future = False
			if self.config.trajectory_enable:
				trajectory_latlon, has_future = _build_future_trajectory(
					pose_lookup,
					sample.timestamp_ns,
					transformer,
					horizon_m=self.config.trajectory_horizon_distance_m,
					sample_rate_hz=self.config.trajectory_sample_rate_hz,
				)
				if not has_future and not processing_complete:
					break
				if not has_future and processing_complete:
					self.logger.debug(
						"Skipping %d: insufficient future trajectory coverage",
						sample.timestamp_ns,
					)
					pending_samples.popleft()
					continue

			# Submit to sample executor
			future = self.sample_executor.submit(
				self._emit_sample,
				bag_name,
				sample,
				pose,
				trajectory_latlon,
				camera_dirs,
				manifest,
				per_camera_counts,
				transformer,
			)
			self._sample_futures.append(future)
			pending_samples.popleft()
			submitted_count += 1

		return 0  # Return 0 as actual writing is async

	def _emit_sample(
		self,
		bag_name: str,
		sample: PendingSample,
		pose: PoseRecord,
		trajectory_latlon: np.ndarray,
		camera_dirs: Dict[str, Path],
		manifest: List[Dict[str, object]],
		per_camera_counts: Dict[str, int],
		transformer: Optional[object],
	) -> int:
		if self._sample_outputs_exist(bag_name, sample.timestamp_ns, camera_dirs):
			self.logger.debug(
				"Skipping %s: FM image and info already exist",
				sample.timestamp_ns,
			)
			return 0

		saved_records = self._write_sample_frames_streaming(
			bag_name,
			sample,
			camera_dirs,
			manifest,
			per_camera_counts,
		)
		if not saved_records:
			return 0

		annotation = self._build_ngii_annotation_payload(pose)
		city_name = getattr(self, '_current_city_name', self.config.city_name)
		payload = _build_openlane_sample(
			bag_name,
			self.config.split_name,
			city_name,
			sample,
			pose,
			saved_records,
			self.camera_calibrations,
			annotation,
		)
		info_dir = (
			self.config.output_root
			/ self.config.split_name
			/ city_name
			/ bag_name
			/ "info"
		)
		info_dir.mkdir(parents=True, exist_ok=True)
		(info_dir / f"{sample.timestamp_ns}-ls.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

		traj_dir: Optional[Path] = None
		if self.config.trajectory_enable:
			traj_dir = (
				self.config.output_root
				/ self.config.split_name
				/ city_name
				/ bag_name
				/ "traj"
			)
		self._save_trajectory_payload(
			traj_dir,
			sample.timestamp_ns,
			pose,
			transformer,
			trajectory_latlon,
		)
		return len(saved_records)

	def _write_sample_frames_streaming(
		self,
		bag_name: str,
		sample: PendingSample,
		camera_dirs: Dict[str, Path],
		manifest: List[Dict[str, object]],
		per_camera_counts: Dict[str, int],
	) -> Dict[str, SavedFrameRecord]:
		written_paths: List[Path] = []
		manifest_entries: List[Dict[str, object]] = []
		records: Dict[str, SavedFrameRecord] = {}

		for camera_name in self.available_camera_names:
			frame = sample.frames.get(camera_name)
			if frame is None:
				self.logger.debug("Sample %d missing camera %s; discarding", sample.timestamp_ns, camera_name)
				self._cleanup_written_files(written_paths)
				return {}
			
			# Check quota with lock
			with self._manifest_lock:
				if (
					self.config.max_frames_per_camera is not None
					and per_camera_counts.get(camera_name, 0) >= self.config.max_frames_per_camera
				):
					self.logger.debug("Skipping %d: %s reached frame quota", sample.timestamp_ns, camera_name)
					self._cleanup_written_files(written_paths)
					return {}

			destination_dir = camera_dirs.get(camera_name)
			if destination_dir is None:
				self.logger.debug("No destination directory mapped for %s", camera_name)
				self._cleanup_written_files(written_paths)
				return {}
			destination = destination_dir / f"{sample.timestamp_ns}.jpg"
			destination_dir.mkdir(parents=True, exist_ok=True)
			if not self.config.json_only:
				try:
					jpeg_bytes = frame.jpeg_bytes
					if frame.processing_future:
						jpeg_bytes = frame.processing_future.result()
					
					if not jpeg_bytes:
						self.logger.warning("Empty JPEG result for %s at %d", camera_name, sample.timestamp_ns)
						self._cleanup_written_files(written_paths)
						return {}

					self._write_bytes(destination, jpeg_bytes)
				except Exception as exc:  # pragma: no cover - disk failure
					self.logger.error("Failed to write %s (%s)", destination, exc)
					self._cleanup_written_files(written_paths)
					return {}
				written_paths.append(destination)
			rel_path = destination.relative_to(self.config.output_root).as_posix()
			record = SavedFrameRecord(
				timestamp_ns=sample.timestamp_ns,
				camera_name=camera_name,
				topic=frame.topic,
				relative_path=rel_path,
				absolute_path=str(destination),
			)
			records[camera_name] = record
			manifest_entries.append(
				{
					"timestamp_ns": sample.timestamp_ns,
					"topic": frame.topic,
					"message_type": frame.message_type,
					"camera": camera_name,
					"segment": bag_name,
					"relative_path": rel_path,
				}
			)

		with self._manifest_lock:
			manifest.extend(manifest_entries)
			for camera_name in records:
				per_camera_counts[camera_name] = per_camera_counts.get(camera_name, 0) + 1

		return records

	def _sample_outputs_exist(
		self,
		bag_name: str,
		timestamp_ns: int,
		camera_dirs: Dict[str, Path],
	) -> bool:
		anchor_dir = camera_dirs.get(self.anchor_camera)
		if anchor_dir is None:
			return False
		frame_path = anchor_dir / f"{timestamp_ns}.jpg"
		city_name = getattr(self, '_current_city_name', self.config.city_name)
		info_path = (
			self.config.output_root
			/ self.config.split_name
			/ city_name
			/ bag_name
			/ "info"
			/ f"{timestamp_ns}-ls.json"
		)
		return frame_path.exists() and info_path.exists()

	@staticmethod
	def _count_existing_info_files(info_dir: Path) -> int:
		if not info_dir.exists():
			return 0
		return sum(1 for entry in info_dir.iterdir() if entry.is_file() and entry.name.endswith("-ls.json"))

	@staticmethod
	def _cleanup_written_files(paths: List[Path]) -> None:
		for path in paths:
			try:
				path.unlink()
			except FileNotFoundError:
				continue
			except Exception:
				continue

	@staticmethod
	def _write_bytes(path: Path, data: bytes) -> None:
		path.write_bytes(data)


def _parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Streaming ROS2 bag → OpenLaneV2 exporter")
	parser.add_argument(
		"--json-only",
		action="store_true",
		help="Skip converting image files and only emit info/metadata outputs.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=None,
		help="Number of worker threads per bag (default: all CPU cores).",
	)
	parser.add_argument(
		"--parallel-bags",
		type=int,
		default=1,
		help="Number of bags to process in parallel (default: 1).",
	)
	parser.add_argument(
		"--bag-path",
		type=Path,
		default=None,
		help="Path to the bag directory or root folder containing bags.",
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = _parse_arguments(argv)
	
	# Build config kwargs, filtering out None values to let defaults shine
	config_kwargs = {
		"json_only": args.json_only,
		"stream_workers": args.workers,
	}
	if args.bag_path is not None:
		config_kwargs["bag_path"] = args.bag_path

	config = ConverterConfig(**config_kwargs)
	# Monkey-patch config to add bag parallelism since it's not in the base class definition
	setattr(config, "stream_bag_parallelism", args.parallel_bags)
	
	converter = StreamingROS2OpenLaneConverter(config)
	return converter.run()


if __name__ == "__main__":
	raise SystemExit(main())
