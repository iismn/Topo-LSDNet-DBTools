#!/usr/bin/env python3
"""OpenLane info → NGII aerial imagery exporter.

This script scans an existing OpenLane-V2 dataset (train/<city>/<bag>/info/*.json)
and downloads aligned NGII aerial tiles for each sample. It no longer
generates virtual-path samples; the sole responsibility is producing
``sd_aerial`` outputs under ``<output_root>/<split>/<city>/<bag>/sd_aerial``.

If the dataset's ``info`` directories live under a different tree than the
desired outputs (for example, re-running imagery where the info files already
sit under ``output_root``), supply ``--info-root`` to target that structure.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2  # type: ignore
from tqdm import tqdm
# 현재 파일 기준 NGII 모듈 경로 추가
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # NGII 폴더
for path in (current_dir, parent_dir):
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))

# 컨버터 임포트 (pkg)
converter_module = importlib.import_module("ros_converter")
NGIIAerialImageDownloaderV4 = converter_module.NGIIAerialImageDownloaderV4
APIRateLimiter = converter_module.APIRateLimiter

# 로컬 상수 경로
OPENLANE_ROOT_DEFAULT = "/home/iismn/Workspace_C/Dataset/LSD_Net/OpenLane-V2"


@dataclass(frozen=True)
class InfoPose:
    x_utm: float
    y_utm: float
    heading_deg: float



class OpenLaneInfoAerialGenerator:
    """Generate aerial imagery for existing OpenLane-V2 samples using info JSON files."""

    def __init__(
        self,
        dataset_root: str,
        info_root: Optional[str] = None,
        split: str = "train",
        city_filter: Optional[str] = None,
        bag_filter: Optional[str] = None,
        output_root: Optional[str] = None,
        limit_per_bag: Optional[int] = None,
        overwrite: bool = False,
        aerial_client_id: str = "",
        aerial_client_secret: str = "",
        physical_size: Tuple[float, float] = (120.0, 80.0),
        zoom_level: int = 20,
        map_type: str = "satellite",
        half_scale_output: bool = False,
        max_workers: int = 1,
        api_retries: int = 3,
        api_retry_backoff: float = 2.0,
        api_requests_per_minute: int = 120,
        api_min_interval: float = 0.5,
        allow_dummy_output: bool = False,
    ):
        self.dataset_root = Path(dataset_root).expanduser()
        self.info_root = Path(info_root).expanduser() if info_root else self.dataset_root
        self.split = split
        self.city_filter = city_filter
        self.bag_filter = bag_filter
        self.limit_per_bag = limit_per_bag
        self.overwrite = overwrite
        self.output_root = Path(output_root).expanduser() if output_root else self.dataset_root
        self.aerial_root = self.output_root / split
        self.aerial_root.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ngii_aerial_exporter")
        self.max_workers = max(1, max_workers)
        self._rate_limiter = APIRateLimiter(
            max_requests_per_minute=max(1, api_requests_per_minute),
            min_interval=max(0.0, api_min_interval),
        )
        self._downloader_kwargs = {
            "client_id": aerial_client_id,
            "client_secret": aerial_client_secret,
            "physical_size": physical_size,
            "zoom_level": zoom_level,
            "map_type": map_type,
            "max_retry_attempts": max(1, api_retries),
            "retry_backoff": max(1.0, api_retry_backoff),
            "rate_limiter": self._rate_limiter,
            "allow_dummy_fallback": allow_dummy_output,
        }
        self.downloader = NGIIAerialImageDownloaderV4(**self._downloader_kwargs)
        self._thread_local: threading.local = threading.local()
        self.half_scale_output = half_scale_output
        self.allow_dummy_output = allow_dummy_output

    def run(self) -> int:
        entries = self._discover_bags()
        if not entries:
            self.logger.warning("No info directories found under %s", self.info_root / self.split)
            return 0
        total_saved = 0
        with tqdm(entries, desc=f"{self.split} bag(s)", unit="bag") as bag_bar:
            for entry in bag_bar:
                saved = self._process_bag(entry)
                total_saved += saved
                bag_bar.set_postfix_str(f"saved={saved}")
        self.logger.info("Saved %d aerial image(s)", total_saved)
        return total_saved

    def _discover_bags(self) -> List[Dict[str, object]]:
        split_dir = self.info_root / self.split
        if not split_dir.is_dir():
            return []
        entries: List[Dict[str, object]] = []
        for city_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            if self.city_filter and city_dir.name != self.city_filter:
                continue
            for bag_dir in sorted([p for p in city_dir.iterdir() if p.is_dir()]):
                if self.bag_filter and bag_dir.name != self.bag_filter:
                    continue
                info_dir = bag_dir / "info"
                if not info_dir.is_dir():
                    continue
                info_files = sorted(info_dir.glob("*-ls.json"))
                if not info_files:
                    continue
                entries.append({
                    "city": city_dir.name,
                    "bag": bag_dir.name,
                    "info_files": info_files,
                })
        return entries

    def _get_downloader(self) -> NGIIAerialImageDownloaderV4:
        if self.max_workers <= 1:
            return self.downloader
        downloader = getattr(self._thread_local, "downloader", None)
        if downloader is None:
            downloader = NGIIAerialImageDownloaderV4(**self._downloader_kwargs)
            self._thread_local.downloader = downloader
        return downloader

    def _process_bag(self, entry: Dict[str, object]) -> int:
        info_files: List[Path] = list(entry.get("info_files", []))  # type: ignore[arg-type]
        if self.limit_per_bag is not None:
            info_files = info_files[: self.limit_per_bag]
        if not info_files:
            return 0
        city = str(entry.get("city", "unknown"))
        bag = str(entry.get("bag", "bag"))
        dest_dir = self.aerial_root / city / bag / "sd_aerial"
        dest_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        if self.max_workers <= 1:
            with tqdm(info_files, desc=f"{city}/{bag}", unit="img", leave=False) as frame_bar:
                for info_path in frame_bar:
                    status = self._process_info_file(info_path, dest_dir)
                    if status == "saved":
                        saved += 1
                    frame_bar.set_postfix_str(status)
            return saved

        frame_bar = tqdm(total=len(info_files), desc=f"{city}/{bag}", unit="img", leave=False)
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_info_file, info_path, dest_dir): info_path
                    for info_path in info_files
                }
                for future in as_completed(future_to_path):
                    info_path = future_to_path[future]
                    try:
                        status = future.result()
                    except Exception as exc:  # pragma: no cover - unexpected worker error
                        self.logger.exception("Parallel worker error for %s: %s", info_path.name, exc)
                        status = "error"
                    if status == "saved":
                        saved += 1
                    frame_bar.update(1)
                    frame_bar.set_postfix_str(status)
        finally:
            frame_bar.close()
        return saved

    def _process_info_file(self, info_path: Path, dest_dir: Path) -> str:
        timestamp = self._timestamp_from_path(info_path)
        output_path = dest_dir / f"{timestamp}.png"
        if output_path.exists():
            if not self.overwrite:
                if self._validate_image(output_path):
                    return "skip"
                self.logger.warning("Corrupt or partial image detected, re-downloading %s", output_path.name)
            try:
                output_path.unlink()
            except OSError as exc:
                self.logger.warning("Failed to remove existing file %s: %s", output_path, exc)
                return "write-fail"
        pose = self._extract_pose(info_path)
        if pose is None:
            return "pose-missing"
        downloader = self._get_downloader()
        try:
            image = downloader.download_and_process_from_utm(
                pose.x_utm,
                pose.y_utm,
                pose.heading_deg,
            )
        except Exception as exc:  # pragma: no cover - network/API failure
            self.logger.warning("Aerial error for %s: %s", info_path.name, exc)
            return "api-error"
        if image is None:
            self.logger.warning(
                "Downloader returned no data for %s (API failed even after retries)",
                info_path.name,
            )
            return "api-miss"
        if self.half_scale_output:
            height, width = image.shape[:2]
            new_width = max(1, width // 2)
            new_height = max(1, height // 2)
            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self._write_image_atomic(image_bgr, output_path):
            return "saved"
        return "write-fail"

    @staticmethod
    def _timestamp_from_path(info_path: Path) -> str:
        name = info_path.name
        if name.endswith("-ls.json"):
            return name[: -len("-ls.json")]
        if name.endswith(".json"):
            return name[:-5]
        return info_path.stem

    def _extract_pose(self, info_path: Path) -> Optional[InfoPose]:
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Failed to read %s: %s", info_path, exc)
            return None
        x_utm: Optional[float] = None
        y_utm: Optional[float] = None
        heading_deg: Optional[float] = None
        meta = payload.get("meta_data")
        if isinstance(meta, dict):
            pos = meta.get("ego_position_utm")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x_utm = float(pos[0])
                y_utm = float(pos[1])
            heading_val = meta.get("ego_heading_deg")
            if isinstance(heading_val, (int, float)):
                heading_deg = float(heading_val)
        pose_block = payload.get("pose")
        if isinstance(pose_block, dict):
            translation = pose_block.get("translation")
            if (
                isinstance(translation, (list, tuple))
                and len(translation) >= 2
                and (x_utm is None or y_utm is None)
            ):
                x_utm = float(translation[0])
                y_utm = float(translation[1])
            if heading_deg is None:
                heading_deg = self._heading_from_pose(pose_block)
        if x_utm is None or y_utm is None:
            self.logger.debug("Missing ego_position_utm in %s", info_path)
            return None
        if heading_deg is None:
            heading_deg = 0.0
        return InfoPose(x_utm=x_utm, y_utm=y_utm, heading_deg=heading_deg)

    @staticmethod
    def _heading_from_pose(pose_block: Dict[str, object]) -> Optional[float]:
        heading_val = pose_block.get("heading")
        if isinstance(heading_val, (int, float)):
            return float(heading_val)
        rotation = pose_block.get("rotation")
        matrix: Optional[List[List[float]]] = None
        if isinstance(rotation, list):
            if len(rotation) == 3 and all(isinstance(row, list) and len(row) == 3 for row in rotation):
                matrix = [[float(value) for value in row[:3]] for row in rotation[:3]]
            elif len(rotation) == 9:
                matrix = [
                    [float(rotation[0]), float(rotation[1]), float(rotation[2])],
                    [float(rotation[3]), float(rotation[4]), float(rotation[5])],
                    [float(rotation[6]), float(rotation[7]), float(rotation[8])],
                ]
        if matrix is None:
            return None
        r00 = matrix[0][0]
        r10 = matrix[1][0]
        return math.degrees(math.atan2(r10, r00))

    def _write_image_atomic(self, image_bgr, output_path: Path) -> bool:
        temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            self.logger.warning("Failed to clear temp file %s", temp_path)
        if not cv2.imwrite(str(temp_path), image_bgr):
            return False
        if not self._validate_image(temp_path):
            try:
                temp_path.unlink()
            except OSError:
                pass
            return False
        temp_path.replace(output_path)
        return True

    @staticmethod
    def _validate_image(path: Path) -> bool:
        try:
            if path.stat().st_size <= 0:
                return False
        except FileNotFoundError:
            return False
        probe = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if probe is None:
            return False
        return probe.size > 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NGII aerial imagery for OpenLane-V2 samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=str, default=OPENLANE_ROOT_DEFAULT)
    parser.add_argument(
        "--info-root",
        type=str,
        default=None,
        help="Optional root containing <split>/<city>/<bag>/info directories (defaults to dataset root)",
    )
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for sd_map_imagery outputs")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--city", type=str, default=None, help="Optional city filter")
    parser.add_argument("--bag", type=str, default=None, help="Optional bag filter")
    parser.add_argument("--limit-per-bag", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing aerial PNGs")
    parser.add_argument(
        "--physical-width",
        type=float,
        default=120.0,
        help="Aerial crop width (lateral, matching ngii_sd_converter dist_y) in meters",
    )
    parser.add_argument(
        "--physical-length",
        type=float,
        default=80.0,
        help="Aerial crop length (forward, matching ngii_sd_converter dist_x) in meters",
    )
    parser.add_argument("--zoom-level", type=int, default=19, help="Naver map zoom level (default lowered by 1 for wider coverage)")
    parser.add_argument(
        "--map-type",
        type=str,
        default="satellite",
        help="Naver map type (e.g., satellite, satellite_base, hybrid). 'satellite' is auto-mapped to satellite_base to remove labels",
    )
    parser.add_argument(
        "--allow-dummy-output",
        action="store_true",
        help="Allow saving synthetic fallback imagery when the API fails (defaults to disabled so failed frames retry next run)",
    )
    parser.add_argument(
        "--half-scale-output",
        action="store_true",
        help="Downscale saved aerial PNGs by 0.5 using Lanczos-4 for sharper smaller files",
    )
    parser.add_argument("--aerial-client-id", type=str, default=None)
    parser.add_argument("--aerial-client-secret", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel download workers (set >1 to enable threading)",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=3,
        help="Maximum number of API retry attempts per frame (>=1)",
    )
    parser.add_argument(
        "--api-retry-backoff",
        type=float,
        default=2.0,
        help="Exponential backoff multiplier between API retries (>=1.0)",
    )
    parser.add_argument(
        "--api-requests-per-minute",
        type=int,
        default=120,
        help="Throttle Naver API calls to this many requests per minute (>=1)",
    )
    parser.add_argument(
        "--api-min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between consecutive API calls (>=0.0)",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    client_id = args.aerial_client_id or os.environ.get("NAVER_MAP_CLIENT_ID", "")
    client_secret = args.aerial_client_secret or os.environ.get("NAVER_MAP_CLIENT_SECRET", "")

    physical_size = (args.physical_width, args.physical_length)
    exporter = OpenLaneInfoAerialGenerator(
        dataset_root=args.dataset_root,
        info_root=args.info_root,
        split=args.split,
        city_filter=args.city,
        bag_filter=args.bag,
        output_root=args.output_root,
        limit_per_bag=args.limit_per_bag,
        overwrite=args.overwrite,
        aerial_client_id=client_id,
        aerial_client_secret=client_secret,
        physical_size=physical_size,
        zoom_level=args.zoom_level,
        map_type=args.map_type,
        half_scale_output=args.half_scale_output,
        max_workers=args.workers,
        api_retries=args.api_retries,
        api_retry_backoff=args.api_retry_backoff,
        api_requests_per_minute=args.api_requests_per_minute,
        api_min_interval=args.api_min_interval,
        allow_dummy_output=args.allow_dummy_output,
    )
    exporter.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
