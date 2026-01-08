#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGII Aerial Image Downloader V4

Downloads aerial imagery for NGII datasets via the Naver Map Static API and
produces rotation-aligned crops that cover the same 120 m (lateral) × 80 m
(forward) footprint used by ``ngii_sd_converter``, aligned with the vehicle
heading.

Main features:
- Compute a safe download size that covers rotated footprints.
- Convert UTM coordinates to latitude/longitude.
- Process odometry CSV files and download imagery for each pose.
- Rotate and crop images to the requested physical dimensions.
"""

import os
import sys
import json
import requests
import time
import math
import hashlib
import threading
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pyproj
from io import BytesIO
import cv2

# Suppress noisy warnings from third-party libraries
import warnings
warnings.filterwarnings('ignore')

# Logging system (borrowed from ngii_openlane_converter)
# ANSI color codes
YELLOW_BOLD = "\033[1;33m"  # Bold yellow
RESET = "\033[0m"           # Reset to default
LOG_TAG = f"{YELLOW_BOLD}[NGII-AERIAL]{RESET}"
_SILENT = False
_DEBUG = False

def set_quiet(flag: bool) -> None:
    global _SILENT
    _SILENT = bool(flag)

def set_debug(flag: bool) -> None:
    global _DEBUG
    _DEBUG = bool(flag)

# Known section labels to align the colon across logs
_LOG_KNOWN_LABELS = [
    "Init",
    "API Setup",
    "Path Check",
    "CSV Load",
    "UTM Convert",
    "Resolution",
    "Download",
    "Cache Hit",
    "Cache Save",
    "API Call",
    "API Error",
    "Rotation",
    "Crop",
    "Save Image",
    "Processing",
    "Statistics",
    "Complete",
    "Error",
    "Warning"
]
_LOG_LABEL_WIDTH = max(len(s) for s in _LOG_KNOWN_LABELS)

def LOGK(label: str, content: str) -> None:
    """Log with aligned colon: [TAG] <label padded>: <content>"""
    if _SILENT:
        return
    print(f"{LOG_TAG} {label.ljust(_LOG_LABEL_WIDTH)}: {content}")

def LOGD(label: str, content: str) -> None:
    """Debug log with aligned colon: [TAG] <label padded>: <content> (only if debug enabled)"""
    if _SILENT or not _DEBUG:
        return
    print(f"{LOG_TAG} {label.ljust(_LOG_LABEL_WIDTH)}: {content}")

def LOG(message: str) -> None:
    """Legacy simple logger (kept for compatibility)."""
    if ":" in message and not message.startswith("http"):
        section, rest = message.split(":", 1)
        LOGK(section.strip(), rest.strip())
    else:
        LOGK("Info", message)

class APIRateLimiter:
    """Thread-safe limiter to throttle shared API usage."""

    def __init__(self, max_requests_per_minute: int = 600, min_interval: float = 0.1):
        self.max_requests_per_minute = max(1, max_requests_per_minute)
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._request_count = 0
        self._window_start = time.time()
        self._last_request_time = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                if now - self._window_start >= 60.0:
                    self._window_start = now
                    self._request_count = 0

                elapsed_since_last = now - self._last_request_time
                need_interval_delay = max(0.0, self.min_interval - elapsed_since_last)

                if self._request_count < self.max_requests_per_minute and need_interval_delay == 0.0:
                    self._request_count += 1
                    self._last_request_time = now
                    return

                wait_for = need_interval_delay if need_interval_delay > 0 else 0.01
                if self._request_count >= self.max_requests_per_minute:
                    remaining_window = 60.0 - (now - self._window_start)
                    wait_for = max(wait_for, max(0.01, remaining_window))

            time.sleep(min(wait_for, 1.0))


class NGIIAerialImageDownloaderV4:
    """Downloader that fetches aerial imagery and aligns it with vehicle heading."""

    _MAP_TYPE_ALIASES = {
        "sat": "satellite_base",
        "satellite": "satellite_base",
        "satellite_base": "satellite_base",
        "hybrid": "hybrid",
        "terrain": "terrain",
        "basic": "basic",
        "roadmap": "basic",
        "standard": "basic",
        "traffic": "traffic",
        "terrain_satellite": "terrain_satellite",
    }
    
    def __init__(self, 
             client_id: str = "",
             client_secret: str = "",
             cache_dir: str = "./cache/aerial_images",
             physical_size: Tuple[float, float] = (120.0, 80.0),
             zoom_level: int = 18,
             map_type: str = "satellite",
             max_retry_attempts: int = 3,
             retry_backoff: float = 2.0,
             rate_limiter: Optional[APIRateLimiter] = None,
             allow_dummy_fallback: bool = True):
        """
        Initialize the downloader.

        Args:
            client_id: Naver Cloud Platform client ID.
            client_secret: Naver Cloud Platform client secret.
            cache_dir: Directory for cached imagery.
            physical_size: Physical footprint (width, height) in meters.
            zoom_level: Naver Map zoom level (1-20).
            map_type: Map style ('satellite', 'basic', 'terrain', etc.).
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store requested footprint and map configuration
        self.physical_size = physical_size  # (width_m, height_m)
        self.zoom_level = zoom_level
        self.map_type = self._normalize_map_type(map_type)
        self.max_retry_attempts = max(1, int(max_retry_attempts))
        self.retry_backoff = max(1.0, float(retry_backoff))
        self.rate_limiter = rate_limiter or APIRateLimiter()
        self.allow_dummy_fallback = allow_dummy_fallback
        
        # Approximate resolution at a default latitude (Seoul)
        default_latitude = 37.5219
        self.default_resolution = self.get_naver_map_resolution(default_latitude, zoom_level)

        # Approximate target image size using the default resolution
        self.target_image_size = (
            int(physical_size[0] / self.default_resolution),  # width in pixels
            int(physical_size[1] / self.default_resolution)   # height in pixels
        )
        
        # Compute a conservative download size that covers the rotated footprint
        self.max_download_size = self._calculate_max_download_size()
        
        # Transformer from UTM Zone 52N to WGS84 latitude/longitude
        self.utm_to_wgs84 = pyproj.Transformer.from_crs(
            'EPSG:32652',
            'EPSG:4326',
            always_xy=True
        )
        
        # Rate limiting configuration
        self.request_delay = 0.1
        
        # Basic statistics for the current session
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'failures': 0
        }

        if not client_id or not client_secret:
            LOGK("Warning", "API credentials missing; dummy images will be generated" if self.allow_dummy_fallback else "API calls will fail without credentials")

    def _calculate_max_download_size(self) -> Tuple[int, int]:
        """
        Calculate maximum download image size considering rotation
        Ensures the 50x100m area is fully covered regardless of rotation angle
        """
        width_m, height_m = self.physical_size
        
        # Calculate diagonal length (meters)
        diagonal_m = math.sqrt(width_m**2 + height_m**2)
        
        # Add margin (20% safety factor)
        margin_factor = 1.2
        max_size_m = diagonal_m * margin_factor
        
        # Calculate pixels using actual Naver Map resolution (Seoul reference)
        actual_resolution = self.get_naver_map_resolution(37.5, self.zoom_level)
        max_size_px = int(max_size_m / actual_resolution)
        
        # Naver Map API limit (max 2048x2048)
        max_allowed = 2048
        final_size = min(max_size_px, max_allowed)
        
        return (final_size, final_size)

    def get_naver_map_resolution(self, latitude: float, zoom_level: int) -> float:
        """
        Calculate actual resolution (meters/pixel) for Naver Map at specific latitude and zoom level
        
        Naver Map uses Web Mercator projection, resolution varies by latitude.
        Reference: https://www.ncloud-forums.com/topic/141/
        
        Args:
            latitude: Latitude
            zoom_level: Zoom level (1-20)
        Returns:
            Resolution in meters/pixel
        """
        # Equatorial resolution in Web Mercator projection (meters/pixel)
        # At zoom level 0, Earth circumference (40,075,016.686m) is represented in 256 pixels
        equatorial_circumference = 40075016.686  # meters
        
        # Resolution by zoom level (equatorial reference)
        equatorial_resolution = equatorial_circumference / (256 * (2 ** (zoom_level + 1)))
        
        # Latitude correction factor (Web Mercator characteristic)
        latitude_factor = math.cos(math.radians(latitude))
        actual_resolution = equatorial_resolution * latitude_factor
        
        # Resolution correction for scale=2 option (2x more detailed)
        actual_resolution = actual_resolution / 2.0
        
        return actual_resolution

    def calculate_optimal_image_size_for_physical_area(self, width_m: float, height_m: float, 
                                                      latitude: float, zoom_level: int) -> Tuple[int, int]:
        """
        Calculate optimal image size for physical dimensions
        
        Args:
            width_m, height_m: Physical size (meters)
            latitude: Latitude
            zoom_level: Zoom level
        Returns:
            (width_px, height_px): Pixel size
        """
        resolution = self.get_naver_map_resolution(latitude, zoom_level)
        
        width_px = int(width_m / resolution)
        height_px = int(height_m / resolution)
        
        return (width_px, height_px)

    def utm_to_latlon(self, utm_x: float, utm_y: float) -> Tuple[float, float]:
        """Convert UTM coordinates to latitude/longitude"""
        try:
            lon, lat = self.utm_to_wgs84.transform(utm_x, utm_y)
            return lat, lon
        except Exception as e:
            LOGK("UTM Convert", f"failed ({utm_x}, {utm_y}): {e}")
            return 37.5219, 126.9240  # Seoul City Hall coordinates as fallback

    def generate_cache_key(self, lat: float, lon: float, 
                          image_size: Tuple[int, int],
                          zoom_level: int,
                          map_type: str) -> str:
        """Generate cache key"""
        key_str = f"{lat:.6f}_{lon:.6f}_{image_size[0]}x{image_size[1]}_{zoom_level}_{map_type}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cached_image(self, cache_key: str) -> Optional[np.ndarray]:
        """Load image from cache"""
        cache_file = self.cache_dir / f"{cache_key}.png"
        if cache_file.exists():
            try:
                image = Image.open(cache_file)
                self.stats['cache_hits'] += 1
                return np.array(image)
            except Exception:
                pass
        return None

    def save_to_cache(self, cache_key: str, image: np.ndarray) -> bool:
        """Save image to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.png"
            Image.fromarray(image).save(cache_file)
            return True
        except Exception:
            return False

    def wait_for_rate_limit(self):
        """Manage API request rate limiting"""
        if self.rate_limiter:
            self.rate_limiter.wait()
        else:
            time.sleep(self.request_delay)

    def download_aerial_image(self, lat: float, lon: float,
                            image_size: Tuple[int, int] = None,
                            zoom_level: int = None,
                            map_type: str = None,
                            use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Download aerial image using Naver Map API
        """
        # Set defaults
        if image_size is None:
            image_size = self.max_download_size
        if zoom_level is None:
            zoom_level = self.zoom_level
        if map_type is None:
            map_type = self.map_type
            
        self.stats['total_requests'] += 1
        
        cache_key: Optional[str] = None
        # Check cache
        if use_cache:
            cache_key = self.generate_cache_key(lat, lon, image_size, zoom_level, map_type)
            cached_image = self.get_cached_image(cache_key)
            if cached_image is not None:
                return cached_image
        
        # Generate dummy image if no API keys
        if not self.client_id or not self.client_secret:
            if self.allow_dummy_fallback:
                return self.generate_dummy_aerial_image(lat, lon, image_size)
            return None
        
        # Call the Naver Cloud Platform Static Map API
        url = "https://naveropenapi.apigw.ntruss.com/map-static/v2/raster"
        
        params = {
            'w': image_size[0],
            'h': image_size[1],
            'center': f"{lon},{lat}",
            'level': zoom_level,
            'maptype': map_type,
            'format': 'png',
            'scale': 2,
            'lang': 'ko'
        }
        
        headers = {
            'X-NCP-APIGW-API-KEY-ID': self.client_id,
            'X-NCP-APIGW-API-KEY': self.client_secret,
            'Content-Type': 'application/json'
        }
        
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                self.wait_for_rate_limit()

                response = requests.get(url, params=params, headers=headers, timeout=30)
                self.stats['api_calls'] += 1

                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')

                    if 'image' in content_type:
                        image = Image.open(BytesIO(response.content))
                        image_array = np.array(image)

                        # Save to cache
                        if use_cache and cache_key:
                            self.save_to_cache(cache_key, image_array)

                        return image_array
                    else:
                        last_error = RuntimeError("response did not contain image data")
                        LOGK("API Error", f"response missing image data (attempt {attempt}/{self.max_retry_attempts})")
                        self.stats['failures'] += 1
                else:
                    last_error = RuntimeError(f"status {response.status_code}")
                    LOGK("API Error", f"request failed with status {response.status_code} (attempt {attempt}/{self.max_retry_attempts})")
                    self.stats['failures'] += 1

            except requests.exceptions.RequestException as exc:
                last_error = exc
                LOGK("API Error", f"Naver API request failed ({lat:.6f}, {lon:.6f}) attempt {attempt}/{self.max_retry_attempts}: {exc}")
                self.stats['failures'] += 1
            except Exception as exc:
                last_error = exc
                LOGK("API Error", f"Unexpected error ({lat:.6f}, {lon:.6f}) attempt {attempt}/{self.max_retry_attempts}: {exc}")
                self.stats['failures'] += 1

            if attempt < self.max_retry_attempts:
                backoff_seconds = min(self.retry_backoff ** (attempt - 1), 10.0)
                jitter = random.uniform(0.0, 0.75)
                time.sleep(backoff_seconds + jitter)
                continue
            break

        LOGK("API Error", f"Naver API request failed ({lat:.6f}, {lon:.6f}) after {self.max_retry_attempts} attempt(s): {last_error}")
        if self.allow_dummy_fallback:
            return self.generate_dummy_aerial_image(lat, lon, image_size)
        return None

    def generate_dummy_aerial_image(self, lat: float, lon: float, 
                                  image_size: Tuple[int, int]) -> np.ndarray:
        """Generate dummy aerial image"""
        width, height = image_size
        
        # Create satellite-like dummy image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Terrain colors (green tones)
        base_color = (34, 139, 34)  # Forest green
        
        # Gradient background
        for y in range(height):
            for x in range(width):
                # Add noise for natural terrain feel
                noise = np.random.randint(-20, 21)
                color = tuple(np.clip(np.array(base_color) + noise, 0, 255))
                img[y, x] = color
        
        # Add road network pattern (gray lines)
        road_color = (128, 128, 128)
        
        # Vertical/horizontal roads
        step = max(80, width // 10)
        for i in range(0, width, step):
            if i + 3 < width:
                img[:, i:i+3] = road_color
        for j in range(0, height, step):
            if j + 3 < height:
                img[j:j+3, :] = road_color
        
        # Add coordinate text
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        coord_text = f"({lat:.4f}, {lon:.4f})"
        dummy_text = f"DUMMY {image_size[0]}x{image_size[1]}"
        
        if font:
            # Coordinate info (top left)
            draw.rectangle([5, 5, 200, 45], fill=(0, 0, 0, 128))
            draw.text((10, 10), coord_text, fill=(255, 255, 255), font=font)
            draw.text((10, 25), dummy_text, fill=(200, 200, 200), font=font)
            
            # Dummy marker (top right)
            draw.rectangle([width-120, 5, width-5, 25], fill=(255, 0, 0, 128))
            draw.text((width-115, 10), "DUMMY", fill=(255, 255, 255), font=font)
        
        return np.array(pil_img)

    def rotate_and_crop_to_exact_size(self, image: np.ndarray, 
                                     heading_deg: float,
                                     target_size: Tuple[int, int]) -> np.ndarray:
        """
        Rotate image according to heading and crop to exact size
        
        Args:
            image: Downloaded large image (numpy array)
            heading_deg: ego vehicle heading (degrees, clockwise from north)
            target_size: Final output size (width_px, height_px)
        Returns:
            Rotated and cropped image with exact size
        """
        target_width, target_height = target_size
        
        # Precise rotation using OpenCV
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix (convert heading to counter-clockwise)
        rotation_matrix = cv2.getRotationMatrix2D(center, -heading_deg, 1.0)
        
        # Calculate new boundaries for rotated image
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation center to new image center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        # Crop target size from center
        rot_center_x, rot_center_y = new_width // 2, new_height // 2
        
        crop_left = rot_center_x - target_width // 2
        crop_top = rot_center_y - target_height // 2
        crop_right = crop_left + target_width
        crop_bottom = crop_top + target_height
        
        # Check boundaries
        if (crop_left >= 0 and crop_top >= 0 and 
            crop_right <= new_width and crop_bottom <= new_height):
            # Normal crop possible
            cropped_image = rotated_image[crop_top:crop_bottom, crop_left:crop_right]
        else:
            # Add padding if boundaries exceeded
            pad_size = max(target_width, target_height)
            padded_image = np.zeros((new_height + pad_size, 
                                   new_width + pad_size, 3), dtype=np.uint8)
            
            # Place original image in center of padded image
            pad_y = pad_size // 2
            pad_x = pad_size // 2
            padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = rotated_image
            
            # Recalculate crop coordinates for padded image
            new_center_x = (new_width + pad_size) // 2
            new_center_y = (new_height + pad_size) // 2
            
            crop_left = new_center_x - target_width // 2
            crop_top = new_center_y - target_height // 2
            crop_right = crop_left + target_width
            crop_bottom = crop_top + target_height
            
            cropped_image = padded_image[crop_top:crop_bottom, crop_left:crop_right]
        
        # Rotate 90 degrees counter-clockwise so forward direction points upward
        rotated_final = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return rotated_final

    def download_and_process_from_utm(self, utm_x: float, utm_y: float,
                                    heading_deg: float = 0.0) -> Optional[np.ndarray]:
        """
        Download aerial image from UTM coordinates and process with rotation/crop
        
        Args:
            utm_x, utm_y: UTM coordinates
            heading_deg: Vehicle heading direction (degrees)
        Returns:
            Processed 50x100m image
        """
        # Convert UTM to lat/lon
        lat, lon = self.utm_to_latlon(utm_x, utm_y)
        
        # Calculate actual resolution at this location
        actual_resolution = self.get_naver_map_resolution(lat, self.zoom_level)
        
        # Recalculate target image size for physical dimensions
        target_width_px, target_height_px = self.calculate_optimal_image_size_for_physical_area(
            self.physical_size[0], self.physical_size[1], lat, self.zoom_level
        )
        
        # Calculate download size considering rotation
        margin_factor = 1.2
        diagonal_m = math.sqrt(self.physical_size[0]**2 + self.physical_size[1]**2)
        download_size_m = diagonal_m * margin_factor
        download_size_px = int(download_size_m / actual_resolution)
        
        # Check API limits
        max_allowed = 2048
        if download_size_px > max_allowed:
            download_size_px = max_allowed
        
        download_size = (download_size_px, download_size_px)
        
        # Download large image (considering rotation)
        large_image = self.download_aerial_image(lat, lon, download_size)
        
        if large_image is not None:
            # Rotate and crop to exact size
            final_image = self.rotate_and_crop_to_exact_size(
                large_image, heading_deg, (target_width_px, target_height_px)
            )
            return final_image
        return None

    def process_csv_file(self, csv_path: str,
                        x_col: str = "x_utm",
                        y_col: str = "y_utm", 
                        heading_col: str = "heading_deg",
                        output_dir: str = "./aerial_images_from_csv",
                        limit_rows: Optional[int] = None) -> int:
        """
        Process CSV file with odometry data to generate aerial images
        """
        try:
            df = pd.read_csv(csv_path)
            LOGK("CSV Load", f"loaded {len(df)} rows")
            
            # Check required columns
            required_cols = [x_col, y_col, heading_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                LOGK("Error", f"missing required columns: {missing_cols}")
                return 0
            
            # Limit rows if specified
            if limit_rows:
                df = df.head(limit_rows)
                LOGK("Processing", f"limited to {len(df)} rows")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            successful_downloads = 0
            
            LOGK("Processing", "starting aerial image processing")
            
            for i, row in df.iterrows():
                utm_x = float(row[x_col])
                utm_y = float(row[y_col])
                heading = float(row[heading_col])
                
                LOGD("Processing", f"[{i+1}/{len(df)}] UTM({utm_x:.1f}, {utm_y:.1f}) heading: {heading:.1f}°")
                
                # Download and process aerial image
                processed_image = self.download_and_process_from_utm(utm_x, utm_y, heading)
                
                if processed_image is not None:
                    # Save file
                    filename = f"frame_{i:06d}_utm_{utm_x:.1f}_{utm_y:.1f}_h_{heading:.1f}.png"
                    output_file = output_path / filename
                    
                    Image.fromarray(processed_image).save(output_file)
                    successful_downloads += 1
                    
                    LOGD("Save Image", f"saved {filename}")
                else:
                    LOGD("Error", "processing failed")
            
            LOGK("Complete", f"{successful_downloads}/{len(df)} images generated")
            return successful_downloads
            
        except Exception as e:
            LOGK("Error", f"processing CSV: {e}")
            return 0

    def get_stats(self) -> Dict[str, int]:
        """Return statistics"""
        return self.stats.copy()

    @classmethod
    def _normalize_map_type(cls, map_type: Optional[str]) -> str:
        value = (map_type or "satellite").strip().lower()
        if value.endswith("_base") and value not in cls._MAP_TYPE_ALIASES:
            return value
        return cls._MAP_TYPE_ALIASES.get(value, value)


def main():
    """Main function"""
    LOGK("Init", "NGII Aerial Image Downloader")
    
    # Naver Cloud Platform API key setup
    client_id = "keo25st5xp"
    client_secret = "RsWsl8Lb6Vacuzi3YA0ru6oNYl7bfghRZZdXWHCn"
    
    LOGK("API Setup", f"completed (ID: {client_id})")
    
    # Initialize downloader
    downloader = NGIIAerialImageDownloaderV4(
        client_id=client_id,
        client_secret=client_secret,
        physical_size=(120.0, 80.0),
        zoom_level=18,
        map_type="satellite_base",
        cache_dir="./cache/aerial_images_v4",
        max_retry_attempts=3,
        retry_backoff=2.0,
    )
    
    # Find first CSV file in specified path
    virtual_paths_dir = "/media/iismn/SSD A/Dataset/inhouse/Dataset/HDMap_Info/NGII_VIRTUAL_DRIVING_PATHS"
    
    LOGK("Path Check", f"checking {virtual_paths_dir}")
    
    if not os.path.exists(virtual_paths_dir):
        LOGK("Error", f"path not found: {virtual_paths_dir}")
        return
    
    # Find CSV files (prioritize odometry files)
    csv_files = list(Path(virtual_paths_dir).glob("*odometry*.csv"))
    if not csv_files:
        # If no odometry files, check all CSV files
        csv_files = list(Path(virtual_paths_dir).glob("*.csv"))
    
    if not csv_files:
        LOGK("Error", f"no CSV files found: {virtual_paths_dir}")
        return
    
    # Prioritize virtual_driving_path_1_odometry.csv
    target_file = Path(virtual_paths_dir) / "virtual_driving_path_1_odometry.csv"
    if target_file.exists():
        first_csv = target_file
    else:
        # Natural sort (by numbers)
        import re
        def natural_sort_key(path):
            numbers = re.findall(r'\d+', path.name)
            return [int(num) for num in numbers] if numbers else [0]
        
        csv_files.sort(key=natural_sort_key)
        first_csv = csv_files[0]
    
    LOGK("CSV Load", f"using {first_csv.name}")
    
    try:
        # Read CSV file
        df = pd.read_csv(first_csv)
        LOGK("CSV Load", f"loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ["x_utm", "y_utm"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            LOGK("Error", f"missing required columns: {missing_cols}")
            LOGK("Error", f"available columns: {list(df.columns)}")
            return
        
        # Extract data from first row
        first_row = df.iloc[0]
        utm_x = float(first_row["x_utm"])
        utm_y = float(first_row["y_utm"])
        
        # Calculate heading (direction between first and second points)
        if "heading_deg" in df.columns:
            heading = float(first_row["heading_deg"])
        elif len(df) >= 2:
            # Calculate heading from first to second point
            second_row = df.iloc[1]
            utm_x2 = float(second_row["x_utm"])
            utm_y2 = float(second_row["y_utm"])
            
            dx = utm_x2 - utm_x
            dy = utm_y2 - utm_y
            heading = math.degrees(math.atan2(dy, dx))
            LOGK("Processing", f"heading calculated: {heading:.1f}° (first->second point)")
        else:
            heading = 0.0
            LOGK("Processing", f"heading default: 0°")
        
        LOGK("Processing", f"position UTM({utm_x}, {utm_y}) heading {heading:.1f}°")
        
        LOGK("Processing", "starting aerial image processing")
        
        # Download and process aerial image
        result_image = downloader.download_and_process_from_utm(utm_x, utm_y, heading)
        
        if result_image is not None:
            # Save file
            output_file = f"aerial_first_position_utm_{utm_x:.1f}_{utm_y:.1f}_h_{heading:.1f}.png"
            Image.fromarray(result_image).save(output_file)
            
            LOGK("Save Image", f"saved {output_file}")
            LOGK("Complete", f"image size: {result_image.shape}")
            LOGK("Complete", f"physical size: 50m x 100m")
        else:
            LOGK("Error", "image processing failed")
            
    except Exception as e:
        LOGK("Error", f"processing CSV: {e}")
    
    # Print statistics
    stats = downloader.get_stats()
    stats_str = ", ".join([f"{key}: {value}" for key, value in stats.items()])
    LOGK("Statistics", f"session summary - {stats_str}")


if __name__ == "__main__":
    main()