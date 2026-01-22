import os
import os.path as osp

import cv2
import math
import pickle
import shapely
import requests
import argparse
import numpy as np
import shapely.affinity

from tqdm import tqdm
from io import BytesIO
from typing import Tuple
from shapely.geometry import box
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation as R
from openlanev2.lanesegment.dataset import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
import av2.geometry.utm as geo_utils


def is_valid_png(path: str) -> bool:
    """Check if PNG file is valid and can be opened."""
    try:
        with Image.open(path) as img:
            img.verify()  # Verify file integrity
        return True
    except Exception:
        return False


# ---------- tqdm settings ----------
POS_BASE = int(os.environ.get("TQDM_POS_BASE", "0"))
PART_IDX = os.environ.get("PART_IDX", "?")
TQDM_KW = dict(dynamic_ncols=True, ascii=True, mininterval=0.2)

# ---------- Global Session for connection pooling ----------
_session = None

def get_session():
    """Get or create a requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=3
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


# ============================================================
#  Tile Download Functions (from image_downloading.py)
# ============================================================

def download_tile(url, headers, channels):
    """Download a single tile from the given URL."""
    session = get_session()
    try:
        response = session.get(url, headers=headers, timeout=10)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        if channels == 3:
            return cv2.imdecode(arr, 1)
        return cv2.imdecode(arr, -1)
    except:
        return None


def project_with_scale(lat, lon, scale):
    """
    Mercator projection
    https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    """
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image_region(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3,
    max_workers: int = 16) -> np.ndarray:
    """
    Downloads a map region. Returns an image stored as a `numpy.ndarray` in BGR or BGRA.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner
    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner
    `zoom` - Zoom level
    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values
    `headers` - Dictionary of HTTP headers
    `tile_size` - Tile size in pixels
    `channels` - Number of channels in the output image
    `max_workers` - Maximum number of parallel download threads
    """
    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    # Collect all tile coordinates
    tiles_to_download = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tiles_to_download.append((tile_x, tile_y))
    
    def download_and_place(tile_coords):
        tile_x, tile_y = tile_coords
        tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)
        return tile_x, tile_y, tile
    
    # Download all tiles in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_and_place, tc) for tc in tiles_to_download]
        
        for future in as_completed(futures):
            tile_x, tile_y, tile = future.result()
            
            if tile is not None:
                # Find the pixel coordinates of the new tile relative to the image
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size

                # Define where the tile will be placed on the image
                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)

                # Define how border tiles will be cropped
                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]
    
    return img


# ============================================================
#  AerialCropper Class (No API Key Required)
# ============================================================

class AerialCropper:
    def __init__(self,
                 zoom: int = 20,
                 dist_x: float = 50,
                 dist_y: float = 100):
        """
        --------------
        AerialCropper       / Tile-based aerial imagery (No API key required)
        --------------
        Parameters:
        zoom                / zoom level
        dist_x, dist_y      / half-extent in meters
        ---------------
        """
        # Store parameters
        self.zoom = zoom
        self.dist_x = dist_x
        self.dist_y = dist_y

        # Tile settings
        self.tile_size = 256
        self.channels = 3
        
        # Google Maps Satellite Tile URL (no API key required)
        self.tile_url = 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        
        # HTTP Headers to mimic browser requests
        self.headers = {
            'cache-control': 'max-age=0',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
        }

        # Coordinate transformers
        self.to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Earth parameters
        self.earth_radius = 6_378_137

        # Print Banner
        ORANGE = "\033[38;5;208m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        LINE_CHAR = "─"

        def center_line(text, width=90):
            return text.center(width)
        
        def solid_line(width=90):
            return LINE_CHAR * width

        diag_m = 2 * np.hypot(self.dist_x, self.dist_y)
        print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")
        print(f"{ORANGE}{BOLD}{center_line('T-LSD Net / SD Map Generator [Imagery - No API]')}{RESET}")
        print(f"{ORANGE}{solid_line()}{RESET}")
        print(f"{ORANGE}{center_line(f'Resolution at zoom {self.zoom}')}{RESET}")
        print(f"{ORANGE}{center_line(f'Max envelope: {diag_m:.1f}x{diag_m:.1f} m')}{RESET}")
        print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")

    def compute_meters_per_pixel(self, lat):
        """Compute meters per pixel at given latitude."""
        lat_rad = np.deg2rad(lat)
        return (2 * np.pi * self.earth_radius * np.cos(lat_rad)) / (256 * (2 ** self.zoom))

    def utm_zone(self, lon):
        return int((lon + 180) // 6) + 1

    def bbox_rot_from_utm(self, lat0, lon0, dist_x, dist_y, rot_deg=0):
        """Compute rotated bounding box corners in WGS84."""
        # A: Select UTM CRS based on center point
        zone = self.utm_zone(lon0)
        hemi = 'north' if lat0 >= 0 else 'south'
        utm_crs = CRS.from_proj4(
            f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs"
        )

        # B: Transformer Coordinate Systems
        to_utm = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
        to_wgs = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

        # C: Center in UTM
        x0, y0 = to_utm.transform(lon0, lat0)

        # D: Local Bounding Box Corners (m) - half extents
        hx, hy = dist_x / 2, dist_y / 2
        local = [
            (-hx,  hy),  # NW
            ( hx,  hy),  # NE
            ( hx, -hy),  # SE
            (-hx, -hy),  # SW
        ]

        # E: Rotation Matrix
        theta = math.radians(rot_deg + 90.0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # F: Ego Vehicle Coordinate to UTM Coordinate
        utm_corners = []
        for dx, dy in local:
            xr = dx * cos_t - dy * sin_t
            yr = dx * sin_t + dy * cos_t
            utm_corners.append((x0 + xr, y0 + yr))

        # G: UTM → WGS84
        corners = []
        for xu, yu in utm_corners:
            lon, lat = to_wgs.transform(xu, yu)
            corners.append((lat, lon))

        # H: Final Bounding Box
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        north, south = max(lats), min(lats)
        east, west = max(lons), min(lons)

        return north, south, east, west, corners

    def get_cropped_EPSG4326(self, lat: float, lon: float, yaw_deg: float):
        """
        Fetch aerial imagery and crop to the rotated bounding box.
        
        Returns:
            img_tile: Full downloaded tile image (PIL Image)
            patch: Cropped and rotated patch (PIL Image)
        """
        # A: Compute resolution and crop size
        mpp = self.compute_meters_per_pixel(lat)
        diag_m = math.hypot(self.dist_x, self.dist_y)
        
        # Target crop size in pixels (dist_x, dist_y = total size in meters)
        ref_w = int(math.ceil(self.dist_x / mpp))
        ref_h = int(math.ceil(self.dist_y / mpp))
        ref_size = (ref_w, ref_h)

        # B: Compute bounding box for download 
        # Need diagonal size + margin for any rotation angle
        download_size = diag_m * 1.3  # Use diagonal with 30% margin
        north, south, east, west, geo_corners = self.bbox_rot_from_utm(
            lat, lon, download_size, download_size, rot_deg=0
        )

        # C: Download the region using tile-based method
        img_np = download_image_region(
            lat1=north, lon1=west,
            lat2=south, lon2=east,
            zoom=self.zoom,
            url=self.tile_url,
            headers=self.headers,
            tile_size=self.tile_size,
            channels=self.channels
        )

        # D: Convert to PIL Image (BGR to RGB)
        img_tile = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        W, H = img_tile.size
        cx, cy = W / 2.0, H / 2.0

        # E: Rotate the image
        img_unrot = img_tile.rotate(
            -(yaw_deg - 90.0),
            resample=Image.BICUBIC,
            center=(cx, cy),
            expand=False
        )

        # F: Crop to target size
        W_ref, H_ref = ref_size
        left = int(round(cx - W_ref / 2))
        top = int(round(cy - H_ref / 2))
        
        # Ensure crop bounds are valid
        left = max(0, min(left, W - W_ref))
        top = max(0, min(top, H - H_ref))
        
        patch = img_unrot.crop((left, top, left + W_ref, top + H_ref))

        return img_tile, patch


# ------- Main Function -------
def main():
    parser = argparse.ArgumentParser(description='Fetch and crop aerial imagery without API key.')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path to OpenLaneV2 data')
    parser.add_argument('--split', type=str, default='test',
                        help='split dir name (e.g., train/val/test)')
    parser.add_argument('--zoom', type=int, default=20,
                        help='Zoom level for aerial crop')
    parser.add_argument('--dist_x', type=float, default=80,
                        help='Half-extent in meters (x direction)')
    parser.add_argument('--dist_y', type=float, default=100,
                        help='Half-extent in meters (y direction)')
    parser.add_argument('--collection', type=str, default='data_dict_subset_A_train')
    parser.add_argument('--root_path_ArgoverseV2', type=str, 
                        default='/home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/ArgoverseV2',
                        help='Path to ArgoverseV2 data for city_dict')
    parser.add_argument('--total_parts', type=int, default=1,
                        help='Total number of parallel parts')
    parser.add_argument('--part', type=int, default=0,
                        help='Index of this parallel part (0-based)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files (default: skip existing)')
    args = parser.parse_args()

    # Auto-generate collection name if it contains {split}
    if '{split}' in args.collection:
        args.collection = args.collection.replace('{split}', args.split)

    # ---------------- banner ----------------
    ORANGE = "\033[38;5;208m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    LINE_CHAR = "─"
    
    def center_line(t, width=90):
        return t.center(width)
    
    def solid_line(width=90):
        return LINE_CHAR * width
    
    print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")
    print(f"{ORANGE}{BOLD}{center_line('Topo-LSD Net / SD Map Generator [Aerial Imagery, No API]')}{RESET}")
    print(f"{ORANGE}{solid_line()}{RESET}")
    for label, val in [
        ('Root Path', args.root_path),
        ('Collection', args.collection),
        ('Split', args.split),
        ('Part', f"{args.part+1}/{args.total_parts}"),
        ('Zoom', args.zoom),
    ]:
        print(f"{ORANGE}{BOLD}{center_line(f'{label}: {val}')}{RESET}")
    print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")

    # ---------------- setup ----------------
    col = Collection(args.root_path, args.root_path, args.collection)

    # -------- Load city_dict for lat/lon conversion -------
    city_pkl = osp.join(args.root_path_ArgoverseV2, f"{args.split}_city.pkl")
    if not osp.exists(city_pkl):
        raise FileNotFoundError(f"City dict not found: {city_pkl}")
    with open(city_pkl, 'rb') as f:
        city_dict = pickle.load(f)
    print(f"Loaded city_dict from {city_pkl} ({len(city_dict)} entries)")

    # -------- process frames (Multi Thread) -----------
    frames_items = list(col.frames.items())
    if args.total_parts > 1:
        per_part = len(frames_items) // args.total_parts + 1
        frames_items = frames_items[per_part * args.part : per_part * (args.part + 1)]

    # -------- Aerial Cropper Setup -------
    aerial = AerialCropper(
        zoom=args.zoom,
        dist_x=args.dist_x,
        dist_y=args.dist_y
    )

    # -------- Process Frame Function (for parallel execution) -------
    def process_frame(frame_data):
        frame_key, frame = frame_data
        split_name = frame_key[0]
        segment_id = frame_key[1]
        ts = frame_key[2]

        # prepare output path
        out_aerial_dir = osp.join(args.root_path, 'sd_map_imagery_all', split_name, segment_id)
        os.makedirs(out_aerial_dir, exist_ok=True)
        out_path = osp.join(out_aerial_dir, f"{ts}.png")
        
        # skip if already exists, valid, and not overwriting
        if osp.exists(out_path) and not args.overwrite:
            if is_valid_png(out_path):
                return 'skipped'
            # File exists but is invalid/corrupted, will be re-downloaded

        try:
            # get pose and rotation
            pose = frame.get_pose()
            rot_deg = R.from_matrix(pose['rotation']).as_euler('zyx', degrees=True)[0]

            # get lat/lon
            meta = frame.meta
            meta_data = meta.get('meta_data', {})
            source_id = meta_data.get('source_id')
            city_name = city_dict.get(source_id, 'PIT')
            
            translation = np.array(pose['translation'][:2]).reshape(1, 2)
            latlon = geo_utils.convert_city_coords_to_wgs84(translation, city_name=city_name)[0]
            lat, lon = latlon[0], latlon[1]

            # fetch and save cropped image
            img_tile, patch = aerial.get_cropped_EPSG4326(lat=lat, lon=lon, yaw_deg=rot_deg)
            patch.save(out_path)
            return 'success'
            
        except Exception as e:
            return f'error: {e}'

    # -------- Process Frames (Parallel) -----------
    GREEN, END = "\033[92m", "\033[0m"
    desc = f"AERIAL [{args.part+1}/{args.total_parts}]"
    
    max_workers = 4  # Number of parallel frame downloads
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frame, item): item[0] for item in frames_items}
        
        with tqdm(total=len(futures), desc=f"{GREEN}{desc}{END}", position=POS_BASE, leave=True, **TQDM_KW) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == 'success':
                    success_count += 1
                elif result == 'skipped':
                    skipped_count += 1
                else:
                    error_count += 1
                pbar.update(1)
    
    print(f"\n[Part {args.part+1}/{args.total_parts}] Done: {success_count} success, {skipped_count} skipped, {error_count} errors")


if __name__ == '__main__':
    main()
