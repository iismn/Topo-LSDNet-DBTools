#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import json
import math
import time
import pickle
import argparse
import warnings
import requests
import os.path as osp
from pathlib import Path

import numpy as np
import osmnx as ox
import geopandas as gpd
import shapely.geometry as geom

from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pyproj import CRS, Transformer
from PIL import Image, ImageDraw
from collections import defaultdict
from urllib3.exceptions import ProtocolError
from scipy.spatial.transform import Rotation as R
from osmnx._errors import EmptyOverpassResponse
from shapely.affinity import translate as affinity_translate, rotate as affinity_rotate

warnings.filterwarnings("ignore", message=".*street_count.*clean_periphery=False.*")

# ---------- tqdm settings ----------
POS_BASE = int(os.environ.get("TQDM_POS_BASE", "0"))
PART_IDX = os.environ.get("PART_IDX", "?")  # 선택
TQDM_KW = dict(dynamic_ncols=True, ascii=True, mininterval=0.2)

# ---------- label/color ----------
HIGHWAY_SETS = {
    'truck_road': ['motorway', 'trunk', 'motorway_link', 'trunk_link'],
    'highway': ['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link'],
    'residential': ['residential', 'living_street'],
    'service': ['service'],
    'road': ['road'],
    'bus_way': ['busway', 'bus_guideway']
}

# ---------- lane colors ----------
LANE_COLORS = {
    # ROAD : YELLOW
    'truck_road':  (255, 255,   0),  # yellow
    'highway':     (255, 255,   0),  # yellow
    'residential': (255, 255,   0),  # yellow
    'service':     (255, 255,   0),  # yellow
    'road':        (255, 255,   0),  # yellow
    'bus_way':     (255, 255,   0),  # yellow
    # BUILDING : RED
    'building':    (255,   0,   0),  # red
    'other':       (  0,   0,   0)   # black
}


class SDMapProcessor:
    """
    -----------------
    SD MAP PROCESSOR    / SD Map OSM API : parse OSMnx graph + rasterize to image
    -----------------
    Parameters:
    earth_radius        / Earth radius in meters (default: WGS84 value)
    --------------
    """
    # ---------- init ----------
    def __init__(self, earth_radius: float = 6_378_137.0):
        self.earth_radius = earth_radius
        self._epsg4326 = CRS.from_epsg(4326)

    # ---------- methods ----------
    def compute_meters_per_pixel(self, zoom: int, lat: float, tile_size: int = 512) -> float:
        R = 6378137.0
        lat_rad = np.deg2rad(lat)
        return (2 * np.pi * R * np.cos(lat_rad)) / (tile_size * (2**zoom))
    
    # ---------- UTM / bbox ----------
    @staticmethod
    def utm_zone(lon: float) -> int:
        return int((lon + 180) // 6) + 1

    # ---------- bbox with rotation ----------
    def bbox_rot_from_utm(self, lat0: float, lon0: float, dist_x: float, dist_y: float, rot_deg: float = 0.0):
        zone = self.utm_zone(lon0)
        hemi = 'north' if lat0 >= 0 else 'south'
        utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs")
        to_utm = Transformer.from_crs(self._epsg4326, utm_crs, always_xy=True)
        to_wgs = Transformer.from_crs(utm_crs, self._epsg4326, always_xy=True)

        x0, y0 = to_utm.transform(lon0, lat0)
        local = [(-dist_x, dist_y), (dist_x, dist_y), (dist_x, -dist_y), (-dist_x, -dist_y)]

        theta = math.radians(rot_deg + 90.0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        utm_corners = []
        for dx, dy in local:
            xr = dx * cos_t - dy * sin_t
            yr = dx * sin_t + dy * cos_t
            utm_corners.append((x0 + xr, y0 + yr))

        corners = []
        for xu, yu in utm_corners:
            lon, lat = to_wgs.transform(xu, yu)
            corners.append((lat, lon))

        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        north, south = max(lats), min(lats)
        east, west = max(lons), min(lons)
        return north, south, east, west, corners

    # ---------- graph parsing (N/A) ----------
    def parse_graph(self, G_simplify, xo, yo, rot_deg, bbox_corners=None):
        # parse roads
        _, gdf_edges = ox.utils_graph.graph_to_gdfs(
            G_simplify, node_geometry=True, fill_edge_geometry=True
        )
        gdf = gdf_edges[['highway','geometry']].explode('highway')
        gdf['HIGHWAY_TYPE'] = gdf.highway.apply(
            lambda h: next((k for k,v in HIGHWAY_SETS.items() if h in v), 'other')
        )
        
        # Set UTM based on centroid
        centroid = gdf.geometry.unary_union.centroid
        lon_c, lat_c = centroid.x, centroid.y
        zone = self.utm_zone(lon_c)
        hemi = 'north' if lat_c >= 0 else 'south'
        utm_crs = CRS.from_proj4(
            f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs"
        )
        to_utm = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
        x0, y0 = to_utm.transform(xo, yo)
        gdf_utm = gdf.to_crs(utm_crs)
        angle = -rot_deg
        gdf_utm['geometry'] = gdf_utm.geometry.apply(
            lambda geom: affinity_rotate(
                affinity_translate(geom, xoff=-x0, yoff=-y0), angle, origin=(0,0)
            )
        )
        graph = {}
        for lane in list(HIGHWAY_SETS.keys()) + ['other']:
            polys = [np.stack(line.coords.xy, axis=-1)
                     for line in gdf_utm[gdf_utm.HIGHWAY_TYPE==lane].geometry]
            graph[lane] = polys
        # parse buildings if bbox provided
        if bbox_corners is not None:
            # build WGS84 polygon from bbox_corners (lat,lon)
            poly_ll = geom.Polygon([(lon, lat) for lat, lon in bbox_corners])
            try:
                bld = ox.features_from_polygon(poly_ll, tags={'building':True})
            except EmptyOverpassResponse:
                # 건물 데이터가 없으면 빈 GeoDataFrame 생성
                bld = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
            bld = bld[bld.geometry.geom_type.isin(['Polygon','MultiPolygon'])]
            bld_utm = bld.to_crs(utm_crs).geometry.apply(
                lambda g: affinity_rotate(
                    affinity_translate(g, xoff=-x0, yoff=-y0), angle, origin=(0,0)
                )
            )
            bpolys = []
            for geom_item in bld_utm:
                if geom_item.geom_type == 'Polygon':
                    bpolys.append(np.stack(geom_item.exterior.coords.xy, axis=-1))
                else:
                    for part in geom_item.geoms:
                        bpolys.append(np.stack(part.exterior.coords.xy, axis=-1))
            graph['building'] = bpolys
        return graph

    # ---------- rasterization OSM ----------
    def rasterize_graph(self, graph, dist_x, dist_y, resolution=0.2360):
        
        # A: Calculate image size
        width_px  = int((2 * dist_y) / resolution)
        height_px = int((2 * dist_x) / resolution)

        # B: Create single-channel binary images for each channel
        chan_imgs = [Image.new('L', (width_px, height_px), 0) for _ in range(3)]
        drawers   = [ImageDraw.Draw(im) for im in chan_imgs]

        # C: densify function
        def densify(coords, max_dist):
            out = []
            for (x0,y0), (x1,y1) in zip(coords[:-1], coords[1:]):
                out.append((x0,y0))
                dx, dy = x1-x0, y1-y0
                d = math.hypot(dx,dy)
                if d > max_dist:
                    n = int(math.ceil(d/max_dist))
                    for i in range(1,n):
                        f = i/n
                        out.append((x0+dx*f, y0+dy*f))
            out.append(coords[-1])
            return out

        # D: Draw polygons
        road_types = set(HIGHWAY_SETS.keys())
        for k, polys in graph.items():
            if k == 'other':
                # skip
                continue
            if k == 'building':
                # building
                ch = 1
                stroke = 8
            else:
                # road
                ch = 0
                stroke = 20

            for poly in polys:
                dense = densify(poly, resolution)
                pts = [((x + dist_y) / resolution, (dist_x - y) / resolution)
                       for x,y in dense]
                drawers[ch].line(pts, fill=255, width=stroke)

        # E: Stack channels
        mask = np.stack([np.array(im, dtype=np.uint8) for im in chan_imgs], axis=0)

        # F: Background Channels Generation
        bg = np.logical_and(mask[0]==0, mask[1]==0).astype(np.uint8) * 255
        mask[2] = bg

        # G: PIL Image Conversion & Rotate
        img = Image.fromarray(mask.transpose(1, 2, 0), mode='RGB')
        return img.transpose(Image.ROTATE_90)


# --------- Yaw & Lat-Lon loaders ---------
def load_yaw_deg_from_info(json_path: str) -> float:
    with open(json_path, 'r') as f:
        info = json.load(f)

    for k in ('yaw', 'heading'):
        if k in info and isinstance(info[k], (int, float)):
            return float(info[k])

    if 'yaw_rad' in info and isinstance(info['yaw_rad'], (int, float)):
        return float(np.degrees(info['yaw_rad']))

    if 'rotation' in info:
        rot = np.array(info['rotation'])
        if rot.size == 9:
            rot = rot.reshape(3, 3)
            return float(R.from_matrix(rot).as_euler('zyx', degrees=True)[0])

    pose = info.get('pose')
    if isinstance(pose, dict):
        rot = pose.get('rotation')
        if rot is not None:
            rot = np.array(rot)
            if rot.size == 9:
                rot = rot.reshape(3, 3)
                return float(R.from_matrix(rot).as_euler('zyx', degrees=True)[0])
        heading = pose.get('heading')
        if isinstance(heading, (int, float)):
            return float(heading)

    if 'ego2global_rotation' in info:
        q = np.array(info['ego2global_rotation'], dtype=float).flatten()
        if q.size == 4:
            w, x, y, z = q.tolist()
            return float(R.from_quat([x, y, z, w]).as_euler('zyx', degrees=True)[0])

    meta = info.get('meta_data')
    if isinstance(meta, dict):
        heading = meta.get('ego_heading_deg')
        if isinstance(heading, (int, float)):
            return float(heading)

    return 0.0

# --------- Lat-Lon loader (NAVSIM) ---------
def load_lat_lon_from_traj(npy_path: str) -> Tuple[float, float]:
    a = np.load(npy_path, allow_pickle=True)
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError(f"{npy_path}: expected [lat, lon], got shape={a.shape}")
    lat, lon = float(a[0]), float(a[1])
    # sanity check
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"{npy_path}: values not in lat/lon range: {(lat, lon)}")
    return lat, lon


def discover_inhouse_bags(
    dataset_root: str,
    split: str,
    city_filter: Optional[str] = None,
    bag_filter: Optional[str] = None,
) -> List[Dict[str, object]]:
    root = Path(dataset_root).expanduser()
    split_dir = root / split
    if not split_dir.is_dir():
        return []
    entries: List[Dict[str, object]] = []
    for city_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        if city_filter and city_dir.name != city_filter:
            continue
        for bag_dir in sorted([p for p in city_dir.iterdir() if p.is_dir()]):
            if bag_filter and bag_dir.name != bag_filter:
                continue
            info_dir = bag_dir / "info"
            traj_dir = bag_dir / "traj"
            image_dir = bag_dir / "image"
            if not (info_dir.is_dir() and traj_dir.is_dir() and image_dir.is_dir()):
                continue
            samples: List[Dict[str, object]] = []
            for json_path in sorted(info_dir.glob("*-ls.json")):
                timestamp = json_path.name
                if timestamp.endswith("-ls.json"):
                    timestamp = timestamp[: -len("-ls.json")]
                traj_path = traj_dir / f"{timestamp}_egopose.npy"
                if not traj_path.exists():
                    continue
                samples.append(
                    {
                        "timestamp": timestamp,
                        "info_json": json_path,
                        "traj_path": traj_path,
                    },
                )
            if samples:
                entries.append(
                    {
                        "split": split,
                        "city": city_dir.name,
                        "bag": bag_dir.name,
                        "base_dir": bag_dir,
                        "samples": samples,
                    },
                )
    return entries

# --------- Graph dict from gdfs (WGS84 → UTM → ego-frame) ---------
def _graph_dict_from_gdfs(edges_wgs: gpd.GeoDataFrame,
                          bld_wgs: gpd.GeoDataFrame,
                          xo: float, yo: float, rot_deg: float,
                          processor: SDMapProcessor) -> dict:
    """
    edges_wgs, bld_wgs : WGS84 (lon/lat)
    xo, yo             : origin lon/lat
    rot_deg            : heading
    processor          : for UTM/crs utils
    """
    # Empty graph Initialization
    graph = {k: [] for k in list(HIGHWAY_SETS.keys()) + ['other', 'building']}
    if edges_wgs is None or edges_wgs.empty:
        return graph

    # HIGHWAY_TYPE
    gdf = edges_wgs.copy()
    if 'highway' in gdf.columns:
        gdf = gdf[['highway', 'geometry']]
        gdf['HIGHWAY_TYPE'] = gdf.highway.map(
            lambda h: next((k for k, v in HIGHWAY_SETS.items() if isinstance(h, str) and h in v), 'other')
        )
    else:
        gdf = gpd.GeoDataFrame(
            {'HIGHWAY_TYPE': 'other', 'geometry': gdf.geometry.values}, crs="EPSG:4326"
        )

    # UTM Selection
    centroid = gdf.geometry.unary_union.centroid
    lon_c, lat_c = centroid.x, centroid.y
    zone = processor.utm_zone(lon_c)
    hemi = 'north' if lat_c >= 0 else 'south'
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs")
    to_utm = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
    x0, y0 = to_utm.transform(xo, yo)

    # WGS84 → UTM → ego-frame
    gdf_utm = gdf.to_crs(utm_crs)
    angle = -rot_deg
    gdf_utm['geometry'] = gdf_utm.geometry.map(
        lambda geom: affinity_rotate(
            affinity_translate(geom, xoff=-x0, yoff=-y0), angle, origin=(0, 0)
        )
    )

    # Line Collection
    for lane in list(HIGHWAY_SETS.keys()) + ['other']:
        sub = gdf_utm[gdf_utm.HIGHWAY_TYPE == lane].geometry
        for geom_item in sub:
            if geom_item.is_empty:
                continue
            if geom_item.geom_type == 'LineString':
                xs, ys = geom_item.coords.xy
                graph[lane].append(np.stack([xs, ys], axis=-1))
            elif geom_item.geom_type == 'MultiLineString':
                for part in geom_item.geoms:
                    xs, ys = part.coords.xy
                    graph[lane].append(np.stack([xs, ys], axis=-1))

    # Building
    bpolys = []
    if bld_wgs is not None and not bld_wgs.empty:
        bld_utm = bld_wgs.to_crs(utm_crs).geometry.map(
            lambda g: affinity_rotate(
                affinity_translate(g, xoff=-x0, yoff=-y0), angle, origin=(0, 0)
            )
        )
        for geom_item in bld_utm:
            if geom_item.geom_type == 'Polygon':
                xs, ys = geom_item.exterior.coords.xy
                bpolys.append(np.stack([xs, ys], axis=-1))
            elif geom_item.geom_type == 'MultiPolygon':
                for part in geom_item.geoms:
                    xs, ys = part.exterior.coords.xy
                    bpolys.append(np.stack([xs, ys], axis=-1))
    graph['building'] = bpolys
    return graph

# --------- main function ----------
def main():

    # -------- args ----------
    parser = argparse.ArgumentParser(
        description="Inhouse SD-map exporter (bag-level OSM fetch + per-frame clip)."
    )
    parser.add_argument('--dataset_root', type=str, default='/home/iismn/Workspace_Share/IEEE_CVF_CVPR/Inhouse',
                        help='Root directory containing split/city/bag folders (default: %(default)s)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split directory name (e.g., train/val)')
    parser.add_argument('--city', type=str, default=None,
                        help='Optional city name filter (process all if omitted)')
    parser.add_argument('--bag', type=str, default=None,
                        help='Optional bag (rosbag) folder filter')
    parser.add_argument('--dist_x', type=float, default=80.0,
            help='Total forward/back coverage in meters for each frame crop')
    parser.add_argument('--dist_y', type=float, default=120.0,
            help='Total left/right coverage in meters for each frame crop')
    parser.add_argument('--retry_delay', type=int, default=5,
                        help='Overpass retry delay seconds')
    parser.add_argument('--total_parts', type=int, default=1,
                        help='Total number of parallel parts')
    parser.add_argument('--part', type=int, default=0,
                        help='Index of this parallel part (0-based)')
    parser.add_argument('--zoom_for_res', type=int, default=18,
                        help='Zoom level used to derive meters-per-pixel')
    parser.add_argument('--log_margin_m', type=float, default=400.0,
                        help='Extra meters added around bag-level bbox before OSM fetch')
    parser.add_argument('--network_type', type=str, default='all',
                        choices=['drive','walk','bike','all','all_private'],
                        help='OSM network type for LOG-level graph fetch')
    parser.add_argument('--graph_only', action='store_true',
                        help='Only generate sd_graph/*.pkl (skip sd_raster/*.png)')
    parser.add_argument('--overwrite', action='store_true',
                help='Regenerate sd_graph/sd_raster even if files already exist')
    parser.add_argument('--verbose', action='store_true',
                help='Enable extra progress logs')
    args = parser.parse_args()

    dataset_root = str(Path(args.dataset_root).expanduser())

    # ---------------- banner ----------------
    ORANGE = "\033[38;5;208m"; BOLD = "\033[1m"; RESET = "\033[0m"; LINE_CHAR = "─"
    def center_line(t, width=90): return t.center(width)
    def solid_line(width=90): return LINE_CHAR * width
    print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")
    print(f"{ORANGE}{BOLD}{center_line('Inhouse SD Map Generator [Graph + Raster, Bag-level fetch]')}{RESET}")
    print(f"{ORANGE}{solid_line()}{RESET}")
    for label, val in [
        ('Dataset Root', dataset_root),
        ('Split', args.split),
        ('City', args.city or 'ALL'),
        ('Bag', args.bag or 'ALL'),
        ('Part', f"{args.part+1}/{args.total_parts}"),
    ]:
        print(f"{ORANGE}{BOLD}{center_line(f'{label}: {val}')}{RESET}")
    print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")

    # ---------------- setup ----------------
    bag_entries = discover_inhouse_bags(dataset_root, args.split, args.city, args.bag)
    if not bag_entries:
        print("No bag directories with image/info/traj found for the given filters.")
        return
    if args.total_parts > 1:
        per_part = max(1, len(bag_entries) // args.total_parts + (1 if len(bag_entries) % args.total_parts else 0))
        start = per_part * args.part
        end = per_part * (args.part + 1)
        bag_entries = bag_entries[start:end]
        if not bag_entries:
            print("No bag directories assigned to this part.")
            return

    processor = SDMapProcessor()
    verbose = bool(args.verbose)
    crop_forward_m = float(args.dist_x)
    crop_lateral_m = float(args.dist_y)
    dx = crop_forward_m * 0.5
    dy = crop_lateral_m * 0.5
    do_raster = not args.graph_only

    # ---------- OSMnx settings ----------
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180
    ox.settings.overpass_rate_limit = True
    ox.settings.cache_folder = "/tmp/osmnx_cache"

    GREEN, END = "\033[92m", "\033[0m"
    outer_desc = f"[part {args.part+1}] Bag-Level"
    for entry in tqdm(bag_entries, desc=f"{GREEN}{outer_desc}{END}", position=POS_BASE, leave=True, **TQDM_KW):
        bag_label = f"{entry['city']}/{entry['bag']}"
        base_dir = Path(entry['base_dir'])
        samples = entry['samples']
        latlons: List[Tuple[float, float]] = []
        valid_samples: List[Dict[str, object]] = []
        for sample in samples:
            traj_path = Path(sample['traj_path'])
            info_json = Path(sample['info_json'])
            ts = sample['timestamp']
            try:
                lat, lon = load_lat_lon_from_traj(str(traj_path))
            except Exception:
                continue
            latlons.append((lat, lon))
            valid_samples.append(
                {
                    "timestamp": ts,
                    "traj_path": traj_path,
                    "info_json": info_json,
                    "lat": lat,
                    "lon": lon,
                }
            )

        if not latlons:
            continue

        lats = np.array([p[0] for p in latlons], dtype=float)
        lons = np.array([p[1] for p in latlons], dtype=float)
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_c = 0.5 * (lat_min + lat_max)
        dlat = (args.log_margin_m / processor.earth_radius) * (180.0 / np.pi)
        dlon = dlat / max(np.cos(np.deg2rad(lat_c)), 1e-6)
        north_big = lat_max + dlat
        south_big = lat_min - dlat
        east_big = lon_max + dlon
        west_big = lon_min - dlon

        for _retry in range(3):
            try:
                G_big = ox.graph_from_bbox(
                    north_big,
                    south_big,
                    east_big,
                    west_big,
                    network_type=args.network_type,
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                    clean_periphery=True,
                )
                break
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                ProtocolError,
                json.JSONDecodeError,
            ):
                time.sleep(args.retry_delay)
        else:
            continue

        _scene_nodes_wgs, scene_edges_wgs = ox.utils_graph.graph_to_gdfs(
            G_big,
            nodes=True,
            edges=True,
            node_geometry=True,
            fill_edge_geometry=True,
        )
        keep_cols = [c for c in ('highway', 'geometry') if c in scene_edges_wgs.columns]
        scene_edges_wgs = scene_edges_wgs[keep_cols].copy()

        try:
            scene_buildings_wgs = ox.features_from_bbox(
                north_big, south_big, east_big, west_big, tags={'building': True}
            )
            if not scene_buildings_wgs.empty:
                scene_buildings_wgs = scene_buildings_wgs[['geometry']].copy()
                scene_buildings_wgs = scene_buildings_wgs[
                    scene_buildings_wgs.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
                ]
            else:
                scene_buildings_wgs = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
        except Exception:
            scene_buildings_wgs = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

        inner_desc = f"[part {args.part+1}] {bag_label}"
        for sample in tqdm(valid_samples, desc=inner_desc, position=POS_BASE + 1, leave=False, **TQDM_KW):
            out_graph_dir = base_dir / 'sd_graph'
            out_graph_dir.mkdir(parents=True, exist_ok=True)
            out_raster_dir = base_dir / 'sd_raster'
            if do_raster:
                out_raster_dir.mkdir(parents=True, exist_ok=True)
            out_pkl = out_graph_dir / f"{sample['timestamp']}.pkl"
            out_png = out_raster_dir / f"{sample['timestamp']}.png" if do_raster else None
            pkl_exists = out_pkl.exists()
            png_exists = out_png.exists() if (do_raster and out_png is not None) else False
            if not args.overwrite and pkl_exists and (not do_raster or png_exists):
                if verbose:
                    print(f"Skip {bag_label}/{sample['timestamp']} (already exists)")
                continue

            lat = float(sample['lat'])
            lon = float(sample['lon'])
            yaw_deg = load_yaw_deg_from_info(str(sample['info_json']))

            north, south, east, west, frame_corners = processor.bbox_rot_from_utm(lat, lon, dx, dy, yaw_deg)
            frame_poly_ll = geom.Polygon([(lo, la) for la, lo in frame_corners])

            try:
                edges_clip = gpd.clip(scene_edges_wgs, frame_poly_ll)
            except Exception:
                edges_clip = scene_edges_wgs[scene_edges_wgs.geometry.intersects(frame_poly_ll)].copy()

            try:
                bld_clip = gpd.clip(scene_buildings_wgs, frame_poly_ll)
            except Exception:
                bld_clip = scene_buildings_wgs[scene_buildings_wgs.geometry.intersects(frame_poly_ll)].copy()

            graph_dict = _graph_dict_from_gdfs(edges_clip, bld_clip, lon, lat, yaw_deg, processor)
            with open(out_pkl, 'wb') as f:
                pickle.dump(graph_dict, f)

            if do_raster:
                res = processor.compute_meters_per_pixel(args.zoom_for_res, lat)
                img = processor.rasterize_graph(graph_dict, dx, dy, resolution=res)
                img.save(out_png)
                del img

            del edges_clip, bld_clip, graph_dict
            gc.collect()

        del G_big, scene_edges_wgs, scene_buildings_wgs
        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
