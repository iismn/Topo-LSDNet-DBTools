#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD Map Generator for OpenLaneV2 Dataset
- Fetches OSM road/building data
- Converts to ego-centric coordinate frame
- Saves graph (.pkl) and raster (.png) outputs
"""
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

import numpy as np
import osmnx as ox
import geopandas as gpd
import shapely.geometry as geom

from tqdm import tqdm
from typing import Tuple
from pyproj import CRS, Transformer
from PIL import Image, ImageDraw
from collections import defaultdict
from urllib3.exceptions import ProtocolError


def is_valid_pkl(path: str) -> bool:
    """Check if pkl file is valid and can be loaded."""
    try:
        with open(path, 'rb') as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def is_valid_png(path: str) -> bool:
    """Check if PNG file is valid and can be opened."""
    try:
        with Image.open(path) as img:
            img.verify()  # Verify file integrity
        return True
    except Exception:
        return False
from scipy.spatial.transform import Rotation as R
from osmnx._errors import EmptyOverpassResponse
from openlanev2.lanesegment.dataset import Collection
from shapely.affinity import translate as affinity_translate, rotate as affinity_rotate
from concurrent.futures import ThreadPoolExecutor, as_completed
import av2.geometry.utm as geo_utils

warnings.filterwarnings("ignore", message=".*street_count.*clean_periphery=False.*")

# ---------- tqdm settings ----------
POS_BASE = int(os.environ.get("TQDM_POS_BASE", "0"))
PART_IDX = os.environ.get("PART_IDX", "?")
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
    'truck_road':  (255, 255,   0),
    'highway':     (255, 255,   0),
    'residential': (255, 255,   0),
    'service':     (255, 255,   0),
    'road':        (255, 255,   0),
    'bus_way':     (255, 255,   0),
    'building':    (255,   0,   0),
    'other':       (  0,   0,   0)
}


class SDMapProcessor:
    """
    SD MAP PROCESSOR - parse OSMnx graph + rasterize to image
    """
    def __init__(self, earth_radius: float = 6_378_137.0):
        self.earth_radius = earth_radius
        self._epsg4326 = CRS.from_epsg(4326)

    def compute_meters_per_pixel(self, zoom: int, lat: float, tile_size: int = 512) -> float:
        R = 6378137.0
        lat_rad = np.deg2rad(lat)
        return (2 * np.pi * R * np.cos(lat_rad)) / (tile_size * (2**zoom))
    
    @staticmethod
    def utm_zone(lon: float) -> int:
        return int((lon + 180) // 6) + 1

    def bbox_rot_from_utm(self, lat0: float, lon0: float, dist_x: float, dist_y: float, rot_deg: float = 0.0):
        zone = self.utm_zone(lon0)
        hemi = 'north' if lat0 >= 0 else 'south'
        utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs")
        to_utm = Transformer.from_crs(self._epsg4326, utm_crs, always_xy=True)
        to_wgs = Transformer.from_crs(utm_crs, self._epsg4326, always_xy=True)

        x0, y0 = to_utm.transform(lon0, lat0)
        # dist_x, dist_y are TOTAL size, so use half for corners
        half_x, half_y = dist_x / 2, dist_y / 2
        local = [(-half_x, half_y), (half_x, half_y), (half_x, -half_y), (-half_x, -half_y)]

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

    def rasterize_graph(self, graph, dist_x, dist_y, resolution=0.2360):
        # dist_x, dist_y are TOTAL size
        half_x, half_y = dist_x / 2, dist_y / 2
        
        width_px  = int(dist_y / resolution)
        height_px = int(dist_x / resolution)

        chan_imgs = [Image.new('L', (width_px, height_px), 0) for _ in range(3)]
        drawers   = [ImageDraw.Draw(im) for im in chan_imgs]

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

        for k, polys in graph.items():
            if k == 'other':
                continue
            if k == 'building':
                ch = 1
                stroke = 16
            else:
                ch = 0
                stroke = 40

            for poly in polys:
                dense = densify(poly, resolution)
                pts = [((x + half_y) / resolution, (half_x - y) / resolution)
                       for x,y in dense]
                drawers[ch].line(pts, fill=255, width=stroke)

        mask = np.stack([np.array(im, dtype=np.uint8) for im in chan_imgs], axis=0)
        bg = np.logical_and(mask[0]==0, mask[1]==0).astype(np.uint8) * 255
        mask[2] = bg

        img = Image.fromarray(mask.transpose(1, 2, 0), mode='RGB')
        return img.transpose(Image.ROTATE_90)


def _graph_dict_from_gdfs(edges_wgs: gpd.GeoDataFrame,
                          bld_wgs: gpd.GeoDataFrame,
                          xo: float, yo: float, rot_deg: float,
                          processor: SDMapProcessor) -> dict:
    """
    edges_wgs, bld_wgs : WGS84 (lon/lat)
    xo, yo             : origin lon/lat
    rot_deg            : heading
    """
    graph = {k: [] for k in list(HIGHWAY_SETS.keys()) + ['other', 'building']}
    if edges_wgs is None or edges_wgs.empty:
        return graph

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

    centroid = gdf.geometry.unary_union.centroid
    lon_c, lat_c = centroid.x, centroid.y
    zone = processor.utm_zone(lon_c)
    hemi = 'north' if lat_c >= 0 else 'south'
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs")
    to_utm = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
    x0, y0 = to_utm.transform(xo, yo)

    gdf_utm = gdf.to_crs(utm_crs)
    angle = -rot_deg
    gdf_utm['geometry'] = gdf_utm.geometry.map(
        lambda geom: affinity_rotate(
            affinity_translate(geom, xoff=-x0, yoff=-y0), angle, origin=(0, 0)
        )
    )

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


def main():
    parser = argparse.ArgumentParser(
        description="OpenLaneV2 SD-map exporter (segment-level OSM fetch + per-frame clip)."
    )
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path to OpenLaneV2 data')
    parser.add_argument('--root_path_ArgoverseV2', type=str, 
                        default='/home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/ArgoverseV2',
                        help='Path to ArgoverseV2 data for city_dict')
    parser.add_argument('--split', type=str, default='train',
                        help='split dir name (e.g., train/val/test)')
    parser.add_argument('--collection', type=str, default='data_dict_subset_A_{split}_ls',
                        help='OpenLaneV2 Collection name (use {split} placeholder)')
    parser.add_argument('--dist_x', type=float, default=50.0,
                        help='Total extent (x, forward/back) in meters')
    parser.add_argument('--dist_y', type=float, default=100.0,
                        help='Total extent (y, left/right) in meters')
    parser.add_argument('--buffer_m', type=float, default=50.0,
                        help='extra buffer meters added around per-frame bbox')
    parser.add_argument('--retry_delay', type=int, default=5,
                        help='Overpass retry delay seconds')
    parser.add_argument('--rasterize', action='store_true',
                        help='also save PNG alongside .pkl')
    parser.add_argument('--total_parts', type=int, default=1,
                        help='Total number of parallel parts')
    parser.add_argument('--part', type=int, default=0,
                        help='Index of this parallel part (0-based)')
    parser.add_argument('--zoom_for_res', type=int, default=20,
                        help='Zoom level used to derive meters-per-pixel')
    parser.add_argument('--log_margin_m', type=float, default=400.0,
                        help='Extra meters added to segment-level big bbox')
    parser.add_argument('--network_type', type=str, default='all',
                        choices=['drive','walk','bike','all','all_private'],
                        help='OSM network type for segment-level graph fetch')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files (default: skip existing)')
    args = parser.parse_args()

    # Auto-generate collection name if it contains {split}
    if '{split}' in args.collection:
        args.collection = args.collection.replace('{split}', args.split)

    # ---------------- banner ----------------
    ORANGE = "\033[38;5;208m"; BOLD = "\033[1m"; RESET = "\033[0m"; LINE_CHAR = "─"
    def center_line(t, width=90): return t.center(width)
    def solid_line(width=90): return LINE_CHAR * width
    print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")
    print(f"{ORANGE}{BOLD}{center_line('Topo-LSD Net / SD Map Generator [Graph + Raster, OpenLaneV2]')}{RESET}")
    print(f"{ORANGE}{solid_line()}{RESET}")
    for label, val in [
        ('Root Path', args.root_path),
        ('Collection', args.collection),
        ('Split', args.split),
        ('Part', f"{args.part+1}/{args.total_parts}"),
        ('Dist', f"{args.dist_x}m x {args.dist_y}m"),
        ('Overwrite', args.overwrite),
    ]:
        print(f"{ORANGE}{BOLD}{center_line(f'{label}: {val}')}{RESET}")
    print(f"{ORANGE}{BOLD}{solid_line()}{RESET}")

    # ---------------- setup ----------------
    col = Collection(args.root_path, args.root_path, args.collection)
    processor = SDMapProcessor()
    dx, dy = args.dist_x, args.dist_y

    # Load city_dict for lat/lon conversion
    city_pkl = osp.join(args.root_path_ArgoverseV2, f"{args.split}_city.pkl")
    if not osp.exists(city_pkl):
        raise FileNotFoundError(f"City dict not found: {city_pkl}")
    with open(city_pkl, 'rb') as f:
        city_dict = pickle.load(f)
    print(f"Loaded city_dict from {city_pkl} ({len(city_dict)} entries)")

    # ---------- OSMnx settings ----------
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180
    ox.settings.overpass_rate_limit = True
    ox.settings.cache_folder = "/tmp/osmnx_cache"

    # --------- FRAME List (part) ----------
    frames_items = list(col.frames.items())
    if args.total_parts > 1:
        per_part = len(frames_items) // args.total_parts + 1
        frames_items = frames_items[per_part*args.part : per_part*(args.part+1)]

    # -------- Segment-level grouping --------
    # key: (split_name, segment_id)
    frames_by_segment = defaultdict(list)
    for frame_key, frame in frames_items:
        split_name = frame_key[0]
        if split_name != args.split:
            continue
        segment_id = frame_key[1]
        ts = frame_key[2]
        frames_by_segment[(split_name, segment_id)].append((frame_key, frame, ts))

    GREEN, END = "\033[92m", "\033[0m"
    outer_desc = f"[part {PART_IDX}] Segment-Level"
    
    for (split_name, segment_id), items in tqdm(frames_by_segment.items(), desc=f"{GREEN}{outer_desc}{END}", position=POS_BASE, leave=True, **TQDM_KW):

        # ---- Lat-Lon Collection for segment bbox ----
        latlons = []
        valid_items = []
        
        for (frame_key, frame, ts) in items:
            try:
                pose = frame.get_pose()
                meta = frame.meta
                meta_data = meta.get('meta_data', {})
                source_id = meta_data.get('source_id')
                city_name = city_dict.get(source_id, 'PIT')
                
                translation = np.array(pose['translation'][:2]).reshape(1, 2)
                latlon = geo_utils.convert_city_coords_to_wgs84(translation, city_name=city_name)[0]
                lat, lon = latlon[0], latlon[1]
                
                rot_deg = R.from_matrix(pose['rotation']).as_euler('zyx', degrees=True)[0]
                
                latlons.append((lat, lon))
                valid_items.append((frame_key, frame, ts, lat, lon, rot_deg))
            except Exception:
                continue

        if not latlons:
            continue

        lats = np.array([p[0] for p in latlons], dtype=float)
        lons = np.array([p[1] for p in latlons], dtype=float)
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

        # ---- Segment-level big bbox with margin ----
        lat_c = 0.5*(lat_min+lat_max)
        dlat = (args.log_margin_m / processor.earth_radius) * (180.0/np.pi)
        dlon = dlat / max(np.cos(np.deg2rad(lat_c)), 1e-6)

        north_big = lat_max + dlat
        south_big = lat_min - dlat
        east_big  = lon_max + dlon
        west_big  = lon_min - dlon

        # ---- Segment-level OSM fetch ----
        # ROAD GRAPH
        for _retry in range(3):
            try:
                G_big = ox.graph_from_bbox(
                    north_big, south_big, east_big, west_big,
                    network_type=args.network_type,
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                    clean_periphery=True
                )
                break
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    ProtocolError,
                    json.JSONDecodeError):
                time.sleep(args.retry_delay)
        else:
            continue

        scene_nodes_wgs, scene_edges_wgs = ox.utils_graph.graph_to_gdfs(
            G_big,
            nodes=True,
            edges=True,
            node_geometry=True,
            fill_edge_geometry=True
        )
        keep_cols = [c for c in ('highway', 'geometry') if c in scene_edges_wgs.columns]
        scene_edges_wgs = scene_edges_wgs[keep_cols].copy()

        # BUILDINGS GRAPH
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

        # ---- CLIP + EXPORT (per-frame) ----
        inner_desc = f"[part {PART_IDX}] {segment_id}"
        for (frame_key, frame, ts, lat, lon, rot_deg) in tqdm(valid_items, desc=inner_desc, position=POS_BASE + 1, leave=False, **TQDM_KW):
            # Output paths: sd_map_graph_all/{split}/{segment_id}/{ts}.pkl
            #               sd_map_raster_all/{split}/{segment_id}/{ts}.png
            out_graph_dir  = osp.join(args.root_path, 'sd_map_graph_all', split_name, segment_id)
            out_raster_dir = osp.join(args.root_path, 'sd_map_raster_all', split_name, segment_id)
            os.makedirs(out_graph_dir, exist_ok=True)
            if args.rasterize:
                os.makedirs(out_raster_dir, exist_ok=True)

            out_pkl = osp.join(out_graph_dir, f"{ts}.pkl")
            out_png = osp.join(out_raster_dir, f"{ts}.png")
            
            # Skip if exists, valid, and not overwriting
            if not args.overwrite:
                pkl_valid = osp.exists(out_pkl) and is_valid_pkl(out_pkl)
                png_valid = (not args.rasterize) or (osp.exists(out_png) and is_valid_png(out_png))
                if pkl_valid and png_valid:
                    continue

            # per-frame rotation bbox
            north, south, east, west, frame_corners = processor.bbox_rot_from_utm(lat, lon, dx, dy, rot_deg)
            
            frame_poly_ll = geom.Polygon([(lo, la) for la, lo in frame_corners])

            # clip
            try:
                edges_clip = gpd.clip(scene_edges_wgs, frame_poly_ll)
            except Exception:
                edges_clip = scene_edges_wgs[scene_edges_wgs.geometry.intersects(frame_poly_ll)].copy()

            try:
                bld_clip = gpd.clip(scene_buildings_wgs, frame_poly_ll)
            except Exception:
                bld_clip = scene_buildings_wgs[scene_buildings_wgs.geometry.intersects(frame_poly_ll)].copy()

            # GDF → ego frame polylines
            graph_dict = _graph_dict_from_gdfs(edges_clip, bld_clip, lon, lat, rot_deg, processor)

            with open(out_pkl, 'wb') as f:
                pickle.dump(graph_dict, f)

            if args.rasterize:
                res = processor.compute_meters_per_pixel(args.zoom_for_res, lat)
                img = processor.rasterize_graph(graph_dict, dx, dy, resolution=res)
                img.save(out_png)
                del img

            del edges_clip, bld_clip, graph_dict
            gc.collect()

        # Segment Memory Cleanup
        del G_big, scene_edges_wgs, scene_buildings_wgs
        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
