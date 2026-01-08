#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGII a2_link Shapefile to OpenLaneV2 Format Converter
서울_여의도_SEC01_2023_a2_link.shp 파일을 OpenLaneV2 형식으로 변환

OpenLaneV2 Format Features:
- Ego-centric coordinate system
- Lane segments with centerline, left/right boundaries
- 100m x 200m region around ego vehicle
- JSON format output
"""

import os
import json
import glob
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
from datetime import datetime

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# Global default directory for NGII Shapefiles (override with env var NGII_SHP_DIR)
DEFAULT_SHP_BASE_DIR = os.getenv(
    "NGII_SHP_DIR",
    "/media/iismn/SSD A/Dataset/inhouse/Dataset/HDMap_Info/NGII_SHP",
)

# Styled logger: bold yellow tag + aligned section labels
ANSI_BOLD_YELLOW = "\033[1;33m"
ANSI_RESET = "\033[0m"
LOG_TAG = f"{ANSI_BOLD_YELLOW}[NGII OpenLaneV2 Converter]{ANSI_RESET}"

# Global quiet flag to suppress logs when running in batch mode
_SILENT = False

# Global debug flag to show detailed debug information
_DEBUG = os.getenv("NGII_DEBUG", "false").lower() in ("true", "1", "yes")

def set_quiet(flag: bool) -> None:
    global _SILENT
    _SILENT = bool(flag)

def set_debug(flag: bool) -> None:
    global _DEBUG
    _DEBUG = bool(flag)

# Known section labels to align the colon across logs
_LOG_KNOWN_LABELS = [
    "Load a2_link",
    "Load b3_surfacemark",
    "Load b2_surfacelinemark",
    "Load shapefile",
    "Virtual path",
    "Start",
    "Links in view",
    "Conversion",
    "Topology",
    "Snap lanelines",
    "Crop polygon",
    "Save JSON",
    "Summary",
    "Done",
    "Result",
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
    # If message already contains a colon immediately after a section, try to split once
    if ":" in message and not message.startswith("http"):
        section, rest = message.split(":", 1)
        LOGK(section.strip(), rest.strip())
    else:
        print(f"{LOG_TAG} {message}")

def _find_shp_in(base_dir: str, pattern: str) -> Optional[str]:
    """Find a shapefile path under base_dir matching the pattern; return first match or None."""
    try:
        candidates = sorted(glob.glob(os.path.join(base_dir, pattern)))
        return candidates[0] if candidates else None
    except Exception:
        return None

def load_shapefile_safely(shp_file, target_crs='EPSG:32652'):
    """Safely load a shapefile and convert to UTM Zone 52N (EPSG:32652)."""
    try:
        # SHAPE_RESTORE_SHX 환경변수 설정
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        # UTF-8 인코딩으로 먼저 시도
        gdf = gpd.read_file(shp_file, encoding='utf-8')
        
        # Ensure target CRS
        if gdf.crs is not None:
            gdf = gdf.to_crs(target_crs)
        else:
            gdf = gdf.set_crs('EPSG:5179').to_crs(target_crs)
            
        return gdf
    except Exception as e1:
        try:
            # CP949 인코딩으로 시도
            gdf = gpd.read_file(shp_file, encoding='cp949')
            
            # 좌표계를 UTM Zone 52N (EPSG:32652)로 변환
            if gdf.crs is not None:
                gdf = gdf.to_crs(target_crs)
            else:
                gdf = gdf.set_crs('EPSG:5179').to_crs(target_crs)
                
            return gdf
        except Exception as e2:
            try:
                # 인코딩 지정 없이 시도
                gdf = gpd.read_file(shp_file)
                
                # 좌표계를 UTM Zone 52N (EPSG:32652)로 변환
                if gdf.crs is not None:
                    gdf = gdf.to_crs(target_crs)
                else:
                    gdf = gdf.set_crs('EPSG:5179').to_crs(target_crs)
                    
                return gdf
            except Exception as e3:
                LOG("Load shapefile: all encoding attempts failed (error)")
                return None

# def load_virtual_driving_path_coordinate(csv_file):
#     """Load the last XY and heading from a virtual driving path CSV."""
#     try:
#         import pandas as pd
#         df = pd.read_csv(csv_file)

#         if len(df) < 2:
#             LOG(f"Virtual path: insufficient points {len(df)} (error)")
#             return None

#         last_point = df.iloc[-1]
#         second_last_point = df.iloc[-2]

#         ego_x = last_point['x_utm']
#         ego_y = last_point['y_utm']

#         dx = last_point['x_utm'] - second_last_point['x_utm']
#         dy = last_point['y_utm'] - second_last_point['y_utm']
#         ego_heading = math.degrees(math.atan2(dy, dx))

#         return ego_x, ego_y, ego_heading

#     except Exception as e:
#         LOG(f"Virtual path: load failed ({e})")
#         return None

def load_virtual_driving_path_coordinate(csv_file):
    """Load the first XY and heading from a virtual driving path CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)

        if len(df) < 2:
            LOG(f"Virtual path: insufficient points {len(df)} (error)")
            return None

        first_point = df.iloc[0]
        second_point = df.iloc[1]

        ego_x = first_point['x_utm']
        ego_y = first_point['y_utm']

        dx = second_point['x_utm'] - ego_x
        dy = second_point['y_utm'] - ego_y
        ego_heading = math.degrees(math.atan2(dy, dx))

        return ego_x, ego_y, ego_heading

    except Exception as e:
        LOG(f"Virtual path: load failed ({e})")
        return None

class NGIIToOpenLaneConverter:
    """NGII a2_link shapefile을 OpenLaneV2 형식으로 변환하는 클래스"""
    
    def __init__(self, view_width=100.0, view_length=200.0, front_ratio=0.5):
        """
        초기화
        Args:
            view_width: 차량 진행방향 시야 범위 (미터) - 200m
            view_length: 차량 가로방향 시야 범위 (미터) - 100m  
            front_ratio: 전방 비율 (0.5 = 전방 50%, 후방 50%)
        """
        self.links_gdf = None
        self.surfacemarks_gdf = None  # b3_surfacemark 데이터용
        self.surfacelinemarks_gdf = None  # b2_surfacelinemark 데이터용
        
        # 시야 범위 설정 (차량 진행방향 200m x 차량 가로방향 100m)
        self.view_width = view_width         # 200m (진행방향)
        self.view_length = view_length       # 100m (가로방향)
        self.front_distance = view_width * front_ratio      # 100m (전방)
        self.rear_distance = view_width * (1 - front_ratio) # 100m (후방)
        self.half_width = view_length / 2     # 50m (좌우 각각)
        
        # crop 영역 범위 (ego vehicle 중심) - 정확히 원하는 FoV 크기
        self.crop_x_min = -self.half_width   # -50m (왼쪽)
        self.crop_x_max = self.half_width    # +50m (오른쪽)  
        self.crop_y_min = -self.rear_distance   # -25m (뒤쪽)
        self.crop_y_max = self.front_distance   # +25m (앞쪽)
        
        # 확장된 검색 영역 (충분한 여유 공간으로 연속성 보장)
        self.search_expansion_factor = 120.0  # 8배로 확장하여 충분한 데이터 확보
        self.search_x_min = self.crop_x_min * self.search_expansion_factor  # -400m
        self.search_x_max = self.crop_x_max * self.search_expansion_factor  # +400m
        self.search_y_min = self.crop_y_min * self.search_expansion_factor  # -800m
        self.search_y_max = self.crop_y_max * self.search_expansion_factor  # +800m
        
    def load_ngii_links(self, shp_file_path=None):
        """Load NGII a2_link shapefile."""
        if shp_file_path is None:
            shp_file_path = _find_shp_in(DEFAULT_SHP_BASE_DIR, "*a2_link*.shp")

        if not shp_file_path or not os.path.exists(shp_file_path):
            LOG("Load a2_link: missing (error)")
            raise FileNotFoundError(f"Missing shapefile: {shp_file_path}")

        self.links_gdf = load_shapefile_safely(shp_file_path)

        if self.links_gdf is not None:
            LOG("Load a2_link: ok")
        else:
            LOG("Load a2_link: failed (error)")
            raise Exception("Failed to load NGII a2_link")
    
    def load_ngii_surfacemarks(self, shp_file_path=None):
        """Load NGII b3_surfacemark shapefile (filter to type=5 & kind=5321)."""
        if shp_file_path is None:
            shp_file_path = _find_shp_in(DEFAULT_SHP_BASE_DIR, "*b3_surfacemark*.shp")
        if not shp_file_path or not os.path.exists(shp_file_path):
            LOG("Load b3_surfacemark: missing (error)")
            return

        self.surfacemarks_gdf = load_shapefile_safely(shp_file_path)

        if self.surfacemarks_gdf is not None:
            # filter type='5' and kind='5321'
            type_5_marks = self.surfacemarks_gdf[self.surfacemarks_gdf['type'] == '5']
            pedestrian_crosswalks = type_5_marks[type_5_marks['kind'] == '5321']
            self.surfacemarks_gdf = pedestrian_crosswalks
            LOG("Load b3_surfacemark: ok")
        else:
            LOG("Load b3_surfacemark: failed (error)")
    
    def extract_links_in_ego_view(self, ego_x, ego_y, ego_heading_deg):
        """
        Ego vehicle 기준 확장된 시야 범위 내 링크 추출 (1.4배 확장된 영역)
        Args:
            ego_x: ego vehicle X 좌표 (UTM)
            ego_y: ego vehicle Y 좌표 (UTM)
            ego_heading_deg: ego vehicle 진행방향 (도, 수학적 좌표계)
        Returns:
            extracted_links: 추출된 링크들 (GeoDataFrame)
        """
        if self.links_gdf is None:
            raise Exception("NGII 링크 데이터가 로드되지 않았습니다")
        
        # 확장된 경계 박스 생성 (1.4배 확장)
        from shapely.geometry import Polygon
        from shapely.affinity import rotate, translate
        
        # 확장된 영역 크기 계산
        expanded_front = self.front_distance * self.search_expansion_factor
        expanded_rear = self.rear_distance * self.search_expansion_factor
        expanded_half_width = self.half_width * self.search_expansion_factor
        
        # 차량 중심 기준 확장된 박스 좌표
        expanded_box_coords = [
            (-expanded_rear, -expanded_half_width),      # 뒤쪽 왼쪽
            (expanded_front, -expanded_half_width),      # 앞쪽 왼쪽  
            (expanded_front, expanded_half_width),       # 앞쪽 오른쪽
            (-expanded_rear, expanded_half_width),       # 뒤쪽 오른쪽
            (-expanded_rear, -expanded_half_width)       # 닫기
        ]
        
        # 박스 생성, 회전, 이동
        expanded_box = Polygon(expanded_box_coords)
        rotated_box = rotate(expanded_box, ego_heading_deg, origin=(0, 0), use_radians=False)
        final_expanded_box = translate(rotated_box, xoff=ego_x, yoff=ego_y)
        
        # 링크 추출 (확장된 영역에서 교차하는 것들)
        extracted_links = self.links_gdf[self.links_gdf.geometry.intersects(final_expanded_box)].copy()
        
        return extracted_links, final_expanded_box
    
    def load_ngii_surfacelinemarks(self, shp_file_path=None):
        """Load NGII b2_surfacelinemark shapefile (exclude kind=530)."""
        if shp_file_path is None:
            shp_file_path = _find_shp_in(DEFAULT_SHP_BASE_DIR, "*b2_surfacelinemark*.shp")
        if not shp_file_path or not os.path.exists(shp_file_path):
            LOG("Load b2_surfacelinemark: missing (error)")
            return

        self.surfacelinemarks_gdf = load_shapefile_safely(shp_file_path)

        if self.surfacelinemarks_gdf is not None:
            # 1) 컬럼명 소문자화 (대소문자 혼용 대비)
            try:
                self.surfacelinemarks_gdf.columns = [str(c).lower() for c in self.surfacelinemarks_gdf.columns]
            except Exception:
                pass

            # 1-1) 동의어 컬럼명을 표준화
            rename_map = {}
            cols = set(self.surfacelinemarks_gdf.columns)
            if 'l_link_id' in cols and 'l_linkid' not in cols:
                rename_map['l_link_id'] = 'l_linkid'
            if 'r_link_id' in cols and 'r_linkid' not in cols:
                rename_map['r_link_id'] = 'r_linkid'
            if 'llinkid' in cols and 'l_linkid' not in cols:
                rename_map['llinkid'] = 'l_linkid'
            if 'rlinkid' in cols and 'r_linkid' not in cols and 'r_linkid' not in rename_map and 'r_linkid' in cols:
                pass
            if 'rlinkid' in cols and 'r_linkid' not in cols and 'r_linkid' not in rename_map and 'r_linkid' not in cols:
                # fallback to r_linkid standard name
                rename_map['rlinkid'] = 'r_linkid'
            if 'ID' in cols and 'id' not in cols:
                rename_map['ID'] = 'id'
            if 'b2_id' in cols and 'id' not in cols:
                rename_map['b2_id'] = 'id'
            if rename_map:
                try:
                    self.surfacelinemarks_gdf = self.surfacelinemarks_gdf.rename(columns=rename_map)
                except Exception:
                    pass

            # 2) 매칭 컬럼 문자열화 (타입 불일치로 인한 후보 미스 방지)
            for col in ("l_linkid", "r_linkid", "kind"):
                if col in self.surfacelinemarks_gdf.columns:
                    try:
                        self.surfacelinemarks_gdf[col] = self.surfacelinemarks_gdf[col].astype(str)
                    except Exception:
                        pass

            # 3) exclude kind='530' (stop line)
            if 'kind' in self.surfacelinemarks_gdf.columns:
                kind_filter = self.surfacelinemarks_gdf['kind'] != '530'
                filtered_marks = self.surfacelinemarks_gdf[kind_filter]
            else:
                filtered_marks = self.surfacelinemarks_gdf
            self.surfacelinemarks_gdf = filtered_marks
            LOG("Load b2_surfacelinemark: ok")
        else:
            LOG("Load b2_surfacelinemark: failed (error)")
    
    def get_laneline_type_from_kind(self, kind_str):
        """NGII kind 값을 OpenLaneV2 laneline type으로 매핑"""
        kind_to_type = {
            '501': 1,   # 중앙선 → double solid line
            '5011': 2,  # 가변차선 → dashed line  
            '502': 2,   # 유턴구역선 → dashed line
            '503': 2,   # 차선 → dashed line
            '504': 1,   # 버스전용차선 → solid line
            '505': 1,   # 길가장자리구역선 → road edge
            '506': 1,   # 진로변경제한선 → solid line
            '515': 1,   # 주정차금지선 → solid line
            '525': 0,   # 유도선 → unknown/none type
            '531': 1,   # 안전지대 → solid line
            '535': 1,   # 자전거도로 → solid line
            '599': 0,   # 기타선 → unknown
        }
        return kind_to_type.get(str(kind_str), 0)  # 기본값: unknown
    
    def build_topology_matrix(self, lane_segments, extracted_links):
        """
        NGII A2_LINK 연결성 정보를 이용해 topology matrix 구축 (최적화된 버전)
        Args:
            lane_segments: OpenLaneV2 형식의 lane segment 리스트
            extracted_links: 추출된 NGII a2_link GeoDataFrame
        Returns:
            topology_matrix: lane segment 간 연결성을 나타내는 2D 배열
        """
        num_segments = len(lane_segments)
        topology_matrix = [[0 for _ in range(num_segments)] for _ in range(num_segments)]

        LOGK("Topology", "build (start)")

        # 1. 전처리: 링크 정보를 딕셔너리로 캐시 (O(n) 대신 O(1) 조회)
        link_info_cache = {}
        for _, link_row in extracted_links.iterrows():
            link_id = str(link_row['id'])
            link_info_cache[link_id] = {
                'tonodeid': str(link_row.get('tonodeid', '')),
                'fromnodeid': str(link_row.get('fromnodeid', ''))
            }

        # 2. 세그먼트 정보 전처리
        segment_info = []
        for i, segment in enumerate(lane_segments):
            source_link = segment.get('source_link')
            if not source_link:
                segment_id = segment['id']
                source_link = segment_id.split('_')[0] if '_' in segment_id else segment_id
            
            segment_info.append({
                'index': i,
                'source_link': source_link,
                'link_info': link_info_cache.get(source_link),
                'centerline': segment.get('centerline', [])
            })

        # 3. 노드 기반 연결성 구축 (HashMap 기반 최적화)
        connections_found = 0
        
        # tonodeid → segment 인덱스 매핑
        tonodeid_to_segments = {}
        for seg_info in segment_info:
            if seg_info['link_info'] and seg_info['link_info']['tonodeid'] != 'None':
                tonodeid = seg_info['link_info']['tonodeid']
                if tonodeid not in tonodeid_to_segments:
                    tonodeid_to_segments[tonodeid] = []
                tonodeid_to_segments[tonodeid].append(seg_info['index'])

        # fromnodeid로 연결 찾기
        for seg_info in segment_info:
            if not seg_info['link_info'] or seg_info['link_info']['fromnodeid'] == 'None':
                continue
                
            fromnodeid = seg_info['link_info']['fromnodeid']
            if fromnodeid in tonodeid_to_segments:
                j = seg_info['index']
                for i in tonodeid_to_segments[fromnodeid]:
                    if i != j:
                        topology_matrix[i][j] = 1
                        connections_found += 1

        # 4. 거리 기반 보완 (필요한 경우만)
        if connections_found < num_segments * 0.05:
            LOGK("Topology", f"node-based edges {connections_found} (fallback to distance)")
            
            CONNECTIVITY_THRESHOLD = 2.0
            distance_connections = 0

            # 세그먼트 끝점/시작점 캐시
            segment_endpoints = []
            for seg_info in segment_info:
                centerline = seg_info['centerline']
                if centerline:
                    segment_endpoints.append({
                        'index': seg_info['index'],
                        'end_point': centerline[-1],
                        'start_point': centerline[0]
                    })

            for i_info in segment_endpoints:
                end_point_i = i_info['end_point']
                
                for j_info in segment_endpoints:
                    if i_info['index'] == j_info['index']:
                        continue
                        
                    start_point_j = j_info['start_point']
                    
                    # 빠른 거리 계산 (제곱근 생략한 비교)
                    dx = end_point_i[0] - start_point_j[0]
                    dy = end_point_i[1] - start_point_j[1]
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq < CONNECTIVITY_THRESHOLD * CONNECTIVITY_THRESHOLD:
                        i_idx = i_info['index']
                        j_idx = j_info['index']
                        if topology_matrix[i_idx][j_idx] == 0:
                            topology_matrix[i_idx][j_idx] = 1
                            distance_connections += 1

            LOGK("Topology", f"distance-based edges {distance_connections} (added)")
            connections_found += distance_connections
        else:
            LOGK("Topology", f"node-based edges {connections_found} (sufficient, skip distance)")

        LOGK("Topology", f"total edges {connections_found} (done)")
        return topology_matrix

    def _snap_lanelines_by_topology(self, lane_segments, topology_matrix, threshold_m: float = 10.0):
        """연결된 lane segment들의 좌/우 차선 경계선 끝점을 보수적으로 스냅하여 끊김을 완화.
        - centerline, topology는 변경하지 않음.
        - seg_i -> seg_j가 연결된 경우, seg_i의 끝점과 seg_j의 시작점에 대해
          left/right 각각의 laneline을 가까운 쪽으로 맞추고 필요시 뒤집은 후, 임계값 내면 시작점을 스냅.
        """
        def dist2d(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        fixes = 0
        n = len(lane_segments)
        for i in range(n):
            for j in range(n):
                if not topology_matrix or topology_matrix[i][j] != 1:
                    continue
                seg_i = lane_segments[i]
                seg_j = lane_segments[j]

                # 각 side 처리
                for side in ("left_laneline", "right_laneline"):
                    li = seg_i.get(side) or []
                    lj = seg_j.get(side) or []
                    if len(li) < 1 or len(lj) < 1:
                        continue

                    end_i = li[-1]
                    # seg_j의 시작점이 끝점과 가까워지도록 필요하면 뒤집기
                    d_start = dist2d(lj[0], end_i)
                    d_end = dist2d(lj[-1], end_i)
                    if d_end < d_start:
                        lj.reverse()
                        d_start = d_end

                    # 너무 작은 오차면 스킵, 임계값 내면 시작점을 스냅
                    if 0.5 < d_start < threshold_m:
                        # z는 seg_j 기존 시작점의 z를 유지
                        z_keep = lj[0][2] if len(lj[0]) > 2 else 0.0
                        lj[0] = [end_i[0], end_i[1], z_keep]
                        seg_j[side] = lj
                        fixes += 1

        if fixes:
            LOGK("Snap lanelines", f"{fixes} fixes (threshold {threshold_m}m)")
        else:
            LOGK("Snap lanelines", f"no fixes (threshold {threshold_m}m)")
    
    def extract_surfacelinemarks_in_ego_view(self, ego_x, ego_y, ego_heading_deg):
        """
        Ego vehicle 기준 확장된 시야 범위 내 surfacelinemark 추출 (type='111', '212'만)
        Args:
            ego_x: ego vehicle X 좌표 (UTM)
            ego_y: ego vehicle Y 좌표 (UTM)
            ego_heading_deg: ego vehicle 진행방향 (도, 수학적 좌표계)
        Returns:
            extracted_surfacelinemarks: 추출된 surfacelinemark들 (GeoDataFrame)
        """
        if self.surfacelinemarks_gdf is None or len(self.surfacelinemarks_gdf) == 0:
            return gpd.GeoDataFrame(), None
        
        # 확장된 경계 박스 생성 (링크와 동일한 방식)
        from shapely.geometry import Polygon
        from shapely.affinity import rotate, translate
        
        # 확장된 영역 크기 계산
        expanded_front = self.front_distance * self.search_expansion_factor
        expanded_rear = self.rear_distance * self.search_expansion_factor
        expanded_half_width = self.half_width * self.search_expansion_factor
        
        # 차량 중심 기준 확장된 박스 좌표
        expanded_box_coords = [
            (-expanded_rear, -expanded_half_width),      # 뒤쪽 왼쪽
            (expanded_front, -expanded_half_width),      # 앞쪽 왼쪽  
            (expanded_front, expanded_half_width),       # 앞쪽 오른쪽
            (-expanded_rear, expanded_half_width),       # 뒤쪽 오른쪽
            (-expanded_rear, -expanded_half_width)       # 닫기
        ]
        
        # 박스 생성, 회전, 이동
        expanded_box = Polygon(expanded_box_coords)
        rotated_box = rotate(expanded_box, ego_heading_deg, origin=(0, 0), use_radians=False)
        final_expanded_box = translate(rotated_box, xoff=ego_x, yoff=ego_y)
        
        # surfacelinemark 추출 (확장된 영역에서 교차하는 것들)
        extracted_surfacelinemarks = self.surfacelinemarks_gdf[
            self.surfacelinemarks_gdf.geometry.intersects(final_expanded_box)
        ].copy()
        
        return extracted_surfacelinemarks, final_expanded_box
    
    def extract_surfacemarks_in_ego_view(self, ego_x, ego_y, ego_heading_deg):
        if self.surfacemarks_gdf is None or len(self.surfacemarks_gdf) == 0:
            return gpd.GeoDataFrame(), None
        
        # 확장된 경계 박스 생성 (링크와 동일한 방식)
        from shapely.geometry import Polygon
        from shapely.affinity import rotate, translate
        
        # 확장된 영역 크기 계산
        expanded_front = self.front_distance * self.search_expansion_factor
        expanded_rear = self.rear_distance * self.search_expansion_factor
        expanded_half_width = self.half_width * self.search_expansion_factor
        
        # 차량 중심 기준 확장된 박스 좌표
        expanded_box_coords = [
            (-expanded_rear, -expanded_half_width),      # 뒤쪽 왼쪽
            (expanded_front, -expanded_half_width),      # 앞쪽 왼쪽  
            (expanded_front, expanded_half_width),       # 앞쪽 오른쪽
            (-expanded_rear, expanded_half_width),       # 뒤쪽 오른쪽
            (-expanded_rear, -expanded_half_width)       # 닫기
        ]
        
        # 박스 생성, 회전, 이동
        expanded_box = Polygon(expanded_box_coords)
        rotated_box = rotate(expanded_box, ego_heading_deg, origin=(0, 0), use_radians=False)
        final_expanded_box = translate(rotated_box, xoff=ego_x, yoff=ego_y)
        
        # surfacemark 추출 (확장된 영역에서 교차하는 것들)
        extracted_surfacemarks = self.surfacemarks_gdf[
            self.surfacemarks_gdf.geometry.intersects(final_expanded_box)
        ].copy()
        
        return extracted_surfacemarks, final_expanded_box
    
    def linestring_to_coordinates(self, linestring):
        """LineString을 좌표 리스트로 변환"""
        if hasattr(linestring, 'coords'):
            coords = list(linestring.coords)
            # Z 좌표가 있는 경우 처리
            if len(coords) > 0 and len(coords[0]) == 3:
                return [[x, y, z] for x, y, z in coords]
            else:
                return [[x, y, 0.0] for x, y in coords]
        return []
    
    def polygon_to_coordinates(self, polygon):
        """Polygon을 좌표 리스트로 변환"""
        from shapely.geometry import Polygon as ShapelyPolygon
        
        if hasattr(polygon, 'exterior'):
            # Polygon인 경우 exterior 좌표 추출
            coords = list(polygon.exterior.coords)
            # Z 좌표가 있는 경우 처리
            if len(coords) > 0 and len(coords[0]) == 3:
                return [[x, y, z] for x, y, z in coords]
            else:
                return [[x, y, 0.0] for x, y in coords]
        elif hasattr(polygon, 'coords'):
            # LineString이나 다른 geometry 타입인 경우
            coords = list(polygon.coords)
            if len(coords) > 0 and len(coords[0]) == 3:
                return [[x, y, z] for x, y, z in coords]
            else:
                return [[x, y, 0.0] for x, y in coords]
        return []
    
    def crop_polygon_to_region(self, ego_coordinates):
        """
        Polygon을 OpenLaneV2 crop 영역에 맞게 자르기
        Args:
            ego_coordinates: ego 좌표계의 polygon 좌표들
        Returns:
            cropped_coords: crop된 좌표 리스트
        """
        if not ego_coordinates:
            return []
        
        from shapely.geometry import Polygon as ShapelyPolygon
        
        # crop 영역 생성
        crop_box = ShapelyPolygon([
            (self.crop_x_min, self.crop_y_min),
            (self.crop_x_max, self.crop_y_min),
            (self.crop_x_max, self.crop_y_max),
            (self.crop_x_min, self.crop_y_max),
            (self.crop_x_min, self.crop_y_min)
        ])
        
        # polygon 생성 (Z 좌표 제외하고 XY만 사용)
        xy_coords = [(coord[0], coord[1]) for coord in ego_coordinates]
        
        try:
            if len(xy_coords) < 3:
                return []
                
            # 닫힌 polygon으로 만들기 (첫 점과 마지막 점이 다르면 첫 점 추가)
            if xy_coords[0] != xy_coords[-1]:
                xy_coords.append(xy_coords[0])
                
            input_polygon = ShapelyPolygon(xy_coords)
            
            # 유효하지 않은 polygon인 경우 buffer(0)로 수정 시도
            if not input_polygon.is_valid:
                input_polygon = input_polygon.buffer(0)
            
            # crop 영역과 교차 부분 계산
            intersection = input_polygon.intersection(crop_box)
            
            if intersection.is_empty:
                return []
            
            # 결과가 Polygon인 경우
            if hasattr(intersection, 'exterior'):
                result_coords = list(intersection.exterior.coords)
                # Z 좌표 복원 (원본에서 보간)
                result_with_z = []
                for x, y in result_coords:
                    # 가장 가까운 원본 점의 Z 좌표 사용
                    min_dist = float('inf')
                    z_val = 0.0
                    for orig_coord in ego_coordinates:
                        dist = math.sqrt((x - orig_coord[0])**2 + (y - orig_coord[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            z_val = orig_coord[2] if len(orig_coord) > 2 else 0.0
                    result_with_z.append([x, y, z_val])
                return result_with_z
            
            # 결과가 MultiPolygon인 경우 가장 큰 것 선택
            elif hasattr(intersection, 'geoms'):
                largest_area = 0
                largest_polygon = None
                for geom in intersection.geoms:
                    if hasattr(geom, 'area') and geom.area > largest_area:
                        largest_area = geom.area
                        largest_polygon = geom
                
                if largest_polygon and hasattr(largest_polygon, 'exterior'):
                    result_coords = list(largest_polygon.exterior.coords)
                    result_with_z = []
                    for x, y in result_coords:
                        min_dist = float('inf')
                        z_val = 0.0
                        for orig_coord in ego_coordinates:
                            dist = math.sqrt((x - orig_coord[0])**2 + (y - orig_coord[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                z_val = orig_coord[2] if len(orig_coord) > 2 else 0.0
                        result_with_z.append([x, y, z_val])
                    return result_with_z
            
        except Exception as e:
            LOGK("Crop polygon", f"exception ({e})")
            # 폴백: 단순히 crop 영역 내의 점들만 필터링
            filtered_coords = []
            for coord in ego_coordinates:
                if self.is_coordinate_in_crop_region(coord):
                    filtered_coords.append(coord)
            return filtered_coords
        
        return []
    
    def transform_to_ego_coordinates(self, coordinates, ego_x, ego_y, ego_heading_deg):
        """
        UTM 좌표를 ego vehicle 기준 좌표계로 변환
        Args:
            coordinates: UTM 좌표 리스트 [[x, y, z], ...]
            ego_x, ego_y: ego vehicle UTM 위치
            ego_heading_deg: ego vehicle 진행방향 (도)
        Returns:
            ego_coordinates: ego 기준 좌표 리스트
        """
        if not coordinates:
            return []
        
        ego_coords = []
        heading_rad = math.radians(-ego_heading_deg)  # 반시계방향 회전
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        
        for coord in coordinates:
            x_utm, y_utm = coord[0], coord[1]
            z = coord[2] if len(coord) > 2 else 0.0
            
            # ego 기준으로 평행이동
            dx = x_utm - ego_x
            dy = y_utm - ego_y
            
            # ego 방향 기준으로 회전 (ego가 +Y 방향을 향하도록)
            x_ego = dx * cos_h - dy * sin_h
            y_ego = dx * sin_h + dy * cos_h
            
            ego_coords.append([x_ego, y_ego, z])
        
        return ego_coords
    
    def is_coordinate_in_crop_region(self, coord):
        """좌표가 OpenLaneV2 crop 영역 내에 있는지 확인"""
        x, y = coord[0], coord[1]
        return (self.crop_x_min <= x <= self.crop_x_max and 
                self.crop_y_min <= y <= self.crop_y_max)
    
    def crop_line_to_region(self, ego_coordinates):
        """
        라인을 OpenLaneV2 crop 영역에 맞게 정확히 자르기
        경계를 지나는 선분은 교차점에서 새로운 노드 생성하여 연속성 보장
        """
        if not ego_coordinates:
            return []
        
        cropped_coords = []
        
        for i in range(len(ego_coordinates)):
            current_coord = ego_coordinates[i]
            current_inside = self.is_coordinate_in_crop_region(current_coord)
            
            if i == 0:
                # 첫 번째 점 처리
                if current_inside:
                    cropped_coords.append(current_coord)
                else:
                    # 첫 번째 점이 외부에 있으면, 다음 점과의 교차점 확인
                    if len(ego_coordinates) > 1:
                        next_coord = ego_coordinates[1]
                        next_inside = self.is_coordinate_in_crop_region(next_coord)
                        if next_inside:
                            # 외부->내부: 교차점을 시작점으로 추가
                            intersections = self.find_all_boundary_intersections(current_coord, next_coord)
                            if intersections:
                                closest_intersection = min(intersections, 
                                                         key=lambda p: self.distance_between_points(next_coord, p))
                                cropped_coords.append(closest_intersection)
            else:
                prev_coord = ego_coordinates[i-1]
                prev_inside = self.is_coordinate_in_crop_region(prev_coord)
                
                # 이전 점과 현재 점 사이의 선분 처리
                if prev_inside and current_inside:
                    # 둘 다 내부 - 현재 점 추가
                    cropped_coords.append(current_coord)
                    
                elif prev_inside and not current_inside:
                    # 내부 -> 외부: 경계 교차점 찾아서 끝점으로 추가
                    intersections = self.find_all_boundary_intersections(prev_coord, current_coord)
                    if intersections:
                        # 이전 점에서 가장 가까운 교차점 선택 (끝점으로 사용)
                        closest_intersection = min(intersections, 
                                                 key=lambda p: self.distance_between_points(prev_coord, p))
                        cropped_coords.append(closest_intersection)
                        
                elif not prev_inside and current_inside:
                    # 외부 -> 내부: 경계 교차점을 먼저 추가(시작점)하고 현재 점 추가
                    intersections = self.find_all_boundary_intersections(prev_coord, current_coord)
                    if intersections:
                        # 현재 점에서 가장 가까운 교차점 선택 (시작점으로 사용)
                        closest_intersection = min(intersections, 
                                                 key=lambda p: self.distance_between_points(current_coord, p))
                        cropped_coords.append(closest_intersection)
                    cropped_coords.append(current_coord)
                    
                elif not prev_inside and not current_inside:
                    # 둘 다 외부: 선분이 crop 영역을 관통하는지 확인
                    intersections = self.find_all_boundary_intersections(prev_coord, current_coord)
                    if len(intersections) >= 2:
                        # 영역을 관통하는 경우: 두 교차점 추가 (진입점과 탈출점)
                        intersections.sort(key=lambda p: self.distance_between_points(prev_coord, p))
                        cropped_coords.extend(intersections[:2])
        
        # 결과 검증: 모든 점이 crop 영역 내에 있는지 확인
        validated_coords = []
        for coord in cropped_coords:
            if self.is_coordinate_in_crop_region(coord):
                validated_coords.append(coord)
            else:
                # 경계값으로 클램핑 (부동소수점 오차 보정)
                x = max(self.crop_x_min, min(self.crop_x_max, coord[0]))
                y = max(self.crop_y_min, min(self.crop_y_max, coord[1]))
                z = coord[2] if len(coord) > 2 else 0.0
                validated_coords.append([x, y, z])
        
        return validated_coords
    
    def distance_between_points(self, p1, p2):
        """두 점 사이의 거리 계산"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _pick_all_marks(self, marks_gdf, centerline_geom, max_distance_m=25.0, prefer_ids=None):
        """
        거리 임계값 내의 모든 surfacelinemark들을 반환 (최적화된 버전)
        Args:
            marks_gdf: 후보 GeoDataFrame (geometry 컬럼 필수)
            centerline_geom: 현재 링크의 LineString (UTM 좌표계)
            max_distance_m: 최대 거리 임계값 (미터)
            prefer_ids: 우선 선택할 ID 리스트 (옵션)
        Returns:
            GeoDataFrame: 거리 임계값 내의 모든 마킹들 (우선 ID들 먼저, 나머지는 거리순, 없으면 빈 GeoDataFrame)
        """
        try:
            if marks_gdf is None or len(marks_gdf) == 0:
                return gpd.GeoDataFrame()
            
            # NumPy 벡터화된 거리 계산 (더 빠름)
            distances = marks_gdf.geometry.distance(centerline_geom).values
            
            # 거리 임계값 내의 마킹들 필터링
            valid_indices = distances <= max_distance_m
            if not valid_indices.any():
                return gpd.GeoDataFrame()
                
            filtered_marks = marks_gdf.iloc[valid_indices].copy()
            filtered_distances = distances[valid_indices]
            
            # 우선 ID들이 있는 경우 먼저 정렬
            if prefer_ids:
                prefer_id_set = {str(pid) for pid in prefer_ids}  # set으로 빠른 조회
                is_preferred = filtered_marks['id'].astype(str).isin(prefer_id_set)
                
                # 우선순위별로 정렬
                sort_keys = (~is_preferred, filtered_distances)  # 우선순위 먼저, 그 다음 거리순
                sort_indices = np.lexsort(sort_keys)
                result_gdf = filtered_marks.iloc[sort_indices].copy()
            else:
                # 거리순으로만 정렬
                sort_indices = np.argsort(filtered_distances)
                result_gdf = filtered_marks.iloc[sort_indices].copy()
            
            return result_gdf.reset_index(drop=True)
        except Exception:
            return gpd.GeoDataFrame()
    
    def find_all_boundary_intersections(self, p1, p2):
        """두 점 사이의 선분과 crop 영역 모든 경계의 교차점 찾기"""
        intersections = []
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        z1 = p1[2] if len(p1) > 2 else 0.0
        z2 = p2[2] if len(p2) > 2 else 0.0
        
        # 4개 경계와의 교차점 확인
        boundaries = [
            ('left', self.crop_x_min, 'vertical'),
            ('right', self.crop_x_max, 'vertical'),
            ('bottom', self.crop_y_min, 'horizontal'),
            ('top', self.crop_y_max, 'horizontal')
        ]
        
        for boundary_name, boundary_value, boundary_type in boundaries:
            intersection = self.find_single_boundary_intersection(
                p1, p2, boundary_value, boundary_type)
            if intersection:
                intersections.append(intersection)
        
        return intersections
    
    def find_single_boundary_intersection(self, p1, p2, boundary_value, boundary_type):
        """단일 경계와의 교차점 찾기"""
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        z1 = p1[2] if len(p1) > 2 else 0.0
        z2 = p2[2] if len(p2) > 2 else 0.0
        
        if boundary_type == 'vertical':  # X = boundary_value
            if x1 == x2:  # 수직선
                return None
            
            # 선분이 경계를 지나는지 확인
            if (x1 <= boundary_value <= x2) or (x2 <= boundary_value <= x1):
                t = (boundary_value - x1) / (x2 - x1)
                y_intersect = y1 + t * (y2 - y1)
                z_intersect = z1 + t * (z2 - z1)
                
                # Y 범위 내에 있는지 확인
                if self.crop_y_min <= y_intersect <= self.crop_y_max:
                    return [boundary_value, y_intersect, z_intersect]
                    
        elif boundary_type == 'horizontal':  # Y = boundary_value
            if y1 == y2:  # 수평선
                return None
                
            # 선분이 경계를 지나는지 확인
            if (y1 <= boundary_value <= y2) or (y2 <= boundary_value <= y1):
                t = (boundary_value - y1) / (y2 - y1)
                x_intersect = x1 + t * (x2 - x1)
                z_intersect = z1 + t * (z2 - z1)
                
                # X 범위 내에 있는지 확인
                if self.crop_x_min <= x_intersect <= self.crop_x_max:
                    return [x_intersect, boundary_value, z_intersect]
        
        return None
    
    def convert_to_openlane_format(self, ego_x, ego_y, ego_heading_deg,
                                  segment_id="00000", timestamp=None):
        """Convert a view around ego to OpenLaneV2 using raw surfacelinemark geometry (QGIS-like)."""
        
        start_time = time.time()

        # 1. 링크 추출
        t1 = time.time()
        extracted_links, _ = self.extract_links_in_ego_view(ego_x, ego_y, ego_heading_deg)
        t2 = time.time()
        if _DEBUG:
            LOGD("PERF", f"Link extraction: {(t2-t1)*1000:.1f}ms")
            
        LOGK("Links in view", f"{len(extracted_links)} (expanded {self.search_expansion_factor}x)")
        
        # 디버그: 추출된 링크 정보
        if _DEBUG:
            LOGD("DEBUG", f"Crop region: X=[{self.crop_x_min:.1f}, {self.crop_x_max:.1f}], Y=[{self.crop_y_min:.1f}, {self.crop_y_max:.1f}]")
        
        if len(extracted_links) == 0:
            LOGK("Links in view", f"none at ({ego_x:.1f}, {ego_y:.1f}) (stop)")
            return None

        if timestamp is None:
            timestamp = int(datetime.now().timestamp() * 1e9)
        openlane_data = {
            "version": "OpenLaneV2_V2.0",
            "segment_id": segment_id,
            "meta_data": {
                "source": "NGII",
                "source_id": "seoul_yeouido_SEC01_2023",
                "ego_position_utm": [ego_x, ego_y],
                "ego_heading_deg": ego_heading_deg,
            },
            "timestamp": timestamp,
            "sensor": {},
            "annotation": {"lane_segment": [], "area": []},
        }

        lane_segments: List[Dict[str, Any]] = []
        processed_links = 0
        skipped_links = 0
        cropped_segments = 0

        # 2. 서피스 라인마크 추출
        t3 = time.time()
        marks_in_view = None
        if self.surfacelinemarks_gdf is not None and len(self.surfacelinemarks_gdf) > 0:
            try:
                extracted_marks, _ = self.extract_surfacelinemarks_in_ego_view(ego_x, ego_y, ego_heading_deg)
                if extracted_marks is not None and len(extracted_marks) > 0:
                    marks_in_view = extracted_marks
                    
                    # 디버그: 추출된 surfacelinemark 정보
                    if _DEBUG:
                        extracted_mark_ids = [str(row["id"]) for _, row in extracted_marks.iterrows()]
                        LOGD("DEBUG", f"Extracted surfacelinemarks: {len(extracted_mark_ids)}")
                        LOGD("DEBUG", f"First 10 extracted mark IDs: {extracted_mark_ids[:10]}")
            except Exception as e:
                LOGK("DEBUG", f"Failed to extract surfacelinemarks: {e}")
                marks_in_view = None
        t4 = time.time()
        if _DEBUG:
            LOGD("PERF", f"Mark extraction: {(t4-t3)*1000:.1f}ms")
            
        marks_gdf = marks_in_view if marks_in_view is not None else self.surfacelinemarks_gdf
        
        # 3. 링크별 레인 세그먼트 생성
        t5 = time.time()
        link_processing_time = 0
        geometry_time = 0
        marking_time = 0
        
        for _, row in extracted_links.iterrows():
            link_start = time.time()
            
            processed_links += 1
            link_id = str(row["id"])
            center_geom = row["geometry"]

            # 기하학적 변환 시간 측정
            geo_start = time.time()
            utm_coords = self.linestring_to_coordinates(center_geom)
            if not utm_coords:
                skipped_links += 1
                continue
                
            ego_coords = self.transform_to_ego_coordinates(utm_coords, ego_x, ego_y, ego_heading_deg)
                    
            cropped_centerline = self.crop_line_to_region(ego_coords)
            if len(cropped_centerline) < 2:
                skipped_links += 1
                continue
                
            if len(cropped_centerline) != len(ego_coords):
                cropped_segments += 1
            geo_end = time.time()
            geometry_time += (geo_end - geo_start)

            # 이 링크에 대해 생성할 lane segment
            # 1개 링크 = 1개 segment (동일 centerline)
            # 여러 boundary가 있으면 리스트의 리스트로 저장

            # 마킹 처리 시간 측정
            marking_start = time.time()
            
            # 결과 저장용 (연속된 boundary들만 병합)
            merged_left_coords = []
            merged_right_coords = []
            left_type_lengths = {0: 0.0, 1: 0.0, 2: 0.0}   # type별 누적 길이
            right_type_lengths = {0: 0.0, 1: 0.0, 2: 0.0}  # type별 누적 길이
            left_mark_ids = []
            right_mark_ids = []
            
            def calc_line_length(coords):
                """라인의 총 길이 계산"""
                if len(coords) < 2:
                    return 0.0
                total = 0.0
                for i in range(len(coords) - 1):
                    total += math.hypot(coords[i+1][0] - coords[i][0], 
                                       coords[i+1][1] - coords[i][1])
                return total
            
            def dist2d(a, b):
                return math.hypot(a[0] - b[0], a[1] - b[1])
            
            # 연속 병합을 위한 헬퍼 함수
            def merge_if_connected(existing_coords, new_coords, max_gap=5.0):
                """
                기존 좌표 리스트에 새 좌표들을 연결.
                끝점들이 max_gap 이내면 연속으로 간주하여 병합.
                방향을 자동으로 맞추고 (필요시 reverse), 연결점에서 중복 제거.
                """
                if not new_coords or len(new_coords) < 2:
                    return existing_coords
                
                if not existing_coords:
                    return list(new_coords)
                
                # 마지막 점과 새 좌표의 시작/끝점 거리 비교
                last_pt = existing_coords[-1]
                first_pt = existing_coords[0]
                
                d_last_to_new_start = dist2d(last_pt, new_coords[0])
                d_last_to_new_end = dist2d(last_pt, new_coords[-1])
                d_first_to_new_start = dist2d(first_pt, new_coords[0])
                d_first_to_new_end = dist2d(first_pt, new_coords[-1])
                
                # 가장 가까운 연결 방식 선택
                min_dist = min(d_last_to_new_start, d_last_to_new_end, 
                              d_first_to_new_start, d_first_to_new_end)
                
                # 너무 멀면 (max_gap 초과) 병합하지 않고 가장 긴 것 유지
                if min_dist > max_gap:
                    # 더 긴 쪽 유지
                    if len(new_coords) > len(existing_coords):
                        return list(new_coords)
                    else:
                        return existing_coords
                
                # 연결 방향 결정
                if min_dist == d_last_to_new_end:
                    # 끝→끝: new_coords 뒤집어서 뒤에 붙임
                    new_coords = list(reversed(new_coords))
                    if dist2d(last_pt, new_coords[0]) < 2.0:
                        new_coords = new_coords[1:]
                    existing_coords.extend(new_coords)
                elif min_dist == d_last_to_new_start:
                    # 끝→시작: new_coords 그대로 뒤에 붙임
                    if dist2d(last_pt, new_coords[0]) < 2.0:
                        new_coords = new_coords[1:]
                    existing_coords.extend(new_coords)
                elif min_dist == d_first_to_new_end:
                    # 시작→끝: new_coords 앞에 붙임
                    if dist2d(first_pt, new_coords[-1]) < 2.0:
                        new_coords = new_coords[:-1]
                    existing_coords = new_coords + existing_coords
                else:  # d_first_to_new_start
                    # 시작→시작: new_coords 뒤집어서 앞에 붙임
                    new_coords = list(reversed(new_coords))
                    if dist2d(first_pt, new_coords[-1]) < 2.0:
                        new_coords = new_coords[:-1]
                    existing_coords = new_coords + existing_coords
                
                return existing_coords
            
            if marks_gdf is not None and len(marks_gdf) > 0:
                    
                # 링크 ID로 한 번에 필터링 (DataFrame 인덱싱 최적화)
                right_candidates = gpd.GeoDataFrame()
                left_candidates = gpd.GeoDataFrame()
                
                if "l_linkid" in marks_gdf.columns:
                    right_mask = marks_gdf["l_linkid"] == link_id
                    right_candidates = marks_gdf[right_mask] if right_mask.any() else gpd.GeoDataFrame()
                            
                if "r_linkid" in marks_gdf.columns:
                    left_mask = marks_gdf["r_linkid"] == link_id  
                    left_candidates = marks_gdf[left_mask] if left_mask.any() else gpd.GeoDataFrame()

                # Right boundaries 처리 - 연속된 것만 병합 (none type 제외)
                for idx, mark_row in right_candidates.iterrows():
                    mark_id = str(mark_row.get("id", ""))
                    kind_val = str(mark_row.get("kind", ""))
                    lane_type = self.get_laneline_type_from_kind(kind_val)
                    
                    # none type(0)은 건너뛰기
                    if lane_type == 0:
                        continue
                    
                    mark_coords = self.linestring_to_coordinates(mark_row["geometry"])
                    if mark_coords:
                        ego_coords = self.transform_to_ego_coordinates(mark_coords, ego_x, ego_y, ego_heading_deg)
                        cropped = self.crop_line_to_region(ego_coords)
                        if len(cropped) >= 2:
                            merged_right_coords = merge_if_connected(merged_right_coords, cropped)
                            # 이 boundary의 길이를 해당 type에 누적
                            seg_length = calc_line_length(cropped)
                            right_type_lengths[lane_type] += seg_length
                            right_mark_ids.append(mark_id)
                
                # Left boundaries 처리 - 연속된 것만 병합 (none type 제외)
                for idx, mark_row in left_candidates.iterrows():
                    mark_id = str(mark_row.get("id", ""))
                    kind_val = str(mark_row.get("kind", ""))
                    lane_type = self.get_laneline_type_from_kind(kind_val)
                    
                    # none type(0)은 건너뛰기
                    if lane_type == 0:
                        continue
                    
                    mark_coords = self.linestring_to_coordinates(mark_row["geometry"])
                    if mark_coords:
                        ego_coords = self.transform_to_ego_coordinates(mark_coords, ego_x, ego_y, ego_heading_deg)
                        cropped = self.crop_line_to_region(ego_coords)
                        if len(cropped) >= 2:
                            merged_left_coords = merge_if_connected(merged_left_coords, cropped)
                            # 이 boundary의 길이를 해당 type에 누적
                            seg_length = calc_line_length(cropped)
                            left_type_lengths[lane_type] += seg_length
                            left_mark_ids.append(mark_id)

            # Dominant type 결정: 길이 비율 기반 (가장 긴 타입 선택, none 제외)
            def get_dominant_type_by_length(type_lengths):
                """type별 길이를 보고 가장 긴 타입 반환 (0=none 제외하고 비교)"""
                # none 제외한 타입들만 비교
                solid_len = type_lengths.get(1, 0.0)
                dashed_len = type_lengths.get(2, 0.0)
                
                if solid_len == 0.0 and dashed_len == 0.0:
                    return 0  # 둘 다 없으면 none
                
                # 가장 긴 타입 선택
                if solid_len >= dashed_len:
                    return 1  # solid
                else:
                    return 2  # dashed
            
            left_type_final = get_dominant_type_by_length(left_type_lengths)
            right_type_final = get_dominant_type_by_length(right_type_lengths)

            # ========== 가상 laneline 생성 (빈 경우) ==========
            # left/right laneline이 비어있으면 centerline 기반으로 가상 laneline 생성
            # OpenLane-V2와 동일하게 type=0(none)이어도 유효한 좌표를 갖도록 함
            DEFAULT_LANE_HALF_WIDTH = 1.75  # 기본 차선폭의 절반 (3.5m / 2)
            
            if not merged_left_coords or len(merged_left_coords) < 2:
                # centerline 기반 가상 left laneline 생성
                virtual_left, _ = self.generate_virtual_laneline_from_centerline(
                    cropped_centerline, offset_distance=DEFAULT_LANE_HALF_WIDTH)
                merged_left_coords = virtual_left
                left_type_final = 0  # none type으로 설정
            
            if not merged_right_coords or len(merged_right_coords) < 2:
                # centerline 기반 가상 right laneline 생성
                _, virtual_right = self.generate_virtual_laneline_from_centerline(
                    cropped_centerline, offset_distance=DEFAULT_LANE_HALF_WIDTH)
                merged_right_coords = virtual_right
                right_type_final = 0  # none type으로 설정
            # ================================================

            # 단일 segment 생성 (1 링크 = 1 centerline = 1 segment, 연속 병합된 boundary)
            segment_data = {
                "id": link_id,
                "centerline": cropped_centerline,
                "left_laneline": merged_left_coords,       # 단일 리스트 [[x,y,z], ...]
                "left_laneline_type": left_type_final,     # 단일 정수
                "right_laneline": merged_right_coords,     # 단일 리스트 [[x,y,z], ...]
                "right_laneline_type": right_type_final,   # 단일 정수
                "is_intersection_or_connector": self.is_intersection_link(row),
                "source_link": link_id,
            }
            
            # mark ID 정보 추가 (있는 경우)
            if left_mark_ids:
                segment_data["left_marks"] = left_mark_ids
            if right_mark_ids:
                segment_data["right_marks"] = right_mark_ids
                    
            lane_segments.append(segment_data)
            
            marking_end = time.time()
            marking_time += (marking_end - marking_start)
            
            link_end = time.time()
            link_processing_time += (link_end - link_start)

        t6 = time.time()
        if _DEBUG:
            LOGD("PERF", f"Link processing: {(t6-t5)*1000:.1f}ms total")
            LOGD("PERF", f"  - Geometry ops: {geometry_time*1000:.1f}ms ({geometry_time/(t6-t5)*100:.1f}%)")
            LOGD("PERF", f"  - Marking ops: {marking_time*1000:.1f}ms ({marking_time/(t6-t5)*100:.1f}%)")

        openlane_data["annotation"]["lane_segment"] = lane_segments
        LOGK(
            "Conversion",
            f"processed {processed_links}, skipped {skipped_links}, cropped {cropped_segments} -> lanes {len(lane_segments)}",
        )

        # 4. 토폴로지 구성
        t7 = time.time()
        if lane_segments:
            topology_matrix = self.build_topology_matrix(lane_segments, extracted_links)
            openlane_data["annotation"]["topology_lsls"] = topology_matrix
            total_connections = sum(sum(r) for r in topology_matrix)
            LOGK(
                "Topology",
                f"{len(topology_matrix)}x{len(topology_matrix[0])} ({total_connections} edges)",
            )
        t8 = time.time()
        if _DEBUG:
            LOGD("PERF", f"Topology building: {(t8-t7)*1000:.1f}ms")

        # 5. 영역 처리
        t9 = time.time()
        areas: List[Dict[str, Any]] = []
        if self.surfacemarks_gdf is not None and len(self.surfacemarks_gdf) > 0:
            extracted_surfacemarks, _ = self.extract_surfacemarks_in_ego_view(ego_x, ego_y, ego_heading_deg)
            for idx, row in extracted_surfacemarks.iterrows():
                mark_id = str(row.get("id", f"area_{idx}"))
                geometry = row["geometry"]
                utm_coordinates = self.polygon_to_coordinates(geometry)
                if not utm_coordinates:
                    continue
                ego_coordinates = self.transform_to_ego_coordinates(utm_coordinates, ego_x, ego_y, ego_heading_deg)
                cropped_area = self.crop_polygon_to_region(ego_coordinates)
                if len(cropped_area) < 3:
                    continue
                areas.append({"id": f"s_{mark_id}", "points": cropped_area, "category": 1})
        openlane_data["annotation"]["area"] = areas
        t10 = time.time()
        if _DEBUG:
            LOGD("PERF", f"Area processing: {(t10-t9)*1000:.1f}ms")

        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        if _DEBUG:
            LOGD("PERF", f"Total conversion: {total_time:.1f}ms")

        LOGK("Done", f"lanes {len(lane_segments)}, areas {len(areas)}")
        return openlane_data
    
    def generate_virtual_laneline_from_centerline(self, centerline, offset_distance=1.75):
        """
        Centerline 기반으로 가상의 left/right laneline 생성
        OpenLane-V2와 동일하게 type=0(none)이어도 유효한 좌표를 갖도록 함
        
        Args:
            centerline: centerline 좌표 리스트 [[x,y,z], ...]
            offset_distance: centerline에서 좌/우 offset 거리 (기본 1.75m = 차선폭의 절반)
        
        Returns:
            left_laneline: 왼쪽 laneline 좌표 리스트
            right_laneline: 오른쪽 laneline 좌표 리스트
        """
        if not centerline or len(centerline) < 2:
            return [], []
        
        left_laneline = []
        right_laneline = []
        
        for i in range(len(centerline)):
            # 현재 점
            curr = centerline[i]
            x, y = curr[0], curr[1]
            z = curr[2] if len(curr) > 2 else 0.0
            
            # 방향 벡터 계산 (이전-다음 점 기반)
            if i == 0:
                # 첫 번째 점: 다음 점 방향 사용
                next_pt = centerline[i + 1]
                dx = next_pt[0] - x
                dy = next_pt[1] - y
            elif i == len(centerline) - 1:
                # 마지막 점: 이전 점에서 현재 방향 사용
                prev_pt = centerline[i - 1]
                dx = x - prev_pt[0]
                dy = y - prev_pt[1]
            else:
                # 중간 점: 이전-다음 점 평균 방향 사용
                prev_pt = centerline[i - 1]
                next_pt = centerline[i + 1]
                dx = next_pt[0] - prev_pt[0]
                dy = next_pt[1] - prev_pt[1]
            
            # 방향 벡터 정규화
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                # 방향 계산 불가 - 이전 점 복사
                if left_laneline:
                    left_laneline.append(list(left_laneline[-1]))
                    right_laneline.append(list(right_laneline[-1]))
                continue
            
            dx /= length
            dy /= length
            
            # 법선 벡터 (왼쪽: 90도 반시계 회전, 오른쪽: 90도 시계 회전)
            # ego 좌표계에서 Y+가 전방, X+가 오른쪽
            # 진행방향의 왼쪽 = (-dy, dx), 오른쪽 = (dy, -dx)
            left_x = x - dy * offset_distance
            left_y = y + dx * offset_distance
            right_x = x + dy * offset_distance
            right_y = y - dx * offset_distance
            
            left_laneline.append([left_x, left_y, z])
            right_laneline.append([right_x, right_y, z])
        
        return left_laneline, right_laneline
    
    def is_intersection_link(self, link_row):
        """
        링크가 교차로 내 주행경로인지 판단
        
        NGII A2_LINK LinkType 코드값:
        - 1: 교차로내주행경로 (평면교차로 및 회전교차로 내부)
        - 2: 톨게이트차로(하이패스차로)
        - 3: 톨게이트차로(비하이패스차로)
        - 4: 버스전용차로
        - 5: 가변차선차로
        - 6: 일반주행차로
        - 7~14: 휴게소/졸음쉼터/교차로 진입/진출로
        - 99: 기타차로
        
        Returns:
            True if linktype == '1' (교차로내주행경로)
        """
        linktype = str(link_row.get('linktype', ''))
        
        # 교차로내주행경로: linktype = 1
        return linktype == '1'
    
    def save_openlane_json(self, openlane_data, output_file):
        """OpenLaneV2 데이터를 JSON 파일로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(openlane_data, f, indent=2, ensure_ascii=False)
        LOGK("Save JSON", f"{output_file} (ok)")

def main():
    """Main entry - minimal demo using virtual driving path."""
    converter = NGIIToOpenLaneConverter(view_width=50.0, view_length=100.0, front_ratio=0.5)
    
    converter.load_ngii_links()
    converter.load_ngii_surfacemarks()
    converter.load_ngii_surfacelinemarks()
    
    # 가상 주행 경로에서 첫 번째 좌표 로드
    virtual_path_file = "/media/iismn/SSD A/Dataset/inhouse/Dataset/HDMap_Info/NGII_VIRTUAL_DRIVING_PATHS/virtual_driving_path_1_odometry.csv"
    
    if not os.path.exists(virtual_path_file):
        LOGK("Virtual path", f"missing file {virtual_path_file} (stop)")
        return
    
    coordinate_data = load_virtual_driving_path_coordinate(virtual_path_file)
    if coordinate_data is None:
        LOGK("Virtual path", "failed to extract coordinates (stop)")
        return
    
    ego_x, ego_y, ego_heading = coordinate_data
    
    # 출력 디렉토리 생성
    output_dir = "../test/"
    os.makedirs(output_dir, exist_ok=True)
    
    LOGK("Start", f"ego ({ego_x:.1f}, {ego_y:.1f}), heading {ego_heading:.1f}° (convert)")
    
    # OpenLaneV2 형식으로 변환
    openlane_data = converter.convert_to_openlane_format(
        ego_x=ego_x,
        ego_y=ego_y, 
        ego_heading_deg=ego_heading,
        segment_id="virtual_path_1_start",
        timestamp=int(datetime.now().timestamp() * 1e9)
    )
    
    if openlane_data:
        output_file = os.path.join(output_dir, "ngii_openlane_virtual_path_1.json")
        converter.save_openlane_json(openlane_data, output_file)
        lane_count = len(openlane_data["annotation"]["lane_segment"])
        area_count = len(openlane_data["annotation"].get("area", []))
        LOGK("Summary", f"lanes {lane_count}, areas {area_count} (saved)")
    else:
        LOGK("Result", "no data produced (stop)")

if __name__ == "__main__":
    main()
