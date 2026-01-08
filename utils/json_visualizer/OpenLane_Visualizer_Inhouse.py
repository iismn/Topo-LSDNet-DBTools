#!/usr/bin/env python3
"""
OpenLaneV2 Visualization Tool
Visualizes lane segments and areas from OpenLaneV2 format JSON files in ego vehicle coordinate system.
Based on LaneSegNet visualization code.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image

# Styled logger: bold yellow tag + aligned labels (same style as converter)
ANSI_BOLD_YELLOW = "\033[1;33m"
ANSI_RESET = "\033[0m"
LOG_TAG = f"{ANSI_BOLD_YELLOW}[NGII OpenLaneV2 Visualizer]{ANSI_RESET}"

_LOG_KNOWN_LABELS = [
    "Mode", "Input", "Pose", "Heading", "Aerial image", "HD Map JSON",
    "Alpha", "Topology", "Visualization", "Save image", "Stats", "Error",
    "Warning", "BEV range", "Reshape"
]
_LOG_LABEL_WIDTH = max(len(s) for s in _LOG_KNOWN_LABELS)

def LOGK(label: str, content: str) -> None:
    print(f"{LOG_TAG} {label.ljust(_LOG_LABEL_WIDTH)}: {content}")

COLOR_DICT = {  # RGB [0, 1]
    'centerline': np.array([243, 90, 2]) / 255,
    'laneline': np.array([0, 32, 127]) / 255,
    'ped_crossing': np.array([255, 192, 0]) / 255,
    'road_boundary': np.array([220, 30, 0]) / 255,
}

LINE_PARAM = {
    0: {'color': COLOR_DICT['laneline'], 'alpha': 0.3, 'linestyle': ':'},       # none
    1: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': 'solid'},  # solid
    2: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': '--'},     # dashed
    'ped_crossing': {'color': COLOR_DICT['ped_crossing'], 'alpha': 1, 'linestyle': 'solid'},
    'road_boundary': {'color': COLOR_DICT['road_boundary'], 'alpha': 1, 'linestyle': 'solid'}
}

BEV_RANGE = [-50, 50, -25, 25]  # [y_min, y_max, x_min, x_max] in ego coordinate system (100x50m)
# BEV_RANGE = [-100, 100, -50, 50]  # [y_min, y_max, x_min, x_max] in ego coordinate system (100x50m)

def _is_in_bev_range(points, bev_range=None):
    """Check if any point of the line/centerline is within BEV range"""
    if bev_range is None:
        bev_range = BEV_RANGE
    
    if len(points) == 0:
        return False
    
    points = np.asarray(points)
    if len(points.shape) != 2 or points.shape[1] < 2:
        return False
    
    # Check if any point is within the BEV range
    y_in_range = np.logical_and(points[:, 0] >= bev_range[0], points[:, 0] <= bev_range[1])
    x_in_range = np.logical_and(points[:, 1] >= bev_range[2], points[:, 1] <= bev_range[3])
    in_range = np.logical_and(y_in_range, x_in_range)
    
    return np.any(in_range)

def _draw_centerline(ax, lane_centerline, lane_id=None, with_id=False, with_endpoints=False):
    """Draw lane centerline with arrow indicating direction"""
    points = np.asarray(lane_centerline['points'])
    
    # Skip if no points are within BEV range
    if not _is_in_bev_range(points):
        return
    
    color = COLOR_DICT['centerline']
    
    # draw line
    ax.plot(points[:, 1], points[:, 0], color=color, alpha=1.0, linewidth=0.6)
    
    # draw start and end vertex - original small points
    ax.scatter(points[[0, -1], 1], points[[0, -1], 0], color=color, s=1)
    
    # draw larger, more visible start and end points if requested
    if with_endpoints and len(points) >= 2:
        # Start point (green circle)
        ax.scatter(points[0, 1], points[0, 0], 
                   color='green', s=20, marker='o', 
                   edgecolors='black', linewidth=0.5, alpha=0.8, zorder=10)
        
        # End point (red square)
        ax.scatter(points[-1, 1], points[-1, 0], 
                   color='red', s=25, marker='s', 
                   edgecolors='black', linewidth=0.5, alpha=0.8, zorder=10)
    
    # draw arrow if we have at least 2 points
    if len(points) >= 2:
        ax.annotate('', xy=(points[-1, 1], points[-1, 0]),
                    xytext=(points[-2, 1], points[-2, 0]),
                    arrowprops=dict(arrowstyle='->', lw=0.6, color=color))
    
    # draw lane ID at midpoint if requested
    if with_id and lane_id is not None and len(points) > 0:
        # Find a point within BEV range for ID placement
        mid_idx = len(points) // 2
        mid_point = points[mid_idx]
        
        # Check if midpoint is within BEV range, if not find closest point that is
        if _is_in_bev_range([mid_point]):
            id_point = mid_point
        else:
            # Find the first point that is within BEV range
            id_point = None
            for point in points:
                if _is_in_bev_range([point]):
                    id_point = point
                    break
            
            # If no point is within range, skip ID display
            if id_point is None:
                return
        
        ax.text(id_point[1], id_point[0], str(lane_id), 
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def _draw_line(ax, line, with_endpoints=False, lane_id=None, line_type=None, with_id=False):
    """Draw a line with specified style and optional endpoints and ID"""
    points = np.asarray(line['points'])
    
    # Skip if no points are within BEV range
    if not _is_in_bev_range(points):
        return
    
    config = LINE_PARAM[line['linetype']]
    ax.plot(points[:, 1], points[:, 0], linewidth=0.6, **config)
    
    # Draw endpoints if requested and we have enough points
    if with_endpoints and len(points) >= 2:
        # Start point (blue triangle for lanelines)
        ax.scatter(points[0, 1], points[0, 0], 
                   color='blue', s=15, marker='^', 
                   edgecolors='black', linewidth=0.5, alpha=0.7, zorder=9)
        
        # End point (purple diamond for lanelines)
        ax.scatter(points[-1, 1], points[-1, 0], 
                   color='purple', s=15, marker='D', 
                   edgecolors='black', linewidth=0.5, alpha=0.7, zorder=9)
    
    # Draw lane ID for lanelines if requested
    if with_id and lane_id is not None and line_type is not None and len(points) > 0:
        # Find a point within BEV range for ID placement
        mid_idx = len(points) // 2
        mid_point = points[mid_idx]
        
        # Check if midpoint is within BEV range, if not find closest point that is
        if _is_in_bev_range([mid_point]):
            id_point = mid_point
        else:
            # Find the first point that is within BEV range
            id_point = None
            for point in points:
                if _is_in_bev_range([point]):
                    id_point = point
                    break
            
            # If no point is within range, skip ID display
            if id_point is None:
                return
        
        # Create ID text with line type suffix
        type_suffix = {'left': 'L', 'right': 'R'}[line_type]
        id_text = f"{lane_id}{type_suffix}"
        
        # Use smaller font and different background for laneline IDs
        ax.text(id_point[1], id_point[0], id_text, 
                fontsize=6, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='lightblue', alpha=0.7))

def _draw_lane_segment(ax, lane_segment, lane_index, with_centerline, with_laneline, with_id=False, with_endpoints=False):
    """Draw a complete lane segment with centerline and lane boundaries
    
    새로운 형식 지원:
    - left_laneline: [[coords1], [coords2], ...] (리스트의 리스트)
    - left_laneline_type: [type1, type2, ...] (리스트)
    - right_laneline: [[coords1], [coords2], ...] (리스트의 리스트)
    - right_laneline_type: [type1, type2, ...] (리스트)
    
    기존 형식도 호환:
    - left_laneline: [coords] (단일 리스트)
    - left_laneline_type: int (단일 값)
    """
    # Check if this lane segment has any parts within BEV range
    has_visible_parts = False
    
    if with_centerline and lane_segment['centerline']:
        if _is_in_bev_range(lane_segment['centerline']):
            has_visible_parts = True
            _draw_centerline(ax, {'points': lane_segment['centerline']}, 
                            lane_id=lane_index, with_id=with_id, with_endpoints=with_endpoints)
    
    if with_laneline:
        # Left lanelines 처리
        left_lanelines = lane_segment.get('left_laneline', [])
        left_types = lane_segment.get('left_laneline_type', [])
        
        if left_lanelines:
            # 새로운 형식인지 확인: 첫 번째 요소가 리스트인지 확인
            if len(left_lanelines) > 0 and isinstance(left_lanelines[0], list) and len(left_lanelines[0]) > 0 and isinstance(left_lanelines[0][0], (list, tuple)):
                # 새로운 형식: [[coords1], [coords2], ...]
                for i, laneline in enumerate(left_lanelines):
                    if laneline and _is_in_bev_range(laneline):
                        has_visible_parts = True
                        line_type = left_types[i] if isinstance(left_types, list) and i < len(left_types) else 0
                        _draw_line(ax, {
                            'points': laneline, 
                            'linetype': line_type
                        }, with_endpoints=with_endpoints, lane_id=f"{lane_index}_{i}", line_type='left', with_id=with_id)
            else:
                # 기존 형식: [coords] - 단일 laneline
                if _is_in_bev_range(left_lanelines):
                    has_visible_parts = True
                    line_type = left_types if isinstance(left_types, int) else (left_types[0] if left_types else 0)
                    _draw_line(ax, {
                        'points': left_lanelines, 
                        'linetype': line_type
                    }, with_endpoints=with_endpoints, lane_id=lane_index, line_type='left', with_id=with_id)
        
        # Right lanelines 처리
        right_lanelines = lane_segment.get('right_laneline', [])
        right_types = lane_segment.get('right_laneline_type', [])
        
        if right_lanelines:
            # 새로운 형식인지 확인
            if len(right_lanelines) > 0 and isinstance(right_lanelines[0], list) and len(right_lanelines[0]) > 0 and isinstance(right_lanelines[0][0], (list, tuple)):
                # 새로운 형식: [[coords1], [coords2], ...]
                for i, laneline in enumerate(right_lanelines):
                    if laneline and _is_in_bev_range(laneline):
                        has_visible_parts = True
                        line_type = right_types[i] if isinstance(right_types, list) and i < len(right_types) else 0
                        _draw_line(ax, {
                            'points': laneline, 
                            'linetype': line_type
                        }, with_endpoints=with_endpoints, lane_id=f"{lane_index}_{i}", line_type='right', with_id=with_id)
            else:
                # 기존 형식: [coords] - 단일 laneline
                if _is_in_bev_range(right_lanelines):
                    has_visible_parts = True
                    line_type = right_types if isinstance(right_types, int) else (right_types[0] if right_types else 0)
                    _draw_line(ax, {
                        'points': right_lanelines, 
                        'linetype': line_type
                    }, with_endpoints=with_endpoints, lane_id=lane_index, line_type='right', with_id=with_id)
    
    return has_visible_parts

def _draw_area(ax, area, with_endpoints=False):
    """Draw area (pedestrian crossing or road boundary)"""
    # Skip if no points are within BEV range
    if not _is_in_bev_range(area['points']):
        return
    
    if area['category'] == 1:  # pedestrian crossing
        _draw_line(ax, {'points': area['points'], 'linetype': 'ped_crossing'}, 
                  with_endpoints=with_endpoints)
    elif area['category'] == 2:  # road boundary
        _draw_line(ax, {'points': area['points'], 'linetype': 'road_boundary'}, 
                  with_endpoints=with_endpoints)

def _draw_topology_connections(ax, lane_segments, topology_lsls):
    """Draw topology connections between lane segments - using lane midpoints like reference code"""
    if not topology_lsls:
        LOGK("Topology", "⚠️ topology_lsls is empty")
        return
    
    LOGK("Topology", f"start drawing: {len(topology_lsls)}x{len(topology_lsls[0])} matrix")
    connections_drawn = 0
    
    for l1_idx, row in enumerate(topology_lsls):
        if l1_idx >= len(lane_segments):
            continue
            
        for l2_idx, connected in enumerate(row):
            if connected == 1 and l2_idx < len(lane_segments):
                # Get lane segments
                l1 = lane_segments[l1_idx]
                l2 = lane_segments[l2_idx]
                
                if not l1['centerline'] or not l2['centerline']:
                    continue
                
                # Check if both lanes have points within BEV range
                if not _is_in_bev_range(l1['centerline']) or not _is_in_bev_range(l2['centerline']):
                    continue
                    
                l1_points = np.asarray(l1['centerline'])
                l2_points = np.asarray(l2['centerline'])
                
                if len(l1_points) == 0 or len(l2_points) == 0:
                    continue
                
                # Use midpoints like in the reference code
                l1_mid = len(l1_points) // 2
                l2_mid = len(l2_points) // 2
                
                p1 = l1_points[l1_mid]  # [x, y, z] from midpoint
                p2 = l2_points[l2_mid]  # [x, y, z] from midpoint
                
                # Check if midpoints are within BEV range
                if not (_is_in_bev_range([p1]) and _is_in_bev_range([p2])):
                    continue
                
                # Draw arrow from l1 midpoint to l2 midpoint
                # 다른 함수들과 같은 좌표계 사용: points[:, 1]이 x축, points[:, 0]이 y축
                ax.annotate('', xy=(p2[1], p2[0]), xytext=(p1[1], p1[0]),
                           arrowprops=dict(arrowstyle='->', lw=2.0, color='green', alpha=1.0))
                
                connections_drawn += 1
                # Only log the very first connection
                if connections_drawn == 1:
                    LOGK("Topology", f"conn 1: Lane {l1_idx} -> Lane {l2_idx}")
    # Suppressed final topology count log per request

def draw_annotation_bev(annotation, with_centerline=True, with_laneline=True, with_area=True, 
                       with_topology=False, with_lane_ids=False, with_endpoints=False, save_path=None):
    """ 
    Draw BEV annotation from OpenLaneV2 format
    
    Args:
        annotation: OpenLaneV2 annotation dict
        with_centerline: Whether to draw centerlines
        with_laneline: Whether to draw lane boundaries
        with_area: Whether to draw areas (ped crossings, road boundaries)
        with_topology: Whether to draw topology connections
        with_lane_ids: Whether to draw lane segment IDs
        with_endpoints: Whether to draw start/end points for each lane
        save_path: Path to save the image (optional)
    
    Returns:
        numpy array of the rendered image
    """
    fig, ax = plt.subplots(figsize=(8, 16), dpi=200)
    ax.set_aspect('equal')
    ax.set_ylim([BEV_RANGE[0], BEV_RANGE[1]])
    ax.set_xlim([BEV_RANGE[2], BEV_RANGE[3]])
    ax.invert_xaxis()  # Invert x-axis for proper ego coordinate visualization
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title('OpenLaneV2 BEV Visualization (Ego Coordinate System)', fontsize=12)
    ax.set_facecolor('white')
    
    # Draw ego vehicle position (origin)
    ax.scatter(0, 0, color='red', s=50, marker='o', label='Ego Vehicle')
    
    # Draw lane segments
    visible_lanes = 0
    for i, lane_segment in enumerate(annotation['lane_segment']):
        if _draw_lane_segment(ax, lane_segment, i, with_centerline, with_laneline, with_lane_ids, with_endpoints):
            visible_lanes += 1
    
    # Draw topology connections
    if with_topology and 'topology_lsls' in annotation:
        _draw_topology_connections(ax, annotation['lane_segment'], annotation['topology_lsls'])
    
    # Draw areas
    if with_area and 'area' in annotation:
        for area in annotation['area']:
            _draw_area(ax, area, with_endpoints)
    
    # Add legend
    legend_elements = []
    if with_centerline:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['centerline'], 
                                        linewidth=2, label='Centerline'))
    if with_laneline:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['laneline'], 
                                        linewidth=2, label='Lane Line'))
    if with_topology:
        legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--',
                                        linewidth=2, label='Topology Connection'))
    if with_endpoints:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='green', 
                                        linewidth=0, markersize=6, label='Centerline Start'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='red', 
                                        linewidth=0, markersize=6, label='Centerline End'))
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='blue', 
                                        linewidth=0, markersize=6, label='Laneline Start'))
        legend_elements.append(plt.Line2D([0], [0], marker='D', color='purple', 
                                        linewidth=0, markersize=6, label='Laneline End'))
    if with_area:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['ped_crossing'], 
                                        linewidth=2, label='Ped Crossing'))
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['road_boundary'], 
                                        linewidth=2, label='Road Boundary'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', 
                                    linewidth=0, markersize=8, label='Ego Vehicle'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        LOGK("Save image", f"{save_path}")
        plt.close(fig)
        return None  # Return None when saving to file to avoid memory issues
    
    # Convert to numpy array only if not saving to file
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    try:
        data = data.reshape((height, width, 3))
    except ValueError as e:
        LOGK("Reshape", f"error: {e}")
        LOGK("Reshape", f"data size {data.size}, expected {height}x{width}x3 = {height*width*3}")
        plt.close(fig)
        return None
    plt.close(fig)
    return data

def visualize_aerial_hdmap_overlay(json_path, aerial_image_path, output_path=None, 
                                   with_centerline=True, with_laneline=True, 
                                   with_area=True, with_topology=True, with_lane_ids=False, with_endpoints=False,
                                   aerial_alpha=0.7, hdmap_alpha=0.8):
    """
    Create overlay visualization of aerial image and HD Map
    
    Args:
        json_path: Path to OpenLaneV2 JSON file
        aerial_image_path: Path to aerial BEV image
        output_path: Path to save overlay image
        with_centerline: Whether to draw centerlines on HD Map
        with_laneline: Whether to draw lane boundaries on HD Map
        with_area: Whether to draw areas on HD Map
        with_topology: Whether to draw topology connections on HD Map
        with_lane_ids: Whether to draw lane segment IDs on HD Map
        with_endpoints: Whether to draw start/end points on HD Map
        aerial_alpha: Transparency of aerial image (0.0 = transparent, 1.0 = opaque)
        hdmap_alpha: Transparency of HD Map elements (0.0 = transparent, 1.0 = opaque)
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotation = data['annotation']
    
    # Print information
    LOGK("Mode", "🎯 Aerial & HD Map Overlay")
    if 'pose' in data:
        pose = data['pose']
        if 'translation' in pose:
            translation = pose['translation']
            LOGK("Pose", f"position (x,y,z)=({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f})")
        
        if 'rotation' in pose:
            rotation = pose['rotation']
            import math
            heading_deg = math.degrees(rotation[2])
            LOGK("Heading", f"{heading_deg:.2f}° (z-axis rotation)")
    
    LOGK("Aerial image", f"{aerial_image_path}")
    LOGK("HD Map JSON", f"{json_path}")
    LOGK("Alpha", f"aerial {aerial_alpha:.1f}, hdmap {hdmap_alpha:.1f}")
    
    # Load aerial image
    try:
        aerial_img = Image.open(aerial_image_path)
        # Apply only horizontal flip (left-right) to match coordinate system
        aerial_img = aerial_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        aerial_array = np.array(aerial_img)
        img_height, img_width = aerial_array.shape[:2]
        LOGK("Aerial image", f"loaded & flipped: {img_width}x{img_height} px")
    except Exception as e:
        LOGK("Error", f"aerial image load failed: {e}")
        return False
    
    # Create figure with aerial image as background
    fig, ax = plt.subplots(figsize=(12, 16), dpi=200)
    
    # Display aerial image with transparency
    ax.imshow(aerial_img, alpha=aerial_alpha, extent=[-25, 25, -50, 50])  # BEV_RANGE mapped to image
    
    # Set up coordinate system to match BEV range
    ax.set_aspect('equal')
    ax.set_ylim([BEV_RANGE[0], BEV_RANGE[1]])  # -50 to 50 meters
    ax.set_xlim([BEV_RANGE[2], BEV_RANGE[3]])  # -25 to 25 meters
    ax.invert_xaxis()  # Invert x-axis for proper ego coordinate visualization
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.set_xlabel('X (meters)', fontsize=12, color='white')
    ax.set_ylabel('Y (meters)', fontsize=12, color='white')
    ax.set_title('Aerial Image & HD Map Overlay (Ego Coordinate System)', 
                fontsize=14, fontweight='bold', color='white', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    ax.set_facecolor('black')
    
    # Draw ego vehicle position (origin) - make it prominent
    ax.scatter(0, 0, color='red', s=150, marker='o', 
               edgecolors='white', linewidth=3, label='Ego Vehicle', zorder=20)
    
    # Draw lane segments on HD Map with adjusted alpha
    visible_lanes = 0
    original_alpha_values = {}
    
    # Temporarily increase alpha for better visibility over aerial image
    for key in LINE_PARAM:
        if isinstance(key, int) or key in ['ped_crossing', 'road_boundary']:
            original_alpha_values[key] = LINE_PARAM[key]['alpha']
            LINE_PARAM[key]['alpha'] = min(1.0, LINE_PARAM[key]['alpha'] * hdmap_alpha / 0.75)
    
    for i, lane_segment in enumerate(annotation['lane_segment']):
        if _draw_lane_segment(ax, lane_segment, i, with_centerline, with_laneline, with_lane_ids, with_endpoints):
            visible_lanes += 1
    
    # Draw topology connections with enhanced visibility
    if with_topology and 'topology_lsls' in annotation:
        # Temporarily modify topology connection style for better visibility
        _draw_topology_connections(ax, annotation['lane_segment'], annotation['topology_lsls'])
    
    # Draw areas
    if with_area and 'area' in annotation:
        for area in annotation['area']:
            _draw_area(ax, area, with_endpoints)
    
    # Restore original alpha values
    for key, original_alpha in original_alpha_values.items():
        LINE_PARAM[key]['alpha'] = original_alpha
    
    # Add enhanced legend with background
    legend_elements = []
    if with_centerline:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['centerline'], 
                                        linewidth=3, label='Centerline'))
    if with_laneline:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['laneline'], 
                                        linewidth=3, label='Lane Line'))
    if with_topology:
        legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--',
                                        linewidth=3, label='Topology Connection'))
    if with_area:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['ped_crossing'], 
                                        linewidth=3, label='Ped Crossing'))
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['road_boundary'], 
                                        linewidth=3, label='Road Boundary'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', 
                                    linewidth=0, markersize=10, label='Ego Vehicle'))
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                      facecolor='black', edgecolor='white', framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f"{base_name}_aerial_hdmap_overlay.png"
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    LOGK("Save image", f"{output_path}")
    
    # Print statistics
    LOGK("Stats", "Visualization")
    LOGK("Stats", f"lanes {len(annotation['lane_segment'])} (visible {visible_lanes})")
    if 'area' in annotation:
        visible_areas = sum(1 for area in annotation['area'] if _is_in_bev_range(area['points']))
        LOGK("Stats", f"areas {len(annotation['area'])} (visible {visible_areas})")
    if with_topology and 'topology_lsls' in annotation:
        lsls = annotation['topology_lsls']
        connections = sum(sum(row) for row in lsls)
        LOGK("Topology", f"connections {connections}")
    
    plt.show()
    plt.close(fig)
    return True

def visualize_aerial_and_hdmap_comparison(json_path, aerial_image_path, output_path=None, 
                                         with_centerline=True, with_laneline=True, 
                                         with_area=True, with_topology=True, with_lane_ids=False, with_endpoints=False):
    """
    Create side-by-side comparison of aerial image and HD Map visualization
    
    Args:
        json_path: Path to OpenLaneV2 JSON file
        aerial_image_path: Path to aerial BEV image
        output_path: Path to save comparison image
        with_centerline: Whether to draw centerlines on HD Map
        with_laneline: Whether to draw lane boundaries on HD Map
        with_area: Whether to draw areas on HD Map
        with_topology: Whether to draw topology connections on HD Map
        with_lane_ids: Whether to draw lane segment IDs on HD Map
        with_endpoints: Whether to draw start/end points on HD Map
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotation = data['annotation']
    
    # Print ego vehicle position information
    LOGK("Mode", "📍 Aerial & HD Map Comparison")
    if 'pose' in data:
        pose = data['pose']
        if 'translation' in pose:
            translation = pose['translation']
            LOGK("Pose", f"position (x,y,z)=({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f})")
        
        if 'rotation' in pose:
            rotation = pose['rotation']
            import math
            heading_deg = math.degrees(rotation[2])
            LOGK("Heading", f"{heading_deg:.2f}° (z-axis rotation)")
    
    LOGK("Aerial image", f"{aerial_image_path}")
    LOGK("HD Map JSON", f"{json_path}")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    
    # Left subplot: Aerial Image
    try:
        aerial_img = Image.open(aerial_image_path)
        ax1.imshow(aerial_img)
        ax1.set_title('Aerial Image (Vehicle Heading Based)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Pixels', fontsize=10)
        ax1.set_ylabel('Pixels', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add ego vehicle marker at center of aerial image
        h, w = aerial_img.size[::-1]  # PIL returns (width, height)
        center_x, center_y = w // 2, h // 2
        ax1.scatter(center_x, center_y, color='red', s=100, marker='o', 
                   edgecolors='white', linewidth=2, label='Ego Vehicle', zorder=10)
        ax1.legend(loc='upper right')
        LOGK("Aerial image", f"loaded: {w}x{h} px")
        
    except Exception as e:
        LOGK("Error", f"aerial image load failed: {e}")
        ax1.text(0.5, 0.5, f'Aerial Image\nNot Available\n{os.path.basename(aerial_image_path)}', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax1.set_title('Aerial Image (Not Available)', fontsize=14)
    
    # Right subplot: HD Map
    ax2.set_aspect('equal')
    ax2.set_ylim([BEV_RANGE[0], BEV_RANGE[1]])
    ax2.set_xlim([BEV_RANGE[2], BEV_RANGE[3]])
    ax2.invert_xaxis()  # Invert x-axis for proper ego coordinate visualization
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (meters)', fontsize=10)
    ax2.set_ylabel('Y (meters)', fontsize=10)
    ax2.set_title('HD Map Visualization (Ego Coordinate System)', fontsize=14, fontweight='bold')
    ax2.set_facecolor('white')
    
    # Draw ego vehicle position (origin) on HD Map
    ax2.scatter(0, 0, color='red', s=100, marker='o', 
               edgecolors='white', linewidth=2, label='Ego Vehicle', zorder=10)
    
    # Draw lane segments on HD Map
    visible_lanes = 0
    for i, lane_segment in enumerate(annotation['lane_segment']):
        if _draw_lane_segment(ax2, lane_segment, i, with_centerline, with_laneline, with_lane_ids, with_endpoints):
            visible_lanes += 1
    
    # Draw topology connections on HD Map
    if with_topology and 'topology_lsls' in annotation:
        _draw_topology_connections(ax2, annotation['lane_segment'], annotation['topology_lsls'])
    
    # Draw areas on HD Map
    if with_area and 'area' in annotation:
        for area in annotation['area']:
            _draw_area(ax2, area, with_endpoints)
    
    # Add legend for HD Map
    legend_elements = []
    if with_centerline:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['centerline'], 
                                        linewidth=2, label='Centerline'))
    if with_laneline:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['laneline'], 
                                        linewidth=2, label='Lane Line'))
    if with_topology:
        legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--',
                                        linewidth=2, label='Topology Connection'))
    if with_area:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['ped_crossing'], 
                                        linewidth=2, label='Ped Crossing'))
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_DICT['road_boundary'], 
                                        linewidth=2, label='Road Boundary'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', 
                                    linewidth=0, markersize=8, label='Ego Vehicle'))
    
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f"{base_name}_aerial_hdmap_comparison.png"
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    LOGK("Save image", f"{output_path}")
    
    # Print statistics
    LOGK("Stats", "Visualization")
    LOGK("Stats", f"lanes {len(annotation['lane_segment'])} (visible {visible_lanes})")
    if 'area' in annotation:
        visible_areas = sum(1 for area in annotation['area'] if _is_in_bev_range(area['points']))
    LOGK("Stats", f"areas {len(annotation['area'])} (visible {visible_areas})")
    if with_topology and 'topology_lsls' in annotation:
        lsls = annotation['topology_lsls']
        connections = sum(sum(row) for row in lsls)
        LOGK("Topology", f"connections {connections}")
    
    plt.show()
    plt.close(fig)

def visualize_openlane_json(json_path, output_path=None, with_centerline=True, with_laneline=True, 
                          with_area=True, with_topology=True, with_lane_ids=False, with_endpoints=False):
    """
    Load OpenLaneV2 JSON file and create visualization
    
    Args:
        json_path: Path to OpenLaneV2 JSON file
        output_path: Path to save visualization image
        with_centerline: Whether to draw centerlines
        with_laneline: Whether to draw lane boundaries  
        with_area: Whether to draw areas
        with_topology: Whether to draw topology connections
        with_lane_ids: Whether to draw lane segment IDs
        with_endpoints: Whether to draw visible start/end points for each lane
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotation = data['annotation']
    
    # Suppressed verbose ego/pose info printing per request
    
    # Create visualization
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f"{base_name}_visualization.png"
    
    img_array = draw_annotation_bev(annotation, with_centerline, with_laneline, with_area, 
                                   with_topology, with_lane_ids, with_endpoints, output_path)
    
    # Print statistics
    print(f"Visualization complete!")
    print(f"- Lane segments: {len(annotation['lane_segment'])}")
    if 'area' in annotation:
        visible_areas = sum(1 for area in annotation['area'] if _is_in_bev_range(area['points']))
        ped_crossings = sum(1 for area in annotation['area'] if area['category'] == 1 and _is_in_bev_range(area['points']))
        road_boundaries = sum(1 for area in annotation['area'] if area['category'] == 2 and _is_in_bev_range(area['points']))
        print(f"- Areas: {len(annotation['area'])} (visible in BEV: {visible_areas} - {ped_crossings} ped crossings, {road_boundaries} road boundaries)")
    else:
        print(f"- Areas: 0")
    
    # Count visible lane segments and total lanelines
    visible_lanes = 0
    total_left_lanelines = 0
    total_right_lanelines = 0
    
    for lane_segment in annotation['lane_segment']:
        is_visible = False
        
        # Check centerline
        if lane_segment['centerline'] and _is_in_bev_range(lane_segment['centerline']):
            is_visible = True
        
        # Check left lanelines (새로운 형식: 리스트의 리스트)
        left_lanelines = lane_segment.get('left_laneline', [])
        if left_lanelines:
            if isinstance(left_lanelines[0], list) and len(left_lanelines[0]) > 0 and isinstance(left_lanelines[0][0], (list, tuple)):
                # 새로운 형식
                total_left_lanelines += len(left_lanelines)
                for ll in left_lanelines:
                    if ll and _is_in_bev_range(ll):
                        is_visible = True
            else:
                # 기존 형식
                total_left_lanelines += 1
                if _is_in_bev_range(left_lanelines):
                    is_visible = True
        
        # Check right lanelines (새로운 형식: 리스트의 리스트)
        right_lanelines = lane_segment.get('right_laneline', [])
        if right_lanelines:
            if isinstance(right_lanelines[0], list) and len(right_lanelines[0]) > 0 and isinstance(right_lanelines[0][0], (list, tuple)):
                # 새로운 형식
                total_right_lanelines += len(right_lanelines)
                for rl in right_lanelines:
                    if rl and _is_in_bev_range(rl):
                        is_visible = True
            else:
                # 기존 형식
                total_right_lanelines += 1
                if _is_in_bev_range(right_lanelines):
                    is_visible = True
        
        if is_visible:
            visible_lanes += 1
    
    print(f"- Visible lane segments in BEV range: {visible_lanes}")
    print(f"- Total lanelines: left={total_left_lanelines}, right={total_right_lanelines}")
    
    if with_topology and 'topology_lsls' in annotation:
        lsls = annotation['topology_lsls']
        connections = sum(sum(row) for row in lsls)
        print(f"- Topology connections: {connections}")
    
    return img_array

def main():
    parser = argparse.ArgumentParser(description='Visualize OpenLaneV2 format JSON files with optional aerial image comparison')
    parser.add_argument('json_path', help='Path to OpenLaneV2 JSON file')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--aerial', '-a', help='Path to aerial BEV image for comparison')
    parser.add_argument('--overlay', action='store_true', help='Create overlay instead of side-by-side comparison')
    parser.add_argument('--aerial-alpha', type=float, default=0.6, help='Aerial image transparency (0.0-1.0)')
    parser.add_argument('--hdmap-alpha', type=float, default=0.9, help='HD Map transparency (0.0-1.0)')
    parser.add_argument('--no-centerline', action='store_true', help='Disable centerline visualization')
    parser.add_argument('--no-laneline', action='store_true', help='Disable lane line visualization')
    parser.add_argument('--no-area', action='store_true', help='Disable area visualization')
    parser.add_argument('--topology', action='store_true', help='Enable topology connection visualization')
    parser.add_argument('--lane-ids', action='store_true', help='Show lane segment IDs on centerlines')
    parser.add_argument('--endpoints', action='store_true', help='Show start/end points for each lane segment')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        return
    
    # Check if aerial image comparison is requested
    if args.aerial:
        if not os.path.exists(args.aerial):
            print(f"Error: Aerial image file not found: {args.aerial}")
            return
        
        if args.overlay:
            print(f"🎨 Aerial & HD Map overlay mode")
            print(f"   Aerial image: {args.aerial}")
            print(f"   HD Map JSON: {args.json_path}")
            print(f"   Aerial alpha: {args.aerial_alpha}, HD Map alpha: {args.hdmap_alpha}")
            
            # Create overlay visualization
            visualize_aerial_hdmap_overlay(
                json_path=args.json_path,
                aerial_image_path=args.aerial,
                output_path=args.output,
                with_centerline=not args.no_centerline,
                with_laneline=not args.no_laneline,
                with_area=not args.no_area,
                with_topology=args.topology,
                with_lane_ids=args.lane_ids,
                with_endpoints=args.endpoints,
                aerial_alpha=args.aerial_alpha,
                hdmap_alpha=args.hdmap_alpha
            )
            return
        else:
            print(f"🎯 Aerial & HD Map comparison mode")
            print(f"   Aerial image: {args.aerial}")
            print(f"   HD Map JSON: {args.json_path}")
            
            # Create comparison visualization
            visualize_aerial_and_hdmap_comparison(
                json_path=args.json_path,
                aerial_image_path=args.aerial,
                output_path=args.output,
                with_centerline=not args.no_centerline,
                with_laneline=not args.no_laneline,
                with_area=not args.no_area,
                with_topology=args.topology,
                with_lane_ids=args.lane_ids,
                with_endpoints=args.endpoints
            )
            return
    elif args.overlay:
        # overlay 옵션은 주어졌지만 aerial 이미지가 없는 경우
        print(f"Error: --overlay option requires --aerial <aerial_image_path>")
        print(f"Usage: python OpenLane_Visualizer.py {args.json_path} --overlay --aerial <aerial_image_path>")
        return
    
    # Standard HD Map only visualization
    # Suppressed CLI banner per request
    
    # Create standard visualization
    visualize_openlane_json(
        json_path=args.json_path,
        output_path=args.output,
        with_centerline=not args.no_centerline,
        with_laneline=not args.no_laneline,
        with_area=not args.no_area,
        with_topology=args.topology,
        with_lane_ids=args.lane_ids,
        with_endpoints=args.endpoints
    )

def quick_aerial_hdmap_overlay():
    """
    Quick function to create overlay of aerial image and HD Map
    Uses the default paths from ngii_aerial_downloader_v2.py output
    """
    # Default paths
    json_path = "/home/iismn/Workspace_C/Dataset/LSD_Net/OpenLane-V2/train/01700/info/1757328161589611008-ls.json"
    aerial_path = "/home/iismn/Workspace_C/Dataset/LSD_Net/NGII/AERIAL_IMAGES/aerial_path_1_point_0_heading_based.png"
    output_path = "/home/iismn/Workspace_C/Dataset/LSD_Net/NGII/aerial_hdmap_overlay.png"
    
    print("🎨 Quick Aerial & HD Map Overlay")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"❌ HD Map JSON not found: {json_path}")
        return False
    
    if not os.path.exists(aerial_path):
        print(f"❌ Aerial image not found: {aerial_path}")
        print(f"   Please run ngii_aerial_downloader_v2.py first to generate the aerial image")
        return False
    
    # Create overlay
    try:
        return visualize_aerial_hdmap_overlay(
            json_path=json_path,
            aerial_image_path=aerial_path,
            output_path=output_path,
            with_centerline=True,
            with_laneline=True,
            with_area=True,
            with_topology=True,
            with_lane_ids=False,
            with_endpoints=False,
            aerial_alpha=0.6,  # Semi-transparent aerial image
            hdmap_alpha=0.9    # More opaque HD Map for visibility
        )
    except Exception as e:
        print(f"❌ Overlay creation failed: {e}")
        return False

def quick_aerial_hdmap_comparison():
    """
    Quick function to compare the generated aerial image with HD Map
    Uses the default paths from ngii_aerial_downloader_v2.py output
    """
    # Default paths
    json_path = "/home/iismn/Workspace_C/Dataset/LSD_Net/OpenLane-V2/train/01700/info/1757328161589611008-ls.json"
    aerial_path = "/home/iismn/Workspace_C/Dataset/LSD_Net/NGII/AERIAL_IMAGES/aerial_path_1_point_0_heading_based.png"
    output_path = "/home/iismn/Workspace_C/Dataset/LSD_Net/NGII/aerial_hdmap_comparison.png"
    
    print("🚀 Quick Aerial & HD Map Comparison")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"❌ HD Map JSON not found: {json_path}")
        return False
    
    if not os.path.exists(aerial_path):
        print(f"❌ Aerial image not found: {aerial_path}")
        print(f"   Please run ngii_aerial_downloader_v2.py first to generate the aerial image")
        return False
    
    # Create comparison
    try:
        visualize_aerial_and_hdmap_comparison(
            json_path=json_path,
            aerial_image_path=aerial_path,
            output_path=output_path,
            with_centerline=True,
            with_laneline=True,
            with_area=True,
            with_topology=True,
            with_lane_ids=False,
            with_endpoints=False
        )
        return True
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No arguments provided, show usage
        print("OpenLaneV2 Visualization Tool")
        print("=" * 40)
        print("Usage examples:")
        print("  # Basic HD Map visualization:")
        print("  python OpenLane_Visualizer.py <json_path>")
        print("")
        print("  # Side-by-side comparison with aerial image:")
        print("  python OpenLane_Visualizer.py <json_path> --aerial <aerial_image_path>")
        print("")
        print("  # Overlay aerial image with HD Map:")
        print("  python OpenLane_Visualizer.py <json_path> --overlay --aerial <aerial_image_path>")
        print("")
        print("  # Advanced options:")
        print("  python OpenLane_Visualizer.py <json_path> --topology --lane-ids --endpoints")
        print("")
        print("For full help: python OpenLane_Visualizer.py --help")
    else:
        # Run with command line arguments
        main()
