![header](https://capsule-render.vercel.app/api?type=rect&color=timeGradient&text=TOPO-LSDNET%20DB%20TOOLS&fontSize=20)

## Overview
**Topo-LSDNet DB Tools** is a dataset conversion toolkit for the Topo-LSDNet project. It converts ROS2 Bag data with NGII HD-Map shapefiles into OpenLaneV2 format for lane segment detection training.

### Supported Regions
| Region | Description |
|--------|-------------|
| **Yeouido** | Seoul financial district with complex intersections |
| **Sangam** | Digital Media City area with urban roads |

## Key Components
| Module | Description |
|--------|-------------|
| **ROS2_Converter** | ROS2 Bag → OpenLaneV2 format conversion with 6-camera sync |
| **NGIIToOpenLaneConverter** | NGII SHP → Lane segment with ego-centric transform |
| **ROS2_Converter_streaming** | Memory-efficient streaming mode converter |
| **JSON Visualizer** | OpenLane annotation visualization tools |

## Repository Structure
```
Dataset/
├── HDMap_Info/                    # NGII HD-Map Shapefiles (large - not in Git)
│   ├── Yeouido/                   # *_a2_link*.shp, *_b2_surfacelinemark*.shp
│   └── Sangam/
├── ROSBag_Info/                   # ROS2 Bag data (large - not in Git)
│   ├── Yeouido/ioniq5_topics_*/
│   └── Sangam/ioniq5_topics_*/
├── Vehicle_Info/                  # Camera calibration
│   ├── calib.yaml                 # Intrinsic/extrinsic parameters
│   └── kalibr_viewer.html
└── utils/                         # Conversion tools
    ├── ros_converter/
    │   ├── ROS2_Converter.py      # Main converter
    │   ├── ngii_openlane_converter.py
    │   └── ngii_aerial_converter.py
    ├── ROS2_Converter_streaming.py
    ├── json_visualizer/
    ├── pkl_processor/
    └── NGII_*.py                  # Map downloaders
```

## Quick Start
```python
from ros_converter.ROS2_Converter import ROS2OpenLaneConverter, ConverterConfig

config = ConverterConfig(
    bag_path=Path("/path/to/rosbag"),
    output_root=Path("/path/to/output"),
    city_name="Sangam",
    max_frames_per_camera=100,
)
converter = ROS2OpenLaneConverter(config)
converter.run()
```

## Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `city_name` | `None` | City name (Yeouido/Sangam) - auto-inferred from path |
| `max_frames_per_camera` | `None` | Max frames per camera |
| `ngii_view_width` | `50.0` | Ego-centric view width (m) |
| `ngii_view_length` | `100.0` | Ego-centric view length (m) |
| `ngii_front_ratio` | `0.5` | Front ratio (0.5 = symmetric) |
| `trajectory_horizon_distance_m` | `50.0` | Trajectory horizon (m) |
| `stationary_skip_enable` | `True` | Skip stationary segments |

## Output Format (OpenLaneV2)
```json
{
  "version": "1.0",
  "meta": { "source": "NGII_Inhouse", "city": "Sangam" },
  "ego_pose": {
    "translation": [x, y, z],
    "rotation": [qx, qy, qz, qw]
  },
  "annotation": {
    "lane_segment": [{
      "id": "A219AS319504",
      "centerline": [[x, y, z], ...],
      "left_laneline": [[x, y, z], ...],
      "left_laneline_type": 1,
      "right_laneline": [[x, y, z], ...],
      "right_laneline_type": 2,
      "is_intersection_or_connector": true
    }],
    "area": [{ "id": "crosswalk_001", "category": 1, "points": [...] }]
  }
}
```

## Conversion Pipeline
```
ROS2 Bag (.db3)
     │
     ▼
┌─────────────────┐
│ ROS2_Converter  │ ◄── Camera Calibration (calib.yaml)
└─────────────────┘
     │
     ▼
┌─────────────────────────┐
│ NGIIToOpenLaneConverter │ ◄── NGII Shapefiles (HDMap_Info/)
└─────────────────────────┘
     │
     ▼
OpenLaneV2 Format (.json + images)
```

## Requirements
```
geopandas
shapely
pyproj
opencv-python
numpy
rosbag2_py
rclpy
```

## Changelog

### v1.0.0-inhouse (Current)
- Lane boundary continuous merge logic
- `is_intersection_or_connector`: `true` when `linktype == 1`
- Lane ID: link ID only (no boundary ID suffix)
- Lane type: length-based dominant type selection
- `lane_type == 0` (none) boundary filtering
- City-based SHP auto-selection (Yeouido/Sangam)

### v0.x (Previous Training)
- Separate segment per boundary combination
- `is_intersection_or_connector`: always `false`
- Lane ID: `{link_id}_LB{left_boundary}_RB{right_boundary}` format

## License
Released under the MIT License.
