![header](https://capsule-render.vercel.app/api?type=rect&color=timeGradient&text=TOPO-LSDNET%20DB%20TOOLS&fontSize=20)

# OpenLane-V2 SD Map Generator

OpenLane-V2 데이터셋을 위한 SD Map (Standard Definition Map) 생성 도구입니다. OSM 도로/건물 데이터와 항공 이미지를 다운로드하여 ego-centric 좌표계로 변환합니다.

## Overview

이 브랜치는 **OpenLane-V2 데이터셋**을 위한 SD Map 생성 도구를 포함합니다:
- OSM 기반 도로 네트워크 및 건물 데이터 추출
- 항공 이미지 다운로드 (API 키 불필요)
- Ego-centric 좌표계 변환
- Graph (`.pkl`) 및 Raster (`.png`) 출력 지원

## Repository Structure

```
OpenLane-V2/
├── utils/                              # Conversion tools
│   ├── load_sdmap_OpenLaneV2.py        # SD Map graph/raster generator
│   ├── load_sdmap_Parallel_OpenLaneV2.sh   # Parallel execution script
│   ├── load_aerialmap_OpenLaneV2.py    # Aerial imagery downloader
│   └── load_aerialmap_Parallel_OpenLaneV2.sh   # Parallel execution script
├── train/                              # Training split (large - not in Git)
├── val/                                # Validation split (large - not in Git)
├── test/                               # Test split (large - not in Git)
├── sd_map_graph_all/                   # Generated graph outputs (not in Git)
│   └── {split}/{segment_id}/{timestamp}.pkl
├── sd_map_raster_all/                  # Generated raster outputs (not in Git)
│   └── {split}/{segment_id}/{timestamp}.png
├── sd_map_imagery_all/                 # Generated aerial imagery (not in Git)
│   └── {split}/{segment_id}/{timestamp}.png
├── data_dict_subset_A.json             # Subset A data dictionary
├── data_dict_subset_B.json             # Subset B data dictionary
└── data_process_ls.py                  # Lane segment preprocessing
```

## Key Components

| Tool | Description |
|------|-------------|
| `load_sdmap_OpenLaneV2.py` | OSM 도로/건물 데이터 → Graph (`.pkl`) + Raster (`.png`) 변환 |
| `load_aerialmap_OpenLaneV2.py` | 항공 이미지 다운로드 (Google Maps 타일 기반, API 키 불필요) |
| `data_process_ls.py` | OpenLane-V2 lane segment 전처리 |

## Quick Start

### SD Map Graph/Raster 생성

```bash
# Single process
python utils/load_sdmap_OpenLaneV2.py \
    --collection 'data_dict_subset_A_{split}_ls' \
    --split train \
    --root_path /path/to/OpenLane-V2 \
    --root_path_ArgoverseV2 /path/to/ArgoverseV2 \
    --dist_x 50 \
    --dist_y 100 \
    --rasterize

# Parallel execution (recommended)
bash utils/load_sdmap_Parallel_OpenLaneV2.sh
```

### Aerial Imagery 생성

```bash
# Single process
python utils/load_aerialmap_OpenLaneV2.py \
    --collection 'data_dict_subset_A_{split}_ls' \
    --split train \
    --root_path /path/to/OpenLane-V2 \
    --zoom 20 \
    --dist_x 50 \
    --dist_y 100

# Parallel execution (recommended)
bash utils/load_aerialmap_Parallel_OpenLaneV2.sh
```

## CLI Arguments

### load_sdmap_OpenLaneV2.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_path` | (required) | OpenLane-V2 데이터 경로 |
| `--root_path_ArgoverseV2` | - | ArgoverseV2 city_dict 경로 |
| `--split` | `train` | 데이터 split (train/val/test) |
| `--collection` | `data_dict_subset_A_{split}_ls` | Collection 이름 |
| `--dist_x` | `50.0` | X 방향 전체 크기 (미터) |
| `--dist_y` | `100.0` | Y 방향 전체 크기 (미터) |
| `--rasterize` | `false` | PNG 래스터 출력 활성화 |
| `--zoom_for_res` | `20` | 래스터 해상도 계산용 줌 레벨 |
| `--network_type` | `all` | OSM 네트워크 타입 |
| `--total_parts` | `1` | 병렬 처리 파트 수 |
| `--part` | `0` | 현재 파트 인덱스 |
| `--overwrite` | `false` | 기존 파일 덮어쓰기 |

### load_aerialmap_OpenLaneV2.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_path` | (required) | OpenLane-V2 데이터 경로 |
| `--split` | `test` | 데이터 split (train/val/test) |
| `--zoom` | `20` | 타일 줌 레벨 |
| `--dist_x` | `80` | X 방향 전체 크기 (미터) |
| `--dist_y` | `100` | Y 방향 전체 크기 (미터) |
| `--total_parts` | `1` | 병렬 처리 파트 수 |
| `--part` | `0` | 현재 파트 인덱스 |
| `--overwrite` | `false` | 기존 파일 덮어쓰기 |

## Output Format

### Graph Output (`.pkl`)

```python
{
    'truck_road': [np.array([[x, y], ...]), ...],   # 고속도로/간선도로
    'highway': [np.array([[x, y], ...]), ...],      # 주요 도로
    'residential': [np.array([[x, y], ...]), ...],  # 주거지역 도로
    'service': [np.array([[x, y], ...]), ...],      # 서비스 도로
    'road': [np.array([[x, y], ...]), ...],         # 기타 도로
    'bus_way': [np.array([[x, y], ...]), ...],      # 버스 전용 도로
    'building': [np.array([[x, y], ...]), ...],     # 건물 폴리곤
    'other': [...]                                   # 기타
}
```

### Raster Output (`.png`)

- **RGB 3채널 이미지**
  - R (Red): 도로 네트워크
  - G (Green): 건물
  - B (Blue): 배경

## Requirements

```
numpy
opencv-python
geopandas
osmnx
shapely
pyproj
pillow
scipy
tqdm
requests
av2  # Argoverse2 SDK (for coordinate conversion)
openlanev2  # OpenLane-V2 SDK
```

## Related Branches

- **main**: Inhouse 데이터셋 변환 도구 (ROS2 Bag + NGII HD-Map)
- **OpenLaneV2**: OpenLane-V2 데이터셋용 SD Map 생성 도구 (현재 브랜치)

## License

Released under the MIT License.
