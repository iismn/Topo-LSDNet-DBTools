# Topo-LSDNet Database Tools (Inhouse Dataset)

ROS2 Bag 데이터를 OpenLaneV2 포맷으로 변환하기 위한 도구 모음입니다.

## 📁 폴더 구조

```
Dataset/
├── HDMap_Info/                    # NGII HD맵 Shapefile (대용량 - Git 미포함)
│   ├── Yeouido/                   # 여의도 지역 SHP 파일들
│   │   ├── *_a2_link*.shp         # 도로 링크 정보
│   │   ├── *_b2_surfacelinemark*.shp  # 차선 경계선
│   │   └── *_b3_surfacemark*.shp  # 노면 표시
│   └── Sangam/                    # 상암 지역 SHP 파일들
│
├── ROSBag_Info/                   # ROS2 Bag 원본 데이터 (대용량 - Git 미포함)
│   ├── Yeouido/                   # 여의도 주행 데이터
│   │   └── ioniq5_topics_*/       # 각 주행 세션별 ROS2 bag
│   └── Sangam/                    # 상암 주행 데이터
│
├── Vehicle_Info/                  # 차량 캘리브레이션 정보
│   ├── calib.yaml                 # 카메라 intrinsic/extrinsic 파라미터
│   ├── calib_compress-results-cam.txt
│   └── kalibr_viewer.html         # 캘리브레이션 시각화
│
└── utils/                         # 변환 도구들
    ├── ros_converter/             # ROS2 → OpenLaneV2 변환기
    │   ├── ROS2_Converter.py      # 메인 변환기
    │   ├── ngii_openlane_converter.py  # NGII SHP → OpenLane 변환
    │   └── ngii_aerial_converter.py    # 항공사진 다운로더
    ├── ROS2_Converter_streaming.py     # 스트리밍 방식 변환기
    ├── json_visualizer/           # OpenLane JSON 시각화 도구
    ├── pkl_processor/             # PKL 데이터 처리기
    ├── NGII_Aerial_Donwloader.py  # NGII 항공사진 다운로드
    └── NGII_SD_Donwloader.py      # NGII SD맵 다운로드
```

## 🔧 주요 컴포넌트

### 1. ROS2_Converter
ROS2 Bag 파일에서 이미지와 odometry 데이터를 추출하여 OpenLaneV2 포맷으로 변환합니다.

**주요 기능:**
- 6개 카메라 이미지 동기화 추출
- NGII HD맵 기반 lane annotation 자동 생성
- Trajectory 정보 생성 (WGS84 좌표계)
- 도시별 SHP 파일 자동 선택 (Yeouido/Sangam)

**사용법:**
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

### 2. NGIIToOpenLaneConverter
NGII Shapefile을 OpenLaneV2 lane segment 포맷으로 변환합니다.

**주요 기능:**
- Ego-centric 좌표계 변환 (100m × 200m 영역)
- Lane boundary 자동 병합 (연속 boundary 연결)
- Lane type 결정 (solid/dashed, 길이 기반)
- 교차로 감지 (`linktype == 1`)
- Topology matrix 생성 (predecessor/successor 관계)

### 3. ConverterConfig 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `city_name` | `None` | 도시명 (Yeouido/Sangam) - bag 경로에서 자동 추론 |
| `max_frames_per_camera` | `None` | 카메라당 최대 프레임 수 |
| `ngii_view_width` | `50.0` | Ego 좌표계 폭 (m) |
| `ngii_view_length` | `100.0` | Ego 좌표계 길이 (m) |
| `ngii_front_ratio` | `0.5` | 전방 비율 (0.5 = 앞뒤 동일) |
| `trajectory_horizon_distance_m` | `50.0` | Trajectory 거리 (m) |
| `stationary_skip_enable` | `True` | 정지 구간 스킵 여부 |

## 📊 출력 데이터 구조

```json
{
  "version": "1.0",
  "meta": {
    "source": "NGII_Inhouse",
    "city": "Sangam",
    "timestamp_ns": 1763430741775094595
  },
  "ego_pose": {
    "translation": [x, y, z],
    "rotation": [qx, qy, qz, qw]
  },
  "annotation": {
    "lane_segment": [
      {
        "id": "A219AS319504",
        "centerline": [[x, y, z], ...],
        "left_laneline": [[x, y, z], ...],
        "left_laneline_type": 1,  // 0=none, 1=solid, 2=dashed
        "right_laneline": [[x, y, z], ...],
        "right_laneline_type": 2,
        "is_intersection_or_connector": true
      }
    ],
    "area": [
      {
        "id": "crosswalk_001",
        "category": 1,
        "points": [[x, y, z], ...]
      }
    ]
  },
  "sensor": {
    "ring_front_center": { "image_path": "..." },
    ...
  }
}
```

## 🔄 변환 워크플로우

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

## 📋 의존성

```
geopandas
shapely
pyproj
opencv-python
numpy
rosbag2_py
rclpy
```

## 📝 변경 이력

### V7 (현재)
- Lane boundary 연속 병합 로직 적용
- `is_intersection_or_connector`: `linktype == 1`이면 `true`
- Lane ID: link ID만 사용 (boundary ID 미포함)
- Lane type: 길이 기반 dominant type 결정
- `lane_type == 0` (none) boundary 필터링

### V6 (이전 훈련용)
- Boundary 조합별 별도 segment 생성
- `is_intersection_or_connector`: 항상 `false`
- Lane ID: `{link_id}_LB{left_boundary}_RB{right_boundary}` 형식

## 📄 License

MIT License
