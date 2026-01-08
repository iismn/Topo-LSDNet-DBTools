# ROSBag_Info

ROS2 Bag 원본 데이터가 저장되는 폴더입니다.

## 구조

```
ROSBag_Info/
├── Yeouido/
│   └── ioniq5_topics_YYYYMMDD_HHMMSS/
│       ├── metadata.yaml
│       └── ioniq5_topics_0.db3
└── Sangam/
    └── ioniq5_topics_YYYYMMDD_HHMMSS/
        └── ...
```

## 포함 토픽

| 토픽 | 메시지 타입 | 설명 |
|-----|-----------|------|
| `/UDP_GMSL_FM/image_raw` | sensor_msgs/Image | 전방 중앙 카메라 |
| `/UDP_GMSL_FL/image_raw` | sensor_msgs/Image | 전방 좌측 카메라 |
| `/UDP_GMSL_FR/image_raw` | sensor_msgs/Image | 전방 우측 카메라 |
| `/UDP_GMSL_BM/image_raw` | sensor_msgs/Image | 후방 중앙 카메라 |
| `/UDP_GMSL_BL/image_raw` | sensor_msgs/Image | 후방 좌측 카메라 |
| `/UDP_GMSL_BR/image_raw` | sensor_msgs/Image | 후방 우측 카메라 |
| `/NAVSIGHT/imu/odometry` | nav_msgs/Odometry | Odometry (UTM 좌표) |

**참고**: 실제 Bag 파일은 용량이 크므로 Git에 포함되지 않습니다.
