# HDMap_Info

NGII HD맵 Shapefile 데이터가 저장되는 폴더입니다.

## 구조

```
HDMap_Info/
├── Yeouido/          # 여의도 지역
│   ├── *_a2_link*.shp          # 도로 링크 정보
│   ├── *_b2_surfacelinemark*.shp   # 차선 경계선
│   └── *_b3_surfacemark*.shp   # 노면 표시 (횡단보도 등)
└── Sangam/           # 상암 지역
    └── (동일 구조)
```

## 데이터 획득

NGII (국토지리정보원) HD맵 포털에서 다운로드:
- https://map.ngii.go.kr/

**참고**: 실제 SHP 파일은 용량이 크므로 Git에 포함되지 않습니다.
