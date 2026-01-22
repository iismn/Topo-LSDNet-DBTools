#!/bin/bash
set -e

# Ctrl+C 시 모든 백그라운드 프로세스 종료
cleanup() {
    echo -e "\n[INFO] Ctrl+C detected. Killing all background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo "[INFO] All processes terminated."
    exit 1
}
trap cleanup SIGINT SIGTERM

# Set PYTHONPATH for openlanev2 module
export PYTHONPATH="/home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet:$PYTHONPATH"

# Change to data directory
cd /home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/OpenLane-V2

ntotal_parts=6          # CPU 코어 수만큼
for ((ipart=0; ipart<ntotal_parts; ipart++)); do
    export TQDM_POS_BASE=$((ipart*2))   # 프로세스당 2줄 슬롯 예약
    export PART_IDX=$ipart              # desc에 표시용 (선택)
    PYTHONUNBUFFERED=1 \
    python utils/load_sdmap_OpenLaneV2.py \
            --collection 'data_dict_subset_A_{split}_ls' \
            --split train \
            --root_path /home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/OpenLane-V2 \
            --root_path_ArgoverseV2 /home/iismn/Workspaace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/ArgoverseV2 \
            --dist_x 50 \
            --dist_y 100 \
            --zoom_for_res 19 \
            --rasterize \
            --network_type all \
            --total_parts $ntotal_parts \
            --part $ipart &
done
wait


# ============================================================
# 참고: dist_x, dist_y는 최종 이미지 크기 (미터 단위)
# ------------------------------------------------------------
# dist_x = 가로 크기 (m)
# dist_y = 세로 크기 (m)
# ============================================================
# 저장 경로:
# - sd_map_graph_all/{split}/{segment_id}/{timestamp}.pkl
# - sd_map_raster_all/{split}/{segment_id}/{timestamp}.png
# ============================================================

# Single process example:
# python utils/load_sdmap_OpenLaneV2.py \
#         --collection 'data_dict_subset_A_{split}_ls' \
#         --split train \
#         --root_path /home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/OpenLane-V2 \
#         --root_path_ArgoverseV2 /home/iismn/Workspace_B/IEEE_CVF_CVPR/Study/TopoLSDNet/data/ArgoverseV2 \
#         --dist_x 50 \
#         --dist_y 100 \
#         --rasterize
