import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import random

from shapely.geometry import LineString
from shapely.errors import GEOSException
from openlanev2.lanesegment.io import io

def _fix_pts_interpolate(curve, n_points):
    """Interpolate a curve (list of coordinates) to n_points."""
    # 빈 경우 처리
    if not curve:
        return np.zeros((n_points, 3), dtype=np.float32)
    
    # list를 numpy array로 변환
    curve_np = np.array(curve, dtype=np.float32, copy=True)
    
    if curve_np.ndim != 2:
        dims = curve_np.shape[-1] if curve_np.size else 3
        return np.zeros((n_points, dims), dtype=np.float32)
    if curve_np.shape[0] == 0:
        return np.zeros((n_points, curve_np.shape[1]), dtype=np.float32)
    if curve_np.shape[0] == 1:
        return np.repeat(curve_np, n_points, axis=0)

    try:
        ls = LineString(curve_np)
    except (ValueError, GEOSException):
        return np.repeat(curve_np[0:1], n_points, axis=0)

    if ls.length == 0:
        return np.repeat(curve_np[0:1], n_points, axis=0)

    distances = np.linspace(0, ls.length, n_points)
    coords = []
    for distance in distances:
        point = ls.interpolate(distance)
        try:
            empty = bool(point.is_empty)
        except (TypeError, AttributeError):
            empty = True
        coords.append(curve_np[-1] if empty else point.coords[0])
    return np.asarray(coords, dtype=np.float32)


def _iter_samples(data_dict: dict):
    for split, logs in data_dict.items():
        for log_name, sequences in logs.items():
            if isinstance(sequences, dict):
                for seq_name, filenames in sequences.items():
                    for filename in filenames:
                        timestamp = Path(filename).stem
                        yield split, log_name, seq_name, timestamp
            elif isinstance(sequences, list):
                for filename in sequences:
                    timestamp = Path(filename).stem
                    yield split, log_name, "", timestamp
            else:
                raise TypeError(f"Unsupported data_dict structure under split '{split}'.")


def _process_single(root_path: Path, n_points: dict, sample):
    split, log_name, seq_name, timestamp = sample
    segment_id = f"{log_name}/{seq_name}" if seq_name else log_name
    identifier = (split, segment_id, timestamp)

    data_split = split if split != 'val' else 'test'
    json_dir = root_path / data_split / log_name
    if not json_dir.exists():
        raise FileNotFoundError(f"Could not locate data for split '{split}' at {json_dir}")
    if seq_name:
        json_dir = json_dir / seq_name
    if not json_dir.exists():
        raise FileNotFoundError(f"Sequence '{seq_name}' missing under {json_dir.parent}")

    frame_path = json_dir / "info" / f"{timestamp}-ls.json"
    frame = io.json_load(str(frame_path))

    traj_path = json_dir / "traj" / f"{timestamp}_egopose.npy"
    frame['vehicle2global'] = np.asarray(np.load(traj_path), dtype=np.float64) if traj_path.exists() else None

    pose = frame.get('pose', {})
    for key, value in pose.items():
        pose[key] = np.asarray(value, dtype=np.float64)

    for cam_info in frame.get('sensor', {}).values():
        intrinsic = cam_info.get('intrinsic', {})
        for key, value in intrinsic.items():
            intrinsic[key] = np.asarray(value, dtype=np.float64)

        extrinsic = cam_info.get('extrinsic', {})
        for key, value in extrinsic.items():
            extrinsic[key] = np.asarray(value, dtype=np.float64)

    if 'annotation' not in frame:
        return identifier, frame

    annotation = frame['annotation']
    for idx, area in enumerate(annotation.get('area', [])):
        annotation['area'][idx]['points'] = np.asarray(area['points'], dtype=np.float32)

    for idx, lane_segment in enumerate(annotation.get('lane_segment', [])):
        lane_segment['centerline'] = _fix_pts_interpolate(lane_segment['centerline'], n_points['centerline'])
        lane_segment['left_laneline'] = _fix_pts_interpolate(lane_segment['left_laneline'], n_points['left_laneline'])
        lane_segment['right_laneline'] = _fix_pts_interpolate(lane_segment['right_laneline'], n_points['right_laneline'])

    for idx, traffic_element in enumerate(annotation.get('traffic_element', [])):
        annotation['traffic_element'][idx]['points'] = np.asarray(traffic_element['points'], dtype=np.float32)

    annotation['topology_lsls'] = np.asarray(annotation['topology_lsls'], dtype=np.int8)
    annotation['topology_lste'] = np.asarray(annotation['topology_lste'], dtype=np.int8)

    return identifier, frame


def collect(root_path: str, data_dict: dict, collection: str, n_points: dict, num_workers: int = None, chunk_size: int = 8) -> None:
    samples = list(_iter_samples(data_dict))
    if not samples:
        return

    root_path = Path(root_path)
    meta = {}

    worker_count = num_workers or max(cpu_count() - 1, 1)
    worker_fn = partial(_process_single, root_path, n_points)

    with Pool(worker_count) as pool:
        for identifier, frame in tqdm(pool.imap_unordered(worker_fn, samples, chunksize=chunk_size), total=len(samples), desc=f'collecting {collection}', ncols=100):
            meta[identifier] = frame

    io.pickle_dump(f'{root_path}/{collection}_parallel.pkl', meta)

def _sample_random_bags(data_dict: dict, split: str, num_bags: int = 5, seed: int = 42) -> dict:
    """
    data_dict_Inhouse.json 구조에서 특정 split(train/val/test)의 bag들 중
    랜덤으로 num_bags개 log(bag)를 선택해, 그에 해당하는 부분만 남긴 새 data_dict를 만든다.

    예시:
    {
      "train": {
        "Sangam": { ... },
        "Yeouido": { ... }
      },
      "val": {},
      "test": {}
    }
    """
    rng = random.Random(seed)

    if split not in data_dict:
        raise KeyError(f"Split '{split}' not found in data_dict")

    split_logs = data_dict[split]  # ex) {"Sangam": {...}, "Yeouido": {...}}

    # bag 이름 수집: (log_name, seq_name) or (log_name,) 형태로 저장
    bag_keys = []
    for log_name, sequences in split_logs.items():
        if isinstance(sequences, dict):
            # 예: "Sangam": { "ioniq5_topics_...": [ ... ], ... }
            for seq_name in sequences.keys():
                bag_keys.append((log_name, seq_name))
        elif isinstance(sequences, list):
            # 예: "SomeLog": [ "xxx.json", ... ]
            bag_keys.append((log_name, ))
        else:
            raise TypeError(f"Unsupported data_dict structure under split '{split}'.")

    if len(bag_keys) == 0:
        raise ValueError(f"No bags found for split '{split}'")

    if num_bags > len(bag_keys):
        num_bags = len(bag_keys)

    sampled_bags = rng.sample(bag_keys, num_bags)

    # 새로운 data_dict 생성
    new_data_dict = {split: {}}
    for key in sampled_bags:
        if len(key) == 2:
            log_name, seq_name = key
            # 상위 log가 dict인 경우
            if log_name not in new_data_dict[split]:
                new_data_dict[split][log_name] = {}
            new_data_dict[split][log_name][seq_name] = split_logs[log_name][seq_name]
        else:
            (log_name,) = key
            new_data_dict[split][log_name] = split_logs[log_name]

    return new_data_dict


if __name__ == '__main__':
    root_path = '../../Inhouse'
    file = f'{root_path}/data_dict_Inhouse.json'
    subset = 'inhouse'
    data_dict = io.json_load(file)

    # 1) 기존처럼 train/val/test 전부에 대해 pkl 생성 (원래 있던 동작 유지)
    for split, segments in data_dict.items():
        if segments:  # 빈 dict면 건너뛰기
            collect(
                root_path,
                {split: segments},
                f'{subset}_{split}_ls',
                n_points={
                    'centerline': 10,
                    'left_laneline': 10,
                    'right_laneline': 10
                },
            )

    # 2) Inhouse train에서 랜덤 5개 bag만 뽑아서 validation pkl 생성
    #    -> 결과 파일 이름 예: inhouse_val_random5_ls_parallel.pkl
    try:
        val_data_dict = _sample_random_bags(data_dict, split='train', num_bags=3, seed=12)
    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to sample validation bags from train split: {e}")

    collect(
        root_path,
        val_data_dict,
        f'{subset}_val_ls',
        n_points={
            'centerline': 10,
            'left_laneline': 10,
            'right_laneline': 10
        },
    )
