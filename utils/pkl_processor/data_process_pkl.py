import numpy as np
from tqdm import tqdm
from pathlib import Path

from shapely.geometry import LineString
from shapely.errors import GEOSException
from openlanev2.lanesegment.io import io

"""
This script is used to collect the data from the original OpenLane-V2 dataset.
The results will be saved in OpenLane-V2 folder.
The main difference between this script and the original one is that we don't interpolate the points for ped crossing and road bouadary.
"""

def _fix_pts_interpolate(curve, n_points):
    """Interpolate a curve (list of coordinates) to n_points."""
    # 빈 경우 처리
    if not curve:
        return np.zeros((n_points, 3), dtype=np.float32)
    
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
        except TypeError:
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


def collect(root_path: str, data_dict: dict, collection: str, n_points: dict) -> None:

    samples = list(_iter_samples(data_dict))
    meta = {}
    root = Path(root_path)
    for split, log_name, seq_name, timestamp in tqdm(samples, desc=f'collecting {collection}', ncols=100):
        segment_id = f"{log_name}/{seq_name}" if seq_name else log_name
        identifier = (split, segment_id, timestamp)

        data_split = split if split != 'val' else 'test'  # navsim aliases val to test assets
        json_dir = root / data_split / log_name
        if not json_dir.exists():
            raise FileNotFoundError(f"Could not locate data for split '{split}' at {json_dir}")
        if seq_name:
            json_dir = json_dir / seq_name
        if not json_dir.exists():
            raise FileNotFoundError(f"Sequence '{seq_name}' missing under {json_dir.parent}")
        frame_path = json_dir / "info" / f"{timestamp}-ls.json"
        frame = io.json_load(str(frame_path))

        # Load cached UTM coordinates so downstream loaders can access vehicle2global directly.
        traj_path = json_dir / "traj" / f"{timestamp}_egopose.npy"
        if traj_path.exists():
            frame['vehicle2global'] = np.asarray(np.load(traj_path), dtype=np.float64)
        else:
            frame['vehicle2global'] = None
        for k, v in frame['pose'].items():
            frame['pose'][k] = np.array(v, dtype=np.float64)
        for camera in frame['sensor'].keys():
            for para in ['intrinsic', 'extrinsic']:
                for k, v in frame['sensor'][camera][para].items():
                    frame['sensor'][camera][para][k] = np.array(v, dtype=np.float64)

        if 'annotation' not in frame:
            meta[identifier] = frame
            continue

        # NOTE: We don't interpolate the points for ped crossing and road bouadary.
        for i, area in enumerate(frame['annotation']['area']):
            frame['annotation']['area'][i]['points'] = np.array(area['points'], dtype=np.float32)
        for i, lane_segment in enumerate(frame['annotation']['lane_segment']):
            frame['annotation']['lane_segment'][i]['centerline'] = _fix_pts_interpolate(lane_segment['centerline'], n_points['centerline'])
            frame['annotation']['lane_segment'][i]['left_laneline'] = _fix_pts_interpolate(lane_segment['left_laneline'], n_points['left_laneline'])
            frame['annotation']['lane_segment'][i]['right_laneline'] = _fix_pts_interpolate(lane_segment['right_laneline'], n_points['right_laneline'])
        for i, traffic_element in enumerate(frame['annotation']['traffic_element']):
            frame['annotation']['traffic_element'][i]['points'] = np.array(traffic_element['points'], dtype=np.float32)
        frame['annotation']['topology_lsls'] = np.array(frame['annotation']['topology_lsls'], dtype=np.int8)
        frame['annotation']['topology_lste'] = np.array(frame['annotation']['topology_lste'], dtype=np.int8)
        meta[identifier] = frame

    io.pickle_dump(f'{root_path}/{collection}.pkl', meta)

if __name__ == '__main__':
    root_path = '../../NAVSIM_OpenLane'
    file = f'{root_path}/data_dict_navsim.json'
    subset = 'navsim_navtrain'
    for split, segments in io.json_load(file).items():
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
