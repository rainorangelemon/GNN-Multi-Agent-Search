import numpy as np
import cv2
from pathlib import Path
import json
import argparse
import multiprocessing
import multiprocessing.pool as mpp
from tqdm import tqdm
from multiprocessing import Pool
from utils.BoxPlacer import Env


def create_meta(dataset_path, data_i):
    env = Env(
        {
            "n_sensors": 6,
            "box_dim_m": [0.485, 0.31, 0.31],
            "env_size_boxes": [8, 10],
            "env_resolution_m": 0.01,
            "min_dist_obstacles_border": 0.05,
            "min_dist_between_obstacles": 0.05,
            "obstacle_placement_probability": 0.8,
            "obstacle_specification": [
                # box factor and number of boxes to place
                ([3, 1, 1], 1),
                ([2, 1, 1], 2),
                ([1, 2, 1], 2),
                ([1, 1, 1], 3),
            ],
        }
    )

    ###############################################

    # env.compute_border_boxes()

    dataset_path = Path(dataset_path)
    (dataset_path / "meta").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(data_i)
    data_id = f"{data_i:06d}"

    rng = np.random.default_rng(data_i)
    while not env.generate(rng):
        pass

    meta = env.save_state()
    meta["random_seed"] = data_i

    with open(dataset_path / "meta" / f"BoxEnv_{data_id}.json", "w") as outfile:
        json.dump(meta, outfile)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Dataset Meta")
    parser.add_argument("dataset_path")
    parser.add_argument("n_samples", type=int)
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of parallel workers to build visgraph",
    )
    args = parser.parse_args()

    with Pool(args.n_workers) as pool:
        iterable = [(args.dataset_path, i) for i in range(args.n_samples)]
        for _ in tqdm(pool.istarmap(create_meta, iterable), total=args.n_samples):
            pass
