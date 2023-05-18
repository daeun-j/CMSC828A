import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset
import math

# level5_iid_partition with same valication
def iid_partition(
    ori_dataset: Dataset, num_clients: int, num_val: int
) -> Tuple[List[List[int]], Dict, int]:
    partition = {"separation": None, "data_indices": None}
    stats = {}
    data_indices = [[] for _ in range(num_clients)]
    targets_numpy = np.array(ori_dataset.targets, dtype=np.int64)
    idx = list(range(len(targets_numpy)))
    random.shuffle(idx)
    idx, idx_val = idx[num_val:], idx[:num_val]
    idx, idx_test = idx[num_val:], idx[:num_val]
    
    num_levels = 5
    sum_lebels = np.sum(list(range(num_levels+1)))
    # size = int(len(idx) / num_clients / sum_lebels * num_levels)
    size = int(200)
    front_idx = 0
    for i in range(num_clients):
        if i %5 == 0:
            id = i %5
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 1:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 2:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 3:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 4:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        # print(i,i %5 , front_idx, len(data_indices[i]))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats, idx_val, idx_test


# level5_iid_partition with same valication
def level5_iid_partition(
    ori_dataset: Dataset, num_clients: int, num_val: int
) -> Tuple[List[List[int]], Dict, int]:
    partition = {"separation": None, "data_indices": None}
    stats = {}
    data_indices = [[] for _ in range(num_clients)]
    targets_numpy = np.array(ori_dataset.targets, dtype=np.int64)
    idx = list(range(len(targets_numpy)))
    random.shuffle(idx)
    idx, idx_val = idx[num_val:], idx[:num_val]
    # idx, idx_test = idx[num_val:], idx[:num_val]
    
    num_levels = 5
    sum_lebels = np.sum(list(range(num_levels+1)))
    size = int(len(idx) / num_clients / sum_lebels * num_levels)
    # size = int(200)
    front_idx = 0
    for i in range(num_clients):
        if i %5 == 0:
            id = i %5
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)

        elif i %5 == 1:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 2:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 3:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        elif i %5 == 4:
            id = i %5
            
            data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
            front_idx += size * (id + 1)
            
        # print(i,i %5 , front_idx, len(data_indices[i]))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats, idx_val


# level5_iid_partition
# def level5_iid_partition(
#     ori_dataset: Dataset, num_clients: int
# ) -> Tuple[List[List[int]], Dict]:
#     partition = {"separation": None, "data_indices": None}
#     stats = {}
#     data_indices = [[] for _ in range(num_clients)]
#     targets_numpy = np.array(ori_dataset.targets, dtype=np.int64)
#     idx = list(range(len(targets_numpy)))
#     random.shuffle(idx)
#     num_levels = 5
#     sum_lebels = np.sum(list(range(num_levels+1)))
#     size = int(len(idx) / num_clients / sum_lebels * num_levels)
#     front_idx = 0
#     for i in range(num_clients):
#         if i %5 == 0:
#             id = i %5
#             data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
#             stats[i] = {"x": None, "y": None}
#             stats[i]["x"] = len(data_indices[i])
#             stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
#             front_idx += size * (id + 1)
            
#         elif i %5 == 1:
#             id = i %5
            
#             data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
#             stats[i] = {"x": None, "y": None}
#             stats[i]["x"] = len(data_indices[i])
#             stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
#             front_idx += size * (id + 1)
            
#         elif i %5 == 2:
#             id = i %5
            
#             data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
#             stats[i] = {"x": None, "y": None}
#             stats[i]["x"] = len(data_indices[i])
#             stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
#             front_idx += size * (id + 1)
            
#         elif i %5 == 3:
#             id = i %5
            
#             data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
#             stats[i] = {"x": None, "y": None}
#             stats[i]["x"] = len(data_indices[i])
#             stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
#             front_idx += size * (id + 1)
            
#         elif i %5 == 4:
#             id = i %5
            
#             data_indices[i] = idx[front_idx : front_idx + size * (id + 1)]
#             stats[i] = {"x": None, "y": None}
#             stats[i]["x"] = len(data_indices[i])
#             stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
#             front_idx += size * (id + 1)
            
#         # print(i,i %5 , front_idx, len(data_indices[i]))


#     num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
#     stats["sample per client"] = {
#         "std": num_samples.mean(),
#         "stddev": num_samples.std(),
#     }

#     partition["data_indices"] = data_indices

#     return partition, stats


def level3_iid_partition(
    ori_dataset: Dataset, num_clients: int
) -> Tuple[List[List[int]], Dict]:
    partition = {"separation": None, "data_indices": None}
    stats = {}
    data_indices = [[] for _ in range(num_clients)]
    targets_numpy = np.array(ori_dataset.targets, dtype=np.int64)
    idx = list(range(len(targets_numpy)))
    random.shuffle(idx)
    size = int(len(idx) / num_clients)

    for i in range(num_clients):
        # data_indices[i] = idx[size * i : size * (i + 1)]
        # stats[i] = {"x": None, "y": None}
        # stats[i]["x"] = len(data_indices[i])
        # stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
        if i %3 == 0:
            data_indices[i] = idx[size * i :  math.floor(size * (i + 0.1))]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
        if i %3 == 1:
            data_indices[i] = idx[size * i : math.floor(size * (i + 0.01))]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())
        if i %3 == 2:
            data_indices[i] = idx[size * i : size * (i + 1)]
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(data_indices[i])
            stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats

def iid_partition_old(
    ori_dataset: Dataset, num_clients: int
) -> Tuple[List[List[int]], Dict]:
    partition = {"separation": None, "data_indices": None}
    stats = {}
    data_indices = [[] for _ in range(num_clients)]
    targets_numpy = np.array(ori_dataset.targets, dtype=np.int64)
    idx = list(range(len(targets_numpy)))
    random.shuffle(idx)
    size = int(len(idx) / num_clients)

    for i in range(num_clients):
        data_indices[i] = idx[size * i : size * (i + 1)]
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(data_indices[i])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
