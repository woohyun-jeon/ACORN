import numpy as np
import torch

from .data import load_cache, score_key, split_indices
from .metrics import fire_cms, pooled
from .model import expand, fire_latents, predict

USGS = [0.10, 0.27, 0.66]


def sensor_list(text):
    return text.split(",") if text else None


def to_device(array, rows, dev):
    return torch.from_numpy(np.ascontiguousarray(array[rows])).to(dev)


def evaluate(model, enc, norm, data, context, fold, sensors, test_sensors, dev):
    x, y, idx = load_cache(data)
    split = split_indices(idx, fold, sensors, test_sensors)
    keys = [score_key(idx[i]) for i in split["test"]]
    xte = to_device(x, split["test"], dev)
    yte = torch.from_numpy(np.ascontiguousarray(y[split["test"]]))
    cx, _, cidx = load_cache(context)
    csplit = split_indices(cidx, fold, sensors, test_sensors)
    bags = [score_key(cidx[i]) for i in csplit["test"]]
    cxte = to_device(cx, csplit["test"], dev)
    names, zh = fire_latents(model, enc, cxte, bags, norm)
    z = expand(names, zh, keys)
    del cxte
    res = dict(test_fire_cm=fire_cms(predict(model, xte, norm, z), yte, keys))
    res["test"] = pooled(res["test_fire_cm"])
    res["z0_fire_cm"] = fire_cms(predict(model, xte, norm, torch.zeros_like(z)), yte, keys)
    res["test_z0"] = pooled(res["z0_fire_cm"])
    rule = torch.from_numpy(np.digitize(np.asarray(x[split["test"], 12], np.float32), USGS).astype(np.int8))
    res["usgs_fire_cm"] = fire_cms(rule, yte, keys)
    res["usgs_test"] = pooled(res["usgs_fire_cm"])
    res["zhat_test"] = {f: z[keys.index(f)].tolist() for f in sorted(set(keys))}
    return res
