import argparse
import glob
import json
import os

import numpy as np

from acorn.metrics import class_iou, fwiou, miou

CLASSES = ["unburned", "low", "moderate", "high"]


def result_files(patterns):
    files = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            files.append(os.path.join(path, "result.json") if os.path.isdir(path) else path)
    return files


def selected(key, program, sensor):
    fire, _, unit_sensor = key.partition("|")
    if program and not fire.startswith(program + ":"):
        return False
    return not sensor or unit_sensor == sensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--program", default="", choices=["", "mtbs", "ems"])
    ap.add_argument("--sensor", default="", choices=["", "landsat", "s2"])
    ap.add_argument("--field", default="test_fire_cm", choices=["test_fire_cm", "z0_fire_cm", "usgs_fire_cm"])
    a = ap.parse_args()
    by_seed = {}
    for path in result_files(a.results):
        res = json.load(open(path))
        cm = by_seed.setdefault(res["seed"], np.zeros((4, 4), np.int64))
        for key, value in res[a.field].items():
            if selected(key, a.program, a.sensor):
                cm += np.array(value, np.int64)
    seeds = sorted(by_seed)
    m = np.array([miou(by_seed[s]) for s in seeds])
    f = np.array([fwiou(by_seed[s]) for s in seeds])
    sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0
    iou = 100 * class_iou(sum(by_seed.values()))
    print(f"seeds {seeds}")
    print(f"mIoU {m.mean():.2f} +- {sd(m):.2f}  per seed {np.round(m, 2).tolist()}")
    print(f"FWIoU {f.mean():.2f} +- {sd(f):.2f}")
    print("class IoU " + ", ".join(f"{c} {v:.1f}" for c, v in zip(CLASSES, iou)))


if __name__ == "__main__":
    main()
