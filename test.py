import argparse
import json

import torch

from acorn.engine import evaluate, sensor_list
from acorn.model import ACORN, FireEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--test_sensors", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"]
    fold = args["fold"] if a.fold is None else a.fold
    test_sensors = args.get("test_sensors", "") if a.test_sensors is None else a.test_sensors
    model = ACORN(ck["groups"], ck["cuts"]).to(dev)
    model.load_state_dict(ck["state"])
    enc = FireEncoder(model.body.encoder.out_channels[-1] + 20).to(dev)
    enc.load_state_dict(ck["enc"])
    model.eval()
    norm = tuple(t.to(dev) for t in ck["norm"][:2])
    res = dict(checkpoint=a.checkpoint, fold=fold, seed=args.get("seed"), best_epoch=ck["epoch"])
    res.update(evaluate(model, enc, norm, a.data, a.context, fold, sensor_list(args.get("sensors", "")), sensor_list(test_sensors), dev))
    json.dump(res, open(a.out, "w"))
    print(f"fold {fold}: test {res['test']:.2f}, z=0 {res['test_z0']:.2f}, USGS {res['usgs_test']:.2f}")


if __name__ == "__main__":
    main()
