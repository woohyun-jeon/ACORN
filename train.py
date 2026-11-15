import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from acorn.data import augment, class_hist, fit_rule, load_cache, score_key, split_indices
from acorn.engine import evaluate, sensor_list, to_device
from acorn.lovasz import LovaszSoftmaxLoss
from acorn.metrics import fire_cms, pooled
from acorn.model import ACORN, TAU, FireEncoder, dnbr_of, embed_batched, expand, fire_latents, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--enc_fires", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--sensors", default="")
    ap.add_argument("--test_sensors", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    sensors, test_sensors = sensor_list(a.sensors), sensor_list(a.test_sensors)

    x, y, idx = load_cache(a.data)
    split = split_indices(idx, a.fold, sensors, test_sensors)
    fires = {s: [idx[i]["fire_id"] for i in v] for s, v in split.items()}
    keys = {s: [score_key(idx[i]) for i in v] for s, v in split.items()}
    cuts = fit_rule(class_hist(x, y, split["train"]))
    xtr, ytr = to_device(x, split["train"], dev), to_device(y, split["train"], dev)
    xva, yva = to_device(x, split["valid"], dev), torch.from_numpy(np.ascontiguousarray(y[split["valid"]]))
    fire_ids = sorted(set(fires["train"]))
    gidx = torch.tensor([fire_ids.index(f) for f in fires["train"]], device=dev)

    cx, _, cidx = load_cache(a.context)
    csplit = split_indices(cidx, a.fold, sensors, test_sensors)
    crows = {s: [cidx[i] for i in csplit[s]] for s in ("train", "valid", "test")}
    cbags = {s: [score_key(r) for r in crows[s]] for s in ("train", "valid", "test")}
    for s in ("valid", "test"):
        missing = sorted(set(keys[s]) - set(cbags[s]))
        assert not missing, f"{s} context missing {len(missing)} fire-sensor units, first {missing[:3]}"
    bag_tiles = {}
    for j, bag in enumerate(cbags["train"]):
        bag_tiles.setdefault(bag, []).append(j)
    bag_tiles = {bag: torch.tensor(v, device=dev) for bag, v in bag_tiles.items()}
    bags_by_fire, source_by_fire = {}, {}
    for row, bag in zip(crows["train"], cbags["train"]):
        fire = row["fire_id"]
        source_by_fire[fire] = row.get("src", fire.split(":", 1)[0])
        bags_by_fire.setdefault(fire, set()).add(bag)
    bags_by_fire = {fire: sorted(v) for fire, v in bags_by_fire.items()}
    sampling_fires = sorted(bags_by_fire)
    source_counts = {}
    for fire in sampling_fires:
        source_counts[source_by_fire[fire]] = source_counts.get(source_by_fire[fire], 0) + 1
    sampling_weights = torch.tensor([source_counts[source_by_fire[fire]] ** (a.alpha - 1.0) for fire in sampling_fires], device=dev)
    assert set(bags_by_fire) == set(fire_ids), (len(bags_by_fire), len(fire_ids))
    fire_to_group = {fire: i for i, fire in enumerate(fire_ids)}
    cxtr = to_device(cx, csplit["train"], dev)
    cxva = to_device(cx, csplit["valid"], dev)

    flat = xtr[::4].float().permute(1, 0, 2, 3).reshape(13, -1)
    norm = (flat.mean(1)[None, :, None, None], flat.std(1)[None, :, None, None] + 1e-6)
    del flat

    model = ACORN(len(fire_ids), cuts).to(dev)
    enc = FireEncoder(model.body.encoder.out_channels[-1] + 20).to(dev)
    print(f"fold {a.fold}: tiles {({s: len(v) for s, v in split.items()})}, fires {({s: len(set(v)) for s, v in fires.items()})}, cuts {cuts}", flush=True)
    groups = [dict(params=[p for n, p in model.named_parameters() if n.startswith("body.")], lr=a.lr, weight_decay=1e-4),
              dict(params=[model.t1, model.gap, model.log_sigma], lr=1e-3, weight_decay=0.0),
              dict(params=list(model.z.parameters()), lr=5e-3, weight_decay=0.0),
              dict(params=enc.parameters(), lr=1e-3, weight_decay=1e-2)]
    opt = torch.optim.AdamW(groups)
    steps = a.epochs * (len(xtr) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i / max(steps, 1), 1.0))))
    lovasz = LovaszSoftmaxLoss()
    tau = torch.tensor(TAU, device=dev)
    best, history = -1.0, []

    for ep in range(a.epochs):
        model.train()
        enc.train()
        perm = torch.randperm(len(xtr), device=dev)
        total = total_enc = 0.0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            b = perm[i:i + a.bs]
            xb, yb = augment(xtr[b].float(), ytr[b].long())
            yb = torch.where(yb < 0, torch.full_like(yb, -100), yb)
            xn = (xb - norm[0]) / norm[1]
            z = model.z(gidx[b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = model(xn, dnbr_of(xb), z)
            loss = F.nll_loss(lp, yb, ignore_index=-100) + lovasz(lp, yb) + a.beta * 0.5 * ((model.z.weight / tau) ** 2).sum(1).mean()
            chosen = torch.multinomial(sampling_weights, a.enc_fires, replacement=False).cpu().tolist()
            chosen_fires = [sampling_fires[j] for j in chosen]
            parts = []
            for fire in chosen_fires:
                pool = bags_by_fire[fire]
                parts.append(bag_tiles[pool[int(torch.randint(len(pool), (1,), device=dev))]])
            sel = torch.cat(parts)
            grp = torch.cat([torch.full((len(part),), j, device=dev, dtype=torch.long) for j, part in enumerate(parts)])
            fs = torch.tensor([fire_to_group[fire] for fire in chosen_fires], device=dev)
            zh = enc(embed_batched(model, cxtr[sel], norm), grp, len(chosen_fires))
            le = 0.5 * (((zh - model.z(fs).detach()) / tau) ** 2).sum(1).mean()
            loss = loss + a.gamma * le
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            total_enc += le.item()
        model.eval()
        names, zh = fire_latents(model, enc, cxva, cbags["valid"], norm)
        v = pooled(fire_cms(predict(model, xva, norm, expand(names, zh, keys["valid"])), yva, keys["valid"]))
        n = max(len(perm) // a.bs, 1)
        rec = dict(epoch=ep, loss=round(total / n, 4), enc_loss=round(total_enc / n, 4), valid=round(v, 3), t=round(time.time() - t0),
                   boundaries=[round(c, 4) for c in model.boundaries(torch.zeros(1, 3, device=dev))[0].tolist()])
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if v > best:
            best = v
            torch.save(dict(state=model.state_dict(), enc=enc.state_dict(), norm=[t.cpu() for t in norm], cuts=cuts, args=vars(a), epoch=ep, groups=len(fire_ids)),
                       os.path.join(a.out, "best.pt"))

    del xtr, ytr, xva, cxtr, cxva
    torch.cuda.empty_cache()
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["state"])
    enc.load_state_dict(ck["enc"])
    model.eval()
    res = dict(fold=a.fold, seed=a.seed, args=vars(a), best_epoch=ck["epoch"], valid=best, train_cuts=cuts, history=history)
    res.update(evaluate(model, enc, norm, a.data, a.context, a.fold, sensors, test_sensors, dev))
    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"))
    print(f"fold {a.fold} seed {a.seed}: best epoch {ck['epoch']}, valid {best:.2f}, test {res['test']:.2f}, z=0 {res['test_z0']:.2f}, USGS {res['usgs_test']:.2f}", flush=True)


if __name__ == "__main__":
    main()
