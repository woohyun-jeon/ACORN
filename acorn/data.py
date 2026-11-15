import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

K = 4
NF = 5
T = 256
MIN_VALID = 0.10
LO, W, NB = -1.0, 0.005, 600
HLS_LABELS = {
    "mtbs": np.array([-1, 0, 1, 2, 3, 0, -1] + [-1] * 249, np.int8),
    "ems": np.array([0, 1, 1, 2, 3] + [-1] * 251, np.int8),
}


def read_csv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nbr(a):
    return (a[3] - a[5]) / np.maximum(a[3] + a[5], 1e-6)


def crop_padded(a, r0, c0, fill):
    out = np.full(a.shape[:-2] + (T, T), fill, a.dtype)
    h, w = a.shape[-2:]
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + T, h), min(c0 + T, w)
    if re > rs and ce > cs:
        out[..., rs - r0:re - r0, cs - c0:ce - c0] = a[..., rs:re, cs:ce]
    return out


def crop_edge(a, row, col, fill=0):
    out = np.full(a.shape[:-2] + (T, T), fill, a.dtype)
    h, w = a.shape[-2:]
    row1, col1 = min(row + T, h), min(col + T, w)
    out[..., :row1 - row, :col1 - col] = a[..., row:row1, col:col1]
    return out


def starts(length):
    if length <= T:
        return [0]
    values = list(range(0, length - T + 1, T))
    if values[-1] != length - T:
        values.append(length - T)
    return values


def context_windows(valid):
    h, w = valid.shape
    selected = []
    for row in starts(h):
        for col in starts(w):
            fraction = float(crop_edge(valid, row, col, False).mean())
            if fraction >= MIN_VALID:
                selected.append((row, col, fraction))
    if not selected and valid.any():
        rr, cc = np.nonzero(valid)
        row = int(np.clip(round(float(rr.mean())) - T // 2, 0, max(h - T, 0)))
        col = int(np.clip(round(float(cc.mean())) - T // 2, 0, max(w - T, 0)))
        selected.append((row, col, float(crop_edge(valid, row, col, False).mean())))
    return selected


def hls_scene_dir(root, unit, src, sensor):
    return os.path.join(root, "scenes", src, sensor, unit.split(":", 1)[1])


def hls_fire_of(meta):
    ident = meta["unit"].split(":", 1)[1]
    return ident.split("_", 1)[0] if meta["src"] == "ems" else ident


def hls_patch_scene(d, src):
    with rasterio.open(os.path.join(d, "pre.tif")) as s:
        pre = s.read()
    with rasterio.open(os.path.join(d, "post.tif")) as s:
        post = s.read()
    with rasterio.open(os.path.join(d, "label.tif")) as s:
        lab = s.read(1)
    y = HLS_LABELS[src][lab]
    y[(pre[6] != 0) | (post[6] != 0)] = -1
    p, q = pre[:6].astype(np.float32) / 1e4, post[:6].astype(np.float32) / 1e4
    ok = (pre[6] == 0) & (post[6] == 0)
    dn = np.where(ok, nbr(p) - nbr(q), 0.0)
    return np.concatenate([p, q, dn[None]]).astype(np.float16), y


def hls_context_scene(d):
    with rasterio.open(os.path.join(d, "pre.tif")) as s:
        pre = s.read()
    with rasterio.open(os.path.join(d, "post.tif")) as s:
        post = s.read()
    valid = (pre[6] == 0) & (post[6] == 0)
    p, q = pre[:6].astype(np.float32) / 1e4, post[:6].astype(np.float32) / 1e4
    dn = np.where(valid, nbr(p) - nbr(q), 0.0)
    x = np.concatenate([p, q, dn[None]]).astype(np.float16)
    x[:, ~valid] = 0
    return x


def build_hls_patches(root, out):
    os.makedirs(out, exist_ok=True)
    tiles = read_csv(os.path.join(root, "tiles.csv"))
    xm = np.lib.format.open_memmap(os.path.join(out, "x.npy"), mode="w+", dtype=np.float16, shape=(len(tiles), 13, T, T))
    ym = np.lib.format.open_memmap(os.path.join(out, "y.npy"), mode="w+", dtype=np.int8, shape=(len(tiles), T, T))
    last = None
    for i, t in enumerate(tiles):
        d = hls_scene_dir(root, t["unit"], t["src"], t["sensor"])
        if d != last:
            x, y = hls_patch_scene(d, t["src"])
            last = d
        r0, c0 = int(t["row"]), int(t["col"])
        xm[i] = crop_padded(x, r0, c0, 0)
        ym[i] = crop_padded(y, r0, c0, -1)
    xm.flush()
    ym.flush()
    write_csv(os.path.join(out, "index.csv"), tiles)
    return len(tiles)


def build_hls_context(root, out):
    os.makedirs(out, exist_ok=True)
    meta_paths = sorted(glob.glob(os.path.join(root, "scenes", "*", "*", "*", "meta.json")))
    metas = [json.load(open(path)) for path in meta_paths]
    units = {r["unit"]: r for r in read_csv(os.path.join(root, "units.csv"))}
    plan = []
    for path, meta in zip(meta_paths, metas):
        d = os.path.dirname(path)
        with rasterio.open(os.path.join(d, "pre.tif")) as pre, rasterio.open(os.path.join(d, "post.tif")) as post:
            valid = (pre.read(7) == 0) & (post.read(7) == 0)
        for row, col, fraction in context_windows(valid):
            plan.append((meta, d, row, col, fraction))
    xm = np.lib.format.open_memmap(os.path.join(out, "x.npy"), mode="w+", dtype=np.float16, shape=(len(plan), 13, T, T))
    rows = []
    last = None
    for i, (meta, d, row, col, fraction) in enumerate(plan):
        if d != last:
            image = hls_context_scene(d)
            last = d
        xm[i] = crop_edge(image, row, col, 0)
        unit = meta["unit"]
        rows.append(dict(i=i, fire_id=f"{meta['src']}:{hls_fire_of(meta)}", unit=unit, src=meta["src"], sensor=meta["sensor"],
                         row=row, col=col, valid_frac=round(fraction, 6), fold=int(units[unit]["fold"]), group=units[unit]["group"]))
    xm.flush()
    write_csv(os.path.join(out, "index.csv"), rows)
    return len(rows)


def baer_patch(args):
    root, t = args
    d = os.path.join(root, "scenes", t["fire_id"])
    w = Window(int(t["col"]), int(t["row"]), T, T)
    with rasterio.open(os.path.join(d, "pre.tif")) as s:
        pre = s.read(window=w, boundless=True, fill_value=0).astype(np.float32) / 1e4
    with rasterio.open(os.path.join(d, "post.tif")) as s:
        post = s.read(window=w, boundless=True, fill_value=0).astype(np.float32) / 1e4
    with rasterio.open(os.path.join(d, "label.tif")) as s:
        lab = s.read(1, window=w, boundless=True, fill_value=255)
    ok = (pre[3] > 0) & (pre[5] > 0) & (post[3] > 0) & (post[5] > 0)
    dn = np.where(ok, nbr(pre) - nbr(post), 0.0)
    y = np.where(ok & (lab < 4), lab, -1).astype(np.int8)
    return np.concatenate([pre, post, dn[None]]).astype(np.float16), y


def build_baer_patches(root, out, workers=8):
    os.makedirs(out, exist_ok=True)
    fires = {f["fire_id"]: f for f in read_csv(os.path.join(root, "fires.csv"))}
    tiles = read_csv(os.path.join(root, "tiles.csv"))
    xm = np.lib.format.open_memmap(os.path.join(out, "x.npy"), mode="w+", dtype=np.float16, shape=(len(tiles), 13, T, T))
    ym = np.lib.format.open_memmap(os.path.join(out, "y.npy"), mode="w+", dtype=np.int8, shape=(len(tiles), T, T))
    with ProcessPoolExecutor(workers) as ex:
        for i, (x, y) in enumerate(ex.map(baer_patch, [(root, t) for t in tiles], chunksize=8)):
            xm[i], ym[i] = x, y
    xm.flush()
    ym.flush()
    rows = [dict(i=i, fire_id=t["fire_id"], row=t["row"], col=t["col"], fold=t["fold"], res_m=fires[t["fire_id"]]["res_m"], centred=t["centred"])
            for i, t in enumerate(tiles)]
    write_csv(os.path.join(out, "index.csv"), rows)
    return len(tiles)


def baer_context_tile(pre, post, row, col):
    w = Window(col, row, T, T)
    p = pre.read(window=w, boundless=True, fill_value=0).astype(np.float32) / 1e4
    q = post.read(window=w, boundless=True, fill_value=0).astype(np.float32) / 1e4
    ok = (p[3] > 0) & (p[5] > 0) & (q[3] > 0) & (q[5] > 0)
    dn = np.where(ok, nbr(p) - nbr(q), 0.0)
    x = np.concatenate([p, q, dn[None]]).astype(np.float16)
    x[:, ~ok] = 0
    return x


def build_baer_context(root, out):
    os.makedirs(out, exist_ok=True)
    fires = {f["fire_id"]: f for f in read_csv(os.path.join(root, "fires.csv"))}
    benchmark = sorted({t["fire_id"] for t in read_csv(os.path.join(root, "tiles.csv"))})
    plan = []
    for fire in benchmark:
        d = os.path.join(root, "scenes", fire)
        with rasterio.open(os.path.join(d, "pre.tif")) as pre, rasterio.open(os.path.join(d, "post.tif")) as post:
            valid = (pre.read(4) > 0) & (pre.read(6) > 0) & (post.read(4) > 0) & (post.read(6) > 0)
        plan.extend((fire, row, col, fraction) for row, col, fraction in context_windows(valid))
    xm = np.lib.format.open_memmap(os.path.join(out, "x.npy"), mode="w+", dtype=np.float16, shape=(len(plan), 13, T, T))
    rows = []
    current = pre = post = None
    for i, (fire, row, col, fraction) in enumerate(plan):
        if fire != current:
            if pre is not None:
                pre.close()
                post.close()
            d = os.path.join(root, "scenes", fire)
            pre, post = rasterio.open(os.path.join(d, "pre.tif")), rasterio.open(os.path.join(d, "post.tif"))
            current = fire
        xm[i] = baer_context_tile(pre, post, row, col)
        f = fires[fire]
        rows.append(dict(i=i, fire_id=fire, src="baer", row=row, col=col, valid_frac=round(fraction, 6), fold=f["fold"], group=f["group"], res_m=f["res_m"]))
    if pre is not None:
        pre.close()
        post.close()
    xm.flush()
    write_csv(os.path.join(out, "index.csv"), rows)
    return len(rows)


def load_cache(path):
    x = np.load(os.path.join(path, "x.npy"), mmap_mode="r")
    y = np.load(os.path.join(path, "y.npy"), mmap_mode="r") if os.path.exists(os.path.join(path, "y.npy")) else None
    return x, y, read_csv(os.path.join(path, "index.csv"))


def split_indices(idx, fold, sensors=None, test_sensors=None):
    role = {fold: "test", (fold + 1) % NF: "valid"}
    ok = lambda r, keep: keep is None or r.get("sensor") in keep
    return {s: np.array([int(r["i"]) for r in idx if role.get(int(r["fold"]), "train") == s and ok(r, test_sensors if s == "test" else sensors)], dtype=np.int64)
            for s in ("train", "valid", "test")}


def score_key(r):
    return r["fire_id"] + (f"|{r['sensor']}" if r.get("sensor") else "")


def class_hist(x, y, idx, step=2):
    h = np.zeros((K, NB), np.int64)
    for i in range(0, len(idx), 256):
        j = idx[i:i + 256]
        d = np.asarray(x[j, 12, ::step, ::step], np.float32).ravel()
        lab = np.asarray(y[j, ::step, ::step]).ravel().astype(np.int64)
        m = lab >= 0
        b = np.clip(((d[m] - LO) / W).astype(np.int64), 0, NB - 1)
        h += np.bincount(lab[m] * NB + b, minlength=K * NB).reshape(K, NB)
    return h


def fit_rule(h, lo=-0.2, hi=1.3, step=2):
    c = np.concatenate([np.zeros((K, 1)), np.cumsum(h, 1)], 1)
    tot = c[:, -1]
    g = np.arange(int((lo - LO) / W), int((hi - LO) / W), step)
    best, arg = -1, None
    for a in range(len(g) - 2):
        i = g[a]
        jj, kk = np.meshgrid(g[a + 1:], g[a + 1:], indexing="ij")
        m = jj < kk
        jj, kk = jj[m], kk[m]
        p = np.stack([np.broadcast_to(c[:, i:i + 1], (K, len(jj))), c[:, jj] - c[:, i:i + 1], c[:, kk] - c[:, jj], tot[:, None] - c[:, kk]], 1)
        tp = np.stack([p[k, k] for k in range(K)])
        s = (tp / np.maximum(p.sum(1) + p.sum(0) - tp, 1)).mean(0)
        q = int(np.argmax(s))
        if s[q] > best:
            best, arg = s[q], (int(i), int(jj[q]), int(kk[q]))
    return [round(LO + W * x, 4) for x in arg]


def augment(x, y):
    for i in range(x.shape[0]):
        k = int(torch.randint(4, (1,)))
        xi, yi = torch.rot90(x[i], k, (1, 2)), torch.rot90(y[i], k, (0, 1))
        if torch.rand(1) < 0.5:
            xi, yi = xi.flip(2), yi.flip(1)
        x[i], y[i] = xi, yi
    return x, y
