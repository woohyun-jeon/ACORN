import math

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 4
TAU = (0.10, 0.30, 0.30)
QUANTILES = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])


def inv_softplus(x):
    return x + math.log(-math.expm1(-x))


def first_conv(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise ValueError("encoder has no Conv2d")


def widen_first_conv(conv, w3, rgb_slots):
    in_ch = conv.in_channels
    w = w3.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
    for c, j in rgb_slots.items():
        w[:, c] = w3[:, j]
    with torch.no_grad():
        conv.weight.copy_(w * 3 / in_ch)


class ACORN(nn.Module):
    def __init__(self, n_fires, cuts, encoder="resnet34"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.in_ch = 13
        self.body = smp.Unet(encoder, encoder_weights="imagenet", in_channels=self.in_ch, classes=2)
        ref = smp.Unet(encoder, encoder_weights="imagenet", in_channels=3, classes=1).encoder
        widen_first_conv(first_conv(self.body.encoder), first_conv(ref).weight.detach(), {0: 2, 1: 1, 2: 0, 6: 2, 7: 1, 8: 0})
        del ref
        head = self.body.segmentation_head[0]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.t1 = nn.Parameter(torch.tensor(float(cuts[0])))
        self.gap = nn.Parameter(torch.tensor([inv_softplus(cuts[1] - cuts[0]), inv_softplus(cuts[2] - cuts[1])]))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.05)))
        self.z = nn.Embedding(n_fires, 3)
        nn.init.zeros_(self.z.weight)

    def boundaries(self, z):
        g = F.softplus(self.gap)
        c0 = self.t1 + z[:, 0]
        c1 = c0 + g[0] * torch.exp(z[:, 1])
        return torch.stack([c0, c1, c1 + g[1] * torch.exp(z[:, 2])], 1)

    def forward(self, xn, dnbr, z):
        o = self.body(xn[:, :self.in_ch]).float()
        s = dnbr + o[:, 0]
        sigma = torch.exp(self.log_sigma + o[:, 1])
        exceed = torch.sigmoid((s[:, None] - self.boundaries(z)[:, :, None, None]) / sigma[:, None])
        p = torch.cat([1 - exceed[:, :1], exceed[:, :-1] - exceed[:, 1:], exceed[:, -1:]], 1).clamp_min(1e-6)
        return torch.log(p / p.sum(1, keepdim=True))


class FireEncoder(nn.Module):
    def __init__(self, feat_dim, hidden=128, p=0.2, out=3):
        super().__init__()
        self.tile = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(p))
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, emb, group, n_groups):
        h = self.tile(emb)
        a = self.score(h).squeeze(1)
        a = a - torch.zeros(n_groups, device=a.device).scatter_reduce(0, group, a, "amax", include_self=False)[group]
        w = torch.exp(a)
        den = torch.zeros(n_groups, device=a.device).index_add_(0, group, w)
        pooled = torch.zeros(n_groups, h.shape[1], device=a.device).index_add_(0, group, w[:, None] * h) / den[:, None].clamp_min(1e-12)
        return self.head(pooled)


def dnbr_of(x):
    return x[:, 12].clone()


def tile_stats(x):
    ok = (x[:, 3] > 0) & (x[:, 5] > 0) & (x[:, 9] > 0) & (x[:, 11] > 0)
    nd = lambda a, b: (a - b) / (a + b).clamp_min(1e-6)
    maps = [dnbr_of(x), nd(x[:, 3], x[:, 5]), nd(x[:, 3], x[:, 2]), nd(x[:, 9], x[:, 11])]
    out = []
    for i in range(x.shape[0]):
        m = ok[i]
        if m.sum() < 16:
            out.append(torch.zeros(20, device=x.device))
            continue
        v = torch.stack([mp[i][m][::4] for mp in maps])
        out.append(torch.quantile(v.float(), QUANTILES.to(x.device), dim=1).T.reshape(-1))
    return torch.stack(out)


def embed(model, x, norm):
    was = model.training
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        f = model.body.encoder(((x - norm[0]) / norm[1])[:, :model.in_ch])[-1].float().mean((2, 3))
    model.train(was)
    return torch.cat([f, tile_stats(x)], 1)


def embed_batched(model, x, norm, bs=32):
    return torch.cat([embed(model, x[i:i + bs].float(), norm) for i in range(0, len(x), bs)])


@torch.no_grad()
def fire_latents(model, enc, x, bags, norm, bs=32):
    names = sorted(set(bags))
    pos = {name: i for i, name in enumerate(names)}
    g = torch.tensor([pos[name] for name in bags], device=x.device)
    e = embed_batched(model, x, norm, bs)
    enc.eval()
    return names, enc(e, g, len(names))


def expand(names, z, bags):
    pos = {name: i for i, name in enumerate(names)}
    missing = sorted(set(bags) - set(pos))
    assert not missing, f"context missing {len(missing)} fire-sensor units, first {missing[:3]}"
    return z[torch.tensor([pos[name] for name in bags], device=z.device)]


@torch.no_grad()
def predict(model, x, norm, z, bs=32):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].float()
        xn = (xb - norm[0]) / norm[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = model(xn, dnbr_of(xb), z[i:i + bs])
        out.append(lp.float().argmax(1).to(torch.int8).cpu())
    return torch.cat(out)
