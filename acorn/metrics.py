import numpy as np
import torch

K = 4


def class_iou(cm):
    cm = np.asarray(cm, np.float64)
    tp = np.diag(cm)
    return tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)


def miou(cm):
    return 100 * float(np.mean(class_iou(cm)))


def fwiou(cm):
    cm = np.asarray(cm, np.float64)
    true = cm.sum(1)
    return 100 * float((true * class_iou(cm)).sum() / max(true.sum(), 1))


def fire_cms(pred, y, keys):
    y = y.long()
    out = {}
    for f in sorted(set(keys)):
        idx = torch.tensor([i for i, g in enumerate(keys) if g == f])
        yy = y[idx]
        m = yy >= 0
        out[f] = torch.bincount(yy[m] * K + pred[idx].long()[m], minlength=K * K).reshape(K, K).numpy().tolist()
    return out


def pooled(cms):
    return miou(np.sum([np.array(c) for c in cms.values()], 0))
