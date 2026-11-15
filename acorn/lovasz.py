import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    if gt_sorted.numel() > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return jac


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits, targets):
        c = logits.shape[1]
        prob = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, c)
        lab = targets.reshape(-1)
        keep = lab != self.ignore_index
        prob, lab = prob[keep], lab[keep]
        losses = []
        for k in range(c):
            fg = (lab == k).float()
            if fg.sum() == 0:
                continue
            err = (fg - prob[:, k]).abs()
            err_sorted, perm = torch.sort(err, descending=True)
            losses.append(torch.dot(err_sorted, lovasz_grad(fg[perm])))
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0
