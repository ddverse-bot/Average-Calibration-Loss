import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class HardL1ACELoss(torch.nn.Module):
    def __init__(self, n_bins=20):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        bins = torch.linspace(0, 1, self.n_bins + 1, device=preds.device)
        ace = 0.0
        for i in range(self.n_bins):
            mask = (preds >= bins[i]) & (preds < bins[i+1])
            if mask.sum() == 0:
                continue
            e = preds[mask].mean()
            o = targets[mask].float().mean()
            ace += torch.abs(e - o)
        return ace / self.n_bins

class SoftL1ACELoss(torch.nn.Module):
    def __init__(self, n_bins=20):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        bins = torch.linspace(0, 1, self.n_bins + 1, device=preds.device)
        ace = 0.0
        for i in range(self.n_bins):
            mask = (preds >= bins[i]) & (preds < bins[i+1])
            if mask.sum() == 0:
                continue
            e = preds[mask].mean()
            o = targets[mask].float().mean()
            ace += ((e - o) ** 2) ** 0.5
        return ace / self.n_bins
