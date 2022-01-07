import torch.nn as nn
import torch
from sklearn import metrics

class Accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predb, yb):
        return ((predb-yb)==0).float().mean()

class MultiClassificationMeter(nn.Module):
    """
    Computes and stores the average and current value
    https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43
    expects logits
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.reset()

    def reset(self):
        self.predicted = []
        self.ground_truth = []
        self.accuracy = []

    def forward(self, output, target):
        self.predicted.append(output.detach().cpu().numpy())
        self.ground_truth.append(target.detach().cpu().numpy())
        # take in logits
        pred = self.softmax(output)
        truth = target
        # accuracy
        self.accuracy.append(metrics.accuracy_score(truth.detach().cpu().numpy(), torch.argmax(pred, dim=1).detach().cpu().numpy()))
        



class BinaryClassificationMeter(nn.Module):
    """
    Computes and stores the average and current value
    https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43
    expects logits
    """
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0
        self.predicted = []
        self.ground_truth = []

    def forward(self, output, target):
        self.predicted.append(output.detach().cpu().numpy())
        self.ground_truth.append(target.detach().cpu().numpy())
        # take in logits
        pred =  (torch.sigmoid(output) >= 0.5).float()
        truth = (target >= 0.5).float()
        self.tp += pred.mul(truth).sum(0).float()
        self.tn += (1 - pred).mul(1 - truth).sum(0).float()
        self.fp += pred.mul(1 - truth).sum(0).float()
        self.fn += (1 - pred).mul(truth).sum(0).float()
        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
        # self.avg_pre = torch.nanmean(self.pre)
        # self.avg_rec = torch.nanmean(self.rec)
        # self.avg_f1 = torch.nanmean(self.f1)

def metric(name, params):
    f = globals().get(name)
    return f()

if __name__ == '__main__':
    print(globals())