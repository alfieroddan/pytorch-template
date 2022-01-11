import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional import accuracy, precision, recall, f1
import json


class MultiClassificationMeter(nn.Module):
    """
    Computes metrics for multiclasssification problem.
    """
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.softmax = nn.Softmax(dim=1)
        self.metrics = {}
    
    def write_out(self, epoch, stage):
        stage = '/'+stage
        for k, v in self.metrics.items():
            self.writer.add_scalar(k+stage, v, epoch)

    @staticmethod
    def accuracy_fn(true, probs):
        # probs will be shape N, Cout
        # true will be shape N,
        return(accuracy(probs, true))

    @staticmethod
    def precision_fn(true, probs):
        # probs will be shape N, Cout
        # true will be shape N,
        return(precision(probs, true))

    @staticmethod
    def recall_fn(true, probs):
        # probs will be shape N, Cout
        # true will be shape N,
        return(recall(probs, true))
    
    @staticmethod
    def f1_fn(true, probs):
        # probs will be shape N, Cout
        # true will be shape N,
        return(f1(probs, true))

    def forward(self, true, logits, epoch, stage):
        # concat batches to epoch
        # shape will be N(bs), C(out) of model
        true = torch.cat(true, 0)
        logits = torch.cat(logits, 0)
        # take in logits
        # softmax along logit dimension
        probs = self.softmax(logits)
        # accuracy
        self.metrics['Accuracy'] = self.accuracy_fn(true, probs)
        self.metrics['Precision'] = self.precision_fn(true, probs)
        self.metrics['Recall'] = self.recall_fn(true, probs)
        self.metrics['F1'] = self.f1_fn(true, probs)
        # print for end of epoch
        for k,v in self.metrics.items():
            print(f"{k}: {v}")
        # write it all
        self.write_out(epoch, stage)


def metric(name, params):
    f = globals().get(name)
    return f(params)

if __name__ == '__main__':
    print(globals())
