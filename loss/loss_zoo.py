import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

def CrossEntropy():
    return(nn.CrossEntropyLoss())

def BCELoss():
    return(nn.BCELoss())

def BCELogitLoss():
    return(nn.BCEWithLogitsLoss())
    
def loss(name):
    f = globals().get(name)
    return f()


if __name__ == '__main__':
    print(globals())
