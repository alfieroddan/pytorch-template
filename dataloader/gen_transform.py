import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random

class Trans(object):

    def __init__(self, train=False):
        self.train=train

    def __call__(self, inp):
        """
        :param img: (numpy): Image

        :return: z-norm, every channel of frame,
        tensorise
        and apply transforms if training.
        """
        inp = torch.tensor(inp, dtype=torch.float)
        # get shape
        # D, H, W = inp.shape
        # mean = torch.mean(inp)
        # std = torch.std(inp)
        max = torch.max(inp)
        min = torch.min(inp)
        # [0, 1] scaling
        inp = (inp - min) / (max - min)
        # z normilisation
        # inp = (inp- mean) / std
        return(inp)

    def __repr__(self):
        return self.__class__.__name__+'()'


def transform(name):
    f = globals().get(name)
    return f


def get_transform(config):
    # transforms
    if not config['transforms']['exist']:
        transforms = { 'train': None,
                    'val': None}
    # train
    train_fn = transform(config['transforms']['train']['name'])
    # test
    test_fn = transform(config['transforms']['test']['name'])
    # dictionary
    transforms = { 'train': train_fn(**config['transforms']['train']['params']),
                    'val': test_fn(**config['transforms']['test']['params'])}
    return(transforms)
