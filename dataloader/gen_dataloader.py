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

class Video(Dataset):
    def __init__(self, df, id_s, transform=None):
        self.df = df
        self.id_s = id_s
        self.transform = transform

    def __len__(self):
        return len(self.id_s)

    @staticmethod
    def load(path):
        return(np.load(path))

    def __getitem__(self, idx):
        # image_id
        image_id = self.id_s[idx]
        record = self.df[self.df['ID']==image_id]
        # get paths
        image_path = record.filepath_frame.values[0]
        # load
        image = self.load(image_path)
        if self.transform:
            transformed = self.transform(image)
            image = transformed
        # incase of greyscale, add channel size 1
        if len(image.shape) == 2:
            image = torch.unsqueeze(image, 0)

        # target
        target = torch.tensor(record.n_label.values[0], dtype=torch.long)
        return image, target

def get_dataloader(config, df, train_ids, val_ids):
    # transforms
    train_tfms = Trans(train=True)
    val_tfms = Trans(train=False)
    # datasets & loaders
    train_dataset = Video(df, train_ids, transform=train_tfms)
    train_dl = DataLoader(train_dataset, batch_size=config['data']['train_bs'], shuffle=True)
    val_dataset = Video(df, val_ids, transform=val_tfms)
    val_dl = DataLoader(val_dataset, batch_size=config['data']['val_bs'])
    dataloaders = { 'train': train_dl,
                    'val': val_dl}
    return(dataloaders)
