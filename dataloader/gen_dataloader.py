import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random


class FMNIST(Dataset):
    def __init__(self, df, ids, transform=None):
        self.ids = ids
        self.transform = transform
        self.df = df
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # image_id
        image_id = self.ids[idx]
        record = self.df[self.df['ID']==image_id]
        # image        
        image = record.values[0][1:-2].reshape((1,28,28))
        # label
        label = record.label.values[0]
        # transforms
        if self.transform:
            image = self.transform(image)
        return image, label


def dataset(name):
    f = globals().get(name)
    return f


def get_dataloader(config, df, train_ids, val_ids, transforms):
    # datasets & loaders
    # train
    train_dataset_fn = dataset(config['dataset']['name'])
    train_datset = train_dataset_fn(df, train_ids, transform=transforms['train'])
    train_dl = DataLoader(train_datset, batch_size=config['data']['train_bs'], shuffle=True)
    # val
    val_dataset_fn = dataset(config['dataset']['name'])
    val_dataset = val_dataset_fn(df, val_ids, transform=transforms['val'])
    val_dl = DataLoader(val_dataset, batch_size=config['data']['val_bs'])
    # dataloader config
    dataloaders = { 'train': train_dl,
                    'val': val_dl}
    return(dataloaders)
