import torch
import os
import pandas as pd
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter

def summary(config):
    return(SummaryWriter(config['output']['path']))


def save_config(config):
    os.makedirs(config['output']['path'], exist_ok=True)
    path = os.path.join(config['output']['path'], 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f)


def save_checkpoint(config, model, optimizer, scheduler, epoch, train_batch_losses, val_batch_losses):
    # save paths
    save_path = os.path.join(config['output']['path'], config['output']['name'])
    os.makedirs(save_path, exist_ok=True)
    # save model
    model.eval()
    s_dict = {
        'model_state_dict': model.state_dict(), #'model_state_dict': self.model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_loss_mean': Average(train_batch_losses),
        'train_batch_losses': train_batch_losses,
        'train_acc': np.mean(tm.accuracy),
        'val_loss_mean': Average(val_batch_losses),
        'val_batch_losses': val_batch_losses,
        'val_acc': np.mean(vm.accuracy)
        }
    torch.save(s_dict, save_path+'epoch_{epoch}.pth'.format(epoch=epoch))