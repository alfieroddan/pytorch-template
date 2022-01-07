import warnings
import argparse
from utils.config import load
from utils.device import get_device
from model.gen_model import get_model
from metric.gen_metrics import get_metric
from loss.gen_loss import get_loss
from optimizer.gen_optimizer import get_optimizer
from scheduler.gen_scheduler  import get_scheduler
from dataloader.gen_dataloader import get_dataloader
from data.gen_data import get_data
from utils.loop import ValidEpoch, TrainEpoch
from utils.save import save_checkpoint, save_config
import pandas as pd
import numpy as np

def train(DEVICE, config, model, optimizer, scheduler, metrics, loss, dataloaders):
    # reset metrics
    if metrics:
        metrics['train'].reset()
        metrics['val'].reset()
    # batch trainers
    train_epoch = TrainEpoch(model=model,
                            loss=loss,
                            metrics=metrics['train'],
                            optimizer=optimizer,
                            device=DEVICE)

    valid_epoch = ValidEpoch(model=model,
                            loss=loss,
                            metrics=metrics['val'],
                            device=DEVICE)
    # temp logs
    df = pd.DataFrame(columns=['epoch', 'train_loss_mean', 'train_batch_losses', 
                            'val_loss_mean', 'val_batch_losses'])

    for i in range(config['data']['nb_epochs']):
        print('\nEpoch: {}'.format(i))
        train_batch_losses = train_epoch.run(dataloaders['train'])
        val_batch_losses = valid_epoch.run(dataloaders['val'])
        if scheduler:
            scheduler.step()
        print('train/acc: {}'.format(np.mean(metrics['train'].accuracy)))
        print('val/acc: {}'.format(np.mean(metrics['val'].accuracy)))
        save_checkpoint(config, model, optimizer, scheduler, i, 
        train_batch_losses, val_batch_losses, metrics)
        if metrics:
            metrics['train'].reset()
            metrics['val'].reset()


def run(config):
    DEVICE = get_device()
    # get elements
    model = get_model(config).to(DEVICE)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer, -1)
    metrics = {
        'train': get_metric(config),
        'val' : get_metric(config)
    }
    loss = get_loss(config)
    df, train_ids, val_ids = get_data(config)
    dataloaders = get_dataloader(config, df, train_ids, val_ids)
    # main training loop
    train(DEVICE, config, model, optimizer, scheduler, metrics, loss, dataloaders)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    config = load(args.config_file)
    save_config(config)
    print('Config: \n' + str(config))
    run(config)
