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
from utils.save import save_checkpoint, save_df, save_out_target, save_config
import pandas as pd

def train(DEVICE, config, model, loss, dataloaders):
    # batch trainers
    train_epoch = TrainEpoch(model=model,
                            loss=loss,
                            device=DEVICE)

    valid_epoch = ValidEpoch(model=model,
                            loss=loss,
                            device=DEVICE)

    for i in range(config['data']['nb_epochs']):
        print('\nEpoch: {}'.format(i))
        train_batch_losses = train_epoch.run(dataloaders['train'])
        val_batch_losses = valid_epoch.run(dataloaders['val'])
        print('Train Losses: \n')
        print(train_batch_losses)
        print('Val Losses: \n')
        print(val_batch_losses)


def run(config):
    DEVICE = get_device()
    # get elements
    model = get_model(config).to(DEVICE)
    loss = get_loss(config)
    df, train_ids, val_ids = get_data(config)
    dataloaders = get_dataloader(config, df, train_ids, val_ids)
    # main training loop
    train(DEVICE, config, model, loss, dataloaders)


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
