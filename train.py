import warnings
import argparse
from utils.config import load
from utils.device import get_device
from model.gen_model import get_model
from loss.gen_loss import get_loss
from optimizer.gen_optimizer import get_optimizer
from scheduler.gen_scheduler  import get_scheduler
from dataloader.gen_dataloader import get_dataloader
from dataloader.gen_transform import get_transform
from metric.gen_metrics import get_metric
from data.gen_data import get_data
from utils.loop import ValidEpoch, TrainEpoch
from utils.save import save_config, summary
import numpy as np


def train(DEVICE, config, model, optimizer, scheduler, loss, dataloaders, writer, metric):
    # batch trainers
    train_epoch = TrainEpoch(model=model,
                            loss=loss,
                            optimizer=optimizer,
                            device=DEVICE)

    valid_epoch = ValidEpoch(model=model,
                            loss=loss,
                            device=DEVICE)

    for i in range(config['data']['nb_epochs']):
        print('\nEpoch: {}'.format(i))
        # train for epoch
        train_batch_losses, (true, logits) = train_epoch.run(dataloaders['train'])
        mean_train = np.mean(train_batch_losses)
        writer.add_scalar('Loss/train', mean_train, i)
        metric.forward(true, logits, i, 'train')
        # test for epoch
        val_batch_losses, (true, logits) = valid_epoch.run(dataloaders['val'])
        metric.forward(true, logits, i, 'test')
        mean_val = np.mean(val_batch_losses)
        writer.add_scalar('Loss/test', mean_val, i)
        # scheduler and learning rate stats
        if scheduler:
            scheduler.step()
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], i)
        # print values for epoch
        print(f"Loss/train: {mean_train}")
        print(f"Loss/test: {mean_val}")


def run(config):
    DEVICE = get_device()
    # get parts for loop
    model = get_model(config).to(DEVICE)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer, -1)
    loss = get_loss(config)
    # transforms
    transforms = get_transform(config)
    # get dataframe and ids
    df, train_ids, val_ids = get_data(config)
    # get data and transforms
    dataloaders = get_dataloader(config, df, train_ids, val_ids, transforms)
    # main training loop
    writer = summary(config)
    metric = get_metric(config, writer)
    train(DEVICE, config, model, optimizer, scheduler, loss, dataloaders, writer, metric)


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
