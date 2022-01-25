# taken from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/train.py
import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np


class Epoch:

    def __init__(self, model, loss, stage_name, device='cpu'):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.device = device
        self.pred_truth = None
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        # self.metrics.to(self.device)

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def update_pred_truth(self, prediction, y):
        # save truths and logits
        # output logits shape
        N, C = prediction.shape
        self.pred_truth['truth'].append(y)
        self.pred_truth['logits'].append(prediction)

    def run(self, dataloader):
        self.on_epoch_start()
        batch_losses = []
        loop = tqdm(dataloader)
        for idx, (x, y) in enumerate(loop):
            # run model
            x, y = x.to(self.device), y.to(self.device)
            loss = self.batch_update(x, y)
            batch_losses.append(loss.item())
            # update progress bar
            loop.set_postfix(loss=loss.item(), mu_loss=np.mean(batch_losses))
        return batch_losses, (self.pred_truth['truth'], self.pred_truth['logits'])


class TrainEpoch(Epoch):

    def __init__(self, model, loss, optimizer=None, device='cpu'):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='train',
            device=device
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()
        self.pred_truth = {
            'truth': [],
            'logits': []
        }

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        self.update_pred_truth(prediction, y)
        return loss


class ValidEpoch(Epoch):

    def __init__(self, model, loss, device='cpu'):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='valid',
            device=device
        )

    def on_epoch_start(self):
        self.model.eval()
        self.pred_truth = {
            'truth': [],
            'logits': []
        }

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
            self.update_pred_truth(prediction, y)
        return loss
