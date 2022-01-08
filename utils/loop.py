# taken from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/train.py
import sys
import torch
from tqdm import tqdm as tqdm


class Epoch:

    def __init__(self, model, loss, stage_name, device='cpu'):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.device = device
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        # self.metrics.to(self.device)

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()
        batch_losses = []
        for x, y in tqdm(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            loss, y_pred = self.batch_update(x, y)
            # update loss logs
            # loss_value = loss.cpu().detach().numpy()
            batch_losses.append(loss.item())
                
        return batch_losses


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

    def batch_update(self, x, y):
        if self.optimizer:
            self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        if self.optimizer:
            self.optimizer.step()
        return loss, prediction


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

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction