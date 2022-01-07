import torch.optim as optim


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                    amsgrad=amsgrad)


def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
  return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                   nesterov=nesterov)


def get_optimizer(config):
  f = globals().get(config.optimiser.name)
  return f(parameters, **config.optimizer.params)
