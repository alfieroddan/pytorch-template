import torch.optim as optim

def adamW(paramaters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
          amsgrad=False, **_):
    return(optim.AdamW(paramaters, lr=lr, betas=betas, weight_decay=weight_decay,
                     amsgrad=amsgrad))

def adam(paramaters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
          amsgrad=False, **_):
    return(optim.Adam(paramaters, lr=lr, betas=betas, weight_decay=weight_decay,
                     amsgrad=amsgrad))

def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return(optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                     nesterov=nesterov))


def get_optimizer(config, parameters):
    o = globals().get(config['optimizer']['name'])
    print('Optimizer Name: ', config['optimizer']['name'])
    return o(parameters, **config['optimizer']['params'])
