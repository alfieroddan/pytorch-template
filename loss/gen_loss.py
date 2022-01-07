from .loss_zoo import loss

def get_loss(config):
    loss_name = config['loss']['name']
    print('Loss Name: ', loss_name)
    return(loss(loss_name))