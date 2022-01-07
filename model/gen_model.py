from .model_zoo import model

def get_model(config):
    model_name = config['model']['name']
    print('Model Name: ', model_name)
    return(model(config))

def get_test(name='FasterRCNNDetector'):
    print('Model Name: ', name)
    return(model(name))

if __name__ == '__main__':
    model = get_test()
    print(dir(model))

