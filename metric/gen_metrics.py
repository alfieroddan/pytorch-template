from .metric_zoo import metric

def get_metric(config):
    m = config['metric']['name']
    print('Metric Name: ', m)
    return(metric(m, None))

def get_test(name='calculate_image_precision'):
    print('Model Name: ', name)
    return(metric(name))

if __name__ == '__main__':
    metric = get_test()
    print(dir(metric))

