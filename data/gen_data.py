import pandas as pd

def get_data(config):
    df = pd.read_csv(config['data']['path'])
    # ids
    val_fold = config['data']['val_fold']
    train_ids = df[df.fold != val_fold].ID.values
    val_ids = df[df.fold == val_fold].ID.values
    print('DataFrame Loaded...')
    print('train length: {}'.format(len(train_ids)))
    print('val length: {}'.format(len(val_ids)))
    return(df, train_ids, val_ids)