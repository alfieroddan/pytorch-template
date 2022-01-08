## Config
An example config is [here](config/res1.yml)

# Train
Simply run <code>python train.py --config [config name]</code>

# Tensorboard
To show tensorboard output:
pip install tensorboard
tensorboard --logdir=<folder>

## Motivation
I wanted to create a plug and play version of training. By utilising the config file we can keep adding parts (metrics, loss functions, checkpoints etc...) without removing old code.

## To Do
- Early Stopping
