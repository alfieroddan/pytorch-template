import yaml

def load(config_path):
  with open('config/'+config_path, 'r') as file:
    yaml_config = yaml.safe_load(file)
  return yaml_config
