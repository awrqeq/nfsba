import yaml

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Simple way to access dict keys as attributes
    cfg = DictToObject(cfg)
    return cfg

class DictToObject:
    """ Converts dictionary to object for easier attribute access """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"{vars(self)}"

# Example Usage in train_generator.py:
# from utils.config import load_config
# cfg = load_config('configs/nfsba_cifar10.yaml')
# print(cfg.generator.lr)