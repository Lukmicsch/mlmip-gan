import os
import pathlib
import argparse

from utils.dict_functions import load_dict

def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default.json', help="Name of a json config file within configs folder. (default: 'default.json')")
    parser.add_argument("--train", default=False, action='store_true', help="Train the gan. (default: False)")
    parser.add_argument("--load_data", default=False, action='store_true', help="Load data. (default: False)")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Initialize abs path to config folder and data/x
    abs_path = pathlib.Path().absolute()
    conf_path = os.path.join(abs_path, 'configs/')

    # Initialize path to config and load dict
    config_path = os.path.join(conf_path, args.config)
    config = load_dict(config_path)

    if args.train:
        pass