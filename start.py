import os
import pathlib
import argparse

from utils.python_utils import load_config
from pix2pix import run_pix2pix_train



def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default', help="Name of a python config file within configs folder. (default: 'default')")
    parser.add_argument("--train", default=False, action='store_true', help="Train the gan. (default: False)")

    return parser.parse_args()




if __name__ == '__main__':
    args = get_args()

    # Initialize abs path to config folder and data/x
    abs_path = pathlib.Path().absolute()

    # Initialize path to config and load dict
    config_path = 'configs.' + args.config
    config = load_config(config_path)

    if args.train:
        run_pix2pix_train(config)