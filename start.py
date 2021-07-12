import argparse
import os
import pathlib

from utils.python_utils import load_config
from dcgan import run_dcgan_train

def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='dcgan256.py', help="Name of the python config file within configs folder. (default: 'dcgan256.json')")
    parser.add_argument("--train", default=False, action='store_true',
                        help="Train the GAN. (default: False)")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # Paths
    abs_path = pathlib.Path().absolute()

    conf_path = 'configs.' + args.config

    # Config
    config = load_config(conf_path)
    algorithm = config['algorithm']

    if args.train:
        if algorithm == 'dcgan' or algorithm == 'dcgan256': run_dcgan_train(config)
