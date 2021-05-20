import os
import pathlib
import argparse

def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default.json', help="Name of a json config file within configs folder. (default: 'default.json')")
    parser.add_argument("--train", default=False, action='store_true', help="Train the gan. (default: False)")

    return parser.parse_args()