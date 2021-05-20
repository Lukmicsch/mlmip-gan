import json
import os
import pathlib


def load_dict(filename):
    """ Load dictionary from JSON file. """

    print("loading dictionary from ", filename)
    dictionary = None

    dirname, basename = os.path.split(filename)

    if not pathlib.Path(dirname).is_dir():
        raise ValueError("directory: %s does not exist" % (dirname))

    if not pathlib.Path(filename).is_file():
        raise ValueError("file: &s does not exist" % (filename))

    with open(filename, 'r') as fileHandle:
        dictionary = json.loads(fileHandle.read())

    return dictionary