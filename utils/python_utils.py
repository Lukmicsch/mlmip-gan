def load_config(config_module_path):
    """" Load python config by importing it. """

    config = getattr(__import__(config_module_path, fromlist=['config']), 'config')

    return config