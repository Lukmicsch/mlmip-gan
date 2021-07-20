def load_config(config_module_path):
    """"
    Load python config by importing it.

    :param config_module_path: where to find the config
    :return: the config
    """

    config = getattr(__import__(config_module_path, fromlist=['config']), 'config')

    return config
