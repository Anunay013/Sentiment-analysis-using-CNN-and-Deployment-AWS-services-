import json

class ConfigHolder():

    def __init__(self, config_path):

        self.config = json.load(open(config_path, "r"))
