import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file).get('evaluation', {})

    def __getitem__(self, key):
        return self.config[key]

    def __getattr__(self, name):
        try:
            return self.config[name]
        except KeyError:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!s}")

    def __setattr__(self, name, value):
        if name == 'config':
            object.__setattr__(self, name, value)
        else:
            self.config[name] = value

    def update(self, args):
        for key, val in vars(args).items():
            if val is not None:
                self.config[key] = val

    def debug_print(self):
        print("Current Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
