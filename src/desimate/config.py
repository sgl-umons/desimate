import tomllib


CONFIG_PATH = "config.toml"


class Config(dict):
    def __init__(self, config: dict | None = None):
        super().__init__(config or {})


config = Config()
with open(CONFIG_PATH) as f:
    config_map = tomllib.loads(f.read())
config.update(config_map)
