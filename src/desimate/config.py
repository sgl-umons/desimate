import tomllib


class Config(dict):
    def __init__(self, config: dict | None = None):
        super().__init__(config or {})

    def read(self, file):
        with open(file) as f:
            config_map = tomllib.loads(f.read())
        config.update(config_map)


config = Config()
# DEFAULT HARDCODED CONFIG (to be sure the default it works)
config.update(
    {
        "parsing": {
            "markdown": {
                "granularities": {
                    "paragraph": ["paragraph", "list", "table"],
                    "md-line": ["text", "listItem", "table", "tableRow"],
                }
            }
        }
    }
)
