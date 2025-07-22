import yaml

with open("lgbm_config.yaml", "r") as f:
    lgbm_params = yaml.safe_load(f)
