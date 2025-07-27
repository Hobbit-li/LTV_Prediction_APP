# src/config_loader.py
import yaml

"""Module for loading and managing configuration files in YAML format."""
def load_config(path="config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        path (str): Path to the YAML configuration file
        
    Returns:
        dict: Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails

    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def validate_config(config: dict) -> bool:
    """
    Validate the configuration dictionary structure.
    
    Args:
        config (dict): Configuration dictionary to validate
        
    Returns:
        bool: Returns True if config is valid, False otherwise
    """
    required_keys = {
        'path_ref',
        'path_pre',
        'days_list',
        'days_list_existed',
        'cost',
        'percentiles',
        'base_weights',
        'top_num',
        'payer_tag',
        'num_features',
        'cat_features',
        'target_col',
        'id_col',
        'days_list',
        'num_features_map',
        'params_clf',
        'params_reg'
    }
    return all(key in config for key in required_keys)
