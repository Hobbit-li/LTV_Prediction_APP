"""
Module for loading and validating YAML configuration files.
Compatible with PyInstaller's temporary directory when bundled as an EXE.

Handles configuration loading and structural validation.
"""

import yaml
import os
import sys

def resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource file.
    Be compatible with PyInstaller's temporary directory when bundled as an EXE.
    """
    if hasattr(sys, "_MEIPASS"):  # Temporary directory when bundled as EXE
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
    
def load_config():
    """
    Load configuration from a YAML file.
    """
    # First try the PyInstaller bundled path
    config_path = resource_path(os.path.join("config", "config.yaml"))

    # If not found, fall back to the source code path
    if not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project root directory
        config_path = os.path.join(base_dir, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
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
        "path_ref",
        "path_pre",
        "days_list",
        "days_list_existed",
        "cost",
        "percentiles",
        "base_weights",
        "top_num",
        "payer_tag",
        "num_features",
        "cat_features",
        "target_col",
        "id_col",
        "num_features_map",
        "params_clf",
        "params_reg",
    }
    return all(key in config for key in required_keys)
