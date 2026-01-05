# ==================================================
# 📌 PURPOSE OF THIS FILE (utils/common.py)
# ==================================================
# Reusable utility/helper functions used across ML pipelines:
# - YAML configuration loading
# - Directory creation
# - JSON & binary artifact handling
# - Logging
# - File size inspection
# - Base64 image encode / decode (API support)
# ==================================================

import os
import json
import yaml
import joblib
import base64

from pathlib import Path
from typing import Any, List

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnnClassifier import logger


# ==================================================
# read_yaml
# ==================================================
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns content as ConfigBox."""
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


# ==================================================
# create_directories
# ==================================================
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")



# ==================================================
# save_json
# ==================================================
@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


# ==================================================
# load_json
# ==================================================
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


# ==================================================
# save_bin
# ==================================================
@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


# ==================================================
# load_bin
# ==================================================
@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


# ==================================================
# get_size
# ==================================================
@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


# ==================================================
# decode_image
# ==================================================
@ensure_annotations
def decode_image(img_string: str, file_name: Path) -> None:
    img_data = base64.b64decode(img_string)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(img_data)
    logger.info(f"Image decoded and saved at: {file_name}")


# ==================================================
# encode_image_to_base64
# ==================================================
@ensure_annotations
def encode_image_to_base64(image_path: Path) -> bytes:
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    logger.info(f"Image encoded to Base64 from: {image_path}")
    return encoded_image

