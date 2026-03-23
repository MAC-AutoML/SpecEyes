"""
Drop-in replacement for qwen_vl_utils.vision_process.fetch_image.

This function adds support for local paths, file URLs and base64-encoded images.
It is intended to be copied into an installed qwen-vl-utils package by
`scripts/patch_qwen_vl_utils.py`.
"""

import base64
import copy
from io import BytesIO
from typing import Dict, Union

import requests
from PIL import Image

from qwen_vl_utils.vision_process import (
    IMAGE_MAX_TOKEN_NUM,
    IMAGE_MIN_TOKEN_NUM,
    SPATIAL_MERGE_SIZE,
    smart_resize,
    to_rgb,
)


def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, dict) and "url" in image:
        image = image["url"]
        if image.startswith("http://") or image.startswith("https://"):
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        else:
            image_obj = Image.open(image)
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image
