"""Normalization helpers shared by Excel and PT batch generation."""

import ast
import math

import torch


DTYPES = {
    "torch.uint8": torch.uint8,
    "uint8": torch.uint8,
    "torch.float32": torch.float32,
    "float32": torch.float32,
    "torch.int32": torch.int32,
    "int32": torch.int32,
}


def normalize_cell(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in DTYPES:
            return DTYPES[text]
        if text.lower() in ("none", "null", ""):
            return None
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        if text.startswith(("[", "(", "{")):
            return ast.literal_eval(text)
        return text
    return value

