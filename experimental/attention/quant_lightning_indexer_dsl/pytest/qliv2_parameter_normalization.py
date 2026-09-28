"""Normalization helpers shared by Excel and PT batch generation."""

import ast
import math

import torch


DTYPES = {
    "FLOAT4_E2M1": torch.float4_e2m1fn_x2,
    "FLOAT4_E2M1FN_X2": torch.float4_e2m1fn_x2,
    "torch.float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
    "FLOAT8_E8M0": torch.float8_e8m0fnu,
    "FLOAT8_E8M0FNU": torch.float8_e8m0fnu,
    "torch.float8_e8m0fnu": torch.float8_e8m0fnu,
    "FP32": torch.float32,
    "torch.uint8": torch.uint8,
    "uint8": torch.uint8,
    "torch.float32": torch.float32,
    "float32": torch.float32,
    "INT32": torch.int32,
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
