"""Ascend Ops — A5 MTE MoE Operators"""
import torch
import os

_so = os.path.join(os.path.dirname(__file__), "_C.abi3.so")
torch.ops.load_library(_so)
