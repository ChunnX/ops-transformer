import yaml
import os
from pathlib import Path
import json

from src.core.parser_registry import ParserRegistry
from src.core.base_parser import BaseParser

@ParserRegistry.register("mlaProlog")
class MLAPrologParser(BaseParser):
    def __init__(self, op_name = "op_name", json_path = None):
        super().__init__(op_name, json_path)