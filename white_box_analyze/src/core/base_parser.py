import yaml
import json

class BaseParser:
    def __init__(self, op_name="op_name", json_path=None):
        self.op_name = op_name
        self.mapping_file = f"config/mappings/{op_name}_map.yaml"
        self.json_path = json_path
        self.json_data = self._load_json()
        self.version = self._get_version()
        self.version_mapping = self._load_mappings()
        self.params = self._process_json()

    def _load_json(self):
        """读取json文件"""
        with open(self.json_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def _get_version(self):
        """获取算子接口版本号"""
        acl_name = self.json_data["acl_name"]
        if 'V4' in acl_name:
            version = self.op_name + '_v4'
        elif 'V3' in acl_name:
            version = self.op_name + '_v3'
        elif 'V2' in acl_name:
            version = self.op_name + '_v2'
        else:
            version = self.op_name + '_v1'
        return version

    def _load_mappings(self):
        """加载YAML文件映射关系"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                mappings = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"Mapping file not found: {self.mapping_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        if self.version not in mappings:
            available_versions = list(mappings.keys())
            raise ValueError(f"Version '{self.version}' not found. Available versions: {available_versions}")

        version_mapping = mappings[self.version]
        if 'inputs' not in version_mapping or 'outputs' not in version_mapping:
            raise ValueError(f"Version '{self.version}' mapping is missing 'inputs' or 'outputs' section")
        return version_mapping

    def _process_json(self):
        """根据映射关系解析json文件"""
        params = {}
        for item in self.json_data:
            if item != 'input' and item != 'output':
                params[item] = self.json_data[item]
        for item in self.json_data["input"]:
            if 'input' in item["name"]:
                i = int(item["name"][5:])
                params[self.version_mapping['inputs'][i]] = {
                    'dtype': item['dtype'],
                    'shape': item['shape']
                }
            else:
                params[item["name"]] = item["value"]
        for item in self.json_data["output"]:
            if 'output' in item["name"]:
                i = int(item["name"][6:])
                params[self.version_mapping['outputs'][i]] = {
                    'dtype': item['dtype'],
                    'shape': item['shape']
                }
        return params