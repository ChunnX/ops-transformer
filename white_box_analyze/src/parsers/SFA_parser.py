import yaml
import os
from pathlib import Path
import json
import ast

from src.core.parser_registry import ParserRegistry
from src.core.base_parser import BaseParser

@ParserRegistry.register('SFA')
class SFAParser(BaseParser):
    def __init__(self, op_name = "op_name", json_path = None):
        self.op_name = op_name
        self.mapping_file = f"config/mappings/{op_name}_map.yaml"
        self.json_path = json_path
        if isinstance(json_path, dict):
            self.json_data = self.json_path
            self.params = {}
            self._process_dict_params()
        else:
            self.json_data = super()._load_json()
            self.version = self._get_version()
            self.version_mapping = self._load_mappings()
            self.params = self._process_json()
            self._process_params()
    
    def _get_version(self):
        """获取算子接口版本号"""
        acl_name = self.json_data["acl_name"]
        if acl_name == 'aclnnSparseFlashAttention':
            version = acl_name
        else:
            # 抛出包含详细信息的异常
            error_msg = (
                f"不支持的算子接口: {acl_name}\n"
                f"支持的算子: 'aclnnLightningIndexer'\n"
                f"请检查JSON配置中的acl_name字段"
            )
            raise ValueError(error_msg)

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
                if item["name"] == 'actualseqlengths' :
                    params['actual_seq_lengths_query'] = item["value"]
                elif item["name"] == 'actualseqlengthskv' :
                    params['actual_seq_lengths_key'] = item["value"]
                else :
                    params[item["name"]] = item["value"]
        # for item in self.json_data["output"]:
        #     if 'output' in item["name"]:
        #         i = int(item["name"][6:])
        #         params[self.version_mapping['outputs'][i]] = {
        #             'dtype': item['dtype'],
        #             'shape': item['shape']
        #         }
        return params

    def _process_params(self):
        """根据映射关系解析json文件"""
        if 'actual_seq_lengths_query' not in self.params or self.params['actual_seq_lengths_query'] is None or self.params['actual_seq_lengths_query'] == []:
            layout_query = self.params.get('layout_query', '').upper()
            query_shape = self.params['query']['shape']
            if 'BSND' in layout_query:
                s = query_shape[1]
                b = query_shape[0]
            elif 'BNSD' in layout_query:
                s = query_shape[2]
                b = query_shape[0]
            self.params['actual_seq_lengths_query'] = [s] * b

        if 'actual_seq_lengths_key' not in self.params or self.params['actual_seq_lengths_key'] is None or self.params['actual_seq_lengths_key'] == []:
            layout_key = self.params.get('layout_kv', '').upper()
            key_shape = self.params['key']['shape']
            if 'BSND' in layout_key:
                s = key_shape[1]
                b = key_shape[0]
            elif 'BNSD' in layout_key:
                s = key_shape[2]
                b = key_shape[0]
            self.params['actual_seq_lengths_key'] = [s] * b

        if 'TND' in self.params.get('layout_query', '').upper():
            seq_lengths_query = self.params.get('actual_seq_lengths_query', [])
            if seq_lengths_query:
                new_seq_lengths_query = [seq_lengths_query[0]]
                for i in range(1, len(seq_lengths_query)):
                    new_seq_lengths_query.append(seq_lengths_query[i] - seq_lengths_query[i-1])
                self.params['actual_seq_lengths_query'] = new_seq_lengths_query

        if 'TND' in self.params.get('layout_kv', '').upper():
            seq_lengths_key = self.params.get('actual_seq_lengths_key', [])
            if seq_lengths_key:
                new_seq_lengths_key = [seq_lengths_key[0]]
                for i in range(1, len(seq_lengths_key)):
                    new_seq_lengths_key.append(seq_lengths_key[i] - seq_lengths_key[i-1])
                self.params['actual_seq_lengths_key'] = new_seq_lengths_key

        if 'TND' in self.params.get('layout_query', '').upper():
            actual_seq_lengths_query = self.params.get('actual_seq_lengths_query', [])
            B = len(actual_seq_lengths_query)
            self.params['B'] = B
        else :
            query_shape = self.params['query']['shape']
            B = query_shape[0]
            self.params['B'] = B
        
        if 'TND' in self.params.get('layout_query', '').upper():
            query_shape = self.params['query']['shape']
            query_rope_shape = self.params['qrope']['shape']
            N1 = query_shape[1]
            D_nope = query_shape[2]
            D_rope = query_rope_shape[2]
        elif 'BSND' in self.params.get('layout_query', '').upper():
            query_shape = self.params['query']['shape']
            query_rope_shape = self.params['qrope']['shape']
            N1 = query_shape[2]
            D_nope = query_shape[3]
            D_rope = query_rope_shape[3]
        elif 'BNSD' in self.params.get('layout_query', '').upper():
            query_shape = self.params['query']['shape']
            query_rope_shape = self.params['qrope']['shape']
            N1 = query_shape[1]
            D_nope = query_shape[3]
            D_rope = query_rope_shape[3]

        if 'TND' in self.params.get('layout_kv', '').upper():
            key_shape = self.params['key']['shape']
            N2 = key_shape[1]
            PA_BlockSize = 1 
        elif 'PA_BSND' in self.params.get('layout_kv', '').upper():
            key_shape = self.params['key(PA)']['shape']
            N2 = key_shape[2]
            PA_BlockSize = key_shape[1]
        elif 'BSND' in self.params.get('layout_kv', '').upper():
            key_shape = self.params['key']['shape']
            N2 = key_shape[2]
            PA_BlockSize = 1
        elif 'BNSD' in self.params.get('layout_kv', '').upper():
            key_shape = self.params['key']['shape']
            N2 = key_shape[1]
            PA_BlockSize = 1
        topk_num = self.params['sparse_indices']['shape'][-1]
        self.params['G'] = int(N1/N2)
        self.params['N2'] = N2
        self.params['D_nope'] = D_nope
        self.params['D_rope'] = D_rope
        self.params['topk_num'] = topk_num
        self.params['PA_BlockSize'] = PA_BlockSize
        return self.params

    def _process_dict_params(self):
        """根据映射关系解析从excel读取到的单个case"""
        self.params['case_name'] = self.json_data['Testcase_Prefix']
        self.params['G'] = int(self.json_data['N1']/self.json_data['N2'])
        self.params['D_nope'] = self.json_data['D']
        self.params['D_rope'] = self.json_data['rope_head_dim']

        if 'actual_seq_q' not in self.json_data or self.json_data['actual_seq_q'] is None or self.json_data['actual_seq_q'] == []:
            layout_query = self.json_data.get('layout_query', '').upper()
            self.params['actual_seq_lengths_query'] = [self.json_data['S1']] * self.json_data['B']
        if 'actual_seq_kv' not in self.json_data or self.json_data['actual_seq_kv'] is None or self.json_data['actual_seq_kv'] == []:
            layout_kv = self.json_data.get('layout_kv', '').upper()
            self.params['actual_seq_lengths_key'] = [self.json_data['S2']] * self.json_data['B']
        if 'TND' in self.json_data.get('layout_query', '').upper():
            seq_lengths_query = ast.literal_eval(self.json_data['actual_seq_q'])
            if seq_lengths_query:
                new_seq_lengths_query = [seq_lengths_query[0]]
                for i in range(1, len(seq_lengths_query)):
                    new_seq_lengths_query.append(seq_lengths_query[i] - seq_lengths_query[i-1])
                self.params['actual_seq_lengths_query'] = new_seq_lengths_query
        else:
            self.params['actual_seq_lengths_query'] = ast.literal_eval(self.json_data['actual_seq_q'])
        if 'TND' in self.json_data.get('layout_kv', '').upper():
            seq_lengths_key = ast.literal_eval(self.json_data['actual_seq_kv'])
            if seq_lengths_key:
                new_seq_lengths_key = [seq_lengths_key[0]]
                for i in range(1, len(seq_lengths_key)):
                    new_seq_lengths_key.append(seq_lengths_key[i] - seq_lengths_key[i-1])
                self.params['actual_seq_lengths_key'] = new_seq_lengths_key
        else:
            self.params['actual_seq_lengths_key'] = ast.literal_eval(self.json_data['actual_seq_kv'])
        self.params['sparse_blocksize'] = self.json_data['sparse_block_size']
        self.params['topk_num'] = self.json_data['K']
        self.params['sparsemode'] = self.json_data['sparse_mode']
        self.params['PA_BlockSize'] = self.json_data['block_size']
        self.params['B'] = self.json_data['B']
        self.params['layout_query'] = self.json_data['layout_query']
        self.params['layout_kv'] = self.json_data['layout_kv']
        self.params['query'] = {'dtype': 'bf16' if self.json_data['q_type'] == 'torch.bfloat16' else 'fp16'}
        self.params['key'] = {'dtype': 'bf16' if self.json_data['kv_type'] == 'torch.bfloat16' else 'fp16'}
        self.params['value'] = {'dtype': 'bf16' if self.json_data['kv_type'] == 'torch.bfloat16' else 'fp16'}
        return self.params