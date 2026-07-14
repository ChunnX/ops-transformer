import yaml
import os
from pathlib import Path
import json
import ast

from src.core.parser_registry import ParserRegistry
from src.core.base_parser import BaseParser

@ParserRegistry.register('MQSMLA')
class MQSMLAParser(BaseParser):
    def __init__(self, op_name="op_name", json_path=None):
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
        acl_name = self.json_data["acl_name"]
        if acl_name == 'aclnnMixedQuantSparseFlashMla':
            version = acl_name
        else:
            error_msg = (
                f"不支持的算子接口: {acl_name}\n"
                f"支持的算子: 'aclnnMixedQuantSparseFlashMla'\n"
                f"请检查JSON配置中的acl_name字段"
            )
            raise ValueError(error_msg)
        return version

    def _load_mappings(self):
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
        params = {}
        for item in self.json_data:
            if item != 'input' and item != 'output':
                params[item] = self.json_data[item]
        for item in self.json_data["input"]:
            if 'input' in item["name"]:
                i = int(item["name"][5:])
                mapped_name = self.version_mapping['inputs'].get(i)
                if mapped_name is not None:
                    params[mapped_name] = {
                        'dtype': item.get('dtype'),
                        'shape': item.get('shape')
                    }
            else:
                if item["name"] == 'actualseqlengths':
                    params['actual_seq_lengths_query'] = item["value"]
                elif item["name"] == 'actualseqlengthskv':
                    params['actual_seq_lengths_ori_kv'] = item["value"]
                elif item["name"] == 'actualseqlengths cmpkv':
                    params['actual_seq_lengths_cmp_kv'] = item["value"]
                else:
                    params[item["name"]] = item["value"]
        return params

    def _process_params(self):
        layout_q = self.params.get('layout_q', self.params.get('layout_query', '')).upper()
        layout_kv = self.params.get('layout_kv', '').upper()
        self.params['layout_q'] = layout_q
        self.params['layout_kv'] = layout_kv

        if 'actual_seq_lengths_query' not in self.params or self.params['actual_seq_lengths_query'] is None or self.params['actual_seq_lengths_query'] == []:
            q_shape = self.params['q']['shape']
            if 'BSND' in layout_q:
                s = q_shape[1]
                b = q_shape[0]
            elif 'TND' in layout_q:
                s = q_shape[0]
                b = 1
            self.params['actual_seq_lengths_query'] = [s] * b

        if 'TND' in layout_q:
            seq_q = self.params.get('actual_seq_lengths_query', [])
            if seq_q:
                new_seq_q = [seq_q[0]]
                for i in range(1, len(seq_q)):
                    new_seq_q.append(seq_q[i] - seq_q[i - 1])
                self.params['actual_seq_lengths_query'] = new_seq_q

        if 'TND' in layout_q:
            self.params['B'] = len(self.params.get('actual_seq_lengths_query', []))
        else:
            self.params['B'] = self.params['q']['shape'][0]

        if 'TND' in layout_q:
            q_shape = self.params['q']['shape']
            N1 = q_shape[1]
            D = q_shape[2]
        elif 'BSND' in layout_q:
            q_shape = self.params['q']['shape']
            N1 = q_shape[2]
            D = q_shape[3]

        N2 = 1
        rope_head_dim = self.params.get('rope_head_dim', 64)
        D_nope = D - rope_head_dim
        D_rope = rope_head_dim
        G = int(N1 / N2)

        if 'PA_BBND' in layout_kv:
            ori_kv_shape = self.params.get('ori_kv', {}).get('shape', [0, 1, 1, 1])
            ori_block_size = ori_kv_shape[1]
            cmp_kv_shape = self.params.get('cmp_kv', {}).get('shape', [0, 1, 1, 1])
            cmp_block_size = cmp_kv_shape[1] if cmp_kv_shape[0] > 0 else 0
            PA_ori_block_size = ori_block_size
            PA_cmp_block_size = cmp_block_size
        else:
            PA_ori_block_size = 1
            PA_cmp_block_size = 1

        if 'actual_seq_lengths_ori_kv' not in self.params or self.params['actual_seq_lengths_ori_kv'] is None or self.params['actual_seq_lengths_ori_kv'] == []:
            if 'PA_BBND' in layout_kv:
                ori_bt_shape = self.params.get('ori_block_table', {}).get('shape', [0, 0])
                max_block_num = ori_bt_shape[1] if len(ori_bt_shape) > 1 else 0
                s2 = max_block_num * PA_ori_block_size
            elif 'BSND' in layout_kv:
                s2 = self.params.get('ori_kv', {}).get('shape', [0, 0, 0, 0])[1]
            elif 'TND' in layout_kv:
                s2 = self.params.get('ori_kv', {}).get('shape', [0, 0, 0])[0]
            else:
                s2 = 0
            self.params['actual_seq_lengths_ori_kv'] = [s2] * self.params['B']

        if 'TND' in layout_kv:
            seq_k = self.params.get('actual_seq_lengths_ori_kv', [])
            if seq_k:
                new_seq_k = [seq_k[0]]
                for i in range(1, len(seq_k)):
                    new_seq_k.append(seq_k[i] - seq_k[i - 1])
                self.params['actual_seq_lengths_ori_kv'] = new_seq_k

        has_cmp_kv = self.params.get('cmp_kv') is not None and self.params.get('cmp_kv', {}).get('shape') is not None
        has_cmp_sparse_indices = self.params.get('cmp_sparse_indices') is not None and self.params.get('cmp_sparse_indices', {}).get('shape') is not None

        if has_cmp_kv:
            if 'actual_seq_lengths_cmp_kv' not in self.params or self.params['actual_seq_lengths_cmp_kv'] is None or self.params['actual_seq_lengths_cmp_kv'] == []:
                if 'PA_BBND' in layout_kv:
                    cmp_bt_shape = self.params.get('cmp_block_table', {}).get('shape', [0, 0])
                    max_cmp_block_num = cmp_bt_shape[1] if len(cmp_bt_shape) > 1 else 0
                    s3 = max_cmp_block_num * PA_cmp_block_size
                elif 'BSND' in layout_kv:
                    s3 = self.params.get('cmp_kv', {}).get('shape', [0, 0, 0, 0])[1]
                elif 'TND' in layout_kv:
                    s3 = self.params.get('cmp_kv', {}).get('shape', [0, 0, 0])[0]
                else:
                    s3 = 0
                self.params['actual_seq_lengths_cmp_kv'] = [s3] * self.params['B']

            if 'TND' in layout_kv:
                seq_ck = self.params.get('actual_seq_lengths_cmp_kv', [])
                if seq_ck:
                    new_seq_ck = [seq_ck[0]]
                    for i in range(1, len(seq_ck)):
                        new_seq_ck.append(seq_ck[i] - seq_ck[i - 1])
                    self.params['actual_seq_lengths_cmp_kv'] = new_seq_ck

        if not has_cmp_kv:
            template_mode = 0  # SWA
        elif has_cmp_sparse_indices:
            template_mode = 2  # CSA
        else:
            template_mode = 1  # HCA

        sparse_block_count = 0
        if has_cmp_sparse_indices:
            cmp_si_shape = self.params.get('cmp_sparse_indices', {}).get('shape', [])
            if cmp_si_shape:
                sparse_block_count = cmp_si_shape[-1]

        cmp_ratio = self.params.get('cmp_ratio', 1)
        quant_mode = self.params.get('quant_mode', self.params.get('quant_scale_repo_mode', 1))
        ori_mask_mode = self.params.get('ori_mask_mode', 4)
        cmp_mask_mode = self.params.get('cmp_mask_mode', 3)
        ori_win_left = self.params.get('ori_win_left', 127)
        ori_win_right = self.params.get('ori_win_right', 0)

        is_split_g = 1 if G > 64 else 0

        ori_kv_dtype = self.params.get('ori_kv', {}).get('dtype', 'float8_e4m3fn')
        if ori_kv_dtype == 'hifloat8' or ori_kv_dtype == 'DT_HIFLOAT8':
            kv_dtype_code = 1
        else:
            kv_dtype_code = 0

        self.params['G'] = G
        self.params['N1'] = N1
        self.params['N2'] = N2
        self.params['D'] = D
        self.params['D_nope'] = D_nope
        self.params['D_rope'] = D_rope
        self.params['PA_ori_block_size'] = PA_ori_block_size
        self.params['PA_cmp_block_size'] = PA_cmp_block_size
        self.params['template_mode'] = template_mode
        self.params['is_split_g'] = is_split_g
        self.params['quant_mode'] = quant_mode
        self.params['cmp_ratio'] = cmp_ratio
        self.params['ori_mask_mode'] = ori_mask_mode
        self.params['cmp_mask_mode'] = cmp_mask_mode
        self.params['ori_win_left'] = ori_win_left
        self.params['ori_win_right'] = ori_win_right
        self.params['sparse_block_count'] = sparse_block_count
        self.params['has_cmp_kv'] = has_cmp_kv
        self.params['has_cmp_sparse_indices'] = has_cmp_sparse_indices
        self.params['kv_dtype_code'] = kv_dtype_code
        return self.params

    def _process_dict_params(self):
        self.params['case_name'] = self.json_data['Testcase_Prefix']
        self.params['G'] = int(self.json_data['N1'] / self.json_data['N2'])
        self.params['N1'] = self.json_data['N1']
        self.params['N2'] = self.json_data['N2']
        self.params['D'] = self.json_data.get('D', 512)
        self.params['D_nope'] = self.params['D'] - self.json_data.get('rope_head_dim', 64)
        self.params['D_rope'] = self.json_data.get('rope_head_dim', 64)

        self.params['actual_seq_lengths_query'] = ast.literal_eval(self.json_data.get('actual_seq_q', '[]')) or [self.json_data['S1']] * self.json_data['B']
        self.params['actual_seq_lengths_ori_kv'] = ast.literal_eval(self.json_data.get('actual_seq_kv', '[]')) or [self.json_data['S2']] * self.json_data['B']
        self.params['actual_seq_lengths_cmp_kv'] = ast.literal_eval(self.json_data.get('actual_seq_cmp_kv', '[]')) or [self.json_data.get('S3', 0)] * self.json_data['B']

        self.params['PA_ori_block_size'] = self.json_data.get('ori_block_size', 1)
        self.params['PA_cmp_block_size'] = self.json_data.get('cmp_block_size', 0)
        self.params['B'] = self.json_data['B']
        self.params['layout_q'] = self.json_data.get('layout_q', 'BSND')
        self.params['layout_kv'] = self.json_data.get('layout_kv', 'PA_BBND')
        self.params['quant_mode'] = self.json_data.get('quant_mode', 1)
        self.params['cmp_ratio'] = self.json_data.get('cmp_ratio', 1)
        self.params['ori_mask_mode'] = self.json_data.get('ori_mask_mode', 4)
        self.params['cmp_mask_mode'] = self.json_data.get('cmp_mask_mode', 3)
        self.params['ori_win_left'] = self.json_data.get('ori_win_left', 127)
        self.params['ori_win_right'] = self.json_data.get('ori_win_right', 0)
        self.params['sparse_block_count'] = self.json_data.get('K2', 0)
        self.params['template_mode'] = self.json_data.get('template_mode', 0)
        self.params['is_split_g'] = 1 if self.params['G'] > 64 else 0
        self.params['has_cmp_kv'] = self.json_data.get('has_cmp_kv', False)
        self.params['has_cmp_sparse_indices'] = self.json_data.get('has_cmp_sparse_indices', False)
        self.params['kv_dtype_code'] = 0
        self.params['query'] = {'dtype': 'bf16'}
        self.params['ori_kv'] = {'dtype': self.json_data.get('kv_type', 'float8_e4m3fn')}
        self.params['cmp_kv'] = {'dtype': self.json_data.get('kv_type', 'float8_e4m3fn')}
        return self.params
