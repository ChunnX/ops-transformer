#!/usr/bin/env python3
"""ATK JSON → TTK CSV 转换脚本 — MlaPreprocess 算子

将 all_aclnnMlaPreprocess.json 转为 TTK ACLNN 模式 CSV。

C API 参数顺序 (aclnnMlaPreprocessGetWorkspaceSize):
  tensor inputs (24): input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0, bias0,
                      gamma1, beta1, quantScale1, quantOffset1, wuq, deScale1, bias1, gamma2,
                      cos, sin, wuk, kvCache, kvCacheRope, slotmapping, ctkvScale, qNopeScale
  attr (13):         wdqDim, qRopeDim, kRopeDim, epsilon, qRotaryCoeff, kRotaryCoeff,
                      transposeWdq, transposeWuq, transposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount
  tensor outputs (4): qOutOut, kvCacheOutOut, qRopeOutOut, krCacheOutOut

用法:
  python3 atk_json_to_ttk_csv_mla_preprocess.py input.json output.csv [start_id] [end_id]
  默认转换前 10 条用例 (id 0-9)
"""

import csv
import json
import sys
import ast

DTYPE_MAP = {
    'fp16': 'float16',
    'bf16': 'bfloat16',
    'fp32': 'float32',
    'int8': 'int8',
    'int32': 'int32',
    'int64': 'int64',
    'bool': 'bool',
    'double': 'float64',
}

# C 头文件 tensor 参数名 (24 输入 + 4 输出 = 28)
TENSOR_PARAM_NAMES = [
    'input', 'gamma0', 'beta0', 'quantScale0', 'quantOffset0',
    'wdqkv', 'deScale0', 'bias0', 'gamma1', 'beta1',
    'quantScale1', 'quantOffset1', 'wuq', 'deScale1', 'bias1',
    'gamma2', 'cos', 'sin', 'wuk', 'kvCache', 'kvCacheRope',
    'slotmapping', 'ctkvScale', 'qNopeScale',
    # outputs:
    'qOutOut', 'kvCacheOutOut', 'qRopeOutOut', 'krCacheOutOut',
]

ATTR_NAMES = [
    'wdqDim', 'qRopeDim', 'kRopeDim', 'epsilon',
    'qRotaryCoeff', 'kRotaryCoeff', 'transeposeWdq', 'transeposeWuq',
    'transeposeWuk', 'cacheMode', 'quantMode', 'doRmsNorm', 'wdkvSplitCount',
]

# ATK JSON 名称 → C API 参数名映射 (C 头文件中有拼写错误 transeposeWdq)
ATTR_NAME_MAP = {
    'transposeWdq': 'transeposeWdq',
    'transposeWuq': 'transeposeWuq',
    'transposeWuk': 'transeposeWuk',
}

SCALAR_DTYPES = [
    'int64', 'int64', 'int64', 'float64',
    'int64', 'int64', 'bool', 'bool',
    'bool', 'int64', 'int64', 'bool', 'int64',
]

BLOCK_NUM = 192
BLOCK_SIZE = 128


def _normalize_attr_value(val):
    """将 attr 的 range_values 归一化为标量"""
    if isinstance(val, list):
        return val[0] if len(val) > 0 else 0
    return val


def _normalize_range(rv):
    """将 tensor 的 range_values 归一化为 (min, max) 元组"""
    if isinstance(rv, list) and len(rv) == 2:
        return tuple(rv)
    if isinstance(rv, (int, float)):
        return (rv, rv)
    return (-1, 1)


def _compute_output_shapes(input_shape, wuk_shape, kvCache_shape, kvCacheRope_shape, cacheMode):
    """根据输入参数计算 4 个输出 tensor 的 shape

    NPU 输出的 kvCacheOut/krCacheOut 在 cacheMode>1 时是 NZ 格式:
      NZ shape = [BLOCK_NUM, last_dim//16, BLOCK_SIZE, 16]

    Returns:
        list of 4 shapes: [qOutOut_shape, kvCacheOutOut_shape, qRopeOutOut_shape, krCacheOutOut_shape]
    """
    token_num = input_shape[0]
    head_num = wuk_shape[0]
    q_out_dim = wuk_shape[2]  # einsum 输出维度
    rope_dim = 64
    kv_last_dim = kvCache_shape[-1]       # kvCache 最后一维
    kv_rope_last_dim = kvCacheRope_shape[-1]  # kvCacheRope 最后一维

    if cacheMode == 0:
        q_out_shape = [token_num, head_num, q_out_dim + rope_dim]
        kv_cache_out_shape = list(kvCache_shape)
        kv_cache_out_shape[-1] = q_out_dim + rope_dim
        q_rope_out_shape = list(q_out_shape)
        kr_cache_out_shape = list(kvCacheRope_shape)
    elif cacheMode == 1:
        q_out_shape = [token_num, head_num, q_out_dim]
        kv_cache_out_shape = list(kvCache_shape)
        q_rope_out_shape = [token_num, head_num, rope_dim]
        kr_cache_out_shape = list(kvCacheRope_shape)
    elif cacheMode == 2:
        q_out_shape = [token_num, head_num, q_out_dim]
        kv_cache_out_shape = [BLOCK_NUM, kv_last_dim // 16, BLOCK_SIZE, 16]
        q_rope_out_shape = [token_num, head_num, rope_dim]
        kr_cache_out_shape = [BLOCK_NUM, kv_rope_last_dim // 16, BLOCK_SIZE, 16]
    else:  # cacheMode == 3
        q_out_shape = [token_num, head_num, q_out_dim]
        kv_cache_out_shape = [BLOCK_NUM, kv_last_dim // 16, BLOCK_SIZE, 16]
        q_rope_out_shape = [token_num, head_num, rope_dim]
        kr_cache_out_shape = [BLOCK_NUM, kv_rope_last_dim // 16, BLOCK_SIZE, 16]

    return [q_out_shape, kv_cache_out_shape, q_rope_out_shape, kr_cache_out_shape]


def _compute_output_dtypes(input_dtype, cacheMode):
    """根据 cacheMode 计算输出 dtype"""
    if cacheMode == 2:
        # out[0] is int8 (quantized), out[1] is int8 (NZ quantized)
        # out[2] is data_type, out[3] is data_type
        return ['int8', 'int8', input_dtype, input_dtype]
    else:
        return [input_dtype, input_dtype, input_dtype, input_dtype]


def convert_json_to_csv(json_path, csv_path, start_id=0, end_id=9):
    with open(json_path) as f:
        cases = json.load(f)

    # Filter by id range
    selected = [c for c in cases if start_id <= c['id'] <= end_id]
    print(f"Converting {len(selected)} cases (id {start_id}-{end_id}) from {json_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # CSV header
        writer.writerow([
            'testcase_name', 'api_name', 'tensor_view_shapes',
            'tensor_dtypes', 'scalar_dtypes', 'attributes',
            'input_data_ranges', 'tensor_formats',
            'precision_tolerances', 'absolute_precision'
        ])

        for case in selected:
            case_id = case['id']
            inputs = case['inputs']

            # Build tensor name → input map
            input_map = {}
            for inp in inputs:
                if inp.get('type') in ('tensor', 'attr', 'attr_tuple'):
                    input_map[inp['name']] = inp

            # Build attr dict (use API param names, mapping from JSON names)
            attrs = {}
            for name in ATTR_NAMES:
                # Try API name first, then JSON name via reverse map
                json_name = name
                for jn, an in ATTR_NAME_MAP.items():
                    if an == name:
                        json_name = jn
                        break
                if json_name in input_map:
                    val = input_map[json_name].get('range_values')
                    attrs[name] = _normalize_attr_value(val)

            cacheMode = int(attrs.get('cacheMode', 3))

            # Build tensor shapes/dtypes/ranges (24 inputs + 4 outputs)
            tensor_shapes = []
            tensor_dtypes = []
            input_ranges = []

            for name in TENSOR_PARAM_NAMES[:24]:  # 24 input tensors
                inp = input_map.get(name)
                if inp is None:
                    tensor_shapes.append(None)
                    tensor_dtypes.append(None)
                    input_ranges.append(None)
                    continue

                shape = inp.get('shape')
                dtype_str = DTYPE_MAP.get(inp.get('dtype', 'fp16'), 'float16')
                rv = _normalize_range(inp.get('range_values'))

                tensor_shapes.append(shape)
                tensor_dtypes.append(dtype_str)
                input_ranges.append(rv)

            # Compute output shapes/dtypes
            input_shape = input_map['input']['shape']
            wuk_shape = input_map['wuk']['shape']
            kvCache_shape = input_map['kvCache']['shape']
            kvCacheRope_shape = input_map['kvCacheRope']['shape']
            input_dtype = DTYPE_MAP.get(input_map['input'].get('dtype', 'fp16'), 'float16')

            out_shapes = _compute_output_shapes(
                input_shape, wuk_shape, kvCache_shape, kvCacheRope_shape, cacheMode
            )
            out_dtypes = _compute_output_dtypes(input_dtype, cacheMode)

            for i, shape in enumerate(out_shapes):
                tensor_shapes.append(shape)
                tensor_dtypes.append(out_dtypes[i])
                input_ranges.append((0, 0))  # outputs don't need data ranges

            # Format as Python literals
            def fmt_shapes(shapes):
                parts = []
                for s in shapes:
                    if s is None:
                        parts.append('None')
                    else:
                        parts.append(str(tuple(s)) if len(s) > 1 else f"({s[0]},)")
                return f"({','.join(parts)})"

            def fmt_dtypes(dtypes):
                parts = [f"'{d}'" if d else 'None' for d in dtypes]
                return f"({','.join(parts)})"

            def fmt_ranges(ranges):
                parts = []
                for r in ranges:
                    if r is None:
                        parts.append('None')
                    else:
                        parts.append(f"({r[0]},{r[1]})")
                return f"({','.join(parts)})"

            def fmt_attrs(attr_dict):
                items = [f"'{k}':{repr(v)}" for k, v in attr_dict.items()]
                return '{' + ','.join(items) + '}'

            testcase_name = f"mla_preprocess_{case_id}"

            # Build tensor_formats:
            # wdqkv(5)/wuq(12) 始终 NZ (算子定义 nzFormat)
            # cacheMode=2 时 kvCache(19)/kvCacheRope(20)/kvCacheOutOut(25)/krCacheOutOut(27) 也为 NZ
            # (算子定义 nd_nz_Format 中 cacheMode=2 对应 NZ)
            tensor_formats = ['ND'] * 28
            tensor_formats[5] = 'NZ'   # wdqkv
            tensor_formats[12] = 'NZ'  # wuq
            if cacheMode == 2:
                tensor_formats[19] = 'NZ'  # kvCache
                tensor_formats[20] = 'NZ'  # kvCacheRope
                tensor_formats[25] = 'NZ'  # kvCacheOutOut
                tensor_formats[27] = 'NZ'  # krCacheOutOut

            writer.writerow([
                testcase_name,
                'aclnnMlaPreprocess',
                fmt_shapes(tensor_shapes),
                fmt_dtypes(tensor_dtypes),
                '()',  # scalar_dtypes: API has 0 scalars (all attrs go in attributes dict)
                fmt_attrs(attrs),
                fmt_ranges(input_ranges),
                f"({','.join(repr(f) for f in tensor_formats)})",
                '(0.005,0.005)',
                '0.0001',
            ])

            print(f"  case {case_id}: cacheMode={cacheMode}, input={input_shape}, dtype={input_dtype}")

    print(f"Written to {csv_path}")


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/t00954448/code/ops-transformer_0811/attention/mla_preprocess/tests/atk/all_aclnnMlaPreprocess.json'
    csv_path = sys.argv[2] if len(sys.argv) > 2 else \
        '/home/t00954448/code/ops-transformer_0811/attention/mla_preprocess/tests/atk/mla_preprocess_ttk.csv'
    start_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end_id = int(sys.argv[4]) if len(sys.argv) > 4 else 9

    convert_json_to_csv(json_path, csv_path, start_id, end_id)
