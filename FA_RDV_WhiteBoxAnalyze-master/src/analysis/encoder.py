from typing import Dict, Any

class MlaPrologCaseEncoder:
    """专门用于mlaProlog算子的用例编码器"""
    
    @staticmethod
    def encode_mla_prolog(dimensions_state: Dict[str, Any]) -> str:
        """将mlaProlog白盒逻辑分析结果编码为32位字符串"""
        encoded_parts = []
        
        # 1. 编码cache_mode (1位)
        cache_mode_map = {
            "PA_BSND": "0", "PA_NZ": "1", "PA_BLK_BSND": "2",
            "PA_BLK_NZ": "3", "BSND": "4", "TND": "5"
        }
        cache_mode = dimensions_state.get('cache_mode', '')
        encoded_parts.append(cache_mode_map.get(cache_mode, "0"))
        
        # 2. 编码量化模式 (5位)
        quant_info = dimensions_state.get('quantization_info', {})
        dtype_map = {
            "int8": "0", "fp8": "1", "int16": "2", 
            "bf16": "3", "int32": "4", "fp32": "5"
        }
        
        x_dtype = quant_info.get('x_dtype', '')
        w_uq_qr_dtype = quant_info.get('w_uq_qr_dtype', '')
        out_q_dtype = quant_info.get('out_q_dtype', '')
        kv_cache_dtype = quant_info.get('kv_cache_dtype', '')
        kr_cache_dtype = quant_info.get('kr_cache_dtype', '')
        
        encoded_parts.append(dtype_map.get(x_dtype, "0"))
        encoded_parts.append(dtype_map.get(w_uq_qr_dtype, "0"))
        encoded_parts.append(dtype_map.get(out_q_dtype, "0"))
        encoded_parts.append(dtype_map.get(kv_cache_dtype, "0"))
        encoded_parts.append(dtype_map.get(kr_cache_dtype, "0"))
        
        # 3. 编码dequantOpt标志位 (1位)
        enableDequantOpt = dimensions_state.get('enableDequantOpt', False)
        encoded_parts.append("1" if enableDequantOpt else "0")
        
        # 4. 编码GroupCompute标志位 (1位)
        enableGroupComputeOpt = dimensions_state.get('enableGroupComputeOpt', False)
        encoded_parts.append("1" if enableGroupComputeOpt else "0")
        
        # 5. 编码四次matmul结果 (24位)
        encoded_parts.append(dimensions_state.get('matmul1', '222222'))
        encoded_parts.append(dimensions_state.get('matmul2', '222222'))
        encoded_parts.append(dimensions_state.get('matmul3', '222222'))
        encoded_parts.append(dimensions_state.get('matmul4', '222222'))

        # 6.切M结果 (1位)
        encoded_parts.append(dimensions_state.get('split_M', 0))

        # 7.vector分核结果(1位)
        encoded_parts.append(dimensions_state.get('vector_core_split', False))

        # 合并所有编码部分
        full_code = ''.join(encoded_parts)
        
        # 确保编码长度为34位
        if len(full_code) != 34:
            # 填充或截断到34位
            full_code = full_code.ljust(34, '0')[:34]
        
        # 打印生成的完整编码
        print(f"mlaProlog用例编码结果: {full_code}")
        print(f"编码结构分析:")
        print(f"  cache_mode: {full_code[0]} ({cache_mode})")
        print(f"  量化模式: {full_code[1:6]} (x:{x_dtype}, w_uq_qr:{w_uq_qr_dtype}, out_q:{out_q_dtype}, kv_cache:{kv_cache_dtype}, kr_cache:{kr_cache_dtype})")
        print(f"  dequantOpt: {full_code[6]} ({enableDequantOpt})")
        print(f"  GroupComputeOpt: {full_code[7]} ({enableGroupComputeOpt})")
        print(f"  matmul1切分: {full_code[8:14]}")
        print(f"  matmul2切分: {full_code[14:20]}")
        print(f"  matmul3切分: {full_code[20:26]}")
        print(f"  matmul4切分: {full_code[26:32]}")
        print(f"  M轴切分: {full_code[32:33]}")
        print(f"  vector分核: {full_code[33:34]}")
        
        return full_code
    
    @staticmethod
    def decode_mla_prolog_case(encoded_str: str, case_name: str) -> Dict[str, Any]:
        """从编码字符串解码mlaProlog用例信息"""
        if len(encoded_str) != 32:
            raise ValueError("mlaProlog编码字符串长度必须为32位")
        
        # 反向映射
        cache_mode_reverse = {
            "0": "PA_BSND", "1": "PA_NZ", "2": "PA_BLK_BSND",
            "3": "PA_BLK_NZ", "4": "BSND", "5": "TND"
        }
        
        dtype_reverse = {
            "0": "int8", "1": "fp8", "2": "int16",
            "3": "bf16", "4": "int32", "5": "fp32"
        }
        
        decoded_info = {
            'case_name': case_name,
            'cache_mode': cache_mode_reverse.get(encoded_str[0], "PA_BSND"),
            'x_dtype': dtype_reverse.get(encoded_str[1], "fp32"),
            'w_uq_qr_dtype': dtype_reverse.get(encoded_str[2], "fp32"),
            'out_q_dtype': dtype_reverse.get(encoded_str[3], "fp32"),
            'kv_cache_dtype': dtype_reverse.get(encoded_str[4], "fp32"),
            'kr_cache_dtype': dtype_reverse.get(encoded_str[5], "fp32"),
            'dequantOpt': encoded_str[6] == "1",
            'GroupComputeOpt': encoded_str[7] == "1",
            'matmul1': encoded_str[8:14],
            'matmul2': encoded_str[14:20],
            'matmul3': encoded_str[20:26],
            'matmul4': encoded_str[26:32],
            'split_M': encoded_str[32:33],
            'vector_core_split': encoded_str[33:34],
            'full_code': encoded_str
        }
        
        # 打印解码信息
        print(f"mlaProlog用例解码结果 - {case_name}:")
        print(f"  完整编码: {encoded_str}")
        print(f"  缓存模式: {decoded_info['cache_mode']}")
        print(f"  数据类型: x={decoded_info['x_dtype']}, w_uq_qr={decoded_info['w_uq_qr_dtype']}, out_q={decoded_info['out_q_dtype']}")
        print(f"  优化标志: dequantOpt={decoded_info['dequantOpt']}, GroupComputeOpt={decoded_info['GroupComputeOpt']}")
        print(f"  切分模式: matmul1={decoded_info['matmul1']}, matmul2={decoded_info['matmul2']}, matmul3={decoded_info['matmul3']}, matmul4={decoded_info['matmul4']}")
        print(f"  切M模式: split_M={decoded_info['split_M']}")
        print(f"  vector分核模式: vector_core_split={decoded_info['vector_core_split']}")
        
        return decoded_info