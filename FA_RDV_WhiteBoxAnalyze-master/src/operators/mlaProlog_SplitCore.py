import math
from typing import Dict, Any, List, Tuple
from src.core.base_operator import BaseOperator
from src.core.op_registry import OperatorRegistry
from pprint import pprint

@OperatorRegistry.register("mlaProlog")

class mlaPrologOperator(BaseOperator):
    def __init__(self, op_name, chipversion):
        super().__init__(op_name)
        
    def _get_data_type_size(self, dtype: str) -> int:
        """获取数据类型占用的字节数"""
        size_map = {
            "fp16": 2, "bf16": 2, "fp32": 4, "int8": 1,
            "float8_e4m3fn": 1, "float8_e8m0": 1, "int32": 4,
            "int64": 8
        }
        return size_map.get(dtype, 4)
    
    def _calculate_axis_size(self, dtype: str) -> int:
        """计算轴大小：32除以数据类型字节数"""
        byte_size = self._get_data_type_size(dtype)
        return 32 // byte_size if byte_size > 0 else 32
    
    def _safe_get_param(self, params: Dict, key: str) -> Tuple[str, List[int]]:
        """安全获取参数信息，处理字典格式的参数"""
        try:
            value = params.get(key)
            if value is None:
                return '', []
            
            # 处理字典格式的参数：{'dtype': 'xxx', 'shape': [...]}
            if isinstance(value, dict):
                dtype = value.get('dtype', '')
                shape = value.get('shape', [])
                return dtype, shape if isinstance(shape, list) else []
            
            # 处理元组格式的参数（向后兼容）
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                dtype = value[0] if isinstance(value[0], str) else ''
                shape = value[1] if isinstance(value[1], (list, tuple)) else []
                return dtype, list(shape)
            
            return '', []
            
        except Exception as e:
            print(f"警告: 获取参数 {key} 时出错: {e}")
            return '', []
    
    def _safe_get_scalar_param(self, params: Dict, key: str, default=0):
        """安全获取标量参数"""
        try:
            value = params.get(key, default)
            # 如果值是字典，尝试获取其中的值
            if isinstance(value, dict):
                return value.get('value', default)
            return value
        except Exception as e:
            print(f"警告: 获取标量参数 {key} 时出错: {e}")
            return default
    
    def _global_preprocessing(self, params: Dict[str, Any], original_aicNum: int, original_aivNum: int) -> Tuple[int, bool, bool, int, int]:
        """全局预处理：计算量化模式和关键标志位，可能调整核数配置"""
        
        # 获取必要参数
        x_dtype, x_shape = self._safe_get_param(params, 'x')
        w_uq_qr_dtype, w_uq_qr_shape = self._safe_get_param(params, 'w_uq_qr')
        kv_cache_dtype, kv_cache_shape = self._safe_get_param(params, 'kv_cache')
        w_uk_dtype, w_uk_shape = self._safe_get_param(params, 'w_uk')
        
        # 计算quantMode（根据acl_name版本）
        acl_name = self._safe_get_scalar_param(params, 'acl_name', '')
        quantMode = self._calculate_quant_mode(acl_name, x_dtype, w_uq_qr_dtype, kv_cache_dtype, params)
        
        # 计算tSize（复杂逻辑）
        tSize = self._calculate_tSize(params, x_shape)
        
        # 计算nkvSize
        nkvSize = 0
        if kv_cache_shape:
            if len(kv_cache_shape) == 3:
                nkvSize = kv_cache_shape[1] if len(kv_cache_shape) > 1 else 0
            else:
                nkvSize = kv_cache_shape[2] if len(kv_cache_shape) > 2 else 0
        
        # 计算cvRatio
        cvRatio = original_aicNum / original_aivNum if original_aivNum > 0 else 1.0
        
        # 计算enableGroupComputeOpt（可能修改核数）
        enableGroupComputeOpt = False
        processed_aicNum = original_aicNum
        processed_aivNum = original_aivNum
        
        if (quantMode == 1 or quantMode == 2) and tSize == 1 and nkvSize == 8:
            if original_aivNum >= 32 and original_aicNum >= 16 and cvRatio != 1:
                enableGroupComputeOpt = True
                # 动态调整核数
                processed_aicNum = 16
                processed_aivNum = 32
        
        # 计算enableDequantOpt（如果enableGroupComputeOpt为false）
        enableDequantOpt = False
        if not enableGroupComputeOpt:
            nSize = w_uk_shape[0] if w_uk_shape and len(w_uk_shape) > 0 else 0
            if (w_uq_qr_dtype == 'int8' and nSize >= 8) or w_uq_qr_dtype == 'float8_e4m3fn':
                enableDequantOpt = True
        
        return quantMode, enableDequantOpt, enableGroupComputeOpt, processed_aicNum, processed_aivNum

    def _calculate_quant_mode(self, acl_name: str, x_dtype: str, w_uq_qr_dtype: str, 
                            kv_cache_dtype: str, params: Dict[str, Any]) -> int:
        """计算量化模式"""
        if "V3WeightNz" in acl_name:
            # V3版本的量化模式计算
            weight_quant_mode = self._safe_get_scalar_param(params, 'weight_quant_mode', 0)
            kv_quant_mode = self._safe_get_scalar_param(params, 'kv_quant_mode', 0)
            
            if weight_quant_mode == 0 and kv_quant_mode == 0:
                return 0
            elif weight_quant_mode == 1 and kv_quant_mode == 0:
                return 1
            elif weight_quant_mode == 1 and kv_quant_mode == 2:
                return 2
            elif weight_quant_mode == 1 and kv_quant_mode == 3:
                return 5
            elif weight_quant_mode == 2 and kv_quant_mode == 0:
                return 3
            elif weight_quant_mode == 2 and kv_quant_mode == 1:
                return 4
            elif weight_quant_mode == 2 and kv_quant_mode == 3:
                return 6
            else:
                return 0
        else:
            # V2版本的量化模式计算
            if x_dtype == 'int8' and kv_cache_dtype == 'int8':
                return 4
            elif x_dtype == 'int8' and kv_cache_dtype != 'int8':
                return 3
            elif x_dtype == 'bf16' and w_uq_qr_dtype == 'int8' and kv_cache_dtype == 'int8':
                return 2
            elif x_dtype == 'bf16' and w_uq_qr_dtype == 'int8' and kv_cache_dtype != 'int8':
                return 1
            else:
                return 0

    def _calculate_tSize(self, params: Dict[str, Any], x_shape: List[int]) -> int:
        """计算tSize（包含复杂的条件判断）"""
        if not x_shape:
            return 0
        
        if len(x_shape) == 3:
            # 3D形状: [batch, sequence, hidden_size]
            return x_shape[0] * x_shape[1] if len(x_shape) > 1 else x_shape[0]
        else:
            # 其他情况
            return x_shape[0] if x_shape else 0

    def _calculate_split_pattern(self, total_size: int, base_size: int, core_num: int, single_core_size: int) -> int:
        """计算切分模式：2=仅整块, 1=仅尾块, 0=混合"""
        if core_num <= 0 or total_size <= 0:
            return 2
            
        # 计算每个核的大小
        core_sizes = []
        for i in range(core_num):
            if i == core_num - 1:  # 最后一个核
                core_size = total_size - (core_num - 1) * single_core_size
            else:
                core_size = single_core_size
            core_sizes.append(core_size)
        
        # 检查所有核的切分情况
        all_whole = True
        all_tail = True
        
        for core_size in core_sizes:
            if base_size == 0:
                all_whole = False
                all_tail = False
                break
                
            if core_size % base_size == 0:  # 整块
                all_tail = False
            else:  # 尾块
                all_whole = False
                
        if all_whole:
            return 2
        elif all_tail:
            return 1
        else:
            return 0
    
    def _calculate_l0_split_pattern(self, l1_base_size: int, l1_tail_size: int, l0_base_size: int) -> int:
        """
        计算L0切分模式，同时考虑L1整块和L1尾块被L0基准切分的情况[1,4](@ref)
        
        参数:
            l1_base_size: L1切分的整块大小
            l1_tail_size: L1切分的尾块大小（可能等于l1_base_size，也可能更小）
            l0_base_size: L0切分的基准大小
        
        返回:
            切分模式编码 (2: 仅整块, 1: 仅尾块, 0: 混合)
        """
        if l0_base_size == 0:
            return 2

        # 计算L1整块被L0切分的情况
        if l1_base_size % l0_base_size == 0:
            l1_whole_pattern = 2  # L1整块被L0切分后全是整块
        elif l1_base_size // l0_base_size == 0:
            l1_whole_pattern = 1  # L1整块被L0切分后全是尾块
        else:
            l1_whole_pattern = 0  # L1整块被L0切分后混合
        
        # 计算L1尾块被L0切分的情况（如果存在尾块且与整块不同）
        if l1_tail_size == l1_base_size:
            # 没有尾块或尾块大小等于整块，只基于整块判断
            return l1_whole_pattern
        else:
            # 分别计算尾块的切分模式
            if l1_tail_size % l0_base_size == 0:
                l1_tail_pattern = 2
            elif l1_tail_size // l0_base_size == 0:
                l1_tail_pattern = 1
            else:
                l1_tail_pattern = 0
            
            # 综合判断：如果整块和尾块的L0切分模式不同，则为混合
            if l1_whole_pattern == l1_tail_pattern:
                return l1_whole_pattern
            else:
                return 0
    
    def _calculate_matmul1_split(self, params: Dict[str, Any], aicNum_: int) -> Tuple[str, str]:
        """计算第一次matmul的切分结果"""
        # 参数获取和n轴计算逻辑保持不变
        w_dq_dtype, w_dq_shape = self._safe_get_param(params, 'w_dq')
        x_dtype, x_shape = self._safe_get_param(params, 'x')
        
        w_dq_axis_size = self._calculate_axis_size(w_dq_dtype)
        
        # 计算hcqSize
        if not w_dq_shape:
            hcqSize = 0
        elif len(w_dq_shape) == 2:
            hcqSize = w_dq_shape[1] if len(w_dq_shape) > 1 else 0
        else:
            hcqSize = (w_dq_shape[0] if w_dq_shape else 0) * w_dq_axis_size
        
        # 计算singlecoreHeadSizeCq和mm1blockNum
        if hcqSize == 0 or aicNum_ == 0:
            singlecoreHeadSizeCq = 64
            mm1blockNum = 1
        else:
            temp_size = math.ceil(hcqSize / (aicNum_ * w_dq_axis_size)) * w_dq_axis_size
            singlecoreHeadSizeCq = max(temp_size, 64)
            mm1blockNum = math.ceil(hcqSize / singlecoreHeadSizeCq)
        
        # 计算heSize
        if not x_shape:
            heSize = 0
        elif len(x_shape) == 3:
            heSize = x_shape[2] if len(x_shape) > 2 else 0
        else:
            heSize = x_shape[1] if len(x_shape) > 1 else 0
        
        # L1切分参数
        nL1Base = 128
        
        # 根据x_dtype确定oriKBase
        if x_dtype in ['int8', 'float8_e4m3fn', 'float8_e8m0']:
            oriKBase = 256
        else:
            oriKBase = 128
            
        kstep = 4
        if heSize > 0 and oriKBase > 0 and (heSize // oriKBase) % 4 != 0:
            kstep = 3
        kL1Base = oriKBase * kstep
        
        # L0切分参数
        nL0Base = nL1Base
        kL0Base = oriKBase
        
        # 计算n轴切分模式
        m_l1 = 2  # m轴不切分
        n_l1 = self._calculate_split_pattern(hcqSize, nL1Base, mm1blockNum, singlecoreHeadSizeCq)
        
        # 修复：计算n轴L0切分模式
        l1_tail_size = hcqSize - (mm1blockNum - 1) * singlecoreHeadSizeCq if mm1blockNum > 1 else singlecoreHeadSizeCq
        n_l0 = self._calculate_l0_split_pattern(singlecoreHeadSizeCq, l1_tail_size, nL0Base)
        
        # 修复：k轴切分逻辑 - k轴在核间不切分，每个核都处理完整的heSize
        # k轴L1切分：总长度heSize，只有1个核，每个核大小heSize
        k_l1 = self._calculate_split_pattern(heSize, kL1Base, 1, heSize)
        
        # k轴L0切分：基于kL1Base进行L0切分
        k_l1_tail_size = heSize % kL1Base if heSize % kL1Base != 0 else kL1Base
        k_l0 = self._calculate_l0_split_pattern(kL1Base, k_l1_tail_size, kL0Base)
        
        m_l0 = 2  # m轴不切分
        
        l1_result = f"{m_l1}{n_l1}{k_l1}"
        l0_result = f"{m_l0}{n_l0}{k_l0}"
        
        return l1_result, l0_result

    def _calculate_matmul2_split(self, params: Dict[str, Any], aicNum_: int) -> Tuple[str, str]:
        """计算第二次matmul的切分结果"""
        # 参数获取和n轴计算逻辑保持不变
        w_dkv_kr_dtype, w_dkv_kr_shape = self._safe_get_param(params, 'w_dkv_kr')
        cos_dtype, cos_shape = self._safe_get_param(params, 'cos')
        x_dtype, x_shape = self._safe_get_param(params, 'x')
        
        w_dkv_kr_axis_size = self._calculate_axis_size(w_dkv_kr_dtype)
        drSize = cos_shape[-1] if cos_shape and len(cos_shape) > 0 else 0
        
        if not w_dkv_kr_shape:
            hckvSize = 0
        elif len(w_dkv_kr_shape) == 4:
            hckvSize = w_dkv_kr_shape[0] * w_dkv_kr_shape[3] - drSize
        else:
            hckvSize = w_dkv_kr_shape[1] - drSize if len(w_dkv_kr_shape) > 1 else 0
        
        total_size = hckvSize + drSize
        
        if aicNum_ >= 9:
            singlecoreHeadSizeCkvKr = 64
            mm2blockNum = math.ceil(total_size / 64) if total_size > 0 else 1
        else:
            if total_size == 0 or aicNum_ == 0:
                singlecoreHeadSizeCkvKr = 64
            else:
                temp_size = math.ceil(total_size / (aicNum_ * w_dkv_kr_axis_size)) * w_dkv_kr_axis_size
                singlecoreHeadSizeCkvKr = temp_size
            mm2blockNum = math.ceil(total_size / singlecoreHeadSizeCkvKr) if total_size > 0 else 1
        
        # 计算heSize
        if not x_shape:
            heSize = 0
        elif len(x_shape) == 3:
            heSize = x_shape[2] if len(x_shape) > 2 else 0
        else:
            heSize = x_shape[1] if len(x_shape) > 1 else 0
        
        # L1切分参数
        nL1Base = 128
        if x_dtype in ['int8', 'float8_e4m3fn', 'float8_e8m0']:
            oriKBase = 256
        else:
            oriKBase = 128
            
        kstep = 4
        if heSize > 0 and oriKBase > 0 and (heSize // oriKBase) % 4 != 0:
            kstep = 3
        kL1Base = oriKBase * kstep
        
        # L0切分参数
        nL0Base = nL1Base
        kL0Base = oriKBase
        
        # 计算n轴切分模式
        m_l1 = 2  # m轴不切分
        n_l1 = self._calculate_split_pattern(total_size, nL1Base, mm2blockNum, singlecoreHeadSizeCkvKr)
        
        # 修复：计算n轴L0切分模式
        l1_tail_size = total_size - (mm2blockNum - 1) * singlecoreHeadSizeCkvKr if mm2blockNum > 1 else singlecoreHeadSizeCkvKr
        n_l0 = self._calculate_l0_split_pattern(singlecoreHeadSizeCkvKr, l1_tail_size, nL0Base)
        
        # 修复：k轴切分逻辑 - k轴在核间不切分，每个核都处理完整的heSize
        k_l1 = self._calculate_split_pattern(heSize, kL1Base, 1, heSize)
        
        # k轴L0切分：基于kL1Base进行L0切分
        k_l1_tail_size = heSize % kL1Base if heSize % kL1Base != 0 else kL1Base
        k_l0 = self._calculate_l0_split_pattern(kL1Base, k_l1_tail_size, kL0Base)
        
        m_l0 = 2  # m轴不切分
        
        l1_result = f"{m_l1}{n_l1}{k_l1}"
        l0_result = f"{m_l0}{n_l0}{k_l0}"
        
        return l1_result, l0_result

    def _calculate_matmul3_split(self, params: Dict[str, Any], aicNum_: int, aivNum_: int,
                                enableDequantOpt: bool, enableGroupComputeOpt: bool) -> Tuple[str, str]:
                # 参数获取逻辑保持不变
        w_uq_qr_dtype, w_uq_qr_shape = self._safe_get_param(params, 'w_uq_qr')
        w_uk_dtype, w_uk_shape = self._safe_get_param(params, 'w_uk')
        x_dtype, x_shape = self._safe_get_param(params, 'x')
        w_dq_dtype, w_dq_shape = self._safe_get_param(params, 'w_dq')
        cos_dtype, cos_shape = self._safe_get_param(params, 'cos')
        
        # 使用传入的预处理标志位，不再重复计算
        # enableDequantOpt和enableGroupComputeOpt现在作为参数传入
        
        # 计算基础参数
        w_uq_qr_axis_size = self._calculate_axis_size(w_uq_qr_dtype)
        dSize = w_uk_shape[1] if w_uk_shape and len(w_uk_shape) > 1 else 0
        nSize = w_uk_shape[0] if w_uk_shape and len(w_uk_shape) > 0 else 0
        drSize = cos_shape[-1] if cos_shape and len(cos_shape) > 0 else 0
        
        # 计算oriM = nSize * (dSize + drSize)
        oriM = nSize * (dSize + drSize)

        w_dq_axis_size = self._calculate_axis_size(w_dq_dtype)
        if not w_dq_shape:
            hcqSize = 0
        elif len(w_dq_shape) == 2:
            hcqSize = w_dq_shape[1] if len(w_dq_shape) > 1 else 0
        else:
            hcqSize = w_dq_shape[0] * w_dq_axis_size if w_dq_shape else 0
        
        # 计算singlecoreHeadSizeQcQr（基于传入的标志位）
        if enableGroupComputeOpt:
            singlecoreHeadSizeQcQr = math.ceil(nSize * dSize / (8 * dSize)) * dSize if dSize > 0 else 0
        elif enableDequantOpt:
            singlecoreHeadSizeQcQr = math.ceil(oriM / (aicNum_ * (dSize + drSize))) * (dSize + drSize) if aicNum_ > 0 else 0
        else:
            singlecoreHeadSizeQcQr = math.ceil(oriM / (aicNum_ * w_uq_qr_axis_size)) * w_uq_qr_axis_size if aicNum_ > 0 else 0
        
        # 计算mm3blockNum
        mm3blockNum = math.ceil(oriM / singlecoreHeadSizeQcQr) if oriM > 0 and singlecoreHeadSizeQcQr > 0 else 1
        
        # L1切分参数
        tSize = self._calculate_tSize(params, x_shape)
        stepBatchSize_ = min(128, tSize)
        if enableGroupComputeOpt:
            nL1Base = 128
        else:
            if x_dtype == 'float8_e4m3fn':
                nL1Base = 128
            else:
                nL1Base = 256 if stepBatchSize_ <= 64 else 128
        
        if x_dtype in ['int8', 'float8_e4m3fn', 'float8_e8m0']:
            oriKBase = 128
        else:
            oriKBase = 64
            
        kstep = 4
        kL1Base = oriKBase * kstep
        kL0Base = oriKBase
        
        # 计算n轴切分模式
        m_l1 = 2  # m轴不切分
        n_l1 = self._calculate_split_pattern(oriM, nL1Base, mm3blockNum, singlecoreHeadSizeQcQr)
        
        # 修复6: 计算n轴L0切分模式
        l1_tail_size = oriM - (mm3blockNum - 1) * singlecoreHeadSizeQcQr if mm3blockNum > 1 else singlecoreHeadSizeQcQr
        n_l0 = self._calculate_l0_split_pattern(singlecoreHeadSizeQcQr, l1_tail_size, nL1Base)
        
        # 修复7: k轴切分逻辑 - k轴在核间不切分，每个核都处理完整的hcqSize
        k_l1 = self._calculate_split_pattern(hcqSize, kL1Base, 1, hcqSize)
        
        # k轴L0切分：基于kL1Base进行L0切分
        k_l1_tail_size = hcqSize % kL1Base if hcqSize % kL1Base != 0 else kL1Base
        k_l0 = self._calculate_l0_split_pattern(kL1Base, k_l1_tail_size, kL0Base)
        
        m_l0 = 2  # m轴不切分
        
        l1_result = f"{m_l1}{n_l1}{k_l1}"
        l0_result = f"{m_l0}{n_l0}{k_l0}"
             
        return l1_result, l0_result

    def _calculate_matmul4_split(self, params: Dict[str, Any], aicNum_: int) -> Tuple[str, str]:
        """计算第四次matmul的切分结果"""
        # 参数获取逻辑保持不变
        w_uk_dtype, w_uk_shape = self._safe_get_param(params, 'w_uk')
        w_dkv_kr_dtype, w_dkv_kr_shape = self._safe_get_param(params, 'w_dkv_kr')
        cos_dtype, cos_shape = self._safe_get_param(params, 'cos')
        
        nSize = w_uk_shape[0] if w_uk_shape and len(w_uk_shape) > 0 else 0
        dSize = w_uk_shape[1] if w_uk_shape and len(w_uk_shape) > 1 else 0
        drSize = cos_shape[-1] if cos_shape and len(cos_shape) > 0 else 0
        
        if not w_dkv_kr_shape:
            hckvSize = 0
        elif len(w_dkv_kr_shape) == 4:
            hckvSize = w_dkv_kr_shape[0] * w_dkv_kr_shape[3] - drSize
        else:
            hckvSize = w_dkv_kr_shape[1] - drSize if len(w_dkv_kr_shape) > 1 else 0
        
        singlecoreNumHeadSize = math.ceil(nSize / aicNum_) if nSize > 0 and aicNum_ > 0 else 1
        mm4BlockNum = math.ceil(nSize / singlecoreNumHeadSize) if nSize > 0 else 1
        
        # Matmul4的特殊处理保持不变：n轴在L1阶段不切分，k轴不切分
        m_l1 = 2  # m轴不切分
        n_l1 = 2  # n轴不切分
        k_l1 = 2  # k轴不切分
        
        # 修复：L0阶段n轴基于hckvSize切分
        nL0Base = 128
        n_l0 = self._calculate_split_pattern(hckvSize, nL0Base, 1, hckvSize)
        
        m_l0 = 2  # m轴不切分
        k_l0 = 2  # k轴不切分
        
        l1_result = f"{m_l1}{n_l1}{k_l1}"
        l0_result = f"{m_l0}{n_l0}{k_l0}"
        
        return l1_result, l0_result

    def _check_m_axis_condition(self, params: Dict[str, Any]):
        """
        检查输入x的M轴维度是否大于128，返回1（是）或0（否）
        
        返回：
            int: 1（M轴>128）/ 0（M轴≤128）
        """
        # x_shape[0]：dtype
        # x_shape[1]：[24, 7168]
        x_shape = self._safe_get_param(params, 'x')
        shape_dim = len(x_shape[1])
        # 仅处理token_x为2维/3维场景，其他维度数默认返回0
        if shape_dim == 2:
            # 2维：M轴大小 = 第一维值
            m_axis_value = x_shape[1][0]
        elif shape_dim == 3:
            # 3维：M轴大小 = 前两维乘积
            m_axis_value = x_shape[1][0] * x_shape[1][1]
        else:
            print(f"输入x的shape维度数不支持（当前{shape_dim}维，仅支持2/3维）")
            return -1

        # 确保M轴值为整数（避免非数值类型导致判断错误）
        try:
            m_axis_value = int(m_axis_value)
        except (ValueError, TypeError):
            print(f"M轴值非整数类型，无法判断（值：{m_axis_value}）")
            return -1

        # 判断是否大于128，返回1/0
        return 1 if m_axis_value > 128 else 0

    def _check_vector_core_split(self, params: Dict[str, Any], aivNum_: int, enableGroupComputeOpt: bool):
        """
        判断vector核间切分是否均匀
        规则：
        1. aivNum=64(950芯片)：M轴为64的倍数 或 <64 → 均匀；M轴>64且非64倍数 → 不均匀
        2. aivNum=48且enableGroupComputeOpt=False：M轴为48的倍数 或 <48 → 均匀；否则不均匀
        3. aivNum=48且enableGroupComputeOpt=True：M轴为32的倍数 或 <32 → 均匀；否则不均匀

        返回：
            bool: True（分核均匀）/ False（分核不均匀）
        """
        # ========== 第一步：计算M轴大小 ==========
        x_shape = self._safe_get_param(params, 'x')
        shape_dim = len(x_shape[1])
        # 仅处理token_x为2维/3维场景，其他维度数默认返回0
        if shape_dim == 2:
            # 2维：M轴大小 = 第一维值
            m_axis_value = x_shape[1][0]
        elif shape_dim == 3:
            # 3维：M轴大小 = 前两维乘积
            m_axis_value = x_shape[1][0] * x_shape[1][1]
        else:
            print(f"输入x的shape维度数不支持（当前{shape_dim}维，仅支持2/3维）")
            raise RuntimeError("获取x的shape失败")

        # ========== 第二步：根据规则判断切分是否均匀 ==========
        # 规则1：aivNum=64 (950芯片)
        if aivNum_ == 64:
            threshold = 64
            if m_axis_value < threshold or (m_axis_value % threshold == 0):
                return True
            else:
                return False

        # 规则2：aivNum=48
        elif aivNum_ == 48:
            # 子规则2.1：enableGroupComputeOpt=True → 阈值32
            if enableGroupComputeOpt:
                threshold = 32
            # 子规则2.2：enableGroupComputeOpt=False → 阈值48
            else:
                threshold = 48

            if m_axis_value < threshold or (m_axis_value % threshold == 0):
                return True
            else:
                return False

        # 其他aivNum值
        else:
            print(f"不支持的AIV核数：{aivNum_}（仅支持64/48）")
            return False
    
    def simulate_whitebox_logic(self, aic_num: int, aiv_num: int, input_params: Dict) -> Dict[str, Any]:
        """模拟C++算子内部的白盒逻辑 - 修复全局预处理架构"""
        
        # 第一步：提取基础参数
        aicNum_ = aic_num
        aivNum_ = aiv_num
        
        # 第二步：全局预处理 - 计算量化模式和关键标志位
        quantMode, enableDequantOpt, enableGroupComputeOpt, processed_aicNum, processed_aivNum = self._global_preprocessing(input_params, aicNum_, aivNum_)
        
        # 第三步：使用处理后的参数进行四次matmul计算
        matmul1_l1, matmul1_l0 = self._calculate_matmul1_split(input_params, processed_aicNum)
        matmul2_l1, matmul2_l0 = self._calculate_matmul2_split(input_params, processed_aicNum)
        matmul3_l1, matmul3_l0 = self._calculate_matmul3_split(input_params, processed_aicNum, processed_aivNum, 
                                                            enableDequantOpt, enableGroupComputeOpt)
        matmul4_l1, matmul4_l0 = self._calculate_matmul4_split(input_params, processed_aicNum)

        # 第四步：判断M轴是否大于128
        m_axis_gt_128 = self._check_m_axis_condition(input_params)

        # 第五步：判断vector核切分情况
        vector_core_split = self._check_vector_core_split(input_params, processed_aivNum, enableGroupComputeOpt)

        
        # 构建返回结果
        dimensions_state = {
            'matmul1': f"{matmul1_l1}{matmul1_l0}",
            'matmul2': f"{matmul2_l1}{matmul2_l0}", 
            'matmul3': f"{matmul3_l1}{matmul3_l0}",
            'matmul4': f"{matmul4_l1}{matmul4_l0}",
            'cache_mode': self._safe_get_scalar_param(input_params, 'cache_mode', 'PA_BSND'),
            'quant_mode': quantMode,
            'enableDequantOpt': enableDequantOpt,
            'enableGroupComputeOpt': enableGroupComputeOpt,
            'processed_aicNum': processed_aicNum,
            'processed_aivNum': processed_aivNum,
            'quantization_info': {
                'x_dtype': self._safe_get_param(input_params, 'x')[0],
                'w_uq_qr_dtype': self._safe_get_param(input_params, 'w_uq_qr')[0],
                'out_q_dtype': self._safe_get_param(input_params, 'out_q')[0],
                'kv_cache_dtype': self._safe_get_param(input_params, 'kv_cache')[0],
                'kr_cache_dtype': self._safe_get_param(input_params, 'kr_cache')[0]
            },
            'm_axis_gt_128': m_axis_gt_128,
            'vector_core_split': vector_core_split
        }

        print("=== Dimensions State ===")
        pprint(dimensions_state, indent=2, width=100, depth=3, sort_dicts=False)
        
        return dimensions_state