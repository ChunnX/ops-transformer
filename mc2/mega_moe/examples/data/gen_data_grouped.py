import os
import sys
import math
import torch
import numpy as np

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 检查FP8支持
has_fp8 = False
try:
    if hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e8m0fnu'):
        has_fp8 = True
        print("FP8 support detected")
    else:
        print("FP8 not supported, using float32 instead")
except Exception as e:
    print(f"Error checking FP8 support: {e}")
    print("Using float32 instead")

# 生成FP8格式数据
def gen_data_fp8(m, k, trans, data_format='E4M3'):
    a, b = 0, 0.002  # 目标范围
    # 生成 [a, b] 范围内的均匀分布
    matrix = a + (b - a) * torch.rand(m, k, dtype=torch.float32)
    # matrix = torch.randn((m, k), dtype=torch.float32)

    quantizer = MXFP8MatrixQuantizer(
        data_format=data_format,
        axis=1,
        block_size=32
    )

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 转换为FP8类型
    if has_fp8:
        if data_format == 'E4M3':
            quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
        else:  # E5M2
            quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
        scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)
        scale_matrix = torch.full_like(scale_matrix, 2**-18)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(
        quantized_matrix.to(torch.float32),
        scale_matrix.to(torch.float32)
    )

    if trans:
        quantized_matrix = quantized_matrix.t()
        dequantized_matrix = dequantized_matrix.t()

    return quantized_matrix, scale_matrix, dequantized_matrix 

# MXFP8矩阵量化器类
class MXFP8MatrixQuantizer:
    def __init__(self, data_format='E4M3', axis=1, block_size=32, epsilon=1e-12):
        self.data_format = data_format
        self.axis = axis
        self.block_size = block_size
        self.epsilon = epsilon
        
        # MXFP8 数据格式定义
        self.DATA_FORMATS = {
            'E4M3': {
                'exp_bits': 4,
                'mantissa_bits': 3,
                'bias': 7,
                'emax': 8,
                'max_value': 448.0,
                'min_value': -448.0
            },
            'E5M2': {
                'exp_bits': 5,
                'mantissa_bits': 2,
                'bias': 15,
                'emax': 15,
                'max_value': 57344.0,
                'min_value': -57344.0
            }
        }
        
        # FP8_E8M0FNU 缩放格式定义
        self.SCALE_FORMAT = {
            'name': 'FP8_E8M0FNU',
            'exp_bits': 8,
            'mantissa_bits': 0,
            'bias': 128,  # 偏置为128，0表示指数-128
            'max_exp': 127,
            'min_exp': -128,
            'max_value': 2**127,   # 约 1.7e38
            'min_value': 2**-128,  # 约 2.9e-39
            'signed': False,       # 无符号，仅正数
            'allow_zero': True     # 允许零值（指数-128表示零）
        }
        
        self.config = self.DATA_FORMATS[data_format]
        
        # 生成FP8查找表
        self.fp8_lut = self._generate_fp8_lut()
        # 预排序查找表，避免每次量化都排序
        self.sorted_fp8_lut, _ = torch.sort(self.fp8_lut)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']
    
    def _build_e4m3_lookup_table(self):
        """构建 E4M3 值查找表"""
        values = []

        # E4M3: 总共 256 个可能值 (0-255)
        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val == 127:  # NaN，替换为最大值
                value = sign * self.config['max_value']
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _build_e5m2_lookup_table(self):
        """构建 E5M2 值查找表"""
        values = []

        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val >= 124 and val <= 127:  # NaN/Inf，替换为最大值
                value = sign * self.config['max_value']
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _generate_fp8_lut(self):
        """生成FP8查找表"""
        if self.data_format == 'E4M3':
            self._build_e4m3_lookup_table()
        else:
            self._build_e5m2_lookup_table()
        return self.fp8_lut
    
    def _quantize_to_fp8(self, data: torch.Tensor) -> torch.Tensor:
        """
        量化数据到 FP8 格式

        使用查找表进行最近邻量化（性能优化版本）
        """
        original_shape = data.shape
        data_flat = data.flatten()
        data_clamped = torch.clamp(data_flat, self.fp8_min, self.fp8_max)
        
        # 向量化实现：一次性处理所有数据点
        # 使用 searchsorted 找到每个值的插入位置
        idx = torch.searchsorted(self.sorted_fp8_lut, data_clamped)
        
        # 处理边界情况
        idx = torch.clamp(idx, 0, len(self.sorted_fp8_lut))
        
        # 创建索引张量
        idx_prev = torch.clamp(idx - 1, 0, len(self.sorted_fp8_lut) - 1)
        idx_curr = torch.clamp(idx, 0, len(self.sorted_fp8_lut) - 1)
        
        # 计算距离
        dist_prev = torch.abs(self.sorted_fp8_lut[idx_prev] - data_clamped)
        dist_curr = torch.abs(self.sorted_fp8_lut[idx_curr] - data_clamped)
        
        # 选择距离较小的索引
        closest_idx = torch.where(dist_prev <= dist_curr, idx_prev, idx_curr)
        
        # 获取量化后的值
        quantized_flat = self.sorted_fp8_lut[closest_idx]
        
        return quantized_flat.view(original_shape)
    
    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> tuple:
        """
        计算 FP8_E8M0FNU 格式的缩放因子

        返回:
            scale: 缩放因子 (浮点数)
            exp: 指数 (E8M0FNU 格式)
        """
        # 计算块的最大绝对值
        max_abs = torch.max(torch.abs(block_data)).item()

        if max_abs < self.epsilon:
            # 全零或接近零的块
            return 1.0, 0  # 指数0表示缩放因子2^0=1

        # 计算对数 (log2)
        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128  # 最小值

        # 四舍五入到最近的整数 (指数)
        exp = int(math.floor(log2_scale)) - self.config['emax']

        # 钳位到 E8M0FNU 范围 [-128, 127]
        exp = max(self.SCALE_FORMAT['min_exp'],
                    min(exp, self.SCALE_FORMAT['max_exp']))

        # 计算缩放因子
        scale = 2.0 ** exp

        return scale, exp

    def _process_block(self, block_data: torch.Tensor) -> tuple:
        """
        处理单个分块

        返回:
            quantized_block: 量化后的块
            scale: 缩放因子
            exp: 指数
        """
        # 计算缩放因子
        scale, exp = self._compute_scale_fp8_e8m0fnu(block_data)

        # 应用缩放
        if abs(scale) > 1e-12:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data  # 缩放因子为0，不缩放

        # 量化到 FP8
        quantized_scaled = self._quantize_to_fp8(scaled_data)

        return quantized_scaled, scale, exp

    def _quantize_by_cols(self, matrix: torch.Tensor, M: int, N: int) -> tuple:
        """
        按列方向量化（部分向量化优化）

        分块: 每 block_size 列一个块
        """
        # 计算分块数量
        num_blocks = (N + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)

        # 处理每个列块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束列
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)
            block_width = end_col - start_col

            # 提取当前块 (M x block_width)
            block_data = matrix[:, start_col:end_col]

            # 逐行计算 scale
            scales = torch.zeros(M, dtype=torch.float32)
            for row in range(M):
                row_data = block_data[row, :]
                scale, exp = self._compute_scale_fp8_e8m0fnu(row_data)
                scales[row] = scale

            # 向量化应用 scale 和量化
            # 将 scale 扩展为 (M, block_width) 用于广播
            scales_expanded = scales.unsqueeze(1).expand(M, block_width)

            # 避免除零
            scales_expanded = torch.where(scales_expanded > 1e-12, scales_expanded, torch.ones_like(scales_expanded))

            # 应用缩放
            scaled_block = block_data / scales_expanded

            # 向量化量化整个块
            quantized_block = self._quantize_to_fp8(scaled_block)

            # 存储结果
            quantized_matrix[:, start_col:end_col] = quantized_block
            scale_matrix[:, block_idx] = scales

        return quantized_matrix, scale_matrix

    def _quantize_by_rows(self, matrix: torch.Tensor, M: int, N: int) -> tuple:
        """
        按行方向量化

        分块: 每 block_size 行一个块
        """
        # 计算分块数量
        num_blocks = (M + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)

        # 处理每个行块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束行
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)

            # 提取当前块
            block_data = matrix[start_row:end_row, :]

            # 对当前块的每一列单独处理（列方向）
            for col in range(N):
                # 提取列数据
                col_data = block_data[:, col]

                # 处理该列
                quantized_col, scale, exp = self._process_block(col_data)

                # 存储结果
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

        return quantized_matrix, scale_matrix

    def quantize_matrix(self, matrix: torch.Tensor) -> tuple:
        """
        量化二维矩阵

        参数:
            matrix: 输入二维矩阵 [M, N]

        返回:
            quantized_matrix: 量化后的矩阵 [M, N]
            scale_matrix: 缩放因子矩阵
        """
        if matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {matrix.dim()}")

        M, N = matrix.shape

        # 根据量化轴确定分块方式
        if self.axis == 0:  # 行方向量化
            return self._quantize_by_rows(matrix, M, N)
        else:  # 列方向量化
            return self._quantize_by_cols(matrix, M, N)

    def dequantize_matrix(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
        """
        反量化二维矩阵

        参数:
            quantized_matrix: 量化后的矩阵 [M, N]
            scale_matrix: 缩放因子矩阵

        返回:
            dequantized_matrix: 反量化后的矩阵 [M, N]
        """
        if quantized_matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {quantized_matrix.dim()}")

        M, N = quantized_matrix.shape

        # 根据量化轴确定分块方式
        if self.axis == 0:  # 行方向量化
            return self._dequantize_by_rows(quantized_matrix, scale_matrix, M, N)
        else:  # 列方向量化
            return self._dequantize_by_cols(quantized_matrix, scale_matrix, M, N)

    def _dequantize_by_cols(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """
        按列方向反量化矩阵
        """
        num_blocks = (N + self.block_size - 1) // self.block_size
        dequantized_matrix = torch.zeros_like(quantized_matrix, dtype=torch.float32)

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, N)
            block = quantized_matrix[:, start:end]
            scale = scale_matrix[:, i:i+1]
            dequantized_matrix[:, start:end] = block * scale

        return dequantized_matrix

    def _dequantize_by_rows(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """
        按行方向反量化矩阵
        """
        num_blocks = (M + self.block_size - 1) // self.block_size
        dequantized_matrix = torch.zeros_like(quantized_matrix, dtype=torch.float32)

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, M)
            block = quantized_matrix[start:end, :]
            scale = scale_matrix[i:i+1, :]
            dequantized_matrix[start:end, :] = block * scale

        return dequantized_matrix

def swish(x):
    return x * torch.sigmoid(x)

def main():
    print("Starting gen_data_grouped_with_topk.py...")
    # 解析命令行参数，支持 --data_format 和 --rank 选项
    data_format = 'E4M3'  # 默认使用 E4M3
    rank = 1  # 默认不分割
    
    if len(sys.argv) < 7:
        print("Usage: python gen_data_grouped_with_topk.py <m> <k> <n> <group> <topk> [--data_format=E4M3|E5M2] [--rank=N]")
        print("Example: python gen_data_grouped_with_topk.py 512 4096 3072 4 2 --data_format=E4M3")
        print("Example: python gen_data_grouped_with_topk.py 512 4096 3072 4 2 --data_format=E5M2")
        print("Example: python gen_data_grouped_with_topk.py 512 4096 3072 4 2 --rank=4")
        sys.exit(1)
    
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    group = int(sys.argv[4])
    topk = int(sys.argv[5])
    rank = int(sys.argv[6])
    trans_a = 0  # 固定为 false
    trans_b = 1  # 固定为 true

    m = m * rank
    group = group * rank
    
    print(f"Generating grouped matmul data with topk selection:")
    print(f"m={m}, k={k}, n={n}, group={group}, trans_a={trans_a}, trans_b={trans_b}, topk={topk}, data_format={data_format}, rank={rank}")
    print("Step 1: Generating input tokens...")
    
    # 1. 生成输入矩阵 input (tokens) [m, k] fp16
    # input_tokens = torch.randn(m, k, dtype=torch.bfloat16)
    a, b = 0, 0.01  # 目标范围
    # 生成 [a, b] 范围内的均匀分布
    input_tokens = a + (b - a) * torch.rand(m, k, dtype=torch.bfloat16)
    print(f"Generated input tokens shape: {input_tokens.shape}")
    
    # 2. 生成 expert_id [m, topk] 每个token选择的topk个group_id
    expert_id = torch.randint(0, group, (m, topk), dtype=torch.int32)
    print(f"Generated expert_id shape: {expert_id.shape}")
    print(f"Expert ID range: [{expert_id.min().item()}, {expert_id.max().item()}]")
    
    # 3. 生成 prob [m, topk] 每个token选择的topk个expert的概率
    prob = torch.rand(m, topk, dtype=torch.float32)
    # 归一化概率
    prob = prob / prob.sum(dim=1, keepdim=True)
    print(f"Generated prob shape: {prob.shape}")
    
    # 4. 对 input 量化为 FP8 和 scale
    quantizer = MXFP8MatrixQuantizer(
        data_format=data_format,
        axis=1,  # 按列方向量化
        block_size=32
    )
    
    # 先转换为float32再量化
    input_tokens_fp32 = input_tokens.to(torch.float32)
    input_quant, input_scale = quantizer.quantize_matrix(input_tokens_fp32)
    
    # 检查是否支持FP8数据类型
    try:
        if has_fp8:
            if data_format == 'E4M3':
                input_quant = input_quant.to(torch.float8_e4m3fn)
            else:  # E5M2
                input_quant = input_quant.to(torch.float8_e5m2)
            input_scale = input_scale.to(torch.float8_e8m0fnu)
    except Exception as e:
        print(f"Warning: FP8 not supported for input matrix, using float32 instead: {e}")
    
    # 5. 按照选中的topk个group_id进行重排，统计每个group的token数
    # 统计每个group的token数（每个token可能选择多个group）
    group_token_counts = torch.zeros(group, dtype=torch.int32)
    for g in range(group):
        # 统计该group被选择的总次数
        count = (expert_id == g).sum().item()
        group_token_counts[g] = count
    
    print(f"Group token counts: {group_token_counts.tolist()}")
    print(f"Total token count: {group_token_counts.sum().item()}, Expected: {m * topk}")
    
    # 构建重排后的矩阵a和对应的索引
    a_reordered_list = []
    token_indices_list = []  # 记录每个group中token的原始索引
    prob_list = []  # 记录每个group中token的概率
    
    # 初始化row_idx矩阵，大小为[m, topk]
    row_idx = torch.zeros(m, topk, dtype=torch.long)
    
    # 记录每个group的起始行索引
    group_start_rows = []
    current_row = 0
    
    for g in range(group):
        # 找出所有选择了该group的token及其位置
        token_indices = []
        group_probs = []
        
        for idx in range(m):
            # 找到该token选择该group的位置
            g_positions = torch.where(expert_id[idx] == g)[0]
            for pos in g_positions:
                token_indices.append(idx)
                group_probs.append(prob[idx, pos])
        
        # 记录当前group的起始行索引
        group_start_rows.append(current_row)
        
        # 更新row_idx矩阵
        for i, idx in enumerate(token_indices):
            # 找到该token在expert_id中的位置
            token_positions = torch.where(expert_id[idx] == g)[0]
            for pos in token_positions:
                row_idx[idx, pos] = current_row + i
        
        # 更新当前行索引
        current_row += len(token_indices)
        
        token_indices = torch.tensor(token_indices, dtype=torch.long)
        
        if len(token_indices) > 0:
            a_reordered_list.append(input_tokens_fp32[token_indices])
            token_indices_list.append(token_indices)
            prob_list.append(torch.tensor(group_probs, dtype=torch.float32))
        else:
            # 如果没有token选择该group，添加空tensor
            a_reordered_list.append(torch.empty(0, k, dtype=torch.float32))
            token_indices_list.append(torch.empty(0, dtype=torch.long))
            prob_list.append(torch.empty(0, dtype=torch.float32))
    
    # 保持列表形式，因为每个 group 的 token 数量可能不同
    print(f"A grouped: {len(a_reordered_list)} groups")
    for i, a in enumerate(a_reordered_list):
        print(f"  Group {i} shape: {a.shape}")
    
    # 5.5 对 a_reordered_list 进行量化（保存用）
    a_quant_list = []
    a_scale_list = []
    for g in range(group):
        if a_reordered_list[g].shape[0] > 0:
            a_quant_g, a_scale_g = quantizer.quantize_matrix(a_reordered_list[g])
            try:
                if has_fp8:
                    if data_format == 'E4M3':
                        a_quant_g = a_quant_g.to(torch.float8_e4m3fn)
                    else:
                        a_quant_g = a_quant_g.to(torch.float8_e5m2)
                    a_scale_g = a_scale_g.to(torch.float8_e8m0fnu)
            except Exception as e:
                print(f"Warning: FP8 not supported for a matrix, using float32 instead: {e}")
            a_quant_list.append(a_quant_g)
            a_scale_list.append(a_scale_g)
        else:
            a_quant_list.append(torch.empty(0, k, dtype=torch.float32))
            a_scale_list.append(torch.empty(0, (k + 31) // 32, dtype=torch.float32))
    
    # 6. 生成B矩阵
    b_fp8_list = []
    b_scale_list = []
    b_fp32_list = []
    
    for g in range(group):
        if trans_b == 1:
            # 当 trans_b=1 时，生成 [n, k] 形状的矩阵
            b_fp8_g, b_scale_g, b_fp32_g = gen_data_fp8(n, k, 0, data_format)
        else:
            # 当 trans_b=0 时，生成 [k, n] 形状的矩阵
            b_fp8_g, b_scale_g, b_fp32_g = gen_data_fp8(k, n, 0, data_format)
        b_fp8_list.append(b_fp8_g)
        b_scale_list.append(b_scale_g)
        b_fp32_list.append(b_fp32_g)
    
    # 堆叠
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_scale_list, dim=0)
    b_fp32 = torch.stack(b_fp32_list, dim=0)
    
    # 7. 执行grouped matmul
    c_fp32_list = []
    c_bf16_list = []
    for g in range(group):
        if a_reordered_list[g].shape[0] > 0:
            a_scale_tmp = torch.repeat_interleave(a_scale_list[g], 32, dim=1).to(torch.float32)
            if trans_b == 1:
                # 当 trans_b=1 时，b 的形状是 [n, k]，需要转置为 [k, n]
                c_g = (a_quant_list[g].to(torch.float32) * a_scale_tmp) @ b_fp32[g].T
            else:
                # 当 trans_b=0 时，b 的形状是 [k, n]，直接使用
                c_g = (a_quant_list[g].to(torch.float32) * a_scale_tmp) @ b_fp32[g]
            c_fp32_list.append(c_g)
            c_bf16_list.append(c_g.to(torch.bfloat16))
        else:
            c_fp32_list.append(torch.empty(0, n, dtype=torch.float32))
            c_bf16_list.append(torch.empty(0, n, dtype=torch.bfloat16))
    
    # 8. 对每个group的结果应用SWIGLU激活
    c_fp32_swiglu_list = []
    c_bf16_swiglu_list = []
    for g in range(group):
        if c_fp32_list[g].shape[0] > 0:
            # 按N方向切分为两个大小相同的矩阵
            c_a_bf16, c_b_bf16 = torch.split(c_bf16_list[g], n//2, dim=1)
            c_a = c_a_bf16.to(torch.float32)
            c_b = c_b_bf16.to(torch.float32)
            # 应用SWIGLU激活
            c_swiglu = swish(c_a) * c_b
            c_fp32_swiglu_list.append(c_swiglu)
            c_bf16_swiglu_list.append(c_swiglu.to(torch.bfloat16))
        else:
            c_fp32_swiglu_list.append(torch.empty(0, n//2, dtype=torch.float32))
            c_bf16_swiglu_list.append(torch.empty(0, n//2, dtype=torch.bfloat16))
    
    # 9. 对SWIGLU结果进行MXFP8量化，得到a2矩阵
    a2_list = []
    a2_scale_list = []
    a2_fp32_list = []
    
    for g in range(group):
        if c_fp32_swiglu_list[g].shape[0] > 0:
            # 量化
            a2_quant, a2_scale = quantizer.quantize_matrix(c_fp32_swiglu_list[g])
            c_bf16_swiglu_abs = torch.abs(c_bf16_swiglu_list[g])
            # c_bf16_swiglu_abs 最大值
            c_bf16_swiglu_max = torch.max(c_bf16_swiglu_abs)
            c_bf16_swiglu_log2 = torch.log2(c_bf16_swiglu_max)
            c_bf16_swiglu_round = torch.floor(c_bf16_swiglu_log2) # 算子具体实现是round还是floor还是远离0取整？
            # 检查是否支持FP8数据类型
            try:
                if has_fp8:
                    if data_format == 'E4M3':
                        c_bf16_swiglu_shared_exp = int(c_bf16_swiglu_round - 8)
                        a2_quant = c_bf16_swiglu_list[g] / (2 ** c_bf16_swiglu_shared_exp)
                        a2_quant = a2_quant.to(torch.float8_e4m3fn)
                        a2_scale = torch.full_like(c_fp32_swiglu_list[g], 2 ** c_bf16_swiglu_shared_exp)
                    else:  # E5M2
                        c_bf16_swiglu_shared_exp = int(c_bf16_swiglu_round - 15)
                        a2_quant = c_bf16_swiglu_list[g] / (2 ** c_bf16_swiglu_shared_exp)
                        a2_quant = a2_quant.to(torch.float8_e5m2)
                        a2_scale = torch.full_like(c_fp32_swiglu_list[g], 2 ** c_bf16_swiglu_shared_exp)
                    a2_scale = a2_scale.to(torch.float8_e8m0fnu)
            except Exception as e:
                print(f"Warning: FP8 not supported for a2 matrix, using float32 instead: {e}")
            
            a2_list.append(a2_quant)
            a2_scale_list.append(a2_scale)
            a2_fp32_list.append(c_fp32_swiglu_list[g])
        else:
            a2_list.append(torch.empty(0, n//2, dtype=torch.float32))
            a2_scale_list.append(torch.empty(0, (n//2 + 31) // 32, dtype=torch.float32))
            a2_fp32_list.append(torch.empty(0, n//2, dtype=torch.float32))
    
    # 10. 生成b2矩阵 [group, n/2, k]
    n_half = n // 2
    b2_fp8_list = []
    b2_scale_list = []
    b2_fp32_list = []
    
    for g in range(group):
        # 当 trans_b=1 时，我们需要生成 [k, n/2] 的矩阵
        if trans_b == 1:
            b2_fp8_g, b2_scale_g, b2_fp32_g = gen_data_fp8(k, n_half, 0, data_format)
        else:
            b2_fp8_g, b2_scale_g, b2_fp32_g = gen_data_fp8(n_half, k, 0, data_format)
        b2_fp8_list.append(b2_fp8_g)
        b2_scale_list.append(b2_scale_g)
        b2_fp32_list.append(b2_fp32_g)
    
    # 堆叠成 [group, n/2, k] 或 [group, k, n/2]
    b2_fp8 = torch.stack(b2_fp8_list, dim=0)
    b2_scale = torch.stack(b2_scale_list, dim=0)
    b2_fp32 = torch.stack(b2_fp32_list, dim=0)
    
    # 11. 计算c2：每个a2子矩阵与对应的b2 group做matmul
    c2_fp32_list = []
    for g in range(group):
        if a2_fp32_list[g].shape[0] > 0:
            if trans_b == 1:
                # 当 trans_b=1 时，b2 的形状是 [k, n/2]，需要转置
                c2_g = (a2_list[g].to(torch.float32) * a2_scale_list[g].to(torch.float32)) @ b2_fp32[g].transpose(0, 1)
            else:
                # 当 trans_b=0 时，b2 的形状是 [n/2, k]
                c2_g = (a2_list[g].to(torch.float32) * a2_scale_list[g].to(torch.float32)) @ b2_fp32[g]
            c2_fp32_list.append(c2_g)
        else:
            c2_fp32_list.append(torch.empty(0, k, dtype=torch.float32))
    c2_fp32_combined = torch.cat(c2_fp32_list, dim=0)
    # 12. 对c2的输出，重排回原有input的顺序，按照prob矩阵进行加权求和
    # 初始化输出矩阵
    c2_final = torch.zeros(m, k, dtype=torch.float32)
    
    # 对每个group的结果进行加权求和
    for idx in range(m):
        for exp_id in range(topk):
            expand_id = row_idx[idx, exp_id]
            c2_final[idx] += c2_fp32_combined[expand_id] * prob[idx, exp_id]
    
    # 13. 生成grouplist，每个值是对应group的token数
    grouplist = group_token_counts.tolist()
    print(f"Group list: {grouplist}")
    
    # 14. 如果 rank > 1，按最外层维度分割数据
    if rank > 1:
        # 计算每个 rank 的样本数
        m_per_rank = m // rank
        remainder = m % rank
        
        # 分割各个变量
        input_tokens_list = torch.split(input_tokens, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        expert_id_list = torch.split(expert_id, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        prob_list_split = torch.split(prob, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        row_idx_list = torch.split(row_idx, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        input_quant_list = torch.split(input_quant, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        input_scale_list = torch.split(input_scale, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        
        # 分割 c2_final (golden 输出)
        c2_final_list = torch.split(c2_final, [m_per_rank + (1 if i < remainder else 0) for i in range(rank)])
        
        print(f"Split data into {rank} ranks, each with approximately {m_per_rank} tokens")
    else:
        input_tokens_list = [input_tokens]
        expert_id_list = [expert_id]
        prob_list_split = [prob]
        row_idx_list = [row_idx]
        input_quant_list = [input_quant]
        input_scale_list = [input_scale]
        c2_final_list = [c2_final]
    
    # 为每个 rank 计算对应的 group 索引
    groups_per_rank = group // rank
    # 15. 保存文件
    for rank_id in range(rank):
        # 获取当前 rank 的数据
        input_tokens_r = input_tokens_list[rank_id]
        expert_id_r = expert_id_list[rank_id]
        prob_r = prob_list_split[rank_id]
        row_idx_r = row_idx_list[rank_id]
        input_quant_r = input_quant_list[rank_id]
        input_scale_r = input_scale_list[rank_id]
        c2_final_r = c2_final_list[rank_id]
        
        # 计算当前 rank 对应的 group 范围
        start_group = rank_id * groups_per_rank
        end_group = start_group + groups_per_rank
        
        # 获取当前 rank 需要的 b, b_scale, b2, b2_scale
        b_fp8_r = b_fp8[start_group:end_group]
        b_scale_r = b_scale[start_group:end_group]
        b2_fp8_r = b2_fp8[start_group:end_group]
        b2_scale_r = b2_scale[start_group:end_group]
        m_r = input_tokens_r.shape[0]
        
        try:
            # 使用与 gen_input_only.py 一致的目录结构
            prefix = f"data_{m}_{k}_{n}_{group}_{topk}_{rank}"
            rankdir = f"{prefix}/{rank_id}"
            os.makedirs(rankdir, exist_ok=True)
            
            # 保存input tokens
            input_tokens_np = input_tokens_r.view(torch.float16).numpy()
            input_tokens_np.tofile(f'{rankdir}/input_tokens.bin')
            print(f"Saved {rankdir}/input_tokens.bin (input tokens, shape: {input_tokens_r.shape}, dtype: {input_tokens_r.dtype})")
            
            # 保存expert_id
            expert_id_np = expert_id_r.numpy()
            expert_id_np.tofile(f'{rankdir}/expert_id.bin')
            print(f"Saved {rankdir}/expert_id.bin (expert IDs, shape: {expert_id_r.shape}, dtype: {expert_id_r.dtype})")
            
            # 保存prob
            prob_r = prob_r.to(torch.bfloat16)
            prob_np = prob_r.view(torch.float16).numpy()
            prob_np.tofile(f'{rankdir}/prob.bin')
            print(f"Saved {rankdir}/prob.bin (probabilities, shape: {prob_r.shape}, dtype: {prob_r.dtype})")
            
            # 保存row_idx
            row_idx_np = row_idx_r.numpy()
            row_idx_np.tofile(f'{rankdir}/row_idx.bin')
            print(f"Saved {rankdir}/row_idx.bin (row indices, shape: {row_idx_r.shape}, dtype: {row_idx_r.dtype})")
            
            # 保存量化后的input
            if input_quant_r.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
                input_quant_np = input_quant_r.view(torch.int8).numpy()
            else:
                input_quant_np = input_quant_r.flatten().numpy()
            input_quant_np.tofile(f'{rankdir}/input_8.bin')
            print(f"Saved {rankdir}/input_8.bin (quantized input, shape: {input_quant_r.shape}, dtype: {input_quant_r.dtype})")
            
            if input_scale_r.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
                input_scale_np = input_scale_r.view(torch.int8).numpy()
            else:
                input_scale_np = input_scale_r.flatten().numpy()
            input_scale_np.tofile(f'{rankdir}/input_scale.bin')
            print(f"Saved {rankdir}/input_scale.bin (input scale factors, shape: {input_scale_r.shape}, dtype: {input_scale_r.dtype})")
            
            # 保存grouplist
            grouplist_r = grouplist[start_group:end_group]
            grouplist_np = np.array(grouplist_r, dtype=np.int32)
            grouplist_np.tofile(f'{rankdir}/grouplist.bin')
            print(f"Saved {rankdir}/grouplist.bin (group list, shape: {grouplist_np.shape}, dtype: {grouplist_np.dtype})")
            
            # 保存b矩阵
            if b_fp8_r.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
                b_np = b_fp8_r.view(torch.int8).numpy()
            else:
                b_np = b_fp8_r.flatten().numpy()
            b_np.tofile(f'{rankdir}/b_8.bin')
            print(f"Saved {rankdir}/b_8.bin (B matrix, shape: {b_fp8_r.shape}, dtype: {b_fp8_r.dtype})")
            
            if b_scale_r.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
                b_scale_np = b_scale_r.view(torch.int8).numpy()
            else:
                b_scale_np = b_scale_r.flatten().numpy()
            b_scale_np.tofile(f'{rankdir}/b_scale.bin')
            print(f"Saved {rankdir}/b_scale.bin (B scale factors, shape: {b_scale_r.shape}, dtype: {b_scale_r.dtype})")
            
            # 保存b2矩阵
            if b2_fp8_r.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
                b2_np = b2_fp8_r.view(torch.int8).numpy()
            else:
                b2_np = b2_fp8_r.flatten().numpy()
            b2_np.tofile(f'{rankdir}/b2_8.bin')
            print(f"Saved {rankdir}/b2_8.bin (b2 matrix, shape: {b2_fp8_r.shape}, dtype: {b2_fp8_r.dtype})")
            
            if b2_scale_r.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
                b2_scale_np = b2_scale_r.view(torch.int8).numpy()
            else:
                b2_scale_np = b2_scale_r.flatten().numpy()
            b2_scale_np.tofile(f'{rankdir}/b2_scale.bin')
            print(f"Saved {rankdir}/b2_scale.bin (b2 scale factors, shape: {b2_scale_r.shape}, dtype: {b2_scale_r.dtype})")
            
            # 保存c2结果
            c2_final_np = c2_final_r.numpy()
            c2_final_np.tofile(f'{rankdir}/c2_fp32.bin')
            print(f"Saved {rankdir}/c2_fp32.bin (c2 final result, shape: {c2_final_r.shape}, dtype: {c2_final_r.dtype})")
            
        except Exception as e:
            print(f"Error saving files for rank {rank_id}: {e}")
    
    # 保存 a 和 a2 的 FP8 数据（不按 rank 分割），使用与 gen_input_only.py 一致的目录结构
    prefix = f"data_{m}_{k}_{n}_{group}_{topk}_{rank}"
    os.makedirs(prefix, exist_ok=True)
    
    # 保存 a_8 和 a_scale
    a_combined = torch.cat(a_quant_list, dim=0)
    if a_combined.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
        a_np = a_combined.view(torch.int8).numpy()
    else:
        a_np = a_combined.flatten().numpy()
    a_np.tofile(f'{prefix}/a_8.bin')
    print(f"Saved {prefix}/a_8.bin (A matrix FP8, shape: {a_combined.shape}, dtype: {a_combined.dtype})")
    
    a_scale_combined = torch.cat(a_scale_list, dim=0)
    if a_scale_combined.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
        a_scale_np = a_scale_combined.view(torch.int8).numpy()
    else:
        a_scale_np = a_scale_combined.flatten().numpy()
    a_scale_np.tofile(f'{prefix}/a_scale.bin')
    print(f"Saved {prefix}/a_scale.bin (A scale factors, shape: {a_scale_combined.shape}, dtype: {a_scale_combined.dtype})")
    
    # 保存 a2_8 和 a2_scale
    a2_combined = torch.cat(a2_list, dim=0)
    if a2_combined.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
        a2_np = a2_combined.view(torch.int8).numpy()
    else:
        a2_np = a2_combined.flatten().numpy()
    a2_np.tofile(f'{prefix}/a2_8.bin')
    print(f"Saved {prefix}/a2_8.bin (a2 matrix FP8, shape: {a2_combined.shape}, dtype: {a2_combined.dtype})")
    
    a2_scale_combined = torch.cat(a2_scale_list, dim=0)
    if a2_scale_combined.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]:
        a2_scale_np = a2_scale_combined.view(torch.int8).numpy()
    else:
        a2_scale_np = a2_scale_combined.flatten().numpy()
    a2_scale_np.tofile(f'{prefix}/a2_scale.bin')
    print(f"Saved {prefix}/a2_scale.bin (a2 scale factors, shape: {a2_scale_combined.shape}, dtype: {a2_scale_combined.dtype})")

    # 保存 c2_fp32_list
    c2_fp32_combined_np = c2_fp32_combined.numpy()
    c2_fp32_combined_np.tofile(f'{prefix}/gmm2_fp32.bin')
    print(f"Saved {prefix}/gmm2_fp32.bin (c2_fp32_list, shape: {c2_fp32_combined.shape}, dtype: {c2_fp32_combined.dtype})")

    c2_bf16_combined_np = c2_fp32_combined.to(torch.bfloat16).view(torch.int16).numpy()
    c2_bf16_combined_np.tofile(f'{prefix}/gmm2_bf16.bin')
    print(f"Saved {prefix}/gmm2_bf16.bin (c2_fp32_list, shape: {c2_fp32_combined.shape}, dtype: {c2_fp32_combined.dtype})")
        
    # 打印结果信息
    print(f"\nGenerated grouped matmul data with topk selection:")
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Expert ID shape: {expert_id.shape}")
    print(f"Prob shape: {prob.shape}")
    print(f"Group list: {grouplist}")
    print(f"Data saved to {prefix}/ directory")

if __name__ == "__main__":
    main()