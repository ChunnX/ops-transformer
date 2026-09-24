import torch
import math

DTYPE = torch.float32
D_MAX = 128

# 硬件级物理切块常量约束
L0A_ROWS = 128
L0B_COLS = 128

# 前向分块参数 (S1=256, S2=1024)
FW_S1_BLOCK = 256
FW_S2_BLOCK = 1024

# 反向分块参数 (S1=512, S2=512)
BW_S1_BLOCK = 512
BW_S2_BLOCK = 512


def cube_matmul_core(matrixA_L1, matrixB_L1, transA=False, transB=False, alpha=1.0, res_L0C=None):
    """
    模拟 NPU Cube 核心的矩阵乘法器。
    数据必须先加载到 L1 才能搬运至 L0A/L0B，并在 L0C 中累加，最后回写。
    约束:
        L0A = 64KB  => 128 * D * 4 bytes (对于fp32, 128*128*4 = 64KB)
        L0B = 64KB  => 128 * D * 4 bytes
        L0C = 128KB => 128 * 128 * 4 bytes
    """
    if transA:
        M, K = matrixA_L1.shape[1], matrixA_L1.shape[0]
    else:
        M, K = matrixA_L1.shape[0], matrixA_L1.shape[1]
        
    if transB:
        K2, N = matrixB_L1.shape[1], matrixB_L1.shape[0]
    else:
        K2, N = matrixB_L1.shape[0], matrixB_L1.shape[1]
    assert K == K2, f"K dimension mismatch: {K} vs {K2}"

    device = matrixA_L1.device
    
    # 模拟分配物理 L0C 缓存
    if res_L0C is None:
        res_L0C = torch.zeros(M, N, dtype=DTYPE, device=device)

    # 按照 L0A_ROWS x L0B_COLS 对 M 和 N 进行切块
    for i in range(0, M, L0A_ROWS):
        i_end = min(i + L0A_ROWS, M)
        rows = i_end - i
        for j in range(0, N, L0B_COLS):
            j_end = min(j + L0B_COLS, N)
            cols = j_end - j
            
            # 实例化分配单次乘加的 L0C 局部累加器块 (128 x 128)
            block_L0C = torch.zeros(rows, cols, dtype=DTYPE, device=device)
            
            # 沿 K 轴切块搬运到 L0A 和 L0B
            for k_start in range(0, K, L0A_ROWS):
                k_end = min(k_start + L0A_ROWS, K)
                
                # 1. 搬运 A 块到 cubeL0A
                if transA:
                    cubeL0A = matrixA_L1[k_start:k_end, i:i_end].T.contiguous()
                else:
                    cubeL0A = matrixA_L1[i:i_end, k_start:k_end].contiguous()
                    
                # 2. 搬运 B 块到 cubeL0B
                if transB:
                    cubeL0B = matrixB_L1[j:j_end, k_start:k_end].T.contiguous()
                else:
                    cubeL0B = matrixB_L1[k_start:k_end, j:j_end].contiguous()
                
                # 3. Cube 矩阵乘加核心计算 (L0C += L0A @ L0B)
                block_L0C += torch.mm(cubeL0A, cubeL0B)
            
            # 将 L0C 结果融合 alpha 累加回主 L0C 空间
            res_L0C[i:i_end, j:j_end] += alpha * block_L0C
            
    return res_L0C


def broadcast_kv(total_tokens, num_kv_heads, num_heads, kvGm):
    """将 kv 展宽的辅助函数"""
    if num_heads == num_kv_heads:
        return kvGm
    factor = num_heads // num_kv_heads
    kv_expanded = kvGm.unsqueeze(2).expand(-1, -1, factor, -1).reshape(total_tokens, num_heads, -1)
    return kv_expanded


def tiled_forward_single(queryGm, keyGm, valueGm, atten_maskGm, softmax_scale, keep_prob):
    """
    单 Batch 前向切块流
    """
    S1, N, D = queryGm.shape
    S2, _, _ = keyGm.shape
    device = queryGm.device

    # 初始化 GM 上的输出和统计量
    yGm = torch.zeros(S1, N, D, dtype=DTYPE, device=device)
    x_maxGm = torch.full((S1, N, 8), -float('inf'), dtype=DTYPE, device=device)
    x_sumGm = torch.zeros((S1, N, 8), dtype=DTYPE, device=device)

    for h in range(N):
        q_hGm = queryGm[:, h, :]   # [S1, D]
        k_hGm = keyGm[:, h, :]   # [S2, D]
        v_hGm = valueGm[:, h, :]   # [S2, D]

        out_hUB = torch.zeros(S1, D, dtype=DTYPE, device=device)
        row_maxUB = torch.full((S1,), -float('inf'), dtype=DTYPE, device=device)
        row_sumUB = torch.zeros(S1, dtype=DTYPE, device=device)

        # 循环外显式申请前向时大小的 workspaceGM
        # workspace = S1_BLOCK * S2_BLOCK + S1_BLOCK * D
        workspaceGM = torch.zeros(FW_S1_BLOCK * FW_S2_BLOCK + FW_S1_BLOCK * D, dtype=DTYPE, device=device)
        
        # 建立 Workspace 内部逻辑切片视图
        S_block_space = workspaceGM[0 : FW_S1_BLOCK * FW_S2_BLOCK]

        for s1_start in range(0, S1, FW_S1_BLOCK):
            s1_end = min(s1_start + FW_S1_BLOCK, S1)
            len_s1 = s1_end - s1_start

            # 从 GM 拷贝 Q 到 L1 (满足 L1 约束: s1_block * D)
            cubeL1_Q = q_hGm[s1_start:s1_end].contiguous()

            for s2_start in range(0, S2, FW_S2_BLOCK):
                s2_end = min(s2_start + FW_S2_BLOCK, S2)
                len_s2 = s2_end - s2_start

                # 从 GM 拷贝 K, V 到 L1 (满足 L1 约束: s2_block * D)
                cubeL1_K = k_hGm[s2_start:s2_end].contiguous()
                cubeL1_V = v_hGm[s2_start:s2_end].contiguous()

                # 1. 计算 Q @ K^T，输出到 L0C，再考回至 Workspace
                res_L0C = cube_matmul_core(cubeL1_Q, cubeL1_K, transB=True)
                
                # 重塑视图接入 workspace 中暂存
                S_block = S_block_space[0 : len_s1 * len_s2].view(len_s1, len_s2)
                S_block.copy_(res_L0C)

                # 2. Vector 单元处理 Mask 和 Scale (从 Workspace 读写)
                if atten_maskGm is not None:
                    mask_block = atten_maskGm[s1_start:s1_end, s2_start:s2_end]
                    S_block += mask_block.bool() * (-4e9)
                S_block *= softmax_scale

                # 3. Vector 单元计算 Flash Softmax 状态更新 (将状态加载到 UB 处理)
                block_maxUB = torch.max(S_block, dim=-1, keepdim=True)[0].squeeze(-1)
                prev_maxUB = row_maxUB[s1_start:s1_end]
                new_maxUB = torch.max(prev_maxUB, block_maxUB)

                exp_oldUB = torch.exp(prev_maxUB - new_maxUB)
                exp_curUB = torch.exp(block_maxUB - new_maxUB)

                exp_S_UB = torch.exp(S_block - block_maxUB.unsqueeze(-1))
                sum_curUB = exp_S_UB.sum(dim=-1)
                prev_sumUB = row_sumUB[s1_start:s1_end]
                new_sumUB = prev_sumUB * exp_oldUB + sum_curUB * exp_curUB

                # 写回全局 UB 状态
                row_maxUB[s1_start:s1_end] = new_maxUB
                row_sumUB[s1_start:s1_end] = new_sumUB

                # 4. 计算 P 矩阵 (保存在临时 UB / 视作中间变量)
                P_blockUB = exp_S_UB / new_sumUB.unsqueeze(-1)

                # 5. 计算 P @ V 并累加到输出中 (P 从 UB 拷至 L1)
                cubeL1_P = P_blockUB.contiguous()
                out_contrib_L0C = cube_matmul_core(cubeL1_P, cubeL1_V)

                out_prevUB = out_hUB[s1_start:s1_end]
                out_hUB[s1_start:s1_end] = out_prevUB * exp_oldUB.unsqueeze(-1) + out_contrib_L0C

        if keep_prob != 1.0:
            out_hUB = out_hUB * (1.0 / keep_prob)

        # 数据写回 Global Memory
        yGm[:, h, :] = out_hUB
        x_maxGm[:, h, :] = row_maxUB.unsqueeze(-1).expand(-1, 8)
        x_sumGm[:, h, :] = row_sumUB.unsqueeze(-1).expand(-1, 8)

    return yGm, x_maxGm, x_sumGm


def tiled_backward_single(dyGm, queryGm, keyGm, valueGm, x_maxGm, x_sumGm, atten_inGm, atten_maskGm, softmax_scale, keep_prob):
    """
    单 Batch 反向切块流
    """
    S1, N, D = queryGm.shape
    S2, _, _ = keyGm.shape
    device = queryGm.device

    dqGm = torch.zeros(S1, N, D, dtype=DTYPE, device=device)
    dkGm = torch.zeros(S2, N, D, dtype=DTYPE, device=device)
    dvGm = torch.zeros(S2, N, D, dtype=DTYPE, device=device)

    # 沿N轴遍历
    for h in range(N):
        q_hGm = queryGm[:, h, :]
        k_hGm = keyGm[:, h, :]
        v_hGm = valueGm[:, h, :]
        dx_hGm = dyGm[:, h, :]
        atten_in_hGm = atten_inGm[:, h, :]
        
        row_max_hUB = x_maxGm[:, h, 0] # (S1,)
        row_sum_hUB = x_sumGm[:, h, 0] # (S1,)

        # Vector1 预计算，保存至 UB
        softmax_gradUB = (dx_hGm * atten_in_hGm).sum(dim=-1, keepdim=True)

        # 循环外显式申请反向 workspaceGM
        # workspace = BW_S1_BLOCK * BW_S2_BLOCK * 2 + BW_S1_BLOCK * BW_S2_BLOCK * 2
        blk_elements = BW_S1_BLOCK * BW_S2_BLOCK
        workspaceGM = torch.zeros(blk_elements * 4, dtype=DTYPE, device=device)
        
        # 划分两块主 Workspace 区域用于 Cube 结果中转
        space_A = workspaceGM[0 : 2 * blk_elements]          # 用于存储 QK -> P mn
        space_B = workspaceGM[2 * blk_elements : 4 * blk_elements]  # 用于存储 dyv -> ds mn
        
        # 沿S1轴遍历，每次分块大小 BW_S1_BLOCK
        for s1_start in range(0, S1, BW_S1_BLOCK):
            s1_end = min(s1_start + BW_S1_BLOCK, S1)
            len_s1 = s1_end - s1_start

            # 搬运 Q, dX 到 L1 块
            cubeL1_Q = q_hGm[s1_start:s1_end].contiguous()
            cubeL1_dX = dx_hGm[s1_start:s1_end].contiguous()

            softmax_grad_blkUB = softmax_gradUB[s1_start:s1_end]
            row_max_blkUB = row_max_hUB[s1_start:s1_end].unsqueeze(-1) # (len_s1, 1)
            row_sum_blkUB = row_sum_hUB[s1_start:s1_end].unsqueeze(-1) # (len_s1, 1)

            # 沿S2轴遍历，每次分块大小 BW_S2_BLOCK
            for s2_start in range(0, S2, BW_S2_BLOCK):
                s2_end = min(s2_start + BW_S2_BLOCK, S2)
                len_s2 = s2_end - s2_start
                curr_blk_shape = (len_s1, len_s2)

                # cube1
                # 搬运 K, V 到 L1 块
                cubeL1_K = k_hGm[s2_start:s2_end].contiguous()
                cubeL1_V = v_hGm[s2_start:s2_end].contiguous()

                # --- 1. 计算 QK = Q @ K^T (即计算S))并写回 Workspace (Path: L0C -> GM) ---
                res_qk_L0C = cube_matmul_core(cubeL1_Q, cubeL1_K, transB=True)
                qk_space = space_A[0 : len_s1 * len_s2].view(curr_blk_shape)
                qk_space.copy_(res_qk_L0C)

                # --- 2. 计算 dyv = dX @ V^T (即dP=dO·V^T)并写回 Workspace ---
                res_dyv_L0C = cube_matmul_core(cubeL1_dX, cubeL1_V, transB=True)
                dyv_space = space_B[0 : len_s1 * len_s2].view(curr_blk_shape)
                dyv_space.copy_(res_dyv_L0C) # 必须先落盘到 Workspace GM

                # vector2
                # --- 3. 计算 P = softmax(S) 并写回 Workspace ---
                qkUB = torch.zeros(curr_blk_shape, dtype=DTYPE, device=device)
                qkUB.copy_(qk_space) # 从 Workspace 搬入 UB
                qkUB *= softmax_scale
                if atten_maskGm is not None:
                    mask_blk = atten_maskGm[s1_start:s1_end, s2_start:s2_end]
                    qkUB += mask_blk.bool() * (-4e9)
                
                # 将 P 写回 space_A 逻辑层（复用qk空间，qk已经使用完毕）
                pUB = torch.exp(qkUB - row_max_blkUB) / row_sum_blkUB
                p_space = space_A[0 : len_s1 * len_s2].view(curr_blk_shape)
                p_space.copy_(pUB)

                # --- 4. 计算 ds 矩阵并写回 Workspace ---
                dyvUB = torch.zeros(curr_blk_shape, dtype=DTYPE, device=device)
                dyvUB.copy_(dyv_space) # 从 Workspace 搬入 UB
                dsUB = pUB * (dyvUB - softmax_grad_blkUB)
                # 将 ds 写回 space_B 逻辑层（复用dyv空间，dyv 已经使用完毕）
                ds_space = space_B[0 : len_s1 * len_s2].view(curr_blk_shape)
                ds_space.copy_(dsUB)

                # cube2
                # --- 5. 计算 dq 贡献并累加 (dq += ds @ K) ---
                cubeL1_ds = ds_space.contiguous()
                res_dq_L0C = cube_matmul_core(cubeL1_ds, cubeL1_K)
                dqGm[s1_start:s1_end, h, :] += res_dq_L0C * softmax_scale

                # --- 6. 计算 dk 贡献并累加 (dk += ds^T @ Q) ---
                cubeL1_ds_T = ds_space.T.contiguous()
                res_dk_L0C = cube_matmul_core(cubeL1_ds_T, cubeL1_Q)
                dkGm[s2_start:s2_end, h, :] += res_dk_L0C * softmax_scale

                # --- 7. 计算 dv 贡献并累加 (dv += p^T @ dX) ---
                cubeL1_p_T = p_space.T.contiguous()
                res_dv_L0C = cube_matmul_core(cubeL1_p_T, cubeL1_dX)
                dvGm[s2_start:s2_end, h, :] += res_dv_L0C

    return dqGm.detach(), dkGm.detach(), dvGm.detach()


def my_flash_attention(
    q, k, v, num_heads,
    pse=None, padding_mask=None, atten_mask=None,
    scale=1.0, keep_prob=1.0,
    input_layout="TND",
    actual_seq_qlen=None, actual_seq_kvlen=None,
    pre_tockens=65536, next_tockens=65536,
    inner_precise=0, sparse_mode=0, prefix=None
):
    assert input_layout == "TND", "Only TND layout supported"
    assert pse is None and padding_mask is None and prefix is None, "Unsupported arguments"
    
    device = q.device
    total_q_tokens, num_heads_q, head_dim = q.shape
    total_kv_tokens, num_heads_kv, _ = k.shape
    assert head_dim <= D_MAX

    # 模拟 GQA 广播到统一的头数
    if num_heads != num_heads_kv:
        k = broadcast_kv(total_kv_tokens, num_heads_kv, num_heads, k)
        v = broadcast_kv(total_kv_tokens, num_heads_kv, num_heads, v)

    if actual_seq_qlen is None or actual_seq_kvlen is None:
        cu_q = [total_q_tokens]
        cu_k = [total_kv_tokens]
    else:
        cu_q = list(actual_seq_qlen)
        cu_k = list(actual_seq_kvlen)

    batch_size = len(cu_q)
    q_starts = [0] + cu_q[:-1]
    q_ends = cu_q
    k_starts = [0] + cu_k[:-1]
    k_ends = cu_k

    out_list, max_list, sum_list = [], [], []

    for i in range(batch_size):
        q_start, q_end = q_starts[i], q_ends[i]
        k_start, k_end = k_starts[i], k_ends[i]
        
        queryGm = q[q_start:q_end]
        keyGm = k[k_start:k_end]
        valueGm = v[k_start:k_end]

        maskGm = None
        if sparse_mode == 1:
            S1 = q_end - q_start
            S2 = k_end - k_start
            maskGm = torch.triu(torch.ones(S1, S2, dtype=torch.bool, device=device), diagonal=S2 - S1 + 1)

        out_i, max_i, sum_i = tiled_forward_single(queryGm, keyGm, valueGm, maskGm, scale, keep_prob)
        out_list.append(out_i)
        max_list.append(max_i)
        sum_list.append(sum_i)

    return torch.cat(out_list, dim=0), torch.cat(max_list, dim=0), torch.cat(sum_list, dim=0)


def my_flash_attention_grad(
    q, k, v, dy,
    num_heads,
    input_layout="TND",
    pse=None, padding_mask=None, atten_mask=None,
    softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None,
    scale_value=1.0, keep_prob=1.0,
    pre_tockens=65536, next_tockens=65536,
    inner_precise=0, prefix=None,
    actual_seq_qlen=None, actual_seq_kvlen=None,
    sparse_mode=0
):
    assert input_layout == "TND"
    assert pse is None and padding_mask is None and prefix is None
    device = q.device

    total_q_tokens, num_heads_q, head_dim = q.shape # T N D
    total_kv_tokens, num_heads_kv, _ = k.shape

    if num_heads != num_heads_kv:
        k = broadcast_kv(total_kv_tokens, num_heads_kv, num_heads, k)
        v = broadcast_kv(total_kv_tokens, num_heads_kv, num_heads, v)

    if actual_seq_qlen is None or actual_seq_kvlen is None:
        cu_q = [total_q_tokens]
        cu_k = [total_kv_tokens]
    else:
        cu_q = list(actual_seq_qlen)
        cu_k = list(actual_seq_kvlen)

    batch_size = len(cu_q)
    q_starts = [0] + cu_q[:-1]
    q_ends = cu_q
    k_starts = [0] + cu_k[:-1]
    k_ends = cu_k

    dq_list, dk_list, dv_list = [], [], []

    for i in range(batch_size):
        q_start, q_end = q_starts[i], q_ends[i]
        k_start, k_end = k_starts[i], k_ends[i]
        
        queryGm = q[q_start:q_end]
        keyGm = k[k_start:k_end]
        valueGm = v[k_start:k_end]
        dyGm = dy[q_start:q_end]
        atten_inGm = attention_in[q_start:q_end]
        maxGm = softmax_max[q_start:q_end]
        sumGm = softmax_sum[q_start:q_end]

        maskGm = None
        if sparse_mode == 1:
            S1 = q_end - q_start
            S2 = k_end - k_start
            maskGm = torch.triu(torch.ones(S1, S2, dtype=torch.bool, device=device), diagonal=S2 - S1 + 1)

        dq_i, dk_i, dv_i = tiled_backward_single(
            dyGm, queryGm, keyGm, valueGm, maxGm, sumGm, atten_inGm, maskGm, scale_value, keep_prob
        )
        dq_list.append(dq_i)
        dk_list.append(dk_i)
        dv_list.append(dv_i)

    return torch.cat(dq_list, dim=0), torch.cat(dk_list, dim=0), torch.cat(dv_list, dim=0)

