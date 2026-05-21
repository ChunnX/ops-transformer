#!/usr/bin/env python3
"""
Python方案核心伪代码 - SparseAttnSharedkvMetadata

数据流: CPU tensor → Python计算 → H2D(metadata) → NPU
优势: 避免D2H传输，端口占用最小(~10-30μs)
"""

# ============================================================
# 时延对比
# ============================================================
# 方案1: 原始AICPU  - 时延~50-100μs  端口占用~30-50μs
# 方案2: Host Op   - 时延~110-250μs 端口占用~90-180μs ❌
# 方案3: Python    - 时延~20-80μs   端口占用~10-30μs  ✅


def compute_metadata_python(
    num_heads_q, num_heads_kv, head_dim,
    cu_seqlens_q, seqused_kv,       # CPU tensor
    batch_size, max_seqlen_q, max_seqlen_kv,
    cmp_topk=None, cmp_ratio=None,
    ori_win_left=127, ori_win_right=0,
    device="npu", arch="A3"
):
    """Python端Metadata计算"""
    
    # Step 1: 基本块大小
    M_BASE = 64
    S2_BASE = 128 
    
    group_size = num_heads_q // num_heads_kv
    has_cmp_kv = (cmp_ratio is not None) and (cmp_topk is not None)
    
    # ============================================================
    # Step 2: 序列长度 (NumPy加速, 无端口占用)
    # ============================================================
    s1_sizes = np.diff(cu_seqlens_q.numpy())  # 每个batch的Q序列长度
    s2_sizes = seqused_kv.numpy()              # 每个batch的KV序列长度
    
    # ============================================================
    # Step 3: 切分信息
    # ============================================================
    # 计算每个batch的S1G块数和S2块数
    s1GBaseNum = np.ceil(s1_sizes * group_size / M_BASE).astype(np.int32)
    s2BaseNum = np.ceil(s2_sizes / S2_BASE).astype(np.int32)
    
    # ============================================================
    # Step 4: 开销计算
    # 计算每行有哪些block有wincost和cmpcost，然后相加得到该行的开销
    # 块开销、行开销、batch开销都要记录，后面的负载均衡要用
    # ============================================================
    # 成本表: typeCost_[行类型][块类型]
    # WIN_NORMAL_BLOCK=0, WIN_TAIL_BLOCK=1, CMP_NORMAL_BLOCK=2, CMP_TAIL_BLOCK=3
    typeCost_ = np.array([
        [6 * 4 + 10 * 8, 6 * 4 + 10 * (S2_BASE//64)],  # WIN_NORMAL_BLOCK
        [6 * 4 + 10 * 8, 6 * 4 + 10 * (S2_BASE//64)],  # WIN_TAIL_BLOCK  
        [6 * 4 + 10 * 8, 6 * 4 + 10 * (S2_BASE//64)],  # CMP_NORMAL_BLOCK
        [6 * 4 + 10 * 8, 6 * 4 + 10 * (S2_BASE//64)],  # CMP_TAIL_BLOCK
    ])
    
    # 记录各层级开销
    block_costs = []      # 块开销
    row_costs = []        # 行开销
    batch_costs = []      # batch开销
    row_info = []         # 行信息: (batch_idx, row_idx, win_blocks, cmp_blocks, cost)
    
    total_cost = 0
    
    for b in range(batch_size):
        batch_cost = 0
        num_rows = s1GBaseNum[b]
        
        for row in range(num_rows):
            # 计算滑动窗口范围
            m_start = row * M_BASE
            m_end = min((row + 1) * M_BASE, s1_sizes[b] * group_size)
            is_tail_row = (m_end < (row + 1) * M_BASE)
            
            # Win部分: 计算覆盖的S2范围
            s2_first = max(0, m_start - ori_win_left)
            s2_last = min(s2_sizes[b] - 1, m_start + ori_win_right)
            
            # Win部分Block数量和尾部大小
            win_block_num = 0
            win_tail_size = 0
            win_cost = 0
            
            if s2_first <= s2_last:
                win_block_num = (s2_last - s2_first) // S2_BASE + 1
                win_tail_size = (s2_last - s2_first + 1) % S2_BASE
                win_normal_blocks = win_block_num - (1 if win_tail_size != 0 else 0)
                win_tail_blocks = 1 if win_tail_size != 0 else 0
                
                # 计算Win开销
                row_type = 1 if is_tail_row else 0  # WIN_TAIL_BLOCK=1, WIN_NORMAL_BLOCK=0
                win_cost = (typeCost_[row_type][0] * win_normal_blocks +
                           typeCost_[row_type][1] * win_tail_blocks)
            
            # Cmp部分开销
            cmp_block_num = 0
            cmp_tail_size = 0
            cmp_cost = 0
            
            if has_cmp_kv and s2_last >= 0:
                # 计算Cmp的token数量
                cmp_token_size = (s2_last + 1) // cmp_ratio
                act_cmp_token_size = min(cmp_token_size, cmp_topk)
                
                if act_cmp_token_size > 0:
                    cmp_block_num = (act_cmp_token_size - 1) // S2_BASE + 1
                    cmp_tail_size = act_cmp_token_size % S2_BASE
                    cmp_normal_blocks = cmp_block_num - (1 if cmp_tail_size != 0 else 0)
                    cmp_tail_blocks = 1 if cmp_tail_size != 0 else 0
                    
                    # 计算Cmp开销
                    row_type_cmp = 3 if is_tail_row else 2  # CMP_TAIL_BLOCK=3, CMP_NORMAL_BLOCK=2
                    cmp_cost = (typeCost_[row_type_cmp][0] * cmp_normal_blocks +
                               typeCost_[row_type_cmp][1] * cmp_tail_blocks)
            
            # 行总开销
            row_cost = win_cost + cmp_cost
            
            # 记录开销
            block_costs.append({
                'win': {'num': win_block_num, 'tail': win_tail_size, 'cost': win_cost},
                'cmp': {'num': cmp_block_num, 'tail': cmp_tail_size, 'cost': cmp_cost}
            })
            row_costs.append(row_cost)
            row_info.append({
                'batch_idx': b,
                'row_idx': row,
                'm_start': m_start,
                'm_end': m_end,
                'win_blocks': win_block_num,
                'cmp_blocks': cmp_block_num,
                'cost': row_cost
            })
            
            batch_cost += row_cost
        
        batch_costs.append(batch_cost)
        total_cost += batch_cost
    
    # ============================================================
    # Step 5: 负载均衡 (三级分配策略)
    # 对于每个aicore：能不切分就不切分，尽量按大粒度的往里塞
    # 分配优先级: batch -> row -> block
    # ============================================================
    # 计算平均负载上限
    avg_cost = total_cost / AICORE_NUM if AICORE_NUM > 0 else 0
    max_row_cost = max(row_costs) if row_costs else 0
    cost_limit = max(avg_cost, max_row_cost)  # 确保能放下最大行
    
    # 初始化核心分配状态
    core_alloc = []  # 每个核心的分配信息
    for _ in range(AICORE_NUM):
        core_alloc.append({
            'cost': 0,
            'batches': [],      # 分配的batch
            'rows': [],         # 分配的行索引
            'bN2Start': 0,
            'mStart': 0,
            'bN2End': 0,
            'mEnd': 0
        })
    
    # 当前处理位置
    current_batch = 0
    current_row_in_batch = 0
    total_rows = len(row_info)
    current_global_row = 0
    
    for core_idx in range(AICORE_NUM):
        if current_global_row >= total_rows:
            break
        
        # 记录分配起始位置
        first_row = row_info[current_global_row]
        core_alloc[core_idx]['bN2Start'] = first_row['batch_idx']
        core_alloc[core_idx]['mStart'] = first_row['m_start']
        
        # 三级分配策略
        # 1. 优先尝试分配完整batch
        while current_batch < batch_size:
            batch_idx = current_batch
            batch_remaining_rows = s1GBaseNum[batch_idx] - current_row_in_batch
            batch_remaining_cost = sum(row_info[r]['cost'] for r in 
                                      range(current_global_row, 
                                            current_global_row + batch_remaining_rows))
            
            if core_alloc[core_idx]['cost'] + batch_remaining_cost <= cost_limit * 1.5:
                # 可以分配完整batch
                for _ in range(batch_remaining_rows):
                    if current_global_row < total_rows:
                        core_alloc[core_idx]['rows'].append(current_global_row)
                        core_alloc[core_idx]['cost'] += row_info[current_global_row]['cost']
                        current_global_row += 1
                current_row_in_batch = 0
                current_batch += 1
            else:
                break
        
        # 2. 尝试分配完整行
        while current_global_row < total_rows:
            row_cost = row_info[current_global_row]['cost']
            if core_alloc[core_idx]['cost'] + row_cost <= cost_limit * 2:
                core_alloc[core_idx]['rows'].append(current_global_row)
                core_alloc[core_idx]['cost'] += row_cost
                current_global_row += 1
                
                # 更新当前batch和行位置
                if current_global_row < total_rows:
                    current_batch = row_info[current_global_row]['batch_idx']
                    current_row_in_batch = row_info[current_global_row]['row_idx']
            else:
                break
        
        # 3. 尝试分配部分块（简化处理，实际会更复杂）
        # 这里省略块级切分逻辑，保持代码简洁
        
        # 记录分配结束位置
        if core_alloc[core_idx]['rows']:
            last_row = row_info[core_alloc[core_idx]['rows'][-1]]
            core_alloc[core_idx]['bN2End'] = last_row['batch_idx'] + 1
            core_alloc[core_idx]['mEnd'] = last_row['m_end']
    
    # 统计使用的核心数
    used_cores = sum(1 for core in core_alloc if core['cost'] > 0)
    
    # ============================================================
    # Step 6: 生成metadata[1024]
    # ============================================================
    metadata = np.zeros(1024, dtype=np.int32)
    
    for core_idx in range(AICORE_NUM):
        core = core_alloc[core_idx]
        offset = core_idx * 8  # 每个核心占用8个int32
        
        if core['cost'] > 0:
            metadata[offset + 0] = 1           # CORE_ENABLE: 核心启用标志
            metadata[offset + 1] = core['bN2Start']  # BN2_START: batch起始索引
            metadata[offset + 2] = core['mStart']    # M_START: M轴起始位置
            metadata[offset + 3] = 0           # RESERVED: 保留
            metadata[offset + 4] = core['bN2End']    # BN2_END: batch结束索引
            metadata[offset + 5] = core['mEnd']      # M_END: M轴结束位置
            metadata[offset + 6] = 0           # RESERVED: 保留
            metadata[offset + 7] = 0           # RESERVED: 保留
        else:
            metadata[offset + 0] = 0           # CORE_ENABLE: 核心禁用
    
    # ============================================================
    # Step 7: H2D传输 (唯一端口占用 ~10-30μs)
    # ============================================================
    if device == "npu":
        return torch.from_numpy(metadata).to("npu")  # ~10-30μs
    return torch.from_numpy(metadata)


# ============================================================
# 调用方式对比
# ============================================================

# 原代码
cu_seqlens_q = torch.tensor([0,128,256,384,512]).npu()  # H2D
seqused_kv = torch.tensor([4096]*4).npu()               # H2D
# metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
#     cu_seqlens_q, seqused_kv, ...)

# Python方案
cu_seqlens_q = torch.tensor([0,128,256,384,512])  # CPU
seqused_kv = torch.tensor([4096]*4)                # CPU  
# metadata = compute_metadata_python(                 # CPU计算
#     cu_seqlens_q, seqused_kv, arch="A5", device="npu")

# ============================================================
# A3 vs A5
# ============================================================
# A3: S2_BASE=512, AICORE_NUM=36, 时延~20-80μs
# A5: S2_BASE=128, AICORE_NUM=48, 时延~40-120μs
# 端口占用: 都是~10-30μs (最小)