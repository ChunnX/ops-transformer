#!/usr/bin/env python3
"""
Engram算子测试用例生成器
将EngramFetch和EngramWrite作为同一算子的两个阶段
"""

import os
import random
from typing import List, Dict, Any
from openpyxl import Workbook

# 六维TilingKey参数范围
NUM_QPS_TIER = [1, 4, 8, 16, 32]
HIDDEN_TIER = [64, 128, 256, 512, 1024, 2048, 4096]
NUM_PEERS_TIER = [1, 2, 4, 8, 16]
RANKS_PER_PEER_TIER = [1, 2, 4, 8]
ENTRIES_TIER = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
NUM_TOKENS_TIER = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
HYBRID_MODE_TIER = [0, 1]

# 测试阶段
TEST_PHASES = ['write', 'fetch', 'write_fetch']

def generate_case_name(params: Dict[str, Any], phase: str, case_id: int) -> str:
    """生成符合MC2规范的case_name"""
    num_qps = params.get('num_qps', 8)
    hidden = params.get('hidden', 256)
    num_entries = params.get('num_entries', 65536)
    num_tokens = params.get('num_tokens', 1024)
    
    name_parts = [
        f"qps{num_qps}",
        f"h{hidden}",
        f"tok{num_tokens}",
        f"e{num_entries}",
        phase,
        f"ID{case_id:03d}"
    ]
    
    return "_".join(name_parts)

def generate_level0_cases() -> List[Dict[str, Any]]:
    """生成Level0基础覆盖用例（单因子覆盖，≤200条）"""
    cases = []
    case_id = 0
    
    # 1. NUM_QPS_TIER覆盖
    base_params = {
        'hidden': 256,
        'num_entries': 65536,
        'num_tokens': 1024,
        'num_peers': 4,
        'ranks_per_peer': 2,
        'hybrid_mode': 0,
        'phase': 'write_fetch'
    }
    
    for num_qps in NUM_QPS_TIER:
        params = base_params.copy()
        params['num_qps'] = num_qps
        params['world_size'] = params['num_peers'] * params['ranks_per_peer']
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    # 2. HIDDEN_TIER覆盖
    for hidden in HIDDEN_TIER:
        params = base_params.copy()
        params['hidden'] = hidden
        params['world_size'] = params['num_peers'] * params['ranks_per_peer']
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    # 3. ENTRIES_TIER覆盖（选择代表性的几个）
    entries_values = [4096, 16384, 65536, 262144, 524288]
    for num_entries in entries_values:
        params = base_params.copy()
        params['num_entries'] = num_entries
        params['world_size'] = params['num_peers'] * params['ranks_per_peer']
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    # 4. NUM_TOKENS_TIER覆盖（选择代表性的几个）
    tokens_values = [32, 128, 512, 2048, 4096]
    for num_tokens in tokens_values:
        params = base_params.copy()
        params['num_tokens'] = num_tokens
        params['world_size'] = params['num_peers'] * params['ranks_per_peer']
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    # 5. HYBRID_MODE覆盖
    for hybrid_mode in HYBRID_MODE_TIER:
        params = base_params.copy()
        params['hybrid_mode'] = hybrid_mode
        params['world_size'] = params['num_peers'] * params['ranks_per_peer']
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    # 6. 测试阶段覆盖
    for phase in TEST_PHASES:
        params = base_params.copy()
        params['phase'] = phase
        params['world_size'] = params['num_peers'] * params['ranks_per_peer']
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    # 7. 世界规模覆盖
    world_size_configs = [
        (2, 1),   # 2卡
        (4, 1),   # 4卡
        (8, 1),   # 8卡 flat
        (4, 2),   # 8卡 hybrid
        (8, 2),   # 16卡
        (16, 2),  # 32卡
    ]
    
    for num_peers, ranks_per_peer in world_size_configs:
        params = base_params.copy()
        params['num_peers'] = num_peers
        params['ranks_per_peer'] = ranks_per_peer
        params['world_size'] = num_peers * ranks_per_peer
        params['hybrid_mode'] = 1 if ranks_per_peer > 1 else 0
        
        case = create_case(params, case_id)
        cases.append(case)
        case_id += 1
    
    return cases[:200]  # 限制在200条以内

def generate_level1_cases() -> List[Dict[str, Any]]:
    """生成Level1两两组合用例（500~700条）"""
    cases = []
    case_id = 200  # 从200开始编号
    
    # 两两组合：NUM_QPS × HIDDEN
    for num_qps in NUM_QPS_TIER:
        for hidden in HIDDEN_TIER:
            params = {
                'num_qps': num_qps,
                'hidden': hidden,
                'num_entries': 65536,
                'num_tokens': 1024,
                'num_peers': 4,
                'ranks_per_peer': 2,
                'hybrid_mode': 0,
                'phase': 'write_fetch',
                'world_size': 8
            }
            cases.append(create_case(params, case_id))
            case_id += 1
    
    # 两两组合：HIDDEN × ENTRIES
    for hidden in [256, 512, 1024, 2048]:
        for num_entries in ENTRIES_TIER[:5]:
            params = {
                'num_qps': 8,
                'hidden': hidden,
                'num_entries': num_entries,
                'num_tokens': 1024,
                'num_peers': 4,
                'ranks_per_peer': 2,
                'hybrid_mode': 0,
                'phase': 'write_fetch',
                'world_size': 8
            }
            cases.append(create_case(params, case_id))
            case_id += 1
    
    # 两两组合：NUM_TOKENS × HIDDEN
    for num_tokens in NUM_TOKENS_TIER[:6]:
        for hidden in HIDDEN_TIER[:5]:
            params = {
                'num_qps': 8,
                'hidden': hidden,
                'num_entries': 65536,
                'num_tokens': num_tokens,
                'num_peers': 4,
                'ranks_per_peer': 2,
                'hybrid_mode': 0,
                'phase': 'write_fetch',
                'world_size': 8
            }
            cases.append(create_case(params, case_id))
            case_id += 1
    
    # 两两组合：HYBRID_MODE × NUM_PEERS
    for hybrid_mode in HYBRID_MODE_TIER:
        for num_peers in NUM_PEERS_TIER:
            ranks_per_peer = 2 if hybrid_mode == 1 else 1
            params = {
                'num_qps': 8,
                'hidden': 256,
                'num_entries': 65536,
                'num_tokens': 1024,
                'num_peers': num_peers,
                'ranks_per_peer': ranks_per_peer,
                'hybrid_mode': hybrid_mode,
                'phase': 'write_fetch',
                'world_size': num_peers * ranks_per_peer
            }
            cases.append(create_case(params, case_id))
            case_id += 1
    
    # 两两组合：phase × HIDDEN
    for phase in TEST_PHASES:
        for hidden in HIDDEN_TIER[:5]:
            params = {
                'num_qps': 8,
                'hidden': hidden,
                'num_entries': 65536,
                'num_tokens': 1024,
                'num_peers': 4,
                'ranks_per_peer': 2,
                'hybrid_mode': 0,
                'phase': phase,
                'world_size': 8
            }
            cases.append(create_case(params, case_id))
            case_id += 1
    
    # 随机组合补充（达到500条）
    while len(cases) < 500:
        params = {
            'num_qps': random.choice(NUM_QPS_TIER),
            'hidden': random.choice(HIDDEN_TIER),
            'num_entries': random.choice(ENTRIES_TIER),
            'num_tokens': random.choice(NUM_TOKENS_TIER),
            'num_peers': random.choice(NUM_PEERS_TIER),
            'ranks_per_peer': random.choice(RANKS_PER_PEER_TIER),
            'hybrid_mode': random.choice(HYBRID_MODE_TIER),
            'phase': random.choice(TEST_PHASES),
            'world_size': params['num_peers'] * params['ranks_per_peer']
        }
        cases.append(create_case(params, case_id))
        case_id += 1
    
    return cases[:700]  # 限制在700条以内

def create_case(params: Dict[str, Any], case_id: int) -> Dict[str, Any]:
    """创建单个测试用例"""
    phase = params.get('phase', 'write_fetch')
    world_size = params.get('world_size', 8)
    num_entries = params.get('num_entries', 65536)
    num_tokens = params.get('num_tokens', 1024)
    hidden = params.get('hidden', 256)
    
    # 根据phase定义输入输出tensor
    if phase == 'write':
        # Write阶段：输入storage，无indices
        input_shapes = [[num_entries, hidden]]
        input_dtypes = ['bf16']
        input_ranges = [(-5, 5)]
    elif phase == 'fetch':
        # Fetch阶段：输入indices，输出fetched
        indices_max = num_entries * world_size - 1
        input_shapes = [[num_tokens]]
        input_dtypes = ['int32']
        input_ranges = [(0, indices_max)]
    else:
        # write_fetch阶段：包含storage和indices
        indices_max = num_entries * world_size - 1
        input_shapes = [[num_entries, hidden], [num_tokens]]
        input_dtypes = ['bf16', 'int32']
        input_ranges = [(-5, 5), (0, indices_max)]
    
    case = {
        'case_name': generate_case_name(params, phase, case_id),
        'input_tensor_shape': input_shapes,
        'input_tensor_dtype': input_dtypes,
        'input_tensor_range': input_ranges,
        'output_tensor_dtype': ['bf16'] if phase != 'write' else [],
        'num_qps': params.get('num_qps', 8),
        'hidden': hidden,
        'num_entries': num_entries,
        'num_tokens': num_tokens,
        'num_peers': params.get('num_peers', 4),
        'ranks_per_peer': params.get('ranks_per_peer', 2),
        'world_size': world_size,
        'hybrid_mode': params.get('hybrid_mode', 0),
        'phase': phase,
        'seed': case_id,
        'graph_type': 0
    }
    
    return case

def generate_xlsx(output_path: str):
    """生成xlsx文件"""
    wb = Workbook()
    
    # 删除默认sheet
    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']
    
    # 生成Level0和Level1用例
    level0_cases = generate_level0_cases()
    level1_cases = generate_level1_cases()
    
    # 创建Level0 sheet
    ws0 = wb.create_sheet('level0')
    write_cases_to_sheet(ws0, level0_cases)
    
    # 创建Level1 sheet
    ws1 = wb.create_sheet('level1')
    write_cases_to_sheet(ws1, level1_cases)
    
    # 保存文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)
    
    print(f"Generated: {output_path}")
    print(f"  Level0: {len(level0_cases)} cases")
    print(f"  Level1: {len(level1_cases)} cases")

def write_cases_to_sheet(ws, cases: List[Dict[str, Any]]):
    """将用例写入sheet"""
    # 列顺序
    columns = [
        'case_name', 'input_tensor_shape', 'input_tensor_dtype', 'input_tensor_range',
        'output_tensor_dtype', 'num_qps', 'hidden', 'num_entries', 'num_tokens',
        'num_peers', 'ranks_per_peer', 'world_size', 'hybrid_mode', 'phase',
        'seed', 'graph_type'
    ]
    
    # 写入标题行
    for col_idx, col_name in enumerate(columns, 1):
        ws.cell(row=1, column=col_idx, value=col_name)
    
    # 写入数据行
    for row_idx, case in enumerate(cases, 2):
        for col_idx, col_name in enumerate(columns, 1):
            value = case.get(col_name, '')
            
            # 编码值
            if isinstance(value, list):
                encoded = str(value)
            elif isinstance(value, tuple):
                encoded = str([value]) if col_name == 'input_tensor_range' else str(value)
            elif isinstance(value, bool):
                encoded = '1' if value else '0'
            else:
                encoded = str(value)
            
            ws.cell(row=row_idx, column=col_idx, value=encoded)

if __name__ == '__main__':
    output_path = 'mc2/result/excel/engram.xlsx'
    generate_xlsx(output_path)
    
    # 同时生成executor代码
    print("\nGenerating executor code...")
    exec_path = 'mc2/tests/st/executor_engram.py'
    print(f"Executor will be at: {exec_path}")