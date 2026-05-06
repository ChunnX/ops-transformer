# 只有 3 个核心，每个 100 个打点
y_timestamps = y.cpu().view(torch.uint64).flatten()[:300].reshape(3, 100).to(torch.int64)
# 全局核心索引映射
# globalCoreIdx 0: AIC core 0 (subBlockIdx_=0)
# globalCoreIdx 1: AIV core 0 (subBlockIdx_=0)
# globalCoreIdx 2: AIV core 1 (subBlockIdx_=1)
MAX_EXPERT = 16
DISPATCH_BASE = 2
GMM1_BASE = 2 + 2 * MAX_EXPERT
GMM2_BASE = 2 + 2 * MAX_EXPERT + 2 * MAX_EXPERT
COMBINE_BASE = 2 + 4 * MAX_EXPERT + 2 * MAX_EXPERT
def calc_latency(num_experts=4):
    aic_ts = y_timestamps[0]   # AIC core 0
    aiv_sub0_ts = y_timestamps[1]  # AIV sub0
    aiv_sub1_ts = y_timestamps[2]  # AIV sub1
    
    result = {
        'routing': aiv_sub0_ts[1] - aiv_sub0_ts[0],
        'combine': aiv_sub0_ts[COMBINE_BASE + 1] - aiv_sub0_ts[COMBINE_BASE],
    }
    
    for i in range(num_experts):
        # dispatch: AIC 结束 - AIV sub1 开始
        result[f'dispatch_{i}'] = (
            aic_ts[DISPATCH_BASE + 2*i + 1] - 
            aiv_sub1_ts[DISPATCH_BASE + 2*i]
        )
        # gmm1: AIV sub0 结束 - AIC 开始
        result[f'gmm1_{i}'] = (
            aiv_sub0_ts[GMM1_BASE + 2*i + 1] - 
            aic_ts[GMM1_BASE + 2*i]
        )
        # gmm2: AIV sub0 结束 - AIC 开始
        result[f'gmm2_{i}'] = (
            aiv_sub0_ts[GMM2_BASE + 2*i + 1] - 
            aic_ts[GMM2_BASE + 2*i]
        )
    
    result['dispatch_total'] = sum(result[f'dispatch_{i}'] for i in range(num_experts))
    result['gmm1_total'] = sum(result[f'gmm1_{i}'] for i in range(num_experts))
    result['gmm2_total'] = sum(result[f'gmm2_{i}'] for i in range(num_experts))
    
    return result
print("Latency (AIC core 0 + AIV core 0/1):")
for name, val in calc_latency(num_experts=4).items():
    print(f"  {name}: {val} us")



import torch
import csv
import sys
def save_timestamps(y, output_file="timestamps.csv"):
    """
    保存时间戳到 CSV 文件
    y: 输出 tensor (torch.Tensor)
    output_file: 输出 CSV 文件路径
    """
    # 提取时间戳数据（3个核心，每个100个点）
    y_timestamps = y.cpu().view(torch.uint64).flatten()[:300].reshape(3, 100)
    
    # globalCoreIdx: 0=AIC, 1=AIV sub0, 2=AIV sub1
    core_names = ["AIC_core0", "AIV_sub0", "AIV_sub1"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ["core_name"] + [f"point_{i}" for i in range(100)]
        writer.writerow(header)
        
        # 写入每个核心的时间戳
        for i, name in enumerate(core_names):
            row = [name] + y_timestamps[i].tolist()
            writer.writerow(row)
    
    print(f"时间戳已保存到 {output_file}")
    print(f"格式: 每行一个核心，共 {len(core_names)} 行")
    print(f"列: core_name, point_0, point_1, ..., point_99")


def parse_timestamps(csv_file="timestamps.csv"):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = {row['core_name']: [int(row[f'point_{i}']) for i in range(100)] for row in reader}
    
    aic = data['AIC_core0']
    aiv_sub0 = data['AIV_sub0']
    aiv_sub1 = data['AIV_sub1']
    
    # 常量定义（与 kernel 代码一致）
    MAX_EXPERT_PER_RANK = 16
    GMM1_BASE = 2
    GMM2_BASE = 2 + 2 * MAX_EXPERT_PER_RANK  # = 34
    COMBINE_BASE = 2 + 4 * MAX_EXPERT_PER_RANK  # = 66
    
    # 检查初始时间
    print("=== 进入Process的初始时间 ===")
    print(f"AIC core 0: {aic[99]} us")
    print(f"AIV sub0:   {aiv_sub0[99]} us")
    print(f"AIV sub1:   {aiv_sub1[99]} us")
    print(f"diff (sub0 - aic): {aiv_sub0[99] - aic[99]} us")
    print(f"diff (sub1 - aic): {aiv_sub1[99] - aic[99]} us")
    
    # 计算各模块耗时
    print("\n=== 各模块耗时 ===")
    print(f"routing: {aiv_sub0[1] - aiv_sub0[0]} us")
    print(f"combine: {aiv_sub0[COMBINE_BASE + 1] - aiv_sub0[COMBINE_BASE]} us")
    
    for i in range(4):
        print(f"dispatch_{i}: {aic[GMM1_BASE + 2*i] - aiv_sub1[2*i]} us")
        print(f"gmm1_{i}: {aiv_sub0[GMM1_BASE + 2*i] - aic[GMM1_BASE + 2*i]} us")
        print(f"gmm2_{i}: {aiv_sub0[GMM2_BASE + 2*i] - aic[GMM2_BASE + 2*i]} us")
parse_timestamps()
    
# //
# // 打点布局（40个槽位，MAX_EXPERT_PER_RANK=4）：
# // AIC 核：
# //   0: routing 开始
# //   1: routing 结束
# //   2+2*i: gmm1 开始 (i=0-3 → 2,4,6,8)
# //   3+2*i: dispatch 结束（GroupMatmulSwigluQuant 的 while 循环后）
# //   10+2*i: gmm2 开始（GroupMatmul2 的 while 循环后）
# //   10+2*i+1: gmm2 结束（EndGMM2Sync 后）
# // AIV sub0（偶数core）：
# //   0: routing 开始
# //   1: routing 结束
# //   18: combine 开始
# //   19: combine 结束
# // AIV sub1（奇数core）：
# //   2*i: dispatch 开始
# //   2*i+1: dispatch 结束
    
# def parse_timestamps(csv_file="timestamps.csv"):
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         data = {row['core_name']: [int(row[f'point_{i}']) for i in range(40)] for row in reader}
    
#     NUM_GROUPS = 10
#     AIV_OFFSET = NUM_GROUPS  # AIV 的 globalCoreIdx 从 10 开始
#     NUM_EXPERTS = 4
    
#     # 常量（与 kernel 一致）
#     GMM1_BASE = 2
#     GMM2_BASE = 2 + 2 * NUM_EXPERTS  # = 10
#     COMBINE_BASE = 2 + 4 * NUM_EXPERTS  # = 18
    
#     print(f"=== 前10组group各模块耗时 (us) ===")
#     print(f"布局: routing(0-1), gmm1+dispatch({GMM1_BASE}-{GMM1_BASE+2*NUM_EXPERTS-1}), "
#           f"gmm2({GMM2_BASE}-{GMM2_BASE+2*NUM_EXPERTS-1}), combine({COMBINE_BASE}-{COMBINE_BASE+1})\n")
    
#     for group_idx in range(NUM_GROUPS):
#         aic_key = f"AIC_core{group_idx}"
#         aiv_sub0_key = f"AIV_core{AIV_OFFSET + 2 * group_idx}"
#         aiv_sub1_key = f"AIV_core{AIV_OFFSET + 2 * group_idx + 1}"
        
#         if aic_key not in data or aiv_sub0_key not in data or aiv_sub1_key not in data:
#             print(f"Group {group_idx}: 数据缺失")
#             continue
        
#         aic = data[aic_key]
#         aiv_sub0 = data[aiv_sub0_key]
#         aiv_sub1 = data[aiv_sub1_key]
        
#         routing = aiv_sub0[1] - aiv_sub0[0]
#         combine = aiv_sub0[COMBINE_BASE + 1] - aiv_sub0[COMBINE_BASE]
        
#         print(f"--- Group {group_idx} ---")
#         print(f"  routing: {routing}, combine: {combine}")
        
#         dispatch_total = 0
#         gmm1_total = 0
#         gmm2_total = 0
        
#         for i in range(NUM_EXPERTS):
#             d = aic[GMM1_BASE + 2*i + 1] - aiv_sub1[2*i]
#             g1 = aic[GMM2_BASE + 2*i] - aic[GMM1_BASE + 2*i]
#             g2 = aic[GMM2_BASE + 2*i + 1] - aic[GMM2_BASE + 2*i]
            
#             dispatch_total += d
#             gmm1_total += g1
#             gmm2_total += g2
            
#             print(f"  expert_{i}: dispatch={d}, gmm1={g1}, gmm2={g2}")
        
#         print(f"  --- 汇总 ---")
#         print(f"  dispatch_total={dispatch_total}, gmm1_total={gmm1_total}, gmm2_total={gmm2_total}")
#         print()
# parse_timestamps()