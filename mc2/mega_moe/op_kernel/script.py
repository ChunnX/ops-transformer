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
    
    # 检查初始时间
    print("=== 进入Process的初始时间 ===")
    print(f"AIC core 0: {aic[99]} us")
    print(f"AIV sub0:   {aiv_sub0[99]} us")
    print(f"AIV sub1:   {aiv_sub1[99]} us")
    
    # 计算各模块耗时
    GMM1_BASE = 2
    GMM2_BASE = 34
    COMBINE_BASE = 66
    
    print("\n=== 各模块耗时 ===")
    print(f"routing: {aiv_sub0[1] - aiv_sub0[0]} us")
    print(f"combine: {aiv_sub0[COMBINE_BASE + 1] - aiv_sub0[COMBINE_BASE]} us")
    
    for i in range(4):
        print(f"dispatch_{i}: {aic[GMM1_BASE + 2*i] - aiv_sub1[2*i]} us")
        print(f"gmm1_{i}: {aiv_sub0[GMM1_BASE + 2*i] - aic[GMM1_BASE + 2*i]} us")
        print(f"gmm2_{i}: {aiv_sub0[GMM2_BASE + 2*i] - aic[GMM2_BASE + 2*i]} us")
parse_timestamps()