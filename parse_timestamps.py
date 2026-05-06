import csv

def parse_timestamps(csv_file="timestamps.csv"):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = {row['core_name']: [int(row[f'point_{i}']) for i in range(100)] for row in reader}
    
    aic = data['AIC_core0']
    aiv_sub0 = data['AIV_sub0']
    aiv_sub1 = data['AIV_sub1']
    
    # 常量定义
    GMM1_BASE = 2
    GMM2_BASE = 34
    COMBINE_BASE = 66
    
    # 验证：检查 AIC 和 AIV 的 routing 开始时间是否一致
    print("=== 验证：routing 开始时间 ===")
    print(f"AIC core 0 point_0: {aic[0]}")
    print(f"AIV sub0 point_0:   {aiv_sub0[0]}")
    print(f"AIV sub1 point_0:   {aiv_sub1[0]}")
    print(f"diff (sub0 - aic):  {aiv_sub0[0] - aic[0]}")
    print(f"diff (sub1 - aic):  {aiv_sub1[0] - aic[0]}")
    
    # 验证：检查 AIC 的 gmm1 开始时间和 AIV sub0 的 gmm1 结束时间
    print("\n=== 验证：gmm1 时间 ===")
    for i in range(4):
        aic_gmm1_start = aic[GMM1_BASE + 2*i]
        aiv_gmm1_end = aiv_sub0[GMM1_BASE + 2*i]
        print(f"gmm1_{i}: AIC开始={aic_gmm1_start}, AIV结束={aiv_gmm1_end}, 耗时={aiv_gmm1_end - aic_gmm1_start} us")
    
    # 计算各模块耗时
    print("\n=== 各模块耗时 ===")
    print(f"routing: {aiv_sub0[1] - aiv_sub0[0]} us")
    print(f"combine: {aiv_sub0[COMBINE_BASE + 1] - aiv_sub0[COMBINE_BASE]} us")
    
    for i in range(4):
        print(f"dispatch_{i}: {aic[GMM1_BASE + 2*i] - aiv_sub1[2*i]} us")
        print(f"gmm1_{i}: {aiv_sub0[GMM1_BASE + 2*i] - aic[GMM1_BASE + 2*i]} us")
        print(f"gmm2_{i}: {aiv_sub0[GMM2_BASE + 2*i] - aic[GMM2_BASE + 2*i]} us")

parse_timestamps()
