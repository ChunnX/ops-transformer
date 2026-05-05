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