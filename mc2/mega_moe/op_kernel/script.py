# // def calc_latency(core_idx, expert_idx=0):
# //     ts = y_timestamps[core_idx]
# //     return {
# //         'routing': ts[1] - ts[0],
# //         'dispatch': ts[2 + 2*expert_idx + 1] - ts[2 + 2*expert_idx],
# //         'gmm1': ts[34 + 2*expert_idx + 1] - ts[34 + 2*expert_idx],
# //         'gmm2': ts[66 + 2*expert_idx + 1] - ts[66 + 2*expert_idx],
# //         'combine': ts[99] - ts[98],
# //     }
# // for core_idx in range(12):
# //     core_type = "AIC" if core_idx < 4 else "AIV"
# //     print(f"{core_type} Core {core_idx}: {calc_latency(core_idx)}")


# // # 读取时间戳
# // y_timestamps = y.cpu().view(torch.uint64).flatten()[:1200].reshape(12, 100)
# // # 打点布局常量
# // MAX_EXPERT = 16
# // DISPATCH_BASE = 2
# // GMM1_BASE = 2 + 2 * MAX_EXPERT
# // GMM2_BASE = 2 + 2 * MAX_EXPERT + 2 * MAX_EXPERT
# // COMBINE_BASE = 2 + 4 * MAX_EXPERT + 2 * MAX_EXPERT
# // def calc_all_latencies(core_idx, num_experts=4):
# //     ts = y_timestamps[core_idx]
# //     result = {
# //         'routing': ts[1] - ts[0],
# //         'combine': ts[COMBINE_BASE + 1] - ts[COMBINE_BASE],
# //     }
    
# //     # 每个专家的耗时
# //     for i in range(num_experts):
# //         result[f'dispatch_{i}'] = ts[DISPATCH_BASE + 2*i + 1] - ts[DISPATCH_BASE + 2*i]
# //         result[f'gmm1_{i}'] = ts[GMM1_BASE + 2*i + 1] - ts[GMM1_BASE + 2*i]
# //         result[f'gmm2_{i}'] = ts[GMM2_BASE + 2*i + 1] - ts[GMM2_BASE + 2*i]
    
# //     # 汇总统计
# //     result['dispatch_total'] = sum(result[f'dispatch_{i}'] for i in range(num_experts))
# //     result['gmm1_total'] = sum(result[f'gmm1_{i}'] for i in range(num_experts))
# //     result['gmm2_total'] = sum(result[f'gmm2_{i}'] for i in range(num_experts))
    
# //     return result
# // # 打印所有核心的结果
# // for core_idx in range(12):
# //     core_type = "AIC" if core_idx < 4 else "AIV"
# //     print(f"\n{core_type} Core {core_idx}:")
# //     latencies = calc_all_latencies(core_idx, num_experts=4)
# //     for name, val in latencies.items():
# //         print(f"  {name}: {val} us")

# // y_timestamps = y.cpu().view(torch.uint64).flatten()[:1200].reshape(12, 100)
# // # 核心索引映射
# // # AIC 核心 0-3 (subBlockIdx_=0) → globalCoreIdx 0-3
# // # AIV 核心 (subBlockIdx_=0) → globalCoreIdx 4, 6, 8, 10
# // # AIV 核心 (subBlockIdx_=1) → globalCoreIdx 5, 7, 9, 11
# // MAX_EXPERT = 16
# // DISPATCH_BASE = 2
# // GMM1_BASE = 2 + 2 * MAX_EXPERT
# // GMM2_BASE = 2 + 2 * MAX_EXPERT + 2 * MAX_EXPERT
# // COMBINE_BASE = 2 + 4 * MAX_EXPERT + 2 * MAX_EXPERT
# // def calc_latency(num_experts=4):
# //     # subBlockIdx_==0 的 AIV 核（用于 gmm 结束、routing、combine）
# //     aiv_sub0_cores = [4, 6, 8, 10]  # globalCoreIdx
# //     # subBlockIdx_==1 的 AIV 核（用于 dispatch 开始）
# //     aiv_sub1_cores = [5, 7, 9, 11]
# //     # AIC 核（用于 dispatch 结束、gmm 开始）
# //     aic_cores = [0, 1, 2, 3]
    
# //     # routing/combine 只看 subBlockIdx_==0 的 AIV 核（取第一个）
# //     aiv_sub0_ts = y_timestamps[aiv_sub0_cores[0]]
# //     result = {
# //         'routing': aiv_sub0_ts[1] - aiv_sub0_ts[0],
# //         'combine': aiv_sub0_ts[COMBINE_BASE + 1] - aiv_sub0_ts[COMBINE_BASE],
# //     }
    
# //     for i in range(num_experts):
# //         # dispatch: AIC 结束 - subBlockIdx_==1 的 AIV 开始
# //         dispatch_ends = [y_timestamps[aic][DISPATCH_BASE + 2*i + 1] for aic in aic_cores]
# //         dispatch_starts = [y_timestamps[aiv][DISPATCH_BASE + 2*i] for aiv in aiv_sub1_cores]
# //         result[f'dispatch_{i}'] = (sum(dispatch_ends)/4 - sum(dispatch_starts)/4).item()
        
# //         # gmm1: subBlockIdx_==0 的 AIV 结束 - AIC 开始
# //         gmm1_starts = [y_timestamps[aic][GMM1_BASE + 2*i] for aic in aic_cores]
# //         gmm1_ends = [y_timestamps[aiv][GMM1_BASE + 2*i + 1] for aiv in aiv_sub0_cores]
# //         result[f'gmm1_{i}'] = (sum(gmm1_ends)/4 - sum(gmm1_starts)/4).item()
        
# //         # gmm2: subBlockIdx_==0 的 AIV 结束 - AIC 开始
# //         gmm2_starts = [y_timestamps[aic][GMM2_BASE + 2*i] for aic in aic_cores]
# //         gmm2_ends = [y_timestamps[aiv][GMM2_BASE + 2*i + 1] for aiv in aiv_sub0_cores]
# //         result[f'gmm2_{i}'] = (sum(gmm2_ends)/4 - sum(gmm2_starts)/4).item()
    
# //     # 汇总
# //     result['dispatch_total'] = sum(result[f'dispatch_{i}'] for i in range(num_experts))
# //     result['gmm1_total'] = sum(result[f'gmm1_{i}'] for i in range(num_experts))
# //     result['gmm2_total'] = sum(result[f'gmm2_{i}'] for i in range(num_experts))
    
# //     return result
# // print("Latency (avg across cores):")
# // for name, val in calc_latency(num_experts=4).items():
# //     print(f"  {name}: {val} us")