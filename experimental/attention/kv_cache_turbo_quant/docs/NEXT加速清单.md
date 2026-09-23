# 下一步（加速清单）· 2026-09-23

## 你现在处在哪

| 已完成 | 未完成 |
|--------|--------|
| 金样 / 孪生 / Kernel 迁入 | 任务书 5 case **NPU** 全量打包对拍 |
| 挂仓编译 + 安装 + smoke 字节绿 | Kernel **向量化**（否则 10× 够不着） |
| CPU 精度矩阵 5 case ALL PASS | 设计文档 PR + 腾讯自测报告 + 个人仓邀审 |

## 今天立刻（人做 / 5～15 分钟）

1. **广场更新进展**（别人已更新 1 次，你别落后）  
   文案可粘贴：  
   `Encode Kernel 已挂 ops-transformer；A3 NPU smoke 与 CPU 孪生 idx/qjl 字节一致；CPU 五档 case 精度矩阵 pack+IP95 已绿。下一步：NPU 五档对拍 + Kernel 向量化冲 10×。`
2. 确认「我的任务」是否已从审核中变为可开发。  
3. **Fork** `ops-transformer` 到个人仓，邀请 `Ascend-CANN`（交验硬条件，越早越好）。

## Agent 下一条主线（自动加速）

1. NPU：把 smoke 扩成 case.json 五档打包对拍 + 写 `logs/npu_precision_*.log`  
2. Kernel：去掉热路径 GetValue，改 DataCopy/Cast/向量点积（否则性能必挂）  
3. 设计文档草稿按 cann-competitions 模板填进 `docs/`

## 本机可复现命令

```bash
# CPU 精度矩阵
cd /root/ascend-work/deliver/kv_cache_turbo_quant/tests
python3 run_cpu_precision_matrix.py
# 期望：ALL PASS；日志 tests/logs/cpu_precision_matrix.log

# NPU smoke（已绿）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/vendors/custom_transformer/bin/set_env.bash
/tmp/test_aclnn_kv_cache_turbo_quant /root/ascend-work/deliver/kv_cache_turbo_quant/tests/data
```
