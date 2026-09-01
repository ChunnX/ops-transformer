# SageAttention2 测试说明

代码仓只保留一条代表性 NPU smoke 和全 tiling-key 构建清单。完整 500 用例、精度/性能测试辅助代码、
有限值/LSE 压力脚本及原始日志作为独立自测附件 `sageAttentionTest` 交付，不提交到算子代码仓。

当前阶段只关闭功能、参数校验、全 tiling-key 可执行性和运行稳定性问题。GPU L-A/L-B 精度门禁及相对
FIA 的双接口性能门禁已安排到后续阶段，本阶段不宣称达标。

## 仓内内容

- `pytest/test_sage_attention2_smoke.py`：HND、FP16、GQA 的代表性 NPU 冒烟用例；
- `tiling_keys.txt`：完整 145 个唯一 tiling key，用于构建阶段包。

运行 smoke：

```bash
source /home/developer/Ascend/cann-9.2.0-beta.1/set_env.sh
source /path/to/sageattention2_opp/vendors/custom_transformer/bin/set_env.bash
export SAGEATTN_NPU_MODULE=sageattention2
export SAGEATTN_DEVICE=npu:0

python3 -m pytest -q -rA \
  experimental/attention/sageattention2/tests/pytest/test_sage_attention2_smoke.py
```

## 独立自测附件

`sageAttentionTest` 固定包含 500 条用例：380 条功能正向、100 条精度配置、14 条性能大形状和 6 条
参数错误用例。功能与精度配置的 480 条正向用例覆盖全部 145 个 tiling key。

附件需与以下阶段报告一并保存：

- `docs/stage_delivery_report_20260824.md`：环境、范围、结果与复现入口；
- `correctness_stage3_report_20260824.md`：本地归档报告；
- `remote_logs/full_correctness_v53_20260824/`：远端原始日志归档。

本阶段已验证结果：finite/LSE 480/480、功能 380/380、负向 6/6、大形状 14/14、ECC 定向压力
20/20。完整测试附件不得复制到代码仓。
