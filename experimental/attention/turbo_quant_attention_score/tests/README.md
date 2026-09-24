# 自测

依赖 CANN 9.1.0+、Ascend 910B、配套 PyTorch/torch_npu、numpy。
`task_golden.py` 和 `case.json` 来自任务附件；仅补齐许可证头、格式和文件末尾换行，计算逻辑及用例数据保持不变。脚本按 case.json 的五组形状生成输入，
随机种子固定为 20260923；页表使用物理块随机排列，而不是假定连续映射。

先按上级 README 构建并安装算子，加载 CANN 和算子包环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /path/to/install/vendors/turboquant_transformer/bin/set_env.bash
export TURBOQUANT_OPAPI=/path/to/install/vendors/turboquant_transformer/op_api/lib/libcust_opapi.so
export TASK_QUEUE_ENABLE=0
cd experimental/attention/turbo_quant_attention_score/tests
python test_turbo_quant_attention_score.py --suite smoke --task-golden --output smoke.json
python test_turbo_quant_attention_score.py --suite edge --output edge.json
python test_turbo_quant_attention_score.py --suite invalid --output invalid.json
python test_turbo_quant_attention_score.py --suite full --task-golden --repeats 10 --output full.json
```

五组指定 case 全部纳入精度与性能测试；edge 额外覆盖 2/3/4 bit、MHA/GQA、FP16/BF16 norm、
batch decode、单序列 prefill、尾块、跨页、长度不一致及空输出。
smoke 同时检查 CPU 长度数组与 INT64 Tensor 路径；invalid 使用 CPU 长度数组路径验证非法长度和输出形状。
`TASK_QUEUE_ENABLE=0` 确保 torch 的事件记录与 ctypes 直接提交的 aclnn 调用按同一 stream 顺序执行。
输入先按声明 dtype 舍入，再参与参考计算。独立 CPU 参考使用整数移位解包和 FP32 点积。

`device_us_median` 是预热后 NPU Event 耗时，包含解码和输出写入；
`wall_us_median` 额外包含 GetWorkspaceSize、执行提交与同步，但不包含输入生成/传输和输出分配。
记录每次原始计时样本，不以单次最小值代替中位数。
输出文件同时给出真实 workspace 返回值及 cache/output 字节数。
`npu_peak_call_bytes` 是输入已驻留后的调用峰值增量，包含输出及分配器对齐开销，不含参考实现内存。

参考计算为 FP32，向量树形归约和矩阵乘法可能有末位差异。
脚本的混合比较阈值为 `abs_error <= 1e-5 + 1e-3*abs(reference)`，并报告相对 L2 误差。
仓库贡献指南引用的 `opbase/docs/zh/ops_precision_standard/experimental_standard.md` 明确规定 FP32
使用 `atol=2^-16, rtol=2^-10, matched_ratio>=0.99, max_abs_error<=1e-2`。
当实测最大绝对误差不超过 `2^-16` 时，可直接证明所有元素满足该仓库标准；报告单独列出此判定。
任务附件 `[0.001,0]` 若另有专项比较器，应另行执行，不能将仓库标准与纯相对检查混为一谈。

`--task-golden` 直接调用未修改的任务 golden，额外记录其最大绝对误差和纯相对超限数量。
`--perf-only` 仅供调优时计时，输出明确标记跳过精度，不能用于生成验收报告。

报告依赖 openpyxl 和 matplotlib，通过以下命令从日志数据生成，保留在线模板字段及额外明细：

```bash
python generate_reports.py --full full.json --edge edge.json --invalid invalid.json \
  --output /path/to/task_submission
```
