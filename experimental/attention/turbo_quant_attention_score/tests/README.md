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

## 最终安装包回归

在上述 CANN、算子包及 `TURBOQUANT_OPAPI` 环境配置完成后执行：

```bash
python test_submission_regression.py --output extended.json
python test_submission_regression.py --perf --output performance.json
```

扩展测试覆盖 2/3/4-bit、BF16/FP16 norm、INT32/INT64 Tensor 长度、CPU 长度数组、单序列 prefill 与批量 decode，共 48 组配置；每组重新创建 executor 连续调用三次，并要求结果逐元素一致。
输入按 physical page、位置、head、query token 构造可区分的数据，页表倒序；每组还故意交换页表，要求参考比较能够检出错误，避免恒定 cache 掩盖地址映射错误。
另有 12 组负向测试，分别缩短 qjl、norm、gamma 的物理块数，覆盖两种 norm dtype 和两种长度接口，要求第一阶段返回错误。
CPU 长度数组接口不以 Tensor 的 INT32/INT64 作为其底层类型；矩阵中对应重复项用于确认输入准备一致，不表示存在额外的 CPU-array dtype 分派。

性能模式独立于精度测试，对每组输入预热 5 次、采样 20 次，报告设备事件 median、p95 和原始样本，包含短尾块、长 decode、prefill。每次调用重新创建 executor。
性能模式本身不做精度验收；应先完成 smoke/edge/invalid/full 及扩展精度测试。对照两个安装包时分别在独立进程加载，记录包 SHA256、源码提交和环境版本。

## GQA 分组及维度拒绝回归

```bash
python test_gqa_regression.py --output gqa.json
```

覆盖 GQA 分组 1/2/3/4/5/8，KV heads=2，2/3/4-bit、BF16/FP16 norm、两种长度接口；单序列 query token 数为 7/8/9（覆盖 8-token 分组边界），另含 batch=3 decode，共 288 组配置。Tensor 长度在不同 token 配置下覆盖 INT32/INT64，并非两种 dtype 的全交叉组合。
每组连续调用三次，每次重新创建 executor、将输出预填 NaN；检查所有输出均被写入、有效位置逐元素匹配独立 CPU 参考、无效位置为负无穷，并要求重复结果完全一致。页表倒序，输入区分 page/head/token。该测试可以检出漏写及导致输出错误的跨组映射，但重复结果一致不等同于证明不存在任何数据竞争。
另有 24 组负向测试：分别将 query、query residual 或两者的末维设为 64/127/129/256，在两种长度接口下均要求 GetWorkspaceSize 阶段拒绝。两者同时改变的用例排除仅因 query 形状不一致而被拒绝的情况。

## 2026-09-30 用例版本

case.json 已同步新版附件：gqa_decode_kv32k 改为 gqa_decode_kv64k（KV=65536），其它四组形状不变；golden 公式未变。历史 9 月 26 日的 32K 结果不能改名为 64K 结果。主测试脚本按 case.json 读取形状，自行生成随机压缩域输入，不读取 data_path；配对 Q/K 验证需要另外加载新版 generate_data.py 产物。

新版 baseline 为 BF16 Paged Full-QK score：decode 4K/64K/128K 为 126.892/2321.657/5386.388 μs，prefill 4K/32K 为 627.391/5910.060 μs。decode 要求不超过 1.2 倍；不能再沿用旧基线的通过结论。PPL 已删除，KL<0.01 仍保留，但未明确其缩放和统计方式。


## 32-key tile 与批量输出回归

`python test_tile_regression.py --output tile.json` 使用最终安装包运行 108 个用例，
覆盖 2/3/4 bit、BF16/FP16 norm、GQA 3/4、query token 分组边界、7/8/9、31/32/33、63/65
长度，以及多 batch 的不同有效长度和全掩码输出。页表乱序，page/head/token 内容可区分，
输出预填 NaN，沿用既有比较器，不修改误差阈值。

## norm/gamma 连续搬运与回退路径回归

```bash
python test_norm_regression.py --output norm.json
```

覆盖 2/3/4-bit、BF16/FP16 norm、KV heads=1/2/3/7/8/15/16/17/32，
以及长度 1/31/32/33/127/128/129，共 378 个安装包用例。KV heads 不超过 16 时，
内核连续搬运包含目标 head 的 norm/gamma 区间，并在 UB 中选取对应元素；更大 head 数使用原有逐行搬运。
测试使用可区分的 norm/gamma 值、乱序页表、batch=3 的不同有效长度及全掩码请求，
检查 16/17 路径切换、32-key tile 尾部和物理页边界，输出预填 NaN，沿用独立参考及既有误差阈值。

## Decode packed QJL 与预取回归

```bash
python test_packed_qjl.py --output packed_qjl.json
```

覆盖 2/3/4-bit、BF16/FP16 norm、GQA 1/3/4/5/8，以及长度
31/32/33/127/128/129/4095/4096/4097，共 270 个配置，每组重复调用两次。
QJL cache 使用遍历字节值的模式，query residual 同时包含正负值和第 63/64/127 维的非零值。
输出预填 NaN，验证 mask 两半、最高位、页尾、长预取链、所有 head 输出与重复调用稳定性。

单 token decode 的 QJL 路径直接用 packed bit 选择正负 query residual，保持原来的 FP32 归约顺序。
压缩 idx 输入使用独立 2 KiB UB 缓冲区预取下一 tile，并在消费前等待 MTE2 到 Vector 的事件。
decode UB 总量为 187 KiB，prefill 保留原来的 185 KiB 路径；GM workspace 仍为零。
3-bit 独立整数解包试验没有纳入此实现，2/3/4-bit 的主量化解包仍使用已验证的公共路径。
