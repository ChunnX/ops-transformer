# AllGatherMatmulV2 仿真精度比对通道（通道 A，2.1 §1.3/§7 T3）

> 本目录实现「cannsim 仿真 → example dump → CPU golden → §2.3 口径判定」
> 的完整链路。精度判定的**唯一取值处**是 `compare_dump.py` 顶部的
> `PRECISION_CRITERIA`（实例化自 2.1 §2.3 口径表，不得放宽/换指标）。

## 组成

| 文件 | 用途 |
|------|------|
| `run_sim_cases.sh` | 用例驱动：逐例注入参数 → `build.sh --run_example`（有卡直跑；`--sim` 回退 `--simulator=camodel` cannsim 仿真）→ 比对 → 汇总 |
| `compare_dump.py` | 比对脚本：§2.3 权威断言（output 混合容差 + gatherOut 逐位字节一致）+ 交叉核对 + 逐张量报告落盘 |
| `selftest_compare.py` | 管线自检（无 NPU/无 example）：合成 dump 验证 golden 与断言可执行、负例可检出 |
| `gen_st_json.py` | 从 `../assets/sim_cases.py` 派生 arch35 非量化 ST 用例 json（28 例参数域镜像，T4） |
| `perf/` | 性能采集框架（A8 备用，见 `perf/README.md`） |
| `../assets/sim_cases.py` | L0/L1/L2 用例表（唯一用例源，与 2.1 §5 矩阵逐条对应） |
| `../assets/impl/golden.py` | golden：通道 A CPU 实现（`agm_v2_sim_golden`）+ TTK 适配层 transB 修复（G-T1/T1） |

## 用法

```bash
# 0) 管线自检（无需 NPU / 无需 example，验证 golden+断言可执行）
python3 tests/sim/selftest_compare.py

# 1) 全量 L0+L1 执行 + 比对（需参数化 example 就绪，见下）
bash tests/sim/run_sim_cases.sh                 # L0+L1 非条件例（34 例），有卡直跑
bash tests/sim/run_sim_cases.sh --case L0-01    # 单例
bash tests/sim/run_sim_cases.sh --include-conditional   # 追加 L1-142（8 卡）
bash tests/sim/run_sim_cases.sh --include-informational # 追加 INFO_CASES（大幅值
                                                # informational 变体，判定不计门禁）
bash tests/sim/run_sim_cases.sh --dry-run       # 只打印将执行的参数
bash tests/sim/run_sim_cases.sh --sim           # 无卡回退：--simulator=camodel cannsim 仿真

# 2) 结果
#    tests/sim/reports/<case>/report_<case>.json  逐张量「张量/指标/实测值/阈值/结论」
#    tests/sim/reports/summary.jsonl              compare_dump 汇总（有 dump 的用例）
#    tests/sim/reports/driver_summary.jsonl       驱动级逐例汇总（每例一行，含
#                                                 example 退出状态/权威判定/关键
#                                                 指标/失败原因摘要——失败例不缺行）
```

执行前置条件：build 目录空闲（与其他构建任务互斥）；CANN 环境已 source；
自定义算子包已出包安装（cust 模式依赖 vendors/<vendor>_transformer 或
ASCEND_CUSTOM_OPP_PATH）。

## 与参数化 example 的接口契约（对 3.1 种子 example）

### 1. case 参数注入（环境变量）

`run_sim_cases.sh` 以如下 env 驱动 example（`build.sh --run_example` 的
子进程继承）：

| env | 含义 | 取值 |
|-----|------|------|
| `AGM_V2_CASE_ID` | 用例号（2.1 §5） | `L0-01` … |
| `AGM_V2_DUMP_DIR` | dump 输出目录 | 路径 |
| `AGM_V2_DTYPE` | dtype | `fp16` / `bf16` |
| `AGM_V2_M` / `AGM_V2_K` / `AGM_V2_N` | shape | 整数（m=0/x2 n=0 合法） |
| `AGM_V2_RANK_SIZE` | 仿真卡数 | 2 / 4 / 8 |
| `AGM_V2_COMM_MODE` | 通信引擎 | `ai_cpu` / `ccu` |
| `AGM_V2_BIAS` / `AGM_V2_TRANS_B` / `AGM_V2_GATHER_OUT` | 因子开关 | 0 / 1 |
| `AGM_V2_VALUE_SPEC` | 输入数值语义 | `clean` / `signed_zero` / `inf_nan` / `magnitude_bands` |
| `AGM_V2_VALUE_SCALE` | 输入幅值缩放系数（乘在 VALUE_SPEC 生成值上，P1 处置 2026-09-20） | (0,1]，默认 1.0 |
| `AGM_V2_INFORMATIONAL` | informational 用例属性透传（驱动识别；example 不消费） | 0 / 1 |
| `AGM_V2_REPEAT_LAUNCH` | 确定性重复 launch 次数 | ≥1（>1 时逐 run dump） |
| `AGM_V2_SEED` | 确定性伪随机种子（per-rank 派生区分） | 整数 |

未设置任一 `AGM_V2_*` 时 example 回退到内置默认（等价 `L0-01`，scale=1.0），
保证既有调用方式不受影响。

`AGM_V2_VALUE_SPEC` 语义（与 `selftest_compare.py::gen_values` 一致）：
- `clean`：80% uniform(-5,5) + 20% uniform(-1,1)（小值带分层，含负镜像）；
- `signed_zero`：约 20% 元素为 +0/-0（符号位随机）；
- `inf_nan`：约 0.1% +inf、0.1% -inf、0.3% nan；
- `magnitude_bands`：[1,5]/[1e-2,1e-1]/[1e-4,1e-3]/[0,1e-6] 四带等分，
  符号随机。

`AGM_V2_VALUE_SCALE` 语义（P1 处置）：对上述 spec 生成值**统一乘**该系数
（clean ±5→±0.5、小值带 ±1→±0.1；x1/x2/bias 全部生成值含全 rank 重放，
单点应用于 `GenSpecValue`）；±0/±inf/nan 位形经 IEEE 乘法保持
（-0×s=-0、±inf×s=±inf、nan×s=nan）。缩放后输入语义仍为 clean 分层采样，
仅幅值量级变化；golden（compare_dump/golden.py）从 dump 字节重算，天然
与缩放无关。informational 用例（`INFO_CASES`，大幅值原幅值变体）刻意
**不缩放**且与母用例同种子，逐位复现缩放前输入，用于「§2.3 容差口径 +
fp64 归因」双指标记录（3.4 报告 v1.2）；判定加 `INFO_` 前缀，不计入
驱动级 PASS/FAIL 门禁。

### 2. dump 落盘格式

```
<AGM_V2_DUMP_DIR>/
  meta.json              # case 参数 + x2 视图 shape/strides（元素单位）
  x1_rank{r}.bin         # per-rank 输入，行主序原始字节（dtype 同声明）
  x2_rank{r}.bin         # x2 API 实参的完整 storage 字节（见下）
  bias_rank{r}.bin       # AGM_V2_BIAS=1 时存在
  output_rank{r}.bin     # (m*R, n)
  gather_out_rank{r}.bin # AGM_V2_GATHER_OUT=1 时存在，(m*R, k)
  output_rank{r}_run{i}.bin  # REPEAT_LAUNCH>1 时 i≥1 追加
```

- **transB（`AGM_V2_TRANS_B=1`）**：x2 按 **950 CCU/ai_cpu 外层 aclnn 实参
  形态**落盘——storage 为 **W.t() 连续（(n,k) 行主序）**，`meta.json
  ["tensors"]["x2"] = {"shape": [k, n], "strides": [1, k]}`（op_api
  CheckShape 按 view dim0 取 k 轴、IsTransposeLastTwoDims 按 stride 判定、
  TransX2Tensor 重建 inner (n,k) desc）。比对侧经 `torch.as_strided` 还原
  视图后直接作为逻辑 W 参与计算；inner/TTK 的 (n,k) 形态同样接受
  （golden 按视图 shape 与 k 判别，见 `golden.py::logical_weight_from_view`）；
- **空 tensor**：0 字节文件（m=0 / n=0 / gatherOut 空传入）；
- `meta.json` 必含：`case_id/desc/dtype/m/k/n/rank_size/comm_mode/bias/
  trans_b/gather_out/value_spec/value_scale/repeat_launch`。

### 3. 计时打印（perf 备用，T7）

每次 launch 后打印一行：`AGM_V2_TIMING_US: <us>`（`perf/collect_perf.sh`
采集、`perf_report.py` 解析）。

## 驱动级判定与汇总（P2 处置 2026-09-20）

- **权威判定 = `compare_dump.py`**（§2.3 口径表实现处）。example 退出码仅
  记录：dump 完整（`meta.json` + `output_rank0.bin`）即执行权威比对，不因
  example 内嵌自校验失败而跳过（v1.1 曾因此失败例 summary 缺行）；example
  内嵌自校验已对齐 §2.3（混合容差 + matched_ratio≥0.99 + max_abs_error
  上限，见参数化 example 头注释），与权威口径单一语义。
- **不掩盖真实崩溃**：example 异常退出且 dump 不完整（未到 kernel 输出/
  建链失败/段错误）→ 直接计 FAIL（driver_summary 记 `crash/launch
  failure`），不做权威比对。
- **informational 用例**（`INFO_CASES` 大幅值原幅值变体）：判定
  `INFO_PASS`/`INFO_FAIL`，不计入驱动级门禁与退出码；报告按「§2.3 容差
  口径 + fp64 归因」双指标记录（3.4 报告 v1.2）。
- **driver_summary.jsonl**：每例一行（`driver_case_summary.py` 生成）——
  `case_id / informational / example_exit_ok / compare_pass / verdict /
  reason / metrics`；失败例另有驱动日志逐例一行汇总（末尾
  `non-PASS case summary` 段）。

## 精度判定实现位置（§2.3 口径表 → 可执行断言）

| 输出张量 | 指标 | 阈值 | 判定代码 |
|----------|------|------|---------|
| output (fp16) | 混合容差 + matched_ratio + max_abs_error | rtol=atol=2^-9；matched_ratio≥0.99；max_abs_error≤0.1 | `compare_dump.py::check_mixed_tolerance`（`PRECISION_CRITERIA["fp16"]`） |
| output (bf16) | 同上 | rtol=atol=2^-6；matched_ratio≥0.99；max_abs_error≤1.0 | 同上（`PRECISION_CRITERIA["bf16"]`） |
| gatherOut | 逐位字节一致（tobytes 级，NaN 位形/±0 符号位严格检出） | 0 | `compare_dump.py::check_bitwise` |
| output（x1 含 ±inf/nan 特例） | 传播语义核对（信息性，不作通过依据） | — | `compare_dump.py::check_propagation` |
| 确定性（L1-170） | run 间 bitwise 一致 | bitwise | `compare_dump.py::compare_case`（determinism 段） |
| 交叉核对（不作通过依据） | cv_fused_double_benchmark（max_re_ratio≤5 / avg_re_ratio≤1.5，vs 双精度 golden） | 见上 | `compare_dump.py::crosscheck_st_acc`；与权威口径结论不一致时以口径表为准并记录 `crosscheck_divergences` |

## ST 依赖（有卡执行前须解决，T4 资产已就绪）

`gen_st_json.py` 生成的 28 例 json 与 `executor_aclnnAllGatherMatmulV2.py`
（沿用）存在三处执行依赖：

1. **bias 实参置 null**：executor `init_by_input_data` 无条件
   `input_args[2] = null`——bias 用例（remark `is_bias=1`）需 executor
   非量化扩展（bias 透传）后方可执行，否则 golden 加 bias 而设备侧未加；
2. **transB 实参视图构造**：json 声明的 x2 为 (n,k)（TTK/inner 形态，
   与既有 json 约定一致）；950 CCU/ai_cpu 路径 aclnn 实参要求
   (k,n) view + strides (1,k)（op_api CheckShape 按 view dim0 取 k 轴，
   IsTransposeLastTwoDims 按 stride 判定）——executor 需按 remark
   `is_trans_b=1` 构造 `x2.t()` 形态的实参（存储不变，视图转置）；
3. **gatherOut 空 tensor**：`gather_output=0` 用例需传空 tensor 触发
   `IsGatherOut=false` 路径。

json 侧已按语义就绪（remark 标注齐全、bias dtype 与 x1 一致、无 scale），
上述 executor 扩展属公共资产改动，已回传 PM 协调。
