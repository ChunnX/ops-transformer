# MoeTokenUnpermuteWithEp Ascend 950 新工作流验证记录

## 总体结论

状态：**BLOCKED，不是 PASS，也不宣称新工作流已完成**。

已交付通路的 NPU 功能和精度门禁均已闭环：Kernel 白盒 106/106、ACLNN
L0/L1 正向 648/648、ACLNN L2 异常 35/35 正确拒绝、GEIR 静态 14/14、GEIR
动态 `-1/-2` 6/6，均为 0 FAIL、0 SKIP。剩余阻塞是 Step4 外部 GPU/XPU
三方精度与性能，以及本机 TTK 不提供 `geir` 子命令所对应的字面门禁。最终
`ops-transfer-test-scan` 也仍包含 FAIL/REMEDIATE/UNAVAILABLE 项，详见下文。
Ascend950 专属修改已经过平台隔离整改；非 3510 源码分支与 `origin/master`
逐行一致，Ascend910B 全包编译通过，但未在 910B 真机运行。

## 范围与版本

- 仓库：`/home/z50058944/code/ops-transformer_moe`。
- 分支：`codex/moe-token-unpermute-with-ep-ascend950`。
- 实现检查点：`3951ec00487871ea02568d2b8566fe3798bd6b20`。
- 测试发现问题后的独立修复：
  - `f04eecc40`：ACLNN 契约校验与空 range 回退；
  - `b0d50e8a1`：拒绝超出真实契约的 sortedIndices；
  - `5d71404bd`：稳定高 TopK 累加精度；
  - `289067d40`：将前述 950 Kernel、Host tiling 与手写 ACLNN 改动完整隔离，
    非 3510 恢复 `origin/master` 逻辑。
- 目标设备：`Ascend950DT_9592`。
- 仅验证现有 Kernel、ACLNN 和仓内 OpDef/InferShape 对应 GEIR 通路。
- Torch Eager、Torch 图、TensorFlow、ONNX：**不涉及**，未新增接口。

## 公共路径影响与平台隔离

- Kernel 公共头文件中的 indices 断言、高 TopK 稳定归约、4 个 UB buffer、初始化和
  调用点，均由仓内标准条件
  `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)` 包围；`#else` 选择的
  430 个非空源码行与 `origin/master` 完全一致。
- Host tiling 不使用 `__NPU_ARCH__`。空 range、参数校验和稳定归约 UB 预算仅在
  CMake 为 ascend950 注入的 `MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950` 下生效；
  tiling `.cpp/.h` 的非 950 分支分别以 338/88 个非空行逐行等同基线。
- ascend950 使用 `op_host_aclnnExc`、`aclnn_exclude` 和
  `op_host/op_api/ascend950` 手写 API；非 950 仍使用原 `op_host_aclnn`、原编译选项
  和自动 `ACLNNTYPE aclnn`。
- post-`289067d40` Ascend910B 全包编译实际生成自动 ACLNN 源和 3 个 Kernel 对象；
  编译数据库中没有 950 手写 API，也没有 950 Host 宏。run 包 SHA-256 为
  `01ed02ca818629112d2e25af9c04e49604611d3cd463e9a509a13b34aa29ec92`。
- 910B 只完成编译验证；当前未分配 910B 设备，故没有真机回归结论。

完整逐文件审计、源码哈希和 910B 构建快照哈希见
`tests/results/platform_isolation/`；原始 build/package 位于 Git 外
`/home/z50058944/moe_platform_isolation_evidence_289067d40/`。

## 门禁结果

| 门禁 | 结果 | 生成/清单/执行 | PASS/FAIL/SKIP | 说明 |
|---|---|---:|---:|---|
| TTK 环境 `aclnnAdd` | PASS | 1/1/1 | 1/0/0 | 固定个人 CANN、Python、TTK 环境，真机连通 |
| 平台隔离静态审计 | PASS | 4 个公共源/CMake + 1 个 910B 构建快照 | 全部检查通过 | 非 3510 分支逐行等同 `origin/master`；Host 不使用 `__NPU_ARCH__` |
| Ascend950 Host UT | PASS | 30/30/30 | 30/0/0 | tiling 21/21、InferShape 9/9 |
| Ascend910B 全包编译 | PASS（仅编译） | 1/1/1 | 1/0/0 | 原自动 ACLNN、原 Host 宏和 3 个旧平台 Kernel 对象；未做 910B 真机运行 |
| Golden 固定向量 CPU 自检 | PASS | 7/7/7 | 7/0/0 | 有/无 probs、EP range、空 range、range 外屏蔽、混合 dtype、Inf 屏蔽 |
| ACLNN L0 | PASS | 48/48/48 | 48/0/0 | 旧 16 条仅作为其中冒烟种子 |
| ACLNN L1 | PASS | 600/600/600 | 600/0/0 | 去重后的有效正向用例；一次调用前设备故障经原条重跑闭环 |
| ACLNN L2 | PASS | 35/35/35 | 35 正确拒绝/0 未证明/0 SKIP | 接口错误与目标 kernel assert 均有独立证据；随后合法控制 1/1 PASS |
| Kernel 白盒 | PASS | 106/106/106 | 106/0/0 | 105 regular + 1 oversized；无抽样 |
| 确定性重复 | PASS | 2/2/2 | 2/0/0 | 同一输出 264 B，两次 SHA-256 完全一致 |
| GEIR 静态 ES | PASS | 14/14/14 | 14/0/0 | Shape、dtype、逐元素数值全校验 |
| GEIR 动态 Shape `-1` | PASS | 3/3/3 | 3/0/0 | 同一图、同一 Session，3 组 concrete shape |
| GEIR 动态 Rank `-2` | PASS | 3/3/3 | 3/0/0 | 协议 concrete rank 固定为 2/1/2，覆盖 3 组尺寸 |
| 字面 `ttk geir` | BLOCKED | 0/0/0 | 0/0/0 | 当前 TTK 无 `geir` 子命令；不能把直连 ES 冒充 TTK |
| Step4 NPU/CPU/GPU 三方 | BLOCKED | 0/0/0 | 0/0/0 | 外部 XPU endpoint 未配置；未产出三方比值 |
| 共性转测扫描 | FAIL/REMEDIATE/UNAVAILABLE | 1 次完整扫描 | 不适用 | 原始报告保留，未改写为 PASS |

所有 TTK 数值套件均使用 `--seed 42 --pc 1`，先校验、再单例、最后完整执行；
没有使用 `--tc`、quick、max-cases 或其他抽样参数。

早期 ACLNN 全量曾有一条在进入目标 API/kernel 前遇到 `aclrtSetDevice 507033`，
原始失败和同条重跑证据均保留。平台隔离提交后的最终完整原始执行不再需要结果
合并：648 个唯一用例直接 648 PASS、0 FAIL、0 SKIP，生成数、清单数和执行数均为
648。ACLNN 的 NPU 对独立 CPU Golden 门禁显式使用
`--compare close`；`cross_check L0` 仍由 Spec 注册并留给 Step4 三方门禁，后者因
外部 XPU endpoint 缺失保持 BLOCKED。

## 测试设计与覆盖

### ACLNN 黑盒

`ascendc-st-design` 生成链先产生参数、因子、约束、求解配置和因子值中间件，再经
`ascendc-950-testcase` 归一化、去重、合并和审计。canndev 200 条只作为输入种子，
其中 112 条归一化后被吸收；最终独立清单为 L0 48、L1 600、L2 35。

正向矩阵覆盖：

- permutedTokens：FP16、BF16、FP32；sortedIndices：INT32；probs：FP16、BF16、
  FP32 和缺省，共 12 种 tokens/probs 组合；输出 dtype 与 tokens 一致。
- 仅 ND；非连续 tokens、indices、probs 和输出视图按 FP16/BF16/FP32 各 1 条实跑。
- TopK：1、2、8、511、512。
- hidden：1、7、15/16/17、31/32/33、63/64/65、255/256/257、511/512/513、
  1024、2048、4096、7619、7680。
- token 数：1、2、3、4、5、17、64、256、520、525、976、1024、1953、2048、
  16384。
- range：完整、部分、非零 start、上下边界、range 外索引贡献 0、空 range 回退；
  probs 有/无均覆盖。
- 值域：正负零、一般正负值、大值、`+Inf`、`-Inf`、`NaN`。

L2 逐类覆盖并全部由目标接口或目标 kernel 正确拒绝：错误 dtype、4 个张量错误
format、tokens/indices/probs/output 错误 rank、3 个 9 维输入、shape 耦合错误、
TopK 0/-1/513、非法 range 长度/顺序/负值、`paddedMode=true`、非空
`restoreShapeOptional`、4 类空 Tensor，以及 sortedIndices 负值和上界值。空 Tensor
按当前真实契约为“不支持”，不是正向空算子语义。

### Kernel 白盒

白盒资产由 `whitebox-designer` 统一入口生成，未手写 cases/CSV：

- `design_doc_cases.json`：106 条；
- regular CSV：105 条；oversized CSV：1 条；
- `validate-inputs`、regular/oversized `validate-ttk`、`validate-shapes`、
  `coverage-audit` 和语义审计全部 PASS；
- 覆盖 FP16/BF16/FP32、probs 有/无、TopK 1/2/8/511/512、4 个 tilingKey、
  对齐/非对齐/质数非对齐、EP 非零 start、零 start、空 range 回退和 range 外索引；
- oversized 为 `[134212224]` 级总元素用例，独立完整执行并 PASS；
- 初次全量暴露 Golden 对 range 外 `Inf` 的 `0*Inf` 问题，保存失败结果后修正
  Golden/third_party 的屏蔽顺序，定向回归再全量 105/105 PASS。

### Golden

`tests/assets/golden.py` 按实际 raw Kernel、ACLNN、GEIR 注册三个独立 Spec。Kernel 与
GEIR 只在边界调用 TTK 官方 `numpy_to_torch_tensor` / `torch_to_numpy_tensor`；
计算链仅由 Torch `index_select`、mask、multiply、sum 组成。Golden 与
`third_party` 为两套独立实现，均不调用 NPU 被测算子，也不使用 NPU 输出自证。

本算子没有同名 Torch/TF 竞品，因此 FP16/BF16/FP32 均按 `cross_check L0` 注册，
不是 Torch 直接竞品 L2。容差为 FP16/BF16 `rtol=atol=1e-3`，FP32
`rtol=atol=1e-4`。

## 性能结果

以下数据均为本次 NPU 实跑，不包含尚未取得的 GPU 对比：

| 通路 | 条数 | 指标 | min / median / mean / p95 / max（us） |
|---|---:|---|---|
| Kernel TTK | 106 | device time | 3.006 / 5.9305 / 547.133783 / 457.517 / 45077.033 |
| ACLNN | 648 | API avg | 37.990 / 63.285 / 65.997160 / 85.800 / 145.470 |
| ACLNN | 648 | AiCore avg | 2.620 / 10.045 / 74.535216 / 309.160 / 2742.800 |
| GEIR 静态 | 14 | RunGraph 含逐图编译/加载 | 23570100 / 24779150 / 24735221 / 25559300 / 26005900 |
| GEIR 动态稳态 | 4 | 同图后续 RunGraph | 247.068 / 269.514 / 268.009 / 285.939 / 285.939 |

GEIR 两个动态场景的首个调用分别为 24.6666 s 和 24.8590 s，包含图编译/加载，
已与后续稳态调用分开统计。oversized Kernel 的 45.077033 ms 是 Kernel mean 偏大
的主因。ACLNN 合格用例的最低逐元素 close 百分比为
99.9999749855917%，未被四舍五入为 100%。

## 自定义 `_apt` 来源证明

- 全量测试所用隔离 run 包 SHA-256：
  `596cb40f6e33ccb51a6c7da4226984715afe3f22c83030dc4d6fffd57eb435e3`。
- 仓内和隔离安装后的 `_apt.cpp` SHA-256 均为
  `2a55a9090c0f6605cbed53772304934cc1c0ce377cfc2d0456bf304366aa7030`。
- 三个安装对象 SHA-256：
  - FP32：`141789515f510a9780786d338a9820d67a2cf801dab4f36c298d9eb95b4dd50f`；
  - BF16：`df1dd4a471f8d6c49bdb5527b711e59bd76d1035b647d4fab430641ee556a3a6`；
  - FP16：`a8d87d0c0d607c7565f39eddf2e653840ee93f642b7a2258e8ab149d98a0cd57`。
- ACLNN plog 明确从隔离 vendor `SelectBin` 并注册
  `moe_token_unpermute_with_ep_apt` 对象。
- GEIR debug plog 对 `unknown_dim_minus_1` 和 `unknown_rank_minus_2` 均记录隔离
  custom DSL、`custom_transformer_impl.dynamic.moe_token_unpermute_with_ep_apt`
  模块和各 3 次实际 launch，并注册 tilingKey 0/1/2/3。

精确 plog 路径、行号、对象与结果哈希见
`tests/results/provenance/kernel_provenance.json`。原始 plog 与运行日志不入 Git。

## Step4 三方状态

在平台隔离提交及 950/910B 构建门禁完成后，已按工作流重新完整发起 regular
105 条命令：`--compare cross_check --golden-mode Promote --seed 42 --pc 1
--xpu-perf --provider torch`。TTK 在准备 case 前退出：

```text
RuntimeError: --xpu-perf requires remote XPU config (ttk.conf.yaml or --config), but none is configured.
```

本机无 `nvidia-smi`、无 `/dev/nvidia*`，用户配置和 TTK 根目录均无
`ttk.conf.yaml`。完整 regular/oversized CSV、Golden、环境变量和可运行命令保存在
`tests/results/three_party/README.md`。因此三方精度、GPU 性能和 NPU/GPU 比值均为
**BLOCKED/未产出**，不是 PASS。

## GEIR 与转测扫描限制

当前 `/home/z50058944/ops-test-kit` 的 CLI 仅有 `kernel/aclnn/e2e/info/list`；
`python -m ttk geir --help` 返回 invalid choice，故字面 `ttk geir` 门禁保持
**BLOCKED**。本记录中的 GEIR PASS 来自仓内 ES OpDef/InferShape 真机直连程序，
不能替代不存在的 TTK 子命令。

最终 `ops-transfer-test-scan` 原始汇总为：

- PASS：Ascend950 ini、旧 Tiling、def 驱动 dtype；
- FAIL：扫描器规定位置的 prototype、Golden、Infer；
- REMEDIATE：GEIR 动态 example 的机械结构检查；
- UNAVAILABLE：TTK geir/e2e 异常证据 9--13 与属性可选性。

其中 Golden 和 GEIR 有已通过的实际等价资产，但扫描器按固定目录或单文件文本匹配
未识别；prototype/Infer 布局差异则保留为真实扫描 FAIL。`-2` 标准扫描还要求三种
具体 Rank，而本接口协议只允许 2/1/2，不能为满足扫描伪造非法 Rank。原始报告和
逐项说明位于 `tests/results/transfer_scan/`，所有非 PASS 状态均保留。

## 关键产物

- ACLNN 设计与分层用例：`tests/st/design/`、`tests/st/testcases/`。
- 最终 ACLNN CSV：`tests/ttk/aclnn_moe_token_unpermute_with_ep_positive.csv`、
  `tests/ttk/aclnn_moe_token_unpermute_with_ep_negative.csv`。
- Kernel 白盒标准资产：`tests/whitebox/`。
- 标准 Golden：`tests/assets/golden.py`。
- 最终逐通路结果：`tests/results/aclnn/`、`tests/results/kernel/`、
  `tests/results/geir/`。
- 覆盖/条数审计：`tests/st/testcases/final_case_audit.json`、
  `tests/whitebox/audit/final_audit_results.json` 及各通路 execution audit。
- 性能：`tests/results/performance/`。
- 三方阻塞：`tests/results/three_party/README.md`。
- 自定义内核来源：`tests/results/provenance/`。
- 公共路径平台隔离：`tests/platform_isolation/`、
  `tests/results/platform_isolation/`。
- 共性扫描：`tests/results/transfer_scan/`。

构建目录、run 包、TTK 日志、设备 dump 和 plog 均保留在 Git 外。
