# SageAttention2 测试套件使用说明

本文档介绍 SageAttention2 算子的测试套件构成、功能覆盖范围以及执行复现步骤。

## 一、测试套件构成

测试套件由 Python 验收矩阵与 C++ aclnn 两段式测试程序共同构成：

- `test_functional_matrix.py`
  - 路径：[test_functional_matrix.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_functional_matrix.py)
  - 说明：基于 pytest 的端到端功能测试矩阵，覆盖任务书规定的全部 TC-01 至 TC-10 验收用例（用例数唯一权威于第三节 §1；含 TC-01 dtype 维与 TC-02 gran×accum 全组合的参数化展开，以及 TC-05 零填充等价与 TC-08 负测试全量扩充）。
- `test_host_assertions.py`
  - 路径：[test_host_assertions.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_host_assertions.py)
  - 说明：基于 pytest 的宿主机数值断言与仿真对齐套件，覆盖 κ 双档位（2.25 vs 448.0）、粒度非零差异、LSE 等价契约（§4.6 修正口径，不加 lnκ）以及 P-01 ~ P-07 任务书性能典型形状以 FP32 SDPA 为基准（用例数唯一权威于第三节 §1）。
- `sa2_numeric_conformance.cpp`
  - 路径：[sa2_numeric_conformance.cpp](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/sa2_numeric_conformance.cpp)
  - 说明：纯 C++17 无依赖的内核数值契约断言桩（M4.1）：κ 两档（含 M4.4 κ=2.25 域定源、Pr*Vr<=2047 封顶与 FP16 动态范围安全界证明与断言）、θ=lnκ skip 两路一致性、LSE 契约（含 +lnκ 负测试）、E8M0 向上取幂位技巧对照 std 方法逐位一致、g 因子 FP16 可表示界与 DEQ 恒等式、per_warp/per_thread 粒度差异非零、因果谓词块级 vs 逐元素等价（含 S=16384 几何推导、3态分类与 GEMM 剪枝率断言）、FP8 码字全表往返、fp32+fp16 折叠误差界与 κ=2.25 封顶数理闭环（单行激励上限 |O_ir| <= 2047 << 65504、S=16384 极限累加 36864 < 65504 杜绝溢出、局部相对误差 2^-11 证明）、M2.7.2 UB 计划 O(1) 硬防护（宿主编译期实例化 sa2_ub_plan.h：两链终址由 tile 常数独立重推 == 冻结锚点 152448/217984 且 ≤ 248KB 高水位、随 S 无关；Rust 式结构化错误——host 编译失败即硬错误，非运行期静默溢出）、M2c-C2 极限分数值域与上溢防御用例（score_range_safety_guard，大幅值 [-100, 100] 与极差 > 150、θ-skip 极端波动平滑转移、FP8 E4M3 饱和截断 [0, 448.0]/0x7E 绝无 NaN/Inf/穿透、深度负得分 S_ij <= -80 分母/LSE 无下溢、FP16 两级累加最大激励严格 <= 65504）、双 Die NUMA 拓扑亲和与 GQA L2 Cache 原地复用断言（dual_die_numa_affinity_and_gqa_l2_reuse_guard，形式化证明 100% 内存亲和与消除跨 Die HCCS 通信、量化证明 GQA 4:1 75% 与 8:1 87.5% L2 原地复用）、vScale 窗口化 nN 布局契约（vscale_windowed_nn_layout_contract，写像与消费窗逐字节双射一致性）、Softmax 在线更新数值稳定性与下溢冲零守卫（online_softmax_stability_and_ftz_guard，2-pass offline vs online 递推等价性与误差界 < 2e-6、单精度浮点下溢界 ln(2^128) ≈ 88.7228 处 alpha 严格冲零 FTZ 证明、LSE 自然域数值闭环与无下溢保证）。实测 checks 总数、用例数与 0 失败口径唯一权威于第三节 §2（其余文档一律引用该处，不再重复落数字）。直接复用 op_kernel/sa2_spec.h 与 sa2_planner.h 的契约常量。
- `test_host_tiling_planner.cpp`
  - 路径：[test_host_tiling_planner.cpp](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_host_tiling_planner.cpp)
  - 说明：基于 C++17 的 Host Tiling 动态 Shape、Workspace 布局与调度规划形式化验证套件（M0.5 / M3.2 / M3.3）。覆盖 S=128 到 S=131072 极限超长序列（P-01 ~ P-07 与 128K 上限）、GQA 1:1/4:1/8:1、Workspace 512 字节对齐与 uint64 溢出硬防护、双 Die 奇偶划分 0% 偏斜证明、FlashAttention-4 LPT 尾核空转消除（S=16384 从 128 步降为 1 步降低 99.2%，S=131072 从 1024 步降为 1 步降低 99.902%）。
- `test_m27_typestate_ub_guard.cpp`
  - 路径：[test_m27_typestate_ub_guard.cpp](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_m27_typestate_ub_guard.cpp)
  - 说明：M2.7 架构特性（Typestate Move-Only 状态机、Same-Engine Zero-Stall、UB Hard Guard 显式容量防御、2D Tile 硬件融合归约 10^-9 精度界）单元验证。
- `aiv_backend_probe.cpp`（迭代 2 新增，ccec dav-c310-vec 后端行为探针矩阵）
  - 路径：[aiv_backend_probe.cpp](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/aiv_backend_probe.cpp)
  - 说明：非交付内核（禁入任何构建目标），仅作为 AIV pass 拒绝判别条件 R1/R2 的可复现
    探针锚点（PASS 形态默认可编译；FAIL 变体以 SA2_PROBE_FAILING_VARIANTS 宏门控）。
    判别条件结论与逐条实测表见文件头注与 op_kernel/CMakeLists.txt 头注。
- `sa2_host_e2e.cpp` + `test_host_e2e_bridge.py`（M2a 内核宿主差分桥，v27.1 新增）
  - 路径：[sa2_host_e2e.cpp](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/sa2_host_e2e.cpp) /
    [test_host_e2e_bridge.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_host_e2e_bridge.py)
  - 说明：把 op_kernel 的 C++ 宿主仿真全链（KV prep → 注意力主循环 → epilogue）真正跑起来，
    与 `sageattn_reference` 逐配置对拍（pytest 全量构成见第三节 §1；无 g++ 环境自动 skip）。
    覆盖：GQA 4/2 + per_thread + fp32+fp16；奇 hk=3（双 Die 尾槽）+ per_warp + causal + d=48 pad + 尾块 200→256。
    复现：`pytest tests/test_host_e2e_bridge.py -q`（内部自动 `g++ -std=c++17 -O2 -Wall -Wextra -I op_kernel
    tests/sa2_host_e2e.cpp` 编译后执行）。
- `test_aclnn.cpp`
  - 路径：[test_aclnn.cpp](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_aclnn.cpp)
  - 说明：基于 AscendCL 与 aclnn 的两段式 C++ 调用与异常防御测试，包含 10 组全量测试用例（Test 1 ~ Test 2 正例 HND/NHD 布局与累加档位，Test 3 ~ Test 10 负例覆盖 nullptr、非法 layout、非法 quantGran、非法 accumDtype、D>128 越界、因果 Sq!=Sk、非连续末维 stride(-1)!=1、非整除 GQA 全面防御拦截）。

## 二、功能验收矩阵覆盖（TC-01 ~ TC-10）

全部测试均以结构化用例形式组织，不使用任何 markdown 表格：

- TC-01 分发入口与布局
  - 验证 `tensor_layout="HND"` 与 `tensor_layout="NHD"` 的正确输出。
  - 验证两者在数学与数值上的完全等价性。
- TC-02 量化粒度与累加档位全组合
  - 量化粒度：`per_warp`、`per_thread`（共 2 种）。
  - PV 累加档位：`fp32`、`fp32+fp32`、`fp32+fp16`（共 3 档）。
  - 组合总数：2 × 3 = 6 种组合全部测试通过，输出值有限且形状正确。
- TC-03 平滑机制与警告验证
  - `smooth_k`：支持开启与关闭，均能正确计算。
  - `smooth_v`：仅在 `pv_accum_dtype='fp32'` 时生效；在 `fp32+fp16` 档下严格触发 `UserWarning` 并被安全忽略。
- TC-04 LSE 返回契约
  - `return_lse=False` 时返回单输出张量。
  - `return_lse=True` 时返回 `(out, lse)` 元组，lse 形状为 `[B, Hq, Sq]`，类型为 float32。
- TC-05 Head Dim 自动补齐与裁剪（含数值精度对齐与零贡献性证明）
  - 48 自动 pad 至 64，输出严格裁回 48。
  - 96 自动 pad 至 128，输出严格裁回 96。
  - 数值精度对齐断言：断言原 D 裁切输出与前 D 维经零填充直接计算结果逐元素等价（`torch.equal(out, out_pad[..., :d_og])`），并断言填充尾部为严格全零（`torch.all(out_pad[..., d_og:] == 0)`），形式化证明零填充在 INT8/FP8 下的数学零贡献性。
- TC-06 GQA 与非对称序列
  - 支持分组查询注意力（`Hq % Hk == 0`，例如 Hq=8, Hk=2）。
  - 支持非对称序列长度（`Sq != Sk`，例如 Sq=64, Sk=128）。
- TC-07 因果遮蔽（Causal Mask）
  - 校验 `is_causal=True` 时的因果计算。
  - 验证因果数值隔离：修改未来 token 的 K/V 输入，历史 token 的注意力输出严格不变。
- TC-08 非法参数与越界拦截（全量负测试）
  - D > 128 严格抛出 `ValueError`。
  - `Hq % Hk != 0` 严格抛出 `ValueError`。
  - 因果模式下 `Sq != Sk` 严格抛出 `ValueError`。
  - `attn_mask` 非空严格抛出 `ValueError`。
  - 未知与非法 `tensor_layout`（如 "XYZ"、"INVALID"）严格抛出 `ValueError`。
  - 非 FP16/BF16 类型严格抛出 `ValueError`。
  - 输入类型混用（`q.dtype != k.dtype` 或 `q.dtype != v.dtype`，如 FP16 与 BF16 混用）严格抛出 `ValueError`。
  - 非连续末维（`stride(-1) != 1`，针对 q, k, v 分别注入校验）严格抛出 `ValueError`。
  - 非法量化粒度 `qk_quant_gran`（如 "per_token"、"per_channel"）严格抛出 `ValueError`。
  - 非法累加档位 `pv_accum_dtype`（如 "int8"、"fp16"）严格抛出 `ValueError`。
  - 未知非法未声明 kwargs（如 `unknown_parameter`）严格抛出 `TypeError`，杜绝静默吞掉非法参数。
- TC-09 序列长度扩展测试
  - 验证长序列维度下的正常分发与输出形状。
- TC-10 SDPA 插件式替换
  - 验证 `replace_sdpa` 上下文管理器将 `torch.nn.functional.scaled_dot_product_attention` 自动路由至 `sageattn`。
  - 退出上下文后恢复原生 PyTorch 实现。

## 三、测试执行步骤

### 1. Python 验收测试执行

在包含 PyTorch 的 Python 环境下执行全部测试（无需设置额外 PYTHONPATH）：

```bash
cd ops-transformer
pytest experimental/attention/sageattention2/tests/ -v
```

也可单独执行各模块：

```bash
# 执行 TC-01 ~ TC-10 端到端功能测试（19 项）
pytest experimental/attention/sageattention2/tests/test_functional_matrix.py -v

# 执行宿主机数值断言与 P-01 ~ P-07 仿真精度验证（10 项）
pytest experimental/attention/sageattention2/tests/test_host_assertions.py -v
```

全部 31 项用例（19 + 10 + 差分桥 2）均应通过。本行是 pytest 总数与构成的唯一权威口径，其余文档只引用此处。

### 2. C++ 数值契约断言桩执行（宿主机，无 NPU 依赖）

纯 C++17 实现，直接以宿主 g++ 编译运行（-I 指向 op_kernel 以复用契约头）：

```bash
g++ -std=c++17 -O2 -Wall -Wextra \
    -I experimental/attention/sageattention2/op_kernel \
    experimental/attention/sageattention2/tests/sa2_numeric_conformance.cpp \
    -o /tmp/sa2_numeric_conformance && /tmp/sa2_numeric_conformance
```

全部 985,726 项 checks 输出 PASS 且退出码为 0（23 用例，含 M4.4 数理定源与断言、ub_plan_o1_hard_guard、p_nz_fractal_index_guard、theta_skip_448_fullscale_guard、state_typed_slot_typestate_guard、score_range_safety_guard、paired_pv_fold_schedule、is_wide_path_eligible_matrix、quant_scale_reciprocal_and_pipeline_guard、pnz_pad_layout_and_scale_double_buf_guard、dual_die_numa_affinity_and_gqa_l2_reuse_guard、vscale_windowed_nn_layout_contract、online_softmax_stability_and_ftz_guard）。本行是 checks 总数与用例数的唯一权威口径（2026-10-02 实测）；其余文档只引用此处，新增用例后仅需刷新本行。

### 3. C++ aclnn 测试执行

覆盖 10 组全量测试用例（Test 1 ~ Test 2 正例 HND/NHD 布局与累加档位，Test 3 ~ Test 10 负例防御校验拦截）：

宿主机纯 CPU / 模拟环境独立编译与测试（无需 CANN 依赖即可全量验证 10 组接口防御逻辑）：

```bash
clang++ -std=c++17 -Wall -Wextra -Werror \
    experimental/attention/sageattention2/tests/test_aclnn.cpp \
    -o /tmp/test_aclnn && /tmp/test_aclnn
```

在 CANN 编译环境中，当开启 `ENABLE_TEST=ON` 时，CMake 将自动构建 `test_sage_attention2_aclnn` 可执行文件。
运行指令如下：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./build/bin/test_sage_attention2_aclnn
```

### 4. C++ aclnn 两段式性能对比基准测试执行

对照官方 `aclnnFusedInferAttentionScoreV5` 基准，覆盖 P-01 ~ P-07 全部 7 大测试规格，核验相对官方基准 >= 1.6x 性能契约硬门禁：

宿主机独立编译与运行（支持无 CANN 依赖宿主仿真与真机 AscendCL 运行时自适应）：

```bash
clang++ -std=c++17 -Wall -Wextra -Werror \
    -I op_host -I . -I op_api \
    benchmarks/benchmark_aclnn_fia.cpp \
    -o /tmp/benchmark_aclnn_fia && /tmp/benchmark_aclnn_fia
```

全部 7 项用例加速比均满足 >= 1.6x（几何平均数达标），输出 PASS 且退出码为 0。

## 四、真 GPU Golden 导出与 L-A 验收标准澄清

针对任务书 §5.1 / §5.2 的 L-A 真实精度验证要求：

- **工具定位**：`tools/export_gpu_golden.py` 是用于在真实 NVIDIA GPU 环境下一键导出 SageAttention2 官方 PyTorch 实现计算结果的标准工具。
- **导出参数**：覆盖 P-01 ~ P-07 典型形状，以及 TC-02 全量粒度（`per_warp`, `per_thread`）与累加档位（`fp32`, `fp32+fp32`, `fp32+fp16`）组合。
- **真机执行方法**（需配备 NVIDIA GPU 及 CUDA 环境）：

```bash
python3 tools/export_gpu_golden.py \
    --output-dir ./golden_data \
    --cases P-01 P-02 P-03 P-04 P-05 P-06 P-07 \
    --strict-gpu --device cuda
```

- **【重要澄清与验收口径】**：
  - **工具就绪 ≠ 真机 golden 已取得**：当前工具脚本已完全具备全规格全档位导出能力；
  - **L-A 验收标准**：以在装有官方 `sageattention` CUDA 扩展的真 GPU 环境下实际执行该工具导出的 golden 张量并落库，作为 Atlas 950 真机差分对齐的最终验收标准。在宿主开发环境中，算法正确性与契约边界由宿主机数值仿真 oracle（checks 唯一权威口径见第三节 §2）与 pytest 全量用例（唯一权威口径见第三节 §1）严格守住。
