# SageAttention2（Ascend 950PR 实验算子）

本目录实现面向 Atlas 950PR 的 SageAttention2 前向注意力融合算子：INT8 量化 $QK^T$ 矩阵乘、FP8 e4m3 PV 矩阵乘、exp2 域在线 Softmax 与两级累加输出，基于 AscendC 与 CATLASS 编写，提供 PyTorch Python 接口与两段式 ACLNN C 接口。CUDA 上游对照实现为 thu-ml/SageAttention 的 `sageattn_qk_int8_pv_fp8_cuda` 路径，逐项语义差异见 [与 CUDA 默认接口的差异](./docs/CUDA接口差异说明.md)。

## 产品支持情况

- Atlas 950PR（`ascend950`）：支持，满配 32 AI Core / 双 Die 拓扑，28 核减配拓扑经 `SAGEATTN_NUM_CORES` 环境变量支持
- 其他昇腾产品：未适配

## 目录结构

- `./aclnn/`：闭式 Host Tiling 计算核心与接口编码协议（`./aclnn/sage_attn_host_tiling.h` 闭式 `ComputeHostTiling` 唯一定义、`./aclnn/sage_attn_proto.h` "字符串属性→数值编码"唯一映射、`./aclnn/autotune_tiling_bank.h` P-01..P-07 调优知识库）
- `./sageattn_fp8/`：AscendC 设备侧基础设施（`./sageattn_fp8/ascendc/tiling.h` 为 Host/Device 共享 TilingData，`./sageattn_fp8/catlass/` 为 Cube MMA 封装；设备核入口已按 CANN opc 打包约定迁至 `./op_kernel/sageattention2.cpp`）
- `./op_kernel/`：CANN opc 打包约定的内核源码目录（`./op_kernel/sageattention2.cpp`，1C2V 延迟一拍主流水唯一内核入口）
- `./quant/`：Q/K INT8 与 V FP8 量化 Host 逻辑与可单测的 C++ 量化内核
- `./op_host/`：CANN 标准 REG_OP/InferShape 注册层；交付态 op_api 层两段式 C 接口唯一实现位于 `./op_api/`（字符串属性 ABI，经 `OP_API_DIR` 随 run 包打包安装），TilingImpl 复用 `./aclnn/sage_attn_host_tiling.h` 的闭式 `ComputeHostTiling`
- `./python/sageattention2/`：PyTorch 接入包（`dispatch.py` 前端分派与环境变量解析、`_ops.py` PyTorch 自定义算子注册与后端绑定、`_extension.py` C++ 扩展加载、`error_check.py` 输入校验与预处理、`monkey_patch.py` SDPA 热替换工具）
- `./tests/`、`./benchmarks/`：容器门禁单测与独立基准工具（`./benchmarks/benchmark_aclnn.cpp` aclnn 路径、`./benchmarks/run_validation_performance.py` PyTorch 路径）
- `./tools/sim/`：设备核 camodel 仿真 harness（B2 门禁数值比对；依赖门禁容器 bisheng `--run-mode=sim` 运行，用法见 `./tools/sim/README.md`，不属于 run.sh 容器门禁主链）
- `./docker/`：算子门禁容器（唯一编译与测试环境；Dockerfile/compose/生命周期入口 `sage`/一键初始化 `init.sh`/CATLASS 内聚目录 `third_party/`，卷挂载全部为仓库内相对路径，见 `./docker/README.md`）
- `./run.sh`：门禁统一入口（宿主只依赖 docker，所有目标自动包装进容器，容器内统一 BiSheng 编译）

## 官方 CATLASS 接入与模块边界

- 设备核经 `./sageattn_fp8/catlass/sage_tile.hpp` 无条件 include 官方 cann/catlass 仓 `<catlass/catlass.hpp>`，分形粒度常量（FRACTAL_M0/N0、FRACTAL_K0_*、LOADDATA2D_REPEAT_BYTES）全部由官方 `BYTE_PER_C0` / `C0_NUM_PER_FRACTAL` / `BYTE_PER_FRACTAL` 派生，官方演进自动跟随；
- 构建前置：`CATLASS_ROOT` 指向官方 catlass checkout（解析链：CMake 变量/环境变量 → 算子 `./docker/third_party/catlass`（`docker/init.sh` 置备）→ 仓库根 `./third_party/catlass`，唯一定义点 `./cmake/SageAttnDeps.cmake`，缺失即 fatal）；
- 深化复用（2026-10-03，全部为位元等价的 constexpr/forceinline 替换，指令流零变化）：跨核同步委托官方 `Arch::CrossCoreFlag` / `CrossCoreSetFlag/WaitFlag`（arch/cross_core_sync.hpp，sync_guard.h 单点）；QK/PV MMA 经官方 `Gemm::Tile::TileMmadTla` 发射（L0 FP8 寄存器位型由 PV Compute 的 `mx_fp8_e4m3_t` 张量位型直承载，见 sage_qk/pv_mmad.hpp）；设备 VF 函数标记采用官方 `CATLASS_DEVICE_SIMD_VF` 宏；host tiling 与设备核全链路取整收敛到官方 `::CeilDiv` / `::RoundUp`。设备编译统一注入官方架构选择宏 `-DCATLASS_ARCH=3510`（run.sh kernel、build_opp.sh、tools/sim 同口径）。逐项清单见设计文档"官方组件复用清单"；
- 模块边界：官方 TileCopyTla 等 copy 组件要求 L1 侧 zN 分形布局；本算子 V0 产出为 ND 行布局 + Data/Scale 相邻契约（任务书允许的按 Cube 微调项），故 LoadData2D/Fixpipe 热路径为本算子自研 AscendC 逻辑，MMA 指令语义冻结待真机 L-A 复核，不与官方 copy 组件混用。

## 编译构建

### 仓库整包构建（产出算子 run 包）

- 前置条件：CANN Toolkit（已验证 `9.2.0-beta.1`）、BiSheng 编译器、已完成 `source /usr/local/Ascend/cann/set_env.sh` 或等效环境注入
- 核心动作：在仓库根目录 `./ops-transformer/` 执行完整构建，`--ops` 按算子目录名过滤实验算子

```bash
cd ./ops-transformer
bash build.sh --pkg --experimental --soc=ascend950 --vendor_name=customize --ops=sage_attention2 -j16
```

- 注意：`--ops` 过滤值必须命中 `./ops-transformer/experimental/attention/` 下的算子目录名（本算子目录为 `sageattention2`）；若交付流水线按注册名 `sage_attention2` 传参导致目录未命中，改用 `bash build.sh --pkg --experimental --soc=ascend950 --vendor_name=customize --ops=sageattention2 -j16` 或省略 `--ops` 全量构建
- 产出：`./build_out/` 下的 custom run 包，安装后含 ACLNN 算子库 `libsageattention2.so`（`opFile` 名见 `./op_host/sage_attention2_def.cpp`）
- vendor 名唯一定义点（P2-14）：本算子自建 OPP 树的 vendor 目录固定为 `customize`（`./build_opp.sh` 出包 `vendors/customize`、`./install_opp.sh` 本地部署与 `vendors/config.ini` `load_priority=customize`、`./benchmarks/build_aclnn_benchmark.sh` 链接、`python/sageattention2/_extension.py` 探测，均以此为准）。仓库整包是另一交付物：`build.sh` 的 vendor 目录为 `${vendor_name}_transformer`（上例 `--vendor_name=customize` → `vendors/customize_transformer`；缺省 `custom` → `vendors/custom_transformer`），整包安装后由 CANN 按 `ASCEND_OPP_PATH` 与 `load_priority` 装载，aclnn 调用运行期 dlsym 解析，与本地 `customize` 树并存、互不指代

### 容器门禁（run.sh → BiSheng 容器，唯一编译与测试环境）

- 前置条件：宿主只依赖 Docker 与 Docker Compose 插件；所有编译与测试经 `./docker/` 容器内的 BiSheng 编译器承载，不存在任何宿主编译器路径
- 独立仓库交付契约：卷挂载全部为相对路径且只指向仓库内部目录，唯一外部依赖 CATLASS 经 `bash docker/init.sh` 拉取固定 commit 内聚到 `./docker/third_party/catlass`（commit 固定为 `e420328d38945dea53ac8ee56795215db20b943b`，唯一定义点 `./docker/init.sh` 的 `CATLASS_COMMIT`；见 `./docker/README.md`）
- 命令均在算子目录 `./ops-transformer/experimental/attention/sageattention2/` 执行；`run.sh` 各目标在宿主侧自动包装进容器（首次运行自动补齐 CATLASS 初始化），容器侧统一 BiSheng
- 设备核编译严格基于 BiSheng 编译器（`--npu-arch=dav-3510 -std=c++17 -xasc`）
- 支持的目标：

- `bash run.sh host`：BiSheng 宿主模式编译并运行 `./tests/test_aclnn_host.cpp`、`./tests/test_op_host.cpp` 与 `./tests/test_tiling_struct.cpp`
- `bash run.sh kernel`：BiSheng 交叉编译设备核 `./op_kernel/sageattention2.cpp` 为 `./build/sageattention2_bisheng.o`
- `bash run.sh python`：容器内 `./tests/test_tiling_struct.py`、`./tests/test_quant.py`、`./tests/test_kernel_twin.py` 与 `./tests/test_gqa_chunk.py`（测试动态库由 BiSheng 编译）
- `bash run.sh opp`：OPP vendor 树 BiSheng 出包（`./build_opp.sh`）与本地部署（`./install_opp.sh`）
- `bash run.sh all`：执行全部门禁验证（host → kernel → python → opp）
- 交互调试：`./docker/sage exec`

## 接口说明

### PyTorch 接口（`./python/sageattention2/`）

- 入口一 `sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, **kwargs)`：自动路由到 FP8 PV kernel，`pv_accum_dtype` 等高级参数经关键字参数（kwargs）传入
- 入口二 `sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread", sm_scale=None, pv_accum_dtype=None, smooth_k=True, smooth_v=False, return_lse=False, **kwargs)`：显式全参版本，末位为 kwargs 透传
- 返回值契约：当 `return_lse=False`（默认值）时，直接单张量返回 `o`；仅当 `return_lse=True` 时，才返回 `(o, lse)` 二元组
- 掩码约束：`attn_mask` 仅经 kwargs 接收（SDPA 形态兼容），非 None 掩码（含位置实参误传张量）直接抛出 `ValueError` 异常；同批拒绝的还有非零 `dropout_p`
- SDPA 热替换工具（`./python/sageattention2/monkey_patch.py`）：提供 `replace_sdpa()` 全局热替换、`restore_sdpa()` 恢复还原以及 `sdpa_patch()` 上下文管理器，支持工业级降级回退与嵌套作用域
- 张量形状：

- `tensor_layout="HND"`：q 为 `[B, Hq, Sq, D]`，k/v 为 `[B, Hkv, Sk, D]`
- `tensor_layout="NHD"`：q 为 `[B, Sq, Hq, D]`，k/v 为 `[B, Sk, Hkv, D]`
- 输出 shape/dtype 与 q 相同；`return_lse=True` 时追加返回 FP32 LSE，shape `[B, Hq, Sq]`

- `pv_accum_dtype` 的取值与覆盖优先级：

- 合法取值：`"fp32"`（全程 FP32 单级）、`"fp32+fp32"`（Cube 硬件累加 + FP32 buffer 周期归约）、`"fp32+fp16"`（FP16 buffer 周期归约，默认验收档）
- 覆盖优先级：`kwargs` 中的 `pv_accum_dtype` 高于环境变量 `SAGEATTN_PV_ACCUM_DTYPE`，环境变量高于默认值 `"fp32+fp16"`（实现在 `./python/sageattention2/dispatch.py` 的 `sageattn`）
- 非法取值直接抛 `ValueError`，不做静默回退

- `SAGEATTN_ALLOW_CPU_STUB` 语义：默认关闭；设为 `1`/`true`/`yes` 时，即使检测到 NPU 可用也允许走高保真 CPU 参考实现（与 golden 数据生成算法逐位同源），用于无卡环境的功能验证与 CI。该模式输出为 CPU 参考结果，禁止作为 NPU 性能与精度验收证据

### 两段式 ACLNN C 接口（交付态 `./op_api/` 唯一实现）

- 交付态（随 run 包安装，真机调用方以此为准）：`./op_api/aclnn_sage_attention2.h` 第一段 `aclnnSageAttention2GetWorkspaceSize`：入参 q/k/v、可选 `keyMean`/`lseCorrection`、`tensorLayout`（字符串，仅 `"HND"`/`"NHD"`，非法取值报错）、`isCausal`、`qkQuantGran`（`"per_thread"`/`"per_warp"`）、`smScale`、`pvAccumDtype`（`"fp32"`/`"fp32+fp32"`/`"fp32+fp16"`）、`smoothK`、`smoothV`、out、可选 lse、`workspaceSize`、`executor`；属性顺序与 `./op_host/sage_attention2_def.cpp` 注册的 7 项一致
- 第二段 `aclnnSageAttention2(workspace, workspaceSize, executor, stream)`：校验 workspace 不小于第一段产出后发射 kernel
- `smScale` 冻结语义：默认值 `1/sqrt(head_dim)` 仅在 Python `sm_scale=None` 时由桥接层折叠后传入；C 入口收到非正值返回 `ACLNN_ERR_PARAM_INVALID`
- Tiling 下发：Host Tiling 唯一实现为 `./aclnn/sage_attn_host_tiling.h` 的闭式 `ComputeHostTiling`，经 `./op_host/sage_attention2_tiling.cpp` 注册的 TilingImpl 序列化进 `gert::RawTilingData`，由框架在 `CommonOpExecutorRun` 下发阶段写入 kernel；数值编码（0=NHD/1=HND 等）仅供 Host/Device 内部消费，经 `./aclnn/sage_attn_proto.h` 的 `*CodeFromString` 唯一映射产生，禁止作为跨语言 ABI 形参再引入
- 完整两段式调用示例见 `./benchmarks/benchmark_aclnn.cpp`

### 环境变量汇总

- `SAGEATTN_PV_ACCUM_DTYPE`：`sageattn` 入口的 `pv_accum_dtype` 默认值覆盖
- `SAGEATTN_ALLOW_CPU_STUB`：允许 CPU 参考回退（语义见上）
- `SAGEATTN_NUM_CORES`：Host Tiling 参与调度的 AI Core 数覆盖（如 `28` 用于减配拓扑，实现见 `./aclnn/sage_attn_host_tiling.h` 的 `GetHardwareCoreNum`）
- `SAGEATTN_TUNE_CASE`：强制 Host Tiling 命中调优知识库指定用例（`P-01`..`P-07`），供调优评估与测试覆盖使用（实现见 `./aclnn/sage_attn_host_tiling.h` 的 `ComputeHostTiling` 查询与 `./aclnn/autotune_tiling_bank.h`）；与 `SAGEATTN_FORCE_MPERCORE`/`_BLOCKN`/`_KSTEP`/`_DIE`/`_SPLITKV` 覆盖钩子同属调优会话专用，仅在 `SAGE_AUTOTUNE_HOOKS=1` 调优构建（`build_opp.sh`）中编入生效，正式发布构建不参与编译
- `SAGEATTN_BUILD_VERBOSE`：置 `1` 时输出 C++ 扩展编译日志
- `SAGE_OPAPI_LIBRARY` / `FIA_OPAPI_LIBRARY`：基准工具 dlsym 外部算子库时的路径（仅 `./benchmarks/` 使用）
- `SAGE_BENCH_HEADS` / `SAGE_BENCH_DIM` / `SAGE_SMOOTH_V` / `SAGE_PERF_GATE` / `SAGE_BENCH_ORDER` / `SAGE_BENCH_L2_FLUSH_MB` / `SAGE_COOLDOWN_MS`：基准工具 CLI 前置默认覆盖（同名命令行旗标可覆盖，仅 `./benchmarks/benchmark_aclnn.cpp` 消费，取值语义见其 `--help`）

## 适用边界

- 仅支持前向推理，不支持反向与 autograd
- 计算路径固定为 INT8 $QK^T$ + FP8 e4m3 PV；`qk_quant_gran` 的 `per_thread` 与 `per_warp` 在 NPU 侧均为真实存在的独立实现
- 稠密注意力：不支持 paged KV cache、block table、varlen（`actualSeqLengths` 语义）与任意自定义 mask；`is_causal=True` 仅在 `Sq == Sk` 时合法
- GQA 要求 `Hq` 能被 `Hkv` 整除，且 `Hq / Hkv <= 8`（上限值唯一定义于 `./sageattn_fp8/ascendc/tiling.h` 的 `MAX_GQA_RATIO`，为本交付明示的验收域上限：`Hq / Hkv > 8` 的输入由 Host Tiling（`./aclnn/sage_attn_host_tiling.h`）显式报错拒绝，不做静默截断或回退）；`1 <= D <= 128`，非 64/128 的头维自动 padding
- `smooth_v=True` 仅在 `pv_accum_dtype="fp32"` 档真实生效（V 减均值 + 输出补偿正路径）；其余档位告警后忽略
- 输入要求四维、同设备、同 dtype（FP16/BF16）、末维连续

## 性能基准工具（`./benchmarks/`）

- `benchmark_aclnn.cpp`：独立 C++ ACLNN 基准工具，dlsym 加载 `aclnnSageAttention2`（函数指针签名对齐交付 `./op_api/aclnn_sage_attention2.h` 字符串属性 ABI）与基线 `aclnnFusedInferAttentionScoreV5`（签名逐参数对齐 CANN `9.2.0-beta.1` 头文件），ABBA 轮换 + Trim5 统计 + JSON 报告
- 退出码契约：`0` 为真机运行完成（含 PASS/FAIL 裁决字段），`1` 为运行错误，`2` 为 SIMULATION 模拟运行
- SIMULATION 为纯运行时模式（非编译路径）：容器未透传 NPU 设备时工具仍可运行，所有 `speedup` 字段输出 `NaN`、`verdict` 输出 `"SIMULATION"`、`passed` 与精度对照字段输出 `null`，报告头与 `meta.simulation_warning` 显式声明模拟语义；模拟延迟为合成常数，禁止作为验收证据
- 构建：`bash ./benchmarks/build_aclnn_benchmark.sh`（容器内执行，BiSheng 宿主模式编译并链接 CANN 真机库）
- 单用例示例（算子目录执行）：

```bash
./build/benchmarks/benchmark_aclnn --case P-03 --warmup 50 --repeats 100 \
    --order abba --gate 1.6 --output build/bench_p03.json
```

- `run_validation_performance.py`：PyTorch 路径性能门禁工具，逐条实测 P-01..P-07 的 `sageattn_qk_int8_pv_fp8_asc`（`fp32+fp16` 交付默认档）对基线 `torch_npu.npu_fused_infer_attention_score` 的 Speedup 并判定每条 `>=1.6` 门禁与全用例几何平均
- 用例表唯一定义点为 `./benchmarks/cases.py`（本工具不复制第二份）；FIA 基线配方固化任务书「最小调用示例」：`input_layout="BNSD"`（与本算子 HND 同构，双方共享同一物理张量）、`num_heads`/`num_key_value_heads`/dtype/causal 逐 case 对齐、`pre_tokens=65535`、`next_tokens=65535`、`scale=1/sqrt(D)`；因果 P-03 叠加 `sparse_mode=2` 与 2048×2048 triu bool 掩码（官方 perf harness 因果配方）
- 计时口径：`torch.npu.Event` 事件对逐次预建，warmup（默认 30，下限 10）不计入统计，`--repeats`（默认 100，下限 20）次测量后 synchronize 再取逐迭代 ms 均值；逐 case 打印 `T_FIA / T_sage / Speedup / 门禁判定`，末行输出几何平均
- 环境契约：无 torch_npu 或无 NPU 设备时打印明确跳过信息并以退出码 0 结束（供 CI-off 机器冒烟）；NPU 上为严格模式，任何失败非零退出
- 运行（算子目录执行，真机环境）：

```bash
python3 benchmarks/run_validation_performance.py --warmup 30 --repeats 100 \
    --output build/pyperf_report.json
```

- 产出 JSON 报告：逐 case 的 shape/dtype/causal/`t_fia_ms`/`t_sage_ms`/`speedup`/`passed`、`geomean_speedup`、`all_cases_pass`，以及 `fia_causal_recipe`、torch/torch_npu 版本与设备名等 `meta` 字段，可直接回填自验证报告性能基准用例表（PyTorch 列）

## 设备侧功能门禁与单算子 Profiling

- 设备侧功能门禁套件 `./tests/test_npu_standalone.py`：真机上驱动已安装的两段式 `aclnnSageAttention2`（经 `./python/sageattention2` 公开 API → C++ 桥接），逐条覆盖任务书《功能验收用例》TC-01..TC-10（`sageattn` 分发 FP16/BF16 × HND/NHD、`sageattn_qk_int8_pv_fp8_asc` per_thread/per_warp × `fp32`/`fp32+fp32`/`fp32+fp16`、smooth_k/smooth_v 开关与 warning 语义、`return_lse` 形状 `[B,Hq,Sq]` 与 LSE 公式校验、head_dim 48→64/96→128 pad 与输出裁剪、GQA+Sq≠Sk、因果、D>128/非连续末维/非法 gran/attn_mask 错误注入、S=8192 与 S=16384 长序列、SDPA 热替换冒烟）
- 精度判据：任务书 L-B 风格混合容差（FP16 `2^-9` / BF16 `2^-6`，`matched_ratio >= 0.99`，`max_abs` 上限 `1e-1`/`1e-0`），LSE 另按 FP32 容差校验；参考为自包含 FP32 分块 softmax 注意力与 logsumexp（不导入 `./tests/_golden_utils.py`）
- 运行（算子目录执行，真机环境；长序列用例需较大显存与数分钟量级运行时间）：

```bash
SAGEATTN_TEST_NPU=1 python3 -m pytest tests/test_npu_standalone.py -v
```

- 未设 `SAGEATTN_TEST_NPU=1`、或无 torch/torch_npu/NPU 设备时整模块 pytest.skip 并以退出码 0 结束，宿主机 pytest 冒烟不受影响、不触发设备调用与扩展编译
- 单算子 Profiling 配方 `./benchmarks/profile_single_op.py`：取 `./benchmarks/cases.py` 单 case，经 `sageattn_qk_int8_pv_fp8_asc`（`fp32+fp16` 档）以 torch_npu.profiler（kineto）采集算子级耗时，导出 chrome trace 并打印 op 级汇总表，模块 docstring 附 msprof 整进程附加用法；无 torch/torch_npu/NPU 时打印跳过信息并以退出码 0 结束

```bash
python3 benchmarks/profile_single_op.py --case P-01 --warmup 10 --steps 20 \
    --trace build/profile_P-01.json
```

## 一键复现（容器门禁序列）

- 前置条件：已在算子目录 `./ops-transformer/experimental/attention/sageattention2/`，宿主机已装 docker 与 compose 插件（唯一宿主依赖；镜像/缓存说明见 `./docker/README.md`）
- 核心动作，按序执行，任何一步非零退出即门禁失败：

```bash
cd ./ops-transformer/experimental/attention/sageattention2
bash docker/init.sh                 # 一键初始化（CATLASS 固定 commit → docker/third_party/、.env、基础镜像）
bash run.sh host
bash run.sh kernel
bash run.sh python
bash run.sh opp
```

- golden 基线数据位于 `./docker/golden/`（25 桶交付基线 + manifest，`run.sh python` 的 L-A/L-B 判据输入）。容器镜像不含 golden：由 `bash docker/init.sh --golden <数据集目录>` 幂等置备（目标已存在则绝不改写/重生成；生成器源码唯一定义点 `./tools/generate_golden.py`，随数据集一同置备，禁止重生成 golden）

- 也可一条命令全跑：`bash run.sh all`；交互调试：`./docker/sage exec`

- 质量准出门禁：

- `bash run.sh host` 输出各二进制自检 `[SUCCESS]` 且退出码为 0（当前收敛为 test_aclnn_host / test_op_host / test_tiling_struct 三个二进制，以 `run.sh` 实际清单为准）
- `bash run.sh python` 四个测试文件全部通过（含 `./tests/test_tiling_struct.py` 的 TilingData ctypes 镜像对 160 字节总长、32 字节对齐与逐字段偏移的断言）
- `./tests/test_kernel_twin.py`（已纳入 `run.sh python`）：宿主孪生全流水执行器（twin_reference RunSequential）与 kernel 宿主串行分支（op_kernel SageAttnKernel + SimMemory，TilingData 直驱 `sage_attention2` C 入口）对 golden 25 桶落盘 out/lse 的功能交叉验证——22 桶整桶 + 3 大桶 128 对齐行窗口（P-05/P-07/TC-09，非因果行独立性如实标注；TC-03_smooth_v_fp32 经 smooth_v 正路径整桶执行），判据 L-A（matched_ratio ≥ 0.99、max_abs ≤ 0.1/1.0、LSE ≤ 1e-3）+ L-B（对 SDPA 误差比 ≤ 2.0/1.2/1.2），并附 P-01 量化中间态（q/k INT8、v FP8、scale）位级比对；P-01/TC-03/TC-04/TC-05/TC-07/TC-01(fp16) 代表桶另以 kernel 宿主串行分支复验，TC-01(bf16) 桶覆盖核内 `tiling.input_dtype==2` 位型分派分支（bf16 读写与 km 差值位型舍回）
- 仓库级数值门（kernel 宿主串行路径 vs 孪生参考十用例位级一致、golden 25 桶 L-A/L-B）：由 `bash run.sh python` 中的 `./tests/test_kernel_twin.py` 承载（判据见上条）
- `bash run.sh opp` 完成 BiSheng vendor 树出包与本地部署且退出码为 0
- 真机验收另行执行 `bash ./benchmarks/build_aclnn_benchmark.sh`（容器内）并以真机报告的 `verdict: PASS` 为准；`mode: SIMULATION` 的报告一律无效

## 交付文档索引

- [与 CUDA 默认接口的差异](./docs/CUDA接口差异说明.md)：量化粒度、S_FP8_OFFSET 放大、两级累加三档、舍入模式、smooth_v 正路径的逐项语义对照
- [PyTorch接入说明](./docs/PyTorch接入说明.md)：安装顺序（CANN → torch/torch_npu → OPP 包 `./install_opp.sh` 部署与 `ASCEND_CUSTOM_OPP_PATH` 注入 → Python 包接入）、接口签名与调用示例（HND/NHD、GQA、因果、LSE）、SDPA 热替换、约束与环境变量、图模式注意事项
- 接口原型：交付态 `./op_api/aclnn_sage_attention2.h`（字符串属性，随 run 包安装，两段式 C 接口唯一权威签名）；Host Tiling 与编码协议见 `./aclnn/README.md`
- Tiling 契约：`./sageattn_fp8/ascendc/tiling.h`（160 字节、32 字节对齐，Host/Device 共享只读）
