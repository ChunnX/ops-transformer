# SageAttention2（Ascend 950PR 实验算子）

本目录实现面向 Atlas 950PR 的 SageAttention2 前向注意力融合算子：INT8 量化 $QK^T$ 矩阵乘、FP8 e4m3 PV 矩阵乘、exp2 域在线 Softmax 与两级累加输出，基于 AscendC 与 CATLASS 编写，提供 PyTorch Python 接口与两段式 ACLNN C 接口。CUDA 上游对照实现为 thu-ml/SageAttention 的 `sageattn_qk_int8_pv_fp8_cuda` 路径，逐项语义差异见 [与 CUDA 默认接口的差异](./docs/CUDA接口差异说明.md)。

## 产品支持情况

- Atlas 950PR（`ascend950`）：支持，满配 32 AI Core / 双 Die 拓扑，28 核减配拓扑经 `SAGEATTN_NUM_CORES` 环境变量支持
- 其他昇腾产品：未适配

## 目录结构

- `./aclnn/`：两段式 ACLNN Host 实现（`sage_attn_proto.h` 接口原型、`sage_attn_host.cpp` 参数校验与 Host Tiling）
- `./sageattn_fp8/`：AscendC 设备侧基础设施（`ascendc/tiling.h` 为 Host/Device 共享 TilingData，`catlass/` 为 Cube MMA 封装；设备核入口已按 CANN opc 打包约定迁至 `./op_kernel/sageattention2.cpp`）
- `./op_kernel/`：CANN opc 打包约定的内核源码目录（`sageattention2.cpp`，1C2V 延迟一拍主流水唯一内核入口）
- `./quant/`：Q/K INT8 与 V FP8 量化 Host 逻辑与可单测的 C++ 量化内核
- `./op_host/`：CANN 标准 REG_OP/InferShape 注册层；交付态 op_api 层两段式 C 接口实现位于 `./op_api/`（字符串属性 ABI，经 `OP_API_DIR` 随 run 包打包安装），`./aclnn/` 为移植态本地单测实现（数值编码 ABI，见 `./aclnn/sage_attn_proto.h`）
- `./python/sageattention2/`：PyTorch 接入包（`core.py` 冻结签名、`dispatch.py` 多后端路由、`op_plugin.py` NPU kernel 加载）
- `./tests/`、`./benchmarks/`：本地门禁单测与独立 ACLNN 基准工具
- `./tools/sim/`：设备核 camodel 仿真 harness（B2 门禁数值比对；依赖 CANN 容器 bisheng `--run-mode=sim` 运行，用法见 `./tools/sim/README.md`，不属于 run.sh 主机门禁）
- `./run.sh`：本地门禁统一入口

## 编译构建

### 仓库整包构建（产出算子 run 包）

- 前置条件：CANN Toolkit（已验证 `9.2.0-beta.1`）、BiSheng 编译器、已完成 `source /usr/local/Ascend/cann/set_env.sh` 或等效环境注入
- 核心动作：在仓库根目录 `ops-transformer/` 执行完整构建，`--ops` 按算子目录名过滤实验算子

```bash
cd ops-transformer
bash build.sh --pkg --experimental --soc=ascend950 --vendor_name=customize --ops=sage_attention2 -j16
```

- 注意：`--ops` 过滤值必须命中 `experimental/attention/` 下的算子目录名（本算子目录为 `sageattention2`）；若交付流水线按注册名 `sage_attention2` 传参导致目录未命中，改用 `bash build.sh --pkg --experimental --soc=ascend950 --vendor_name=customize --ops=sageattention2 -j16` 或省略 `--ops` 全量构建
- 产出：`build_out/` 下的 custom run 包，安装后含 ACLNN 算子库 `libsageattention2.so`（`opFile` 名见 `./op_host/sage_attention2_def.cpp`）

### 本地门禁（run.sh，无需 NPU）

- 前置条件：macOS 或 Linux 主机，`clang++` 或 `g++`；Python 门解释器选择链：SAGE_PYTHON 环境变量 → 本目录 .venv → 仓库外 workspace 级 .venv-sage → PATH（2026-10-03 起 .venv 不再入库，避免 kernel 打包拷贝 659MB 环境目录）
- 命令均在算子目录 `experimental/attention/sageattention2/` 执行
- 支持的目标：

- `bash run.sh host`：编译并运行 `tests/test_aclnn_host.cpp`（TilingData 布局、Host Tiling、Die 亲和性、参数校验）、`tests/test_op_host.cpp`（REG_OP/InferShape）、`tests/test_output_queue.cpp`（OutputRetireQueue 成对归约）
- `bash run.sh kernel`：设备 kernel 源码语法检查（主机编译器语法级）
- `bash run.sh python`：`tests/test_tiling_struct.py`（ctypes 镜像与 `tiling.h` 逐字段对齐）、`tests/test_quant.py`（量化内核 ctypes 行为验证）与 `tests/test_kernel_twin.py`（宿主孪生执行器 + kernel 宿主串行分支对 golden 落盘数据的 L-A/L-B 功能交叉验证）
- `bash run.sh opp`：custom OPP 打包 dry-run 与 mock 安装测试
- `bash run.sh container`：容器内 BiSheng 设备二进制编译（`dav-3510`）
- `bash run.sh all`：以上全部

## 接口说明

### PyTorch 接口（`./python/sageattention2/`）

- 入口一 `sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, 其余关键字参数包)`：自动路由到 FP8 PV kernel，`pv_accum_dtype` 等高级参数经关键字参数（kwargs）传入
- 入口二 `sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread", sm_scale=None, pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False, return_lse=False, 其余关键字参数包)`：显式全参版本，末位为 kwargs 透传
- 张量形状：

- `tensor_layout="HND"`：q 为 `[B, Hq, Sq, D]`，k/v 为 `[B, Hkv, Sk, D]`
- `tensor_layout="NHD"`：q 为 `[B, Sq, Hq, D]`，k/v 为 `[B, Sk, Hkv, D]`
- 输出 shape/dtype 与 q 相同；`return_lse=True` 时追加返回 FP32 LSE，shape `[B, Hq, Sq]`

- `pv_accum_dtype` 的取值与覆盖优先级：

- 合法取值：`"fp32"`（全程 FP32 单级）、`"fp32+fp32"`（Cube 硬件累加 + FP32 buffer 周期归约）、`"fp32+fp16"`（FP16 buffer 周期归约，默认验收档）
- 覆盖优先级：`kwargs` 中的 `pv_accum_dtype` 高于环境变量 `SAGEATTN_PV_ACCUM_DTYPE`，环境变量高于默认值 `"fp32+fp16"`（实现在 `./python/sageattention2/dispatch.py` 的 `dispatch_sageattn`）
- 非法取值直接抛 `ValueError`，不做静默回退

- `SAGEATTN_ALLOW_CPU_STUB` 语义：默认关闭；设为 `1`/`true`/`yes` 时，即使检测到 NPU 可用也允许走高保真 CPU 参考实现（与 golden 数据生成算法逐位同源），用于无卡环境的功能验证与 CI。该模式输出为 CPU 参考结果，禁止作为 NPU 性能与精度验收证据

### 两段式 ACLNN C 接口（交付态 `./op_api/`，移植态 `./aclnn/`）

- 交付态（随 run 包安装，真机调用方以此为准）：`./op_api/aclnn_sage_attention2.h` 第一段 `aclnnSageAttention2GetWorkspaceSize`：入参 q/k/v、可选 `keyMean`/`lseCorrection`、`tensorLayout`（字符串 `"HND"`/`"NHD"`，`"BNSD"`/`"BSND"` 同义词域）、`isCausal`、`qkQuantGran`（`"per_thread"`/`"per_warp"`）、`smScale`、`pvAccumDtype`（`"fp32"`/`"fp32+fp32"`/`"fp32+fp16"`）、`smoothK`、`smoothV`、out、可选 lse、`workspaceSize`、`executor`；属性顺序与 `./op_host/sage_attention2_def.cpp` 注册的 7 项一致
- 移植态（本地单测与无 opdev 环境自包含实现，不随 run 包安装）：`./aclnn/sage_attn_proto.h` 第一段同名接口取数值编码：`qkQuantGran`（2=per_warp，3=per_thread）、`pvAccumDtype`（0=fp32，1=fp32+fp32，2=fp32+fp16）、`tensorLayout`（0=NHD，1=HND），属性顺序为 scale, gran, accum, causal, layout, smoothK, smoothV
- 第二段 `aclnnSageAttention2(workspace, workspaceSize, executor, stream)`：校验 workspace 不小于第一段产出后发射 kernel（两态签名一致）
- `smScale` 冻结语义：默认值 `1/sqrt(head_dim)` 仅在 Python `sm_scale=None` 时由桥接层折叠后传入；两态 C 接口收到非正值返回 `ACLNN_ERR_PARAM_INVALID`
- Tiling 下发：Host Tiling 经 `./op_host/sage_attention2_tiling.cpp` 注册的 TilingImpl 序列化进 `gert::RawTilingData`，由框架在 `CommonOpExecutorRun` 下发阶段写入 kernel
- 完整两段式调用示例见 `./benchmarks/benchmark_aclnn.cpp`

### 环境变量汇总

- `SAGEATTN_PV_ACCUM_DTYPE`：`sageattn` 入口的 `pv_accum_dtype` 默认值覆盖
- `SAGEATTN_ALLOW_CPU_STUB`：允许 CPU 参考回退（语义见上）
- `SAGEATTN_NUM_CORES`：Host Tiling 参与调度的 AI Core 数覆盖（如 `28` 用于减配拓扑，实现见 `./aclnn/sage_attn_host_tiling.h` 的 `GetHardwareCoreNum`）
- `SAGEATTN_BUILD_VERBOSE`：置 `1` 时输出 C++ 扩展编译日志
- `SAGE_OPAPI_LIBRARY` / `FIA_OPAPI_LIBRARY`：基准工具 dlsym 外部算子库时的路径（仅 `./benchmarks/` 使用）

## 适用边界

- 仅支持前向推理，不支持反向与 autograd
- 计算路径固定为 INT8 $QK^T$ + FP8 e4m3 PV；`qk_quant_gran` 的 `per_thread` 与 `per_warp` 在 NPU 侧均为真实存在的独立实现
- 稠密注意力：不支持 paged KV cache、block table、varlen（`actualSeqLengths` 语义）与任意自定义 mask；`is_causal=True` 仅在 `Sq == Sk` 时合法
- GQA 要求 `Hq` 能被 `Hkv` 整除；`1 <= D <= 128`，非 64/128 的头维自动 padding
- `smooth_v=True` 仅在 `pv_accum_dtype="fp32"` 档真实生效（V 减均值 + 输出补偿正路径）；其余档位告警后忽略
- 输入要求四维、同设备、同 dtype（FP16/BF16）、末维连续

## 性能基准工具（`./benchmarks/`）

- `benchmark_aclnn.cpp`：独立 C++ ACLNN 基准工具，dlsym 加载 `aclnnSageAttention2`（函数指针签名对齐交付 `./op_api/aclnn_sage_attention2.h` 字符串属性 ABI）与基线 `aclnnFusedInferAttentionScoreV5`（签名逐参数对齐 CANN `9.2.0-beta.1` 头文件），ABBA 轮换 + Trim5 统计 + JSON 报告
- 退出码契约：`0` 为真机运行完成（含 PASS/FAIL 裁决字段），`1` 为运行错误，`2` 为 SIMULATION 模拟运行
- SIMULATION 模式（无 CANN/NPU 的主机 mock 构建）下，所有 `speedup` 字段输出 `NaN`、`verdict` 输出 `"SIMULATION"`、`passed` 与精度对照字段输出 `null`，报告头与 `meta.simulation_warning` 显式声明模拟语义；模拟延迟为合成常数，禁止作为验收证据
- 构建：`bash benchmarks/build_aclnn_benchmark.sh`（有 CANN 环境时链接真机库，否则自动落 mock 模式并容忍冒烟退出码 `2`）
- 单用例示例（算子目录执行）：

```bash
./build/benchmarks/benchmark_aclnn --case P-03 --warmup 50 --repeats 100 \
    --order abba --gate 1.6 --output build/bench_p03.json
```

## 一键复现（本地门禁序列）

- 前置条件：已在算子目录 `experimental/attention/sageattention2/`；Python 门依赖 torch 环境（解释器选择链见上文；本仓库不入库任何 venv）
- 核心动作，按序执行，任何一步非零退出即门禁失败：

```bash
cd ops-transformer/experimental/attention/sageattention2
bash run.sh host
bash run.sh kernel
bash run.sh python
bash run.sh opp
```

- 等价的 pytest 直跑形式：

```bash
SAGE_PYTHON=... pytest tests/ -q（或 run.sh python）
```

- 质量准出门禁：

- `bash run.sh host` 输出三段 `[SUCCESS]`（ACLNN Host 7 例、OP_HOST 5 例、OutputQueue 10 例）且退出码为 0
- `SAGE_PYTHON=... pytest tests/ -q` 全部用例通过（含 `tests/test_tiling_struct.py` 的 TilingData ctypes 镜像对 160 字节总长、32 字节对齐与逐字段偏移的断言）；`run.sh python` 覆盖 test_tiling_struct.py、test_quant.py 与 test_kernel_twin.py 三个快速项，全量口径以 pytest 为准
- `tests/test_kernel_twin.py`（已纳入 `run.sh python`）：宿主孪生全流水执行器（twin_reference RunSequential）与 kernel 宿主串行分支（op_kernel SageAttnKernel + SimMemory，TilingData 直驱 `sage_attention2` C 入口）对 golden 25 桶落盘 out/lse 的功能交叉验证——22 桶整桶 + 3 大桶 128 对齐行窗口（P-05/P-07/TC-09，非因果行独立性如实标注；TC-03_smooth_v_fp32 为 smooth_v 建模边界由 test_function.py 覆盖），判据 L-A（matched_ratio ≥ 0.99、max_abs ≤ 0.1/1.0、LSE ≤ 1e-3）+ L-B（对 SDPA 误差比 ≤ 2.0/1.2/1.2），并附 P-01 量化中间态（q/k INT8、v FP8、scale）位级比对；P-01/TC-04/TC-05/TC-07/TC-01(fp16) 代表桶另以 kernel 宿主串行分支复验，TC-01(bf16) 桶覆盖核内 `tiling.input_dtype==2` 位型分派分支（bf16 读写与 km 差值位型舍回）
- 仓库级数值门（kernel 宿主串行路径 vs 孪生参考十用例位级一致、golden 25 桶 L-A/L-B）：由工程根 `ci/local_test.sh` 承载，依赖仓库外 `./tests/` 与 `./sageAttention2/sageAttentionTest`（交付包内不自带，复现须在完整工程根执行）
- `bash run.sh opp` 完成 dry-run 打包与 mock 安装且退出码为 0
- 真机验收另行执行 `bash benchmarks/build_aclnn_benchmark.sh` 并以真机报告的 `verdict: PASS` 为准；`mode: SIMULATION` 的报告一律无效

## 交付文档索引

- [与 CUDA 默认接口的差异](./docs/CUDA接口差异说明.md)：量化粒度、S_FP8_OFFSET 放大、两级累加三档、舍入模式、smooth_v 正路径的逐项语义对照
- 接口原型：交付态 `./op_api/aclnn_sage_attention2.h`（字符串属性，随 run 包安装）；移植态 `./aclnn/sage_attn_proto.h`（数值编码，本地单测）
- Tiling 契约：`./sageattn_fp8/ascendc/tiling.h`（160 字节、32 字节对齐，Host/Device 共享只读）
