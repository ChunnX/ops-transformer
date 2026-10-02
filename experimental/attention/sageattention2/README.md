# SageAttention2 算子实现

本目录是 SageAttention2（INT8 QK^T + FP8 PV 路径）在华为昇腾 Atlas 950PR 架构下的算子交付工程。当前进展口径（2026-10-02 实测，证据链见下）：接口层（PyTorch 适配 / aclnn 两段式 / op_host 定义与 Tiling）与内核标量数值骨架（算法正确性载体与差分测试 oracle，纯 C++ 循环实现）已就绪——宿主五链全绿（C++ 数值契约断言 0 失败——checks 总数与用例数唯一权威口径见 tests/README.md 第三节 §2、pytest 全量 0 失败——总数与构成唯一权威口径见 tests/README.md 第三节 §1、host-tiling / M2.7 架构断言 / aclnn 两段式入口测试全绿，覆盖面见 tests/README.md 第一节与第三节）。设备侧：1 AIC + 2 AIV 混合核 AscendC + CATLASS 实现按 dav-c310 拆分编译（bisheng -xcce，对齐 ascendc_kernel_cmake 口径）——AIC pass（dav-c310-cube）编译 0 错（对象 .text 48,576B + .ascendc_tiling 节）；AIV pass（dav-c310-vec）编译 0 错（对象 .text 232,032B + .ascendc_tiling 节；attention loop 三 VF 与 Q 量化 / KV 预处理标量浮点核心均已按全向量指令形态改写，AIV 后端拒绝判别条件已被探针矩阵钉死，见 tests/aiv_backend_probe.cpp 与 op_kernel/CMakeLists.txt 头注）。合并（llvm-objcopy --localize-symbol=g_tilingKey 双对象 + ld.lld -x -m aicorelinux --allow-multiple-definition -Ttext=0，AIC 先序）EXIT=0：合并对象 .text 280,672B，nm -u 未定义符号面 = 0（旧 rintf / ldexpf 阻塞面已随 Q 量化 / KV 预处理向量化清零）；双 pass 入口 / g_tilingKey 仲裁与各引擎私有副本的真机语义验证挂 M3。真机（Atlas 950PR NPU）执行尚未验证，属 M3 真机串联阶段。

## 一、目录结构与任务书建议目录映射

为了契合 `ops-transformer` 统一构建体系与仓内既有算子组织规范，交付文件组织如下：

- `op_host/`
  - 对应任务书：`op_host/`
  - 内容：算子定义注册 [sa2_def.cpp](./op_host/sa2_def.cpp)、Tiling 分析与资源校验 [sa2_tiling.cpp](./op_host/sa2_tiling.cpp)、构建配置 [CMakeLists.txt](./op_host/CMakeLists.txt)。
- `op_kernel/`
  - 对应任务书：`op_kernel/`
  - 内容：真实物理内核模块，严格专属于真机 NPU 物理编译（`__AICORE__` 或硬件桩 `-D__CCE_KT_TEST__`）。包含内核参数与常量契约 [sa2_spec.h](./op_kernel/sa2_spec.h)、8 特化模板维 [sa2_template_key.h](./op_kernel/sa2_template_key.h)、FA4 LPT 规划器 [sa2_planner.h](./op_kernel/sa2_planner.h)、UB 槽位内存预算 [sa2_ub_plan.h](./op_kernel/sa2_ub_plan.h)、跨核与本地同步旗号 [sa2_events.h](./op_kernel/sa2_events.h)、KV 预处理与注意力主循环 AscendC + CATLASS 真实硬件流水（[sa2_kv_prep.h](./op_kernel/sa2_kv_prep.h) / [sa2_attention_loop.h](./op_kernel/sa2_attention_loop.h)）与内核入口函数 [sa2_kernel.cpp](./op_kernel/sa2_kernel.cpp)。顶部设置物理硬防线（非设备宏直接 `#error` 拦截），彻底杜绝宿主 STL 与标量代码污染。
- `sim/`
  - 对应任务书：独立宿主高精数值仿真模块
  - 内容：宿主机标量数值仿真引擎 [sa2_sim_engine.h](./sim/sa2_sim_engine.h)、仿真注意力循环 [sa2_sim_loop.h](./sim/sa2_sim_loop.h)、仿真 KV 预处理 [sa2_sim_kv_prep.h](./sim/sa2_sim_kv_prep.h) 与构建配置 [CMakeLists.txt](./sim/CMakeLists.txt)。完全与设备内核物理隔离，专为 pytest、差分桥与离线黄金对拍服务。
- `op_api/`
  - 对应任务书：`op_api/`
  - 内容：两段式 aclnn 接口声明 [aclnn_sage_attention2.h](./op_api/aclnn_sage_attention2.h) 与分发实现 [aclnn_sage_attention2.cpp](./op_api/aclnn_sage_attention2.cpp)。
- `python/`
  - 对应任务书：`python/`
  - 内容：PyTorch Dispatcher 适配层 [sageattention2](./python/sageattention2)（包含自动分发 `_ops.py`、CSRC C++ 绑定 `csrc/attention.cpp` 与纯 CPU 数值参考实现 `reference.py`）。
- `tests/`
  - 对应任务书：`tests/`
  - 内容：端到端功能验收矩阵 [test_functional_matrix.py](./tests/test_functional_matrix.py)（TC-01 ~ TC-10 全覆盖，含零填充数值等价与负测试全面扩充）、宿主机数值断言与仿真对齐 [test_host_assertions.py](./tests/test_host_assertions.py)、C++ 数值契约断言桩 [sa2_numeric_conformance.cpp](./tests/sa2_numeric_conformance.cpp)（0 失败，checks 总数唯一权威口径见 tests/README.md 第三节 §2，含 M4.4 数理证明、M2.7.2 UB 防护、M2c-C2 极限分值域防御、双 Die NUMA 拓扑亲和/GQA L2 Cache 原地复用断言、vScale 窗口化 nN 布局契约与 Softmax 在线更新数值稳定性及 FTZ 冲零断言）、C++ 两段式接口与异常防御全量验证 [test_aclnn.cpp](./tests/test_aclnn.cpp)（覆盖正例 3 项 + 负例 8 项共 11 组全量校验防御，含官方标准大驼峰别名 aclnnSageAttention2）与套件说明 [tests/README.md](./tests/README.md)。
- `benchmarks/`
  - 对应任务书：性能评测程序
  - 内容：C++ aclnn 两段式性能对比测试程序 [benchmark_aclnn_fia.cpp](./benchmarks/benchmark_aclnn_fia.cpp)，覆盖 P-01 ~ P-07 全部规格，明确对照官方 `aclnnFusedInferAttentionScoreV5` 基准，核验 >= 1.6x 性能契约硬门禁。
- `tools/`
  - 对应任务书：`tools/`
  - 内容：真 GPU golden 导出工具 [export_gpu_golden.py](./tools/export_gpu_golden.py)，用于在真实 NVIDIA GPU 环境下一键导出 L-A 验证数据（覆盖 TC-02 全组合 × P-01 ~ P-07 形状）。【澄清】：导出工具已就绪 ≠ 真机 golden 已取得，L-A 验收标准以真 GPU 环境实际导出并落库的 golden 数据为最终准绳。
- `THIRD_PARTY`
  - 对应任务书：开源许可声明与借用登记
  - 内容：[THIRD_PARTY](./THIRD_PARTY) 集中登记借用的 FlashMLA、DeepGEMM-Ascend、TileKernels、tilelang、DeepSelect 5 个开源 MIT 资产信息与完整许可文本。

## 二、API 说明与环境变量

### 1. PyTorch Python API

提供统一自动分发入口 `sageattn` 以及与开源接口逐项对齐的 `sageattn_qk_int8_pv_fp8_asc`：

```python
from sageattention2 import sageattn, sageattn_qk_int8_pv_fp8_asc, replace_sdpa

# 1. 自动分发入口（NPU 上固定分发到 FP8 PV 路径）
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False)

# 2. 显式专用入口（参数与开源 sageattn_qk_int8_pv_fp8_cuda 严格逐项对齐）
out, lse = sageattn_qk_int8_pv_fp8_asc(
    q, k, v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_thread",  # 默认 per_thread；可选 per_warp
    sm_scale=None,
    pv_accum_dtype="fp32+fp16",   # 默认 fp32+fp16；可选 fp32, fp32+fp32
    smooth_k=True,                # 默认开启 K 中心化
    smooth_v=False,               # 默认关闭 V 中心化（仅 fp32 累加档生效）
    return_lse=True,
)

# 3. 即插即用替换 F.scaled_dot_product_attention
with replace_sdpa():
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

### 2. 环境变量控制

可通过环境变量覆盖默认累加档位与量化粒度：

- `SAGEATTN_PV_ACCUM_DTYPE`：指定默认 PV 累加档位，合法值为 `fp32`、`fp32+fp32`、`fp32+fp16`（缺省为 `fp32+fp16`）。
- `SAGEATTN_QK_QUANT_GRAN`：指定默认 QK 量化粒度，合法值为 `per_warp`、`per_thread`（缺省为 `per_thread`）。

### 3. C++ aclnn API

标准两段式接口：

```cpp
#include "aclnn_sage_attention2.h"

// 第一段：解析维度、转换布局并获取所需 workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnSa2Attention2GetWorkspaceSize(
    q, k, v, tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
    smoothK, smoothV, returnLse, out, lse, &workspaceSize, &executor);

// 申请 device 侧 workspace 空间
void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 第二段：在指定的 stream 上执行计算
aclnnSa2Attention2(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

## 三、构建与测试复现

### 1. 部署与编译构建

- **OPP 算子包一键构建与安装**（生产级交付标准）：

```bash
cd experimental/attention/sageattention2

# 1. 构建标准 OPP 自定义算子包（输出 cann-ops-transformer-custom_*.run）
bash build_opp.sh

# 2. 安装 OPP 算子包至 CANN 环境
bash install_opp.sh
```

- **PyTorch C++ 扩展安装**：

```bash
cd experimental/attention/sageattention2
pip install -r requirements.txt
pip install .
```

- **统一全量构建**（通过 ops-transformer 仓顶层脚本）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --experimental --soc=ascend950 --ops=sageattention2 --build-type=Release -O3
```

### 2. 运行测试与性能基线对比

- 执行 Python 验收与仿真断言测试（全量用例，总数与构成唯一权威口径见 tests/README.md 第三节 §1）：

```bash
cd ops-transformer
pytest experimental/attention/sageattention2/tests/ -v
```

- 执行 C++ 数值契约断言桩（100% 通过 0 失败，checks 总数唯一权威口径见 tests/README.md 第三节 §2，含 M4.4 数理定源、M2.7.2 UB 计划 O(1) 硬防护、M2c-C2 极限分数值域防御、双 Die NUMA 拓扑亲和/GQA L2 Cache 原地复用断言、vScale 窗口化 nN 布局契约与 Softmax 在线更新数值稳定性及 FTZ 冲零断言编译期/运行期用例）：

```bash
clang++ -std=c++17 -O3 -Wall -Wextra \
    -I experimental/attention/sageattention2/op_kernel \
    experimental/attention/sageattention2/tests/sa2_numeric_conformance.cpp \
    -o /tmp/sa2_numeric_conformance && /tmp/sa2_numeric_conformance
```

- 执行 C++ aclnn 全量防御与接口契约测试（全量用例 0 失败，构成唯一权威口径见 tests/README.md 第三节 §3）：

```bash
# 宿主机独立编译与运行（无需 CANN 依赖即可全量验证 11 组接口防御）
clang++ -std=c++17 -Wall -Wextra -Werror \
    experimental/attention/sageattention2/tests/test_aclnn.cpp \
    -o /tmp/test_aclnn && /tmp/test_aclnn

# CANN 环境下 CMake 构建产物运行
./build/bin/test_sage_attention2_aclnn
```

- 执行 C++ aclnn 两段式性能对比基准测试（明确对照官方 `aclnnFusedInferAttentionScoreV5`，覆盖 P-01 ~ P-07 形状；注：宿主运行 = 模型口径仿真，门禁结论以 M5.2/M5.3 真机实测为准）：

```bash
clang++ -std=c++17 -Wall -Wextra -Werror \
    -I experimental/attention/sageattention2/op_host \
    -I experimental/attention/sageattention2 \
    -I experimental/attention/sageattention2/op_api \
    experimental/attention/sageattention2/benchmarks/benchmark_aclnn_fia.cpp \
    -o /tmp/benchmark_aclnn_fia && /tmp/benchmark_aclnn_fia
```

## 四、开源资产借用与许可登记

根据开源许可规范与项目合规要求，本项目所参考与借用的第三方开源组件均已在 [THIRD_PARTY](./THIRD_PARTY) 集中明确登记，包括：

- FlashMLA（DeepSeek，MIT 许可）
  - 借用点：三状态 online softmax 状态机、θ 阈值法 skip-scale 调度、UB 累加器驻留与 MMA 重缩放解耦设计。
- DeepGEMM-Ascend（DeepSeek，MIT 许可）
  - 借用点：MX SF UE8M0 K 维相邻两元素 pack 为 int16 的打包约定、L2 hint 缓存命中优化策略。
- TileKernels（Tile-AI，MIT 许可）
  - 借用点：MX 格式 32-token 量化算法与指数域位操作设计。
- tilelang（Tile-AI / DeepSeek，MIT 许可）
  - 借用点：Ascend 950 Cube MMAD 指令调度流水与原型验证参考轨。
- DeepSelect（DeepSeek，MIT 许可）
  - 借用点：单指令复合原语与 AIV 指令错峰流水控制理念。

## 五、工程专题文档索引

- 任务需求契约逐条合规与实现优势论证：[docs/COMPLIANCE_SPEC.md](./docs/COMPLIANCE_SPEC.md)
- 与 CUDA 开源接口差异说明：[docs/CUDA_INTERFACE_SPEC.md](./docs/CUDA_INTERFACE_SPEC.md)（中文稿：[docs/CUDA接口差异说明.md](./docs/CUDA接口差异说明.md)）
- PyTorch 接入范围与深度集成契约说明：[docs/PYTORCH_INTEGRATION_SPEC.md](./docs/PYTORCH_INTEGRATION_SPEC.md)（中文稿：[docs/PyTorch接入说明.md](./docs/PyTorch接入说明.md)）
