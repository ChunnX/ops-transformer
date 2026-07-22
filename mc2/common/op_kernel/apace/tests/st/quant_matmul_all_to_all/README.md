# QuantMatmulAllToAll ST（apace Hcomm kernel 直调）

`mc2/common/op_kernel/apace/kernel/quant_matmul_all_to_all/` 下
`MatmulAllToAllMxKernelE4M3E4M3` kernel 的系统测试（ST）直调工程。

参考：[`matmul_allto_all_story/src`](https://gitcode.com/hzp_gitcode/cann-samples-hzp/blob/master-qbmm-get-combined/Samples/2_Performance/matmul_allto_all_story/src/)（分支 `master-qbmm-get-combined`）。

## 目录结构

```
quant_matmul_all_to_all/
├── CMakeLists.txt              # bisheng 编译配置（apace + CANN Blaze/tensor_api/hccl）
├── run.sh                      # 一键: gen → build → run → verify
├── src/
│   ├── main.cpp                # host launcher: fork 多 rank + HCCL + tiling + <<<>>> 直调
│   ├── comm_context_builder.h  # HCCL CommContext 构建（适配自 mc2/common/op_api/mc2_context.cpp）
│   └── utils.h                 # 日志/ACL 检查/文件读写
└── scripts/
    ├── gen_data.py             # 生成 MXFP8 输入 + CPU golden（可独立运行）
    └── verify_result.py        # BF16 精度比对 rtol=2% atol=0.2（可独立运行）
```

## 数据流与语义假设

依据对 `all_to_all_mx_matmul_udma_impl.h` 的阅读，假设语义如下（**kernel 可编译后需联调确认**）：

- 每个 rank `r` 持有本地 `A_r [M, K]`（fp8_e4m3）+ `scaleA_r [M, ceil(K/64), 2]`（fp8_e8m0）
- 每个 rank 持有完整 `B [rankNum*K, N]`（按 K 切成 rankNum 段，第 r 段属 rank r）+ 完整 `scaleB`
- AllToAll 将各 rank 的 `A_r` 交换到所有 rank
- 每个 rank 计算 `C = Σ_r dequant(A_r) × dequant(B_r)` → `[M, N]` bf16，**各 rank 输出相同（replicated）**

> 参考样例 `matmul_allto_all_story` 走 **aclshmemx** 后端（自己的 `MatmulAllToAllKernel`，
> 输出 AllToAll 沿 N 轴 → `[rankNum*M, N/rankNum]`）。apace kernel 走 **HCCL CommContext**
> 后端，语义不同（AllToAll 沿输入 A 的 M 轴，输出 replicated `[M, N]`）。本 ST 按 apace 语义实现。

## 用法

```bash
# 1. source CANN 环境
source /usr/local/Ascend/cann-9.1.0/set_env.sh

# 2. 一键运行（默认 2 ranks, 2048×3584×4096）
bash run.sh

# 自定义参数: m k n rank_num tile_cnt local_matmul
bash run.sh 2048 3584 4096 4 4 1

# 仅生成 CPU golden（不依赖 kernel）
bash run.sh --gen-only 2048 3584 4096 2

# 仅精度比对（需已有 output/{r}/npu_out.bin）
bash run.sh --verify-only 2048 4096 2
```

**约束**：`K % 32 == 0`；`ceil(K/64) % 2 == 0`；`N % rankNum == 0`；`M % tileCnt == 0`。

## 精度标准

MXFP8 量化 matmul + BF16 输出：`rtol = 2%`，`atol = 0.2`（`torch.allclose`）。

---

## ⚠️ 已知阻塞：apace kernel 不可编译

`all_to_all_mx_matmul_udma_impl.h`（commit `3460fbb "add a2a"`，无 mc2 op 引用）当前存在 **6 处编译 bug**，
经 bisheng + CANN 9.1.0 实际编译验证（`tiling_host.cpp` host 模式编译通过，`main.cpp` kernel 模式失败）。
**build/run 步骤在修复前会失败**；`gen_data.py`/`verify_result.py` 不受影响，可独立运行。

| # | 文件:行 | Bug | 建议修复 |
|---|--------|-----|---------|
| 1 | `block/aiv_comm/collective_comm_base.h:19` | `#include "../tiling/comm_tiling_data.h"` 路径错误（tiling 在 `apace/tiling/`，从 `block/aiv_comm/` 需 `../../tiling/`） | 改为 `#include "../../tiling/comm_tiling_data.h"` |
| 2 | `kernel/all_to_all_quant_matmul/all_to_all_mx_matmul_udma_impl.h:27` | `using Apace::Utils::CommTilingData;` 命名空间不存在（`CommTilingData` 是全局的） | 改为 `using ::CommTilingData;` |
| 3 | 同上 `:108, :110` | `CommAddrLayut` 拼写错误 | 改为 `CommAddrLayout` |
| 4 | 同上 `:419` 等四处 | `Apace::AivComm::CommContext` 类型未定义（kernel 用 `hcommCtx->ubmemCtx`/`udmaCtx`） | 在 `collective_comm_context.h` 补 `struct CommContext { CommUbmemContext ubmemCtx; CommUdmaContext udmaCtx; };` |
| 5 | 同上 `:281` 及 `:201/:211` | `commTilingData.bufferSize` 字段不存在（`CommTilingData` 仅 5 字段）；`commTilingDataA_ = {...6 值...}` 初始化也越界 | 在 `CommTilingData` 补 `uint64_t bufferSize;`，或 kernel 改用其它字段 |
| 6 | 同上 `:426` | `TPipe pipe;` 在 `__global__` kernel 函数内，`TPipe` 是 `AscendC::TPipe`（kernel 端类型，需限定命名空间或 `using`） | 改为 `AscendC::TPipe pipe;` 或文件顶部 `using AscendC::TPipe;` |

> ST 侧已对 #4 做了就地兜底：`src/comm_context_builder.h` 在 `Apace::AivComm` 命名空间补了 `CommContext` 定义，
> 并 include `collective_comm_context.h` 获取 `CommUdmaContext`/`CommUbmemContext`。待 apace 正式定义后
> 移除该兜底即可。#1/#2/#3/#5/#6 需在 apace 侧修复。

修复上述 6 处后，`bash scripts/run.sh` 即可走通完整 gen→build→run→verify 流程。

### 编译架构说明（已验证）

apace tiling 头文件传递包含 CANN `platform_ascendc.h`，在 `-xasc`（kernel 编译模式）下会引入
`__aicore__`/`__gm__` 的 kernel 端定义，与 host 端代码冲突。因此本工程采用 **双 TU 编译**：

| TU | 编译模式 | 内容 | 状态 |
|----|---------|------|------|
| `src/tiling_host.cpp` | host（无 `-xasc`） | apace tiling 计算，导出 `ComputeAllToAllTiling()` | ✅ 编译通过 |
| `src/main.cpp` | kernel（`-xasc`） | kernel 头文件 + `<<<>>>` 直调 + HCCL CommContext | ❌ 受 kernel bug 阻塞 |

两个 TU 编译后链接为同一可执行文件（已验证跨模式链接可行）。

## 依赖

- CANN 9.1.0（`ASCEND_HOME_PATH`）
- Blaze / tensor_api / hcomm：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/`
- bisheng 编译器：`$ASCEND_HOME_PATH/x86_64-linux/ccec_compiler/bin/bisheng`
- NPU：Atlas A3（dav-3510 / Ascend950PR），本机 4 卡
