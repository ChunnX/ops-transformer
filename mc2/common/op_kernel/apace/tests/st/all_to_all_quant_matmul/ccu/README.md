# AllToAll + MXFP8 MatMul 通算融合（CCU 直调变体）

CCU 直调（asc-comm `<<<>>>` host launch）的 AllToAll + Blaze MXFP8 MatMul 融合 ST：
通信运行在 `__ccu_host__` CCU kernel 内（streamCcu，scale 轮 + data 轮多轮事件桥），
AIC 融合 kernel（streamAicore）经 XN/CKE 寄存器 MMIO 直写预写通信参数区 + wake，
matmul 逐 tile 轮询 round-done，tile 级通算 overlap。数据语义与
`../`（urma/hcomm 变体）一致：K-并行 GEMM，Y = Σ_r recvData[r块,:] @ x2[r*K段,:]。

## 目录结构

```
ccu/
├── CMakeLists.txt              # 构建配置（兄弟目录骨架 + hcomm/asccomm_ccu 链接增量）
├── cases.csv                   # 用例表（mode,M,tileM,K,N,loop）
├── run.sh                      # 编译 + 运行（precision/perf；perf 走 msprof）
├── scripts/
│   └── parse_prof.py           # msprof op_summary 解析（AIC kernel 后 5 次取平均）
└── src/
    └── main.cpp                # 单翻译单元编译源（-xasc）：经 include 引入 .asc 链
```

device 侧实现：

```
apace/core/ccu_comm/                                       # CCU 通信公共件（可复用）
├── ccu_comm.h                                             # 槽位/事件契约 + LoopGroup 搬运引擎 + host 资源管理
└── all_to_all/
    ├── ccu_all_to_all_comm.h                              # CcuCommTilingData + AIC 原语（prepare/commit/wait）
    └── ccu_all_to_all_kernel.asc                          # __ccu_host__ CcuAllToAllMesh1DKernel（单锚点，名字冻结）

apace/kernel/fusions/all_to_all_quant_matmul/              # 本融合专属
├── all_to_all_mx_quant_matmul_ccu_tiling_data.h           # tiling 聚合（comm + matmul）
├── all_to_all_mx_quant_matmul_ccu_impl.h                  # AIC 侧：CcuCommWaitPolicy + 融合 matmul impl
├── all_to_all_ccu_kernel_impl.asc                         # 单 TU 锚点：AIC 入口 + Custom（双流 launch）
└── all_to_all_mx_quant_matmul_ccu_host_impl.h             # host 编排：Validate/Prepare/Release
```

> 单翻译单元：`src/main.cpp` 经 `#include` 引入 `all_to_all_ccu_kernel_impl.asc`（锚点），
> 锚点按序引入 core 通信件 → 融合 AIC impl → host 编排。CMake 只编译 `src/main.cpp`
> 一个源文件，不要把各部分拆成独立编译单元；core 的 `.asc` 每个二进制只能被引入一次。

## 编译

```bash
source ~/Ascend/cann/set_env.sh          # CANN >= 9.2.0（ccu_host_launch.h / libasccomm_ccu.so）
cd <tests/st/all_to_all_quant_matmul/ccu>
bash run.sh --skip-build 2>/dev/null;    # 首次直接 bash run.sh（含编译）
# 或手动：
cmake -S ../../.. -B ../../../build
cmake --build ../../../build --target apace_a2a_qmm_ccu_st --parallel 8
```

## 运行

```bash
export HCCL_OP_EXPANSION_MODE=CCU_SCHED   # CCU 调度模式（运行时必须）

./apace_a2a_qmm_ccu_st precision                          # 默认：M=512 tileM=128 K=256 N=256
./apace_a2a_qmm_ccu_st precision 8192 512 2048 3584      # 目标 case（标卡 4P）
./apace_a2a_qmm_ccu_st perf 8192 512 2048 3584 10        # perf：循环 10 次
```

rank 数 = 本机 NPU 数（进程内每卡一线程，单进程多 rank）。

## 验证语义

内置 host 侧行级溯源校验（无需外部 golden）：A/B 用 E4M3FN 2 的幂闭式编码
（源 rank/目标路由/行号三维编进指数），期望 Y = K × Σ_r 2^((2i+r+m)%4 + r%2)，
逐元素比对 BF16 位型（全精确可表示，零舍入噪声）。FAIL 时打印首错行列 +
期望/实际位型（可反解错误类别），并落地 `output/<rank>/npu_out.bin`。

## 运行环境要求

- CANN ≥ 9.2.0，`HCCL_OP_EXPANSION_MODE=CCU_SCHED`
- 至少 2 张 NPU 卡；共享服务器运行须走排队工具（勿裸跑）
- perf 数据解析：`python3 scripts/parse_prof.py --all`（AIC kernel duration，后 5 次取平均）
