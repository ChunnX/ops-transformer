# CCU AllToAll 纯通信验证（阶段1）

2 卡（可扩展）均匀 AllToAll + CCU 直调（FP16，strideCount 型，commTurn 多轮事件桥）。
通信全部运行在 `__ccu_host__` CCU kernel 内（streamCcu），AIC 桥 kernel（streamAicore）
预写参数桥 + wake + 轮询 round done——双流架构与 `../ccu_all_gather_matmul` 同款。

> 设计文档见 `docs/DESIGN_ANALYSIS.md`（含 CANN `ccu_kernel_all_to_all_mesh1d.h` 代码搬运映射表）。
> **本工程 Blaze-free**：纯通信阶段不依赖 ops-tensor/Blaze，编译只需 CANN 环境。

## 目录结构

```
ccu_alltoall/
├── CMakeLists.txt              # 工程根（Blaze-free）
├── README.md                   # 本文件
├── docs/
│   └── DESIGN_ANALYSIS.md      # 设计文档（CANN 代码搬运映射表）
├── scripts/
│   └── run.sh                  # 一键编译 + 运行（唯一入口脚本）
└── src/                        # 单翻译单元：.asc 经相对 include 引入其余三个文件
    ├── ccu_alltoall_kernel.asc # CCU kernel（__ccu_host__）+ AIC 桥 kernel + launch 入口
    ├── a2a_bridge_kernel.h     # AIC 侧：参数桥预写 + wake + per-round done 轮询
    ├── host_launcher.h         # Host 编排：A2AContext + PrepareAllToAll
    └── sample_main.cpp         # Host 入口：溯源编码数据（元素级可区分）+ 断言校验 + 双模式
```

> 注意：CMakeLists 只编译 `src/ccu_alltoall_kernel.asc` 一个源文件——ASC 编译模型下
> host 侧 `.h`/`.cpp` 由 `.asc` 单翻译单元 include 引入（对齐 `../ccu_all_gather_matmul` 结构），
> 不要拆成独立编译单元。

## 编译 + 运行（在用户的编译/运行服务器执行）

```bash
# 确认 CANN 环境
echo $ASCEND_HOME_PATH   # 应指向 CANN 安装目录
source $ASCEND_HOME_PATH/set_env.sh

cd <本目录>
bash scripts/run.sh
```

`scripts/run.sh` 自动完成：source 环境 → `export HCCL_OP_EXPANSION_MODE=CCU_SCHED` → cmake 编译 →
运行默认 precision 用例（每块 65536 元素 FP16 = 128KB，commTurn=1）。运行产物（`output/`、`build/`）
统一落地在工程根目录。

### 手动分步（调试用，在工程根目录执行）

```bash
source $ASCEND_HOME_PATH/set_env.sh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

rm -rf build
cmake -S . -B build
cmake --build build --target ccu_alltoall --parallel 4

./build/ccu_alltoall precision              # 默认：128KB 块，commTurn=1
./build/ccu_alltoall precision 65536 2      # commTurn=2 多轮事件桥验证
./build/ccu_alltoall precision 16385 1      # 32770 字节块：触发 goSize 尾部分支
./build/ccu_alltoall perf 65536 1 10        # perf 模式：循环 10 次不落地
```

## 精度校验说明

程序内置 host 侧逐块逐元素断言（无需外部 golden）：

- **数据设计（溯源编码）**：每个 uint16 元素位型 = `srcRank<<12 | dstBlock<<8 | tPos`，
  其中 `tPos = (t ^ (t>>8)) & 0xFF`（块内位置摘要）——**每个 slice、每个元素都不同**；
  纯字节搬运验证，位型不解释为 FP16 数值（CCU Write/GroupCopy 均为字节级拷贝）
- **发送侧**：rank i 的 sendBuf 第 j 块元素 t = `Encode(src=i, dst=j, t)`
- **期望结果**（AllToAll 语义 `recv_i[r] = send_r[i]`）：rank i 的 recvBuf 第 r 块元素 t
  = `Encode(src=r, dst=i, t)`
- **输出**：每 rank 打印 `AllToAll PASS/FAIL`；FAIL 时打印首错定位
  （块号/元素偏移/期望 vs 实际位型 + 位型直接反解 srcRank/dstBlock，可分类
  src 混淆 / 块混淆 / 块内偏移 / 数据损坏；tPos 非线性混合使"Δ 为 256 整数倍"的
  块内错位亦可检出）
- 落地 `output/<rank>/npu_out.bin`（uint16 位型），可用 python 复核：

```python
import numpy as np
R = 2  # rank 数
elems = 65536
MY_RANK = 0  # 复核哪个 rank 的输出
t = np.arange(elems)
raw = np.fromfile(f"output/{MY_RANK}/npu_out.bin", dtype=np.uint16)
for r in range(R):
    expect = (r << 12) | (MY_RANK << 8) | ((t ^ (t >> 8)) & 0xFF)
    assert np.all(raw[r*elems:(r+1)*elems] == expect), f"block {r} mismatch"
print("PASS")
```

## 关键编译参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--npu-arch` | `dav-3510` | Ascend 950 |
| `HCCL_OP_EXPANSION_MODE` | `CCU_SCHED` | CCU 调度模式（运行时必须） |
| 链接库 | `hcomm asccomm_ccu_dataplane c_sec ascendcl acl_rt ascendc_runtime pthread dl` | 无 Blaze 三件套（platform/tiling_api/register） |

## 运行环境要求

- `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 必须设置（`run.sh` 已包含）
- `ASCEND_HOME_PATH` 指向 CANN 安装目录
- 至少 2 张 NPU 卡（2 rank 单进程多线程）

## 与 ccu_all_gather_matmul 的差异速览

| 项 | AG 示例 | 本工程 |
|---|---|---|
| 通信语义 | AllGather（一份源广播，recvBuf 膨胀 R 倍） | AllToAll（逐对换块，recvBuf 等量） |
| 参数桥 | (commTurn+1)×10 槽（含 scale 轮） | commTurn×11 槽（新增 srcStride） |
| CCU kernel src | 每 round 固定（input+inputOffset） | 逐 rank 递推（input+srcOffset+j*srcStride） |
| 远端 dst 落点 | output+outputOffset（轮次决定） | output+myRank*blockSize+r*sub（发送方 rankId 决定，host 预乘烧入 dstOffset） |
| 本地拷贝 | sendBuf+inputOff → recvBuf+outputOff | 独立累加构造 localSrc（本 rank 块）后 GroupCopy（与远端 j 循环互不复用） |
| dtype | INT8（FP8 复用） | FP16 |
| Blaze 依赖 | 需要（ops-tensor submodule） | 无（阶段3 叠 matmul 时恢复） |

## Blaze 恢复点（阶段3 叠加通算融合时）

`CMakeLists.txt` 恢复以下三项（对照 `../ccu_all_gather_matmul/CMakeLists.txt`）：
1. `OPS_TENSOR_ROOT` 检查与 3 个 include 路径（L25-33, L57-60 对应段）
2. 链接库 `platform tiling_api register`
3. `platform_ascendc.h` 相关 include 路径（GetCoreNumAic 查询，替换 `AIC_BLOCK_NUM=8` 常量）

## 已知问题 / 风险

- **AIC 桥 kernel 固定 8 核**（Blaze-free 无法查询 platform 核数），阶段3 恢复查询
- **token 机制对齐 AG 已验证行为**：仅对 sendBuf 获取 token；远端写 recvBuf 沿用该令牌机制
  （AG 上板验证过同款用法；若阶段2 上板遇到 token 校验问题，优先排查此处）
- **CCU kernel 性能不可被 msprof --ai-core 采集**（沿用 AG 实测结论），perf 模式仅作烟囱
