# LastTranspose NDDMA — 算子设计说明

NDDMA benchmark 算子，用一条 NDDMA 搬运在 UB 内完成最后两轴转置 `permute[0,1,3,2]`。
仅 Ascend 950PR/950DT（arch35 / dav-3510）。由 Kernel 直调工程按 R-054（only-move-not-rewrite）
迁移而来，compute 逻辑逐字保留。

## 语义

```
输入  (dim0, dim1, dim2, dim3)   permute[0,1,3,2]
输出  (dim0, dim1, dim3, dim2)              # 交换最后两轴
```

等价于 `x.transpose(-1, -2)`（精度校验以此为 golden，`torch.equal` bitwise 全等）。

支持 dtype：`0=fp16(b16)` / `1=fp32(b32)` / `2=int8(b8)`。

## 维度角色

| 维 | 角色 |
|----|------|
| `dim0` | 分核维——按 `dim0` 在核间切分（benchmark 取 `dim0=56`＝物理 vector core 数） |
| `dim1` | 核内 for 循环维——每拍发一条 NDDMA；与 `dim0` 拍平成线性块索引 `g` |
| `dim2, dim3` | 一次 UB 内 NDDMA 转置的二维块 `[dim2,dim3]→[dim3,dim2]`；`blockElems = dim2*dim3`（转置前后一致） |

## NDDMA 转置读路径

核心是一条 2D `NdDmaLoopInfo<2>` 的 `DataCopy`，靠**交换 src/dst stride** 在 UB 内完成转置：

```
loopSrcStride = {1,      dim3}     # 内轴沿列步进 1、外轴沿行步进 dim3（按源 [dim2,dim3] 读）
loopDstStride = {dim2,   1}        # 内轴沿列步进 dim2、外轴步进 1（按目的 [dim3,dim2] 写）
loopSize      = {dim3,   dim2}
pad           = 0                  # 默认 NdDmaConfig：常数 0 填充、无 padding
```

读进 UB 时逐元素落到转置后的位置，故 UB 内已是 `[dim3,dim2]` 布局。`isMte3=1` 时再 `DataCopyPad`
写回 GM（精度对比路径）；`isMte3=0` 时跳过写回，隔离出**纯 NDDMA 读带宽**这一个变量，其余（DCI +
全部 SetFlag/WaitFlag + 流水结构）两模式完全一致。

## 分核与 UB 规划

- **分核**：`dim0` 均分到 `GetBlockNum()` 个核，余数前 `rem` 核各 +1；本核负责 `[i0Start, i0Start+i0Count)`。
- **UB 容量**（编译期常量，静态 Tensor 分配）：

  | dtype | `UB_CAP_ELEMS` | halfCap（DB 单半区） |
  |-------|----------------|----------------------|
  | b16 | 131072（2×128KB） | 65536 |
  | b32 | 32768（2×64KB） | 16384 |
  | b8 | 131072（2×64KB） | 65536 |

  DB 双缓冲把整块 UB 切成两半 ping-pong；约束 `blockElems ≤ halfCap`。

## 流水结构（`USE_PINGPONG`）

- **DB（true）**：奇偶 buffer + 奇偶 event 软件流水。read 提前一拍发射落到另一半区，与本块 write 在
  不同 pipe 上重叠；复用半区前等其上一轮写回完成（WAR）。prologue 发第 0 块读、epilogue 排空尾部两个
  写回 flag。
- **SB（false）**：单缓冲同址串行 `read → 同步 → (write) → 同步`，无重叠，作对照基线。

## pta 采集 / 校验使用

PTA 直调最小流程（build → demo）见 [`pta/README.md`](../pta/README.md)。常用命令：

```bash
cd pta
# 1) 构建 launch .so（需 CANN + ASC 工具链）
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target last_transpose_launch -j

# 2) PTA 直调 demo（需真实 NPU + torch_npu；JIT 构建扩展并跑一条 case）
python3 pta_nddma.py
```

## 相关文件

| 路径 | 角色 |
|------|------|
| `op_kernel/arch35/last_transpose_nddma_bench.h` | compute 头（与主线共用） |
| `op_host/op_api/aclnn_last_transpose_nddma_bench.h` | aclnn 二段式接口声明 |
| `docs/aclnn_last_transpose_nddma_bench.md` | aclnn 接口资料说明 |
| `examples/test_aclnn_last_transpose_nddma_bench.cpp` | aclnn 二段式调用示例 |
| `pta/` | PTA 直调 + msprof benchmark harness（见 `pta/README.md`） |
| `docs/perf/PERF_COMPARISON.md` | 迁移版 vs 源 `<<<>>>` 基线性能对照 |
