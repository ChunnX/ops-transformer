# Broadcast NDDMA — 算子设计说明

NDDMA benchmark 算子，用一条 3D NDDMA 沿广播轴复制搬运（靠 `srcStride=0` 实现广播读）。
仅 Ascend 950PR/950DT（arch35 / dav-3510）。由 Kernel 直调工程按 R-054（only-move-not-rewrite）
迁移而来，compute 逻辑逐字保留。3D 单条方案权威定义见源 DESIGN §9（§8 方案 A「dim2 核内 for」已作废）。

## 语义

输入按 scene 的广播轴长度=1 的 5D 形态，输出恒 `(dim0, dim1, dim2, dim3, dim4)`。
等价于 `torch.broadcast_to`（精度校验以此为 golden，`torch.equal` bitwise 全等）。

支持 dtype：`0=fp16(b16)` / `1=fp32(b32)` / `2=int8(b8)`。

### 四场景（`scene`，DESIGN §9.0）

| scene | 值 | 广播轴 | srcStride `[inner,mid,outer]` | 源块（SrcOff = g × …） |
|-------|----|--------|-------------------------------|------------------------|
| AB | 0 | dim4（inner 置零） | `[0, 1, dim3]` | `dim2*dim3` |
| BA | 1 | dim3（中轴置零） | `[1, 0, dim4]` | `dim2*dim4` |
| BAB | 2 | dim2+dim4（inner+outer 双置零） | `[0, 1, 0]` | `dim3` |
| ABA | 3 | dim3（中轴置零，≡ BA §9.1） | `[1, 0, dim4]` | `dim2*dim4` |

> **BA ≡ ABA 同构（DESIGN §9.1，有意为之，非 bug）**：3D 单条方案下二者 loopInfo 与 SrcOff 完全相同，
> `switch` 中 `SCENE_BA` 单列、`ABA` 落 `default` 且分支体逐字节相同——code review 勿判重复。

## 维度角色

> ⚠️ 设计红线：`dim1` 是**唯一**核内 for 循环维；`dim2/dim3/dim4` 全是同一条 NDDMA 的 loopInfo 轴
> （单条 tile = `dim2*dim3*dim4`），**不可**把 dim2/dim3/dim4 降为 for 循环规避。

| 维 | 角色 |
|----|------|
| `dim0` | 分核维——按 `dim0` 在核间切分（`numBlocks = dim0`；benchmark 取 `dim0=56`＝物理 vector core 数，单波 per=1） |
| `dim1` | 唯一核内 for 循环维——与 `dim0` 拍平成线性块号 `g`，每拍发一条 3D NDDMA |
| `dim2` | NDDMA 最外轴 `loopInfo[2]`（单条 NDDMA 内遍历，非 for） |
| `dim3` | NDDMA 中轴 `loopInfo[1]` |
| `dim4` | NDDMA 最内连续轴 `loopInfo[0]`（inner 连续块须 8B 对齐） |

## NDDMA 广播读路径

一条 3D `NdDmaLoopInfo<3>`（索引 0=最内 dim4，2=最外 dim2）。dstStride 恒连续 `[1, dim4, dim3*dim4]`，
`loopSize = [dim4, dim3, dim2]`，置零轴由 `srcStride=0` 实现「读同一份源、写多份目的」的广播复制
（见上表 scene→srcStride）。目的块 `curDstElems = dim2*dim3*dim4`。

`isMte3=1` 时 `DataCopyPad` 写回 GM（精度对比）；`isMte3=0` 跳过写回，隔离**纯 NDDMA 读带宽**，
其余（DCI + 全部同步 + 流水结构）两模式完全一致。

## 分核与 UB 规划

- **分核**：`numBlocks = dim0` 均分到各核，余数前 `rem` 核各 +1；`Gid(j)` 由本核 `unitStart_/unitCount_`
  还原全局块号 `g`。
- **Init 三断言**（DESIGN §9.4，纯回归网拦 host 配错，正常路径不触发）：
  1. `dim2/dim3/dim4 ≥ 2`（loopSize>1 红线，无退化轴）；
  2. `dim4 * sizeof(T)` 须 8B 对齐（inner 连续块对齐）；
  3. `dim2*dim3*dim4 ≤ halfCap`（单条 3D tile ≤ DB 单半区，最严约束）。
- **UB 容量**（编译期常量，沿用参考算子已上板验证定值）：

  | dtype | `UB_CAP_ELEMS` | halfCap（DB 单半区） |
  |-------|----------------|----------------------|
  | b16 | 131072（2×128KB） | 65536 |
  | b32 | 32768（2×64KB） | 16384 |
  | b8 | 131072（2×64KB） | 65536 |

## 流水结构（`USE_PINGPONG`）

- **DB（true）**：奇偶 buffer + 奇偶 event。read 提前一拍发射落到另一半区，与本块 write 重叠；复用半区
  前等其上一轮写回完成（WAR）。prologue 发第 0 块读、epilogue 排空尾部两个写回 flag。每拍按 `k+1` 的
  scene 重建 loopInfo/SrcOff。
- **SB（false）**：单缓冲同址串行 `read → 同步 → (write) → 同步`，无重叠，作对照基线。

## pta 采集 / 校验使用

完整 harness 见 [`pta/README.md`](../pta/README.md)。常用命令：

```bash
cd pta
# 1) 构建 launch .so（需 CANN + ASC 工具链）
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nddma_broadcast_launch -j

# 2) 精度校验（需真实 NPU + torch_npu；bitwise vs torch broadcast_to）
python3 validate_pta.py            # 5D 全集（默认唯一分支）
#   ⚠️ NDDMA_MODE=sb 只切 Python 容量表，SB 精度复跑须先 USE_PINGPONG=false 重编 .so + 清缓存

# 3) 性能采集（需真实 NPU + msprof）
bash collect_matrix.sh    <matrix_out> [iters]              # 全量四配置扫描（自动 toggle+重编）
bash collect_matrix_5d.sh <matrix_out> [iters]              # 5D 场景全量四配置扫描
bash collect_corescan.sh  <out_dir> [iters] [mode] [isMte3] # 核数扫描 blockNum={1,2,4,8,16,28,56}
bash collect_single.sh    <db|sb> <cfg> <isMte3> [iters]    # 单配置（预编 .so）
# 4) 解析（注意 parse_matrix.py 参数顺序：iters 在前、matrix_dir 在后）
python3 parse_matrix.py   [iters] [matrix_dir]
python3 parse_corescan.py <cfg_dir> <isMte3> [t_nddma_us] [T_max_ms] [iters]
```

## 相关文件

| 路径 | 角色 |
|------|------|
| `op_kernel/arch35/broadcast_nddma_bench.h` | compute 头（与主线共用，含 `SCENE_*` 常量 + 3D loopInfo） |
| `op_host/op_api/aclnn_broadcast_nddma_bench.h` | aclnn 二段式接口声明 |
| `docs/aclnn_broadcast_nddma_bench.md` | aclnn 接口资料说明 |
| `pta/` | PTA 直调 + msprof benchmark harness（见 `pta/README.md`） |
| `docs/perf/PERF_COMPARISON.md` | 迁移版 vs 源 `<<<>>>` 基线性能对照 |
