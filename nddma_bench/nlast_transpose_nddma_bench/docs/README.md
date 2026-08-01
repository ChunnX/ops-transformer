# NLastTranspose NDDMA — 算子设计说明

NDDMA benchmark 算子，用一条 3D NDDMA 在 UB 内交换中间两轴 `permute[0,1,3,2,4]`，保留最内连续轴。
仅 Ascend 950PR/950DT（arch35 / dav-3510）。由 Kernel 直调工程按 R-054（only-move-not-rewrite）
迁移而来，compute 逻辑逐字保留。

## 语义

```
输入  (dim0, dim1, dim2, dim3, dim4)   permute[0,1,3,2,4]
输出  (dim0, dim1, dim3, dim2, dim4)              # 交换中间两轴 dim2/dim3，dim4 不动
```

等价于 `x.permute(0,1,3,2,4)`（精度校验以此为 golden，`torch.equal` bitwise 全等）。

支持 dtype：`0=fp16(b16)` / `1=fp32(b32)` / `2=int8(b8)`。

## 维度角色

| 维 | 角色 |
|----|------|
| `dim0` | 分核维——按 `dim0` 在核间切分（benchmark 取 `dim0=56`＝物理 vector core 数） |
| `dim1` | 核内 for 循环维——每拍发一条 3D NDDMA；与 `dim0` 拍平成线性块索引 `g` |
| `dim2, dim3` | 被交换的中间两轴（转置对象） |
| `dim4` | 连续内轴，保留不动、随块一起搬运；`blockElems = dim2*dim3*dim4`（转置前后一致） |

## NDDMA 转置读路径

一条 3D `NdDmaLoopInfo<3>` 的 `DataCopy`（索引 0=最内、2=最外），靠**交换 dim2/dim3 的 stride**
完成转置，`dim4` 作为连续内轴保持 stride 1：

```
              loopSize   srcStride       dstStride
inner (idx0)  dim4       1               1            # dim4 连续，src/dst 都步进 1
middle(idx1)  dim2       dim3*dim4       dim4         # 沿 dim2：src 大步、dst 小步
outer (idx2)  dim3       dim4            dim2*dim4    # 沿 dim3：src 小步、dst 大步
pad = 0                                               # 默认 NdDmaConfig：常数填充、无 padding
```

`dim2/dim3` 的 src stride（`dim3*dim4 ↔ dim4`）与 dst stride（`dim4 ↔ dim2*dim4`）互换即完成中间两轴
转置。`isMte3=1` 时 `DataCopyPad` 写回 GM（精度对比）；`isMte3=0` 跳过写回，隔离**纯 NDDMA 读带宽**，
其余（DCI + 全部同步 + 流水结构）两模式完全一致。

## 分核与 UB 规划

- **分核**：`dim0` 均分到各核，余数前 `rem` 核各 +1；本核负责 `[i0Start, i0Start+i0Count)`。
- **UB 容量**（编译期常量，沿用参考算子 `last_transpose` 已上板验证定值）：

  | dtype | `UB_CAP_ELEMS` | halfCap（DB 单半区） |
  |-------|----------------|----------------------|
  | b16 | 131072（2×128KB） | 65536 |
  | b32 | 32768（2×64KB） | 16384 |
  | b8 | 131072（2×64KB） | 65536 |

  DB 双缓冲把整块 UB 切成两半 ping-pong；约束 `blockElems ≤ halfCap`。

## 流水结构（`USE_PINGPONG`）

- **DB（true）**：奇偶 buffer + 奇偶 event。read 提前一拍发射落到另一半区，与本块 write 在不同 pipe
  上重叠，打破连续 NDDMA 的同址依赖；复用半区前等其上一轮写回完成（WAR）。prologue 发第 0 块读、
  epilogue 排空尾部两个写回 flag。
- **SB（false）**：单缓冲同址串行 `read → 同步 → (write) → 同步`，无重叠，作对照基线。

## pta 采集 / 校验使用

完整 harness 见 [`pta/README.md`](../pta/README.md)。常用命令：

```bash
cd pta
# 1) 构建 launch .so（需 CANN + ASC 工具链）
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nlast_transpose_launch -j

# 2) 精度校验（需真实 NPU + torch_npu；bitwise vs x.permute(0,1,3,2,4)）
python3 validate_pta.py --all                 # 全量 NL_0~47 + B8
python3 validate_pta.py 56 32 128 8 16 0       # 单 case: dim0 dim1 dim2 dim3 dim4 dtype
NDDMA_MODE=sb python3 validate_pta.py --all    # SB 变体（须先按 SB 重编 .so）

# 3) 性能采集（需真实 NPU + msprof）
#   本算子无 FP16+FP32 全 dtype 全量扫描脚本：全量 FP 走 collect_single.sh ×4 配置（db/sb 手动 toggle+重编）；
#   int8 全量走自动化 collect_b8.sh。
bash collect_single.sh <db|sb> <cfg> <isMte3> [iters]   # cfg∈{db_rw,db_r,sb_rw,sb_r}
bash collect_b8.sh     <op_root> <matrix_out> [iters]   # int8-only 全量四配置扫描
# 4) 解析
python3 parse_matrix.py <matrix_dir> [iters]
python3 parse_b8.py     <matrix_dir> [iters]
```

## 相关文件

| 路径 | 角色 |
|------|------|
| `op_kernel/arch35/nlast_transpose_nddma_bench.h` | compute 头（与主线共用） |
| `op_host/op_api/aclnn_nlast_transpose_nddma_bench.h` | aclnn 二段式接口声明 |
| `docs/aclnn_nlast_transpose_nddma_bench.md` | aclnn 接口资料说明 |
| `pta/` | PTA 直调 + msprof benchmark harness（见 `pta/README.md`） |
