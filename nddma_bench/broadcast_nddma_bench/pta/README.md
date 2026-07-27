# Broadcast NDDMA — PTA 直调 + msprof benchmark harness（3D NDDMA 方案，DESIGN §9）

PyTorch 直调（PTA）+ msprof 采集台，从 `ops-direct-invoke` 的 `op_pta/` 结构化迁移而来
（R-054 only-move-not-rewrite：计算逻辑与 benchmark shape 矩阵逐字保留）。

## 与 registry-invoke 主线的关系

- **compute 逻辑**：`../op_kernel/arch35/broadcast_nddma_bench.h`
  （命名空间类 `BroadcastNddmaBenchNs::KernelBroadcastNDDMA<T, ubCapElems>` + `SCENE_AB/BA/BAB/ABA`）。
  本 harness 与主线 registry-invoke 算子**共用同一份 compute 头**，不复制、不改写。
- **本目录 launch.asc**：定义 thin benchmark kernel 入口 `BroadcastNDDMABenchKernel`
  （读 8 字段 attr GM → 按 dtype 派发到命名空间类，scene 透传 `Init` → `Process`）+ 唯一
  C-ABI 函数 `LaunchNddmaBroadcast5d`（本算子无 `*Host` 变体），签名与源保持一致。
  blockNum 由入参 `blockNum` 控制（核数扫描），0 → 缺省 `min(dim0,56)`（`kVectorCoreCnt`）。
- **USE_PINGPONG 开关**随 compute 头迁移；`collect_full_matrix*.sh` / `collect_corescan_5d.sh`
  的 `set_pingpong()` 直接 sed 该头。

## 四场景（DESIGN §9.0）

`scene`：0=AB（dim4 广播）1=BA（dim3 广播）2=BAB（dim2+dim4 广播）3=ABA（≡BA，§9.1）。
输入 shape 随 scene（广播轴长度=1），输出恒 (dim0,dim1,dim2,dim3,dim4)。

## 构建 launch .so（离线需 CANN + ASC 工具链）

```bash
cd pta
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nddma_broadcast_launch -j
# 产物：pta/build/libnddma_broadcast_launch.so
```

`pta_nddma.py` 的 `_BUILD` 指向 `pta/build`，JIT 构建 pybind 扩展时按 C ABI 链接此 .so。

## 运行（需真实 NPU + torch_npu + msprof）

```bash
# 单配置 msprof（自定位）
bash collect_matrix.sh <db|sb> <db_rw|db_r|sb_rw|sb_r> <isMte3> [iters]
# 全量矩阵（DB/SB × isMte3 0/1，自动切 USE_PINGPONG + 重编）
bash collect_full_matrix.sh <matrix_out> [iters]
# 5D 场景全量矩阵（NDDMA_SUITE=5D）
bash collect_full_matrix_5d.sh <matrix_out> [iters]
# 5D 核数扫描（blockNum={1,2,4,8,16,28,56}）
bash collect_corescan_5d.sh <out_dir> [iters] [mode] [isMte3]
# 解析
python3 parse_matrix_5d.py <matrix_out>
python3 parse_corescan_5d.py <out_dir>
# 精度校验（bitwise golden，§9.5）
python3 validate_pta.py
```

> 本机无 NPU/CANN runtime，以上运行步骤仅在真实 Ascend 950PR 环境执行；
> 迁移仅做文件搬运 + 路径重指，不产出性能数据。

## 文件清单

| 文件 | 来源 | 迁移动作 |
|------|------|----------|
| `nddma_broadcast_launch.asc` | op_pta 同名 | include 重指 arch35 头 + 本地 attr(8字段)/entry + C-ABI 逐字保留 |
| `nddma_broadcast_launch.h`   | op_pta 同名 | 逐字复制 |
| `nddma_broadcast_ext.cpp`    | op_pta 同名 | 逐字复制 |
| `pta_nddma.py`               | op_pta 同名 | `_BUILD` 重指 `pta/build` |
| `prof_pta_matrix.py`         | op_pta 同名 | 逐字复制（同目录导入） |
| `parse_matrix_5d.py` / `parse_corescan_5d.py` / `validate_pta.py` | op_pta 同名 | 逐字复制 |
| `collect_matrix.sh`          | op_pta 同名 | 自定位脚本（`HERE`/`ROOT` 随文件自纠），未改逻辑 |
| `collect_full_matrix.sh` / `collect_full_matrix_5d.sh` / `collect_corescan_5d.sh` | op_pta 同名 | `HERE`/`KERNEL_ASC`/`BUILD`/cmake 目标目录重指 |
| `CMakeLists.txt`             | 算子根 CMakeLists 的 BUILD_PTA_LAUNCH 分支 | 抽成 standalone pta/ 工程 |

**有意跳过（未迁移）**：源 `op_pta/` 下 `*.log` / `*.err`（历史采集/评审输出，如 `collect5d.log`、
`5d_prec_db.log`、`db34.err` 等 ~30 个）与 `__pycache__/`——均为陈旧运行产物，非源码。
