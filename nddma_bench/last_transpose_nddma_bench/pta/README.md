# LastTranspose NDDMA — PTA 直调 + msprof benchmark harness

PyTorch 直调（PTA）+ msprof 采集台，从 `ops-direct-invoke` 的 `op_pta/` 结构化迁移而来
（R-054 only-move-not-rewrite：计算逻辑与 benchmark shape 矩阵逐字保留）。

## 与 registry-invoke 主线的关系

- **compute 逻辑**：`../op_kernel/arch35/last_transpose_nddma_bench.h`
  （命名空间类 `LastTransposeNddmaBenchNs::KernelLastTransposeNDDMA<T, ubCapElems>`）。
  本 harness 与主线 registry-invoke 算子**共用同一份 compute 头**，不复制、不改写。
- **本目录 launch.asc**：定义 thin benchmark kernel 入口 `LastTransposeNDDMABenchKernel`
  （读 attr GM → 按 dtype 派发到命名空间类 → `Init`/`Process`）+ 两个 C-ABI 函数
  （`LaunchNddmaTranspose` / `LaunchNddmaTransposeHost`），签名与源保持一致。
- **USE_PINGPONG 开关**随 compute 头迁移；`collect_*.sh` 的 `set_pingpong()` 直接 sed 该头。

## 构建 launch .so（离线需 CANN + ASC 工具链）

```bash
cd pta
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nddma_transpose_launch -j
# 产物：pta/build/libnddma_transpose_launch.so
```

`pta_nddma.py` 的 `_BUILD` 指向 `pta/build`，JIT 构建 pybind 扩展时按 C ABI 链接此 .so。

## 运行（需真实 NPU + torch_npu + msprof）

```bash
# 单点/带宽快测
python3 prof_pta.py
python3 prof_pta_bw.py
# 全量矩阵（DB/SB × isMte3 0/1，FP16+FP32）
bash collect_matrix.sh <op_root> <matrix_out> [iters]
# B8-only 矩阵
bash collect_b8_matrix.sh <op_root> <matrix_out> [iters]
# 解析
python3 parse_matrix.py <matrix_out>
python3 parse_b8_matrix.py <matrix_out>
# 精度校验
python3 validate_pta.py
```

> 本机无 NPU/CANN runtime，以上运行步骤仅在真实 Ascend 950PR 环境执行；
> 迁移仅做文件搬运 + 路径重指，不产出性能数据。

## 文件清单

| 文件 | 来源 | 迁移动作 |
|------|------|----------|
| `nddma_transpose_launch.asc` | op_pta 同名 | include 重指 arch35 头 + 本地 attr/entry + C-ABI 逐字保留 |
| `nddma_transpose_launch.h`   | op_pta 同名 | 逐字复制 |
| `nddma_transpose_ext.cpp`    | op_pta 同名 | 逐字复制 |
| `pta_nddma.py`               | op_pta 同名 | `_BUILD` 重指 `pta/build` |
| `prof_pta.py` / `prof_pta_bw.py` / `prof_pta_matrix.py` | op_pta 同名 | 逐字复制（同目录导入） |
| `parse_matrix.py` / `parse_b8_matrix.py` / `validate_pta.py` | op_pta 同名 | 逐字复制 |
| `collect_matrix.sh` / `collect_b8_matrix.sh` | op_pta 同名 | `HERE`/`KERNEL_ASC`/`BUILD`/cmake 目标目录重指 |
| `CMakeLists.txt`             | 算子根 CMakeLists 的 BUILD_PTA_LAUNCH 分支 | 抽成 standalone pta/ 工程 |
