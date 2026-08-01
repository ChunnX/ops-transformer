# NLastTranspose [0,1,3,2,4] NDDMA — PTA 直调 + msprof benchmark harness

PyTorch 直调（PTA）+ msprof 采集台，从 `ops-direct-invoke` 的 `op_pta/` 结构化迁移而来
（R-054 only-move-not-rewrite：计算逻辑与 benchmark shape 矩阵逐字保留）。

> 三个 NDDMA 算子的 `pta/` 采用**统一脚本命名约定**（见文末《文件清单》）：
> `collect_single.sh`=单配置（须预编 .so）、`collect_b8.sh`=int8 全量四配置扫描（自动 toggle+重编）、
> `parse_*.py`=解析、`validate_pta.py`=精度校验、`prof_pta_matrix.py`=采集驱动（勿直接调用）。
>
> **已知 gap（不臆造新脚本）**：本算子无 FP16+FP32 全 dtype 的全量四配置扫描脚本。
> 全量 FP 采集须**循环 `collect_single.sh` ×4 配置**（db/sb 之间手动 toggle + 重编 .so）；
> int8 全量扫描走自动化的 `collect_b8.sh`。

## 与 registry-invoke 主线的关系

- **compute 逻辑**：`../op_kernel/arch35/nlast_transpose_nddma_bench.h`
  （命名空间类 `NlastTransposeNddmaBenchNs::KernelNLastTransposeNDDMA<T, ubCapElems>`）。
  本 harness 与主线 registry-invoke 算子**共用同一份 compute 头**，不复制、不改写。
- **本目录 `nlast_transpose_launch.asc`**：thin benchmark kernel 入口 `NLastTransposeNDDMABenchKernel`
  （读 attr GM → 按 dtype 派发命名空间类 → `Init`/`Process`）+ C-ABI（`LaunchNlastTranspose` /
  `LaunchNlastTransposeHost`），签名与源一致（含 dim4）。blockDim 按物理 vector core 数 56 发射
  （`kVectorCoreCnt`）。
- **USE_PINGPONG 开关**随 compute 头迁移；`collect_b8.sh` 的 `set_pingpong()` 直接 sed 该头
  （DB=true/SB=false）。

## 构建 launch .so（离线需 CANN + ASC 工具链）

```bash
cd pta
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nlast_transpose_launch -j
# 产物：pta/build/libnlast_transpose_launch.so（pta_nddma.py 的 _BUILD 指向此处）
```

## 精度校验（需真实 NPU + torch_npu）

```bash
python3 validate_pta.py                              # 默认精选 6 例（FP16/FP32，对齐/非对齐、A大B小/A小B大）
python3 validate_pta.py --all                        # 全量 NL_0~47 + B8，bitwise 比对 x.permute(0,1,3,2,4)
python3 validate_pta.py 56 32 128 8 16 0             # 单 case: dim0 dim1 dim2 dim3 dim4 dtype(0=fp16/1=fp32/2=int8)
NDDMA_MODE=sb python3 validate_pta.py --all          # SB 变体（须先按 SB 重编 .so，容量表随之切换）
```

## 性能采集（需真实 NPU + msprof）

```bash
# 单 msprof 配置（须已按 <db|sb> 预编 .so；cfg∈{db_rw,db_r,sb_rw,sb_r}）
bash collect_single.sh <db|sb> <cfg> <isMte3> [iters]
# int8-only 全量四配置扫描（DB/SB × isMte3 0/1，自动 toggle USE_PINGPONG + 重编 .so + 清 torch ext 缓存）
bash collect_b8.sh <op_root> <matrix_out> [iters]
# 解析（每 <matrix_dir> 生成 parsed.json）
python3 parse_matrix.py <matrix_dir> [iters]
python3 parse_b8.py     <matrix_dir> [iters]
```

> 本机若无 NPU/CANN runtime，以上仅在真实 Ascend 950PR 环境执行；迁移仅做文件搬运 + 路径重指。

## 文件清单（名 → 角色）

| 文件 | 角色 |
|------|------|
| `nlast_transpose_launch.asc` / `.h` | ASC launch kernel + C-ABI（compute 头在 `../op_kernel/arch35`） |
| `nlast_transpose_ext.cpp` | pybind11 扩展（`nlast_transpose` / `nlast_transpose_host`） |
| `pta_nddma.py` | JIT 构建扩展 + Python API `nlast_transpose(x, is_mte3)` |
| `validate_pta.py` | 精度校验（bitwise vs torch permute[0,1,3,2,4]；`NDDMA_MODE=db\|sb`） |
| `collect_single.sh` | 单 msprof 配置（预编 .so，不 toggle） |
| `collect_b8.sh` | **int8-only 全量四配置扫描**（自动 toggle+重编） |
| `prof_pta_matrix.py` | 采集驱动（由 collect 经 msprof `--application` 调用，勿直接跑） |
| `parse_matrix.py` / `parse_b8.py` | 解析 op_summary → `<matrix_dir>/parsed.json` |
| `CMakeLists.txt` | standalone pta/ 工程（`find_package(ASC)` + `project()`） |
