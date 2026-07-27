# Broadcast NDDMA — PTA 直调 + msprof benchmark harness（3D NDDMA 方案，DESIGN §9）

PyTorch 直调（PTA）+ msprof 采集台，从 `ops-direct-invoke` 的 `op_pta/` 结构化迁移而来
（R-054 only-move-not-rewrite：计算逻辑与 benchmark shape 矩阵逐字保留）。

> 三个 NDDMA 算子的 `pta/` 采用**统一脚本命名约定**（见文末《文件清单》）：
> `collect_matrix.sh`=全量四配置扫描、`collect_single.sh`=单配置（须预编 .so）、
> `collect_matrix_5d.sh`=5D 场景全量扫描、`collect_corescan.sh`=核数扫描、
> `parse_*.py`=解析、`validate_pta.py`=精度校验、`prof_pta_matrix.py`=采集驱动（勿直接调用）。

## 与 registry-invoke 主线的关系

- **compute 逻辑**：`../op_kernel/arch35/broadcast_nddma_bench.h`
  （命名空间类 `BroadcastNddmaBenchNs::KernelBroadcastNDDMA<T, ubCapElems>` + `SCENE_AB/BA/BAB/ABA`）。
  本 harness 与主线 registry-invoke 算子**共用同一份 compute 头**，不复制、不改写。
- **本目录 `nddma_broadcast_launch.asc`**：thin benchmark kernel 入口 `BroadcastNDDMABenchKernel`
  （读 8 字段 attr GM → 按 dtype 派发命名空间类，scene 透传 `Init` → `Process`）+ 唯一 C-ABI
  `LaunchNddmaBroadcast5d`（本算子无 `*Host` 变体），签名与源一致。blockNum 由入参控制（核数扫描），
  0 → 缺省 `min(dim0,56)`（`kVectorCoreCnt`）。
- **USE_PINGPONG 开关**随 compute 头迁移；`collect_matrix.sh` / `collect_matrix_5d.sh` /
  `collect_corescan.sh` 的 `set_pingpong()` 直接 sed 该头（DB=true/SB=false）。

## 四场景（DESIGN §9.0）

`scene`：0=AB（dim4 广播）1=BA（dim3 广播）2=BAB（dim2+dim4 广播）3=ABA（≡BA，§9.1）。
输入 shape 随 scene（广播轴长度=1），输出恒 (dim0,dim1,dim2,dim3,dim4)。

## 构建 launch .so（离线需 CANN + ASC 工具链）

```bash
cd pta
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nddma_broadcast_launch -j
# 产物：pta/build/libnddma_broadcast_launch.so（pta_nddma.py 的 _BUILD 指向此处）
```

## 精度校验（需真实 NPU + torch_npu）

```bash
python3 validate_pta.py            # 5D 全集（12 例 fp32 + 4 dtype 变体 RUN + 1 fp16 SKIP 对照）——默认唯一分支
python3 validate_pta.py --5d       # 同上（显式）
```

> ⚠️ **变体一致性（S1）**：`NDDMA_MODE=sb` 只切换 Python 侧容量表，**不改 .so 编译变体**。
> SB 精度复跑**必须**先 `USE_PINGPONG=false` 重编 `build/libnddma_broadcast_launch.so`（并清 torch
> ext 缓存）才正确；直接用 DB 编的 .so 跑 `NDDMA_MODE=sb` 会因 tile>DB halfCap 半区溢出误判 FAIL。

## 性能采集（需真实 NPU + msprof）

```bash
# 全量四配置扫描（DB/SB × isMte3 0/1，自动 toggle USE_PINGPONG + 重编 .so + 清 torch ext 缓存）
bash collect_matrix.sh <matrix_out> [iters]
# 5D 场景全量四配置扫描（NDDMA_SUITE=5D，两判据 AND：tile≤cap 且 dim4 8B 对齐）
bash collect_matrix_5d.sh <matrix_out> [iters]
# 5D 核数扫描（blockNum={1,2,4,8,16,28,56}，单配置）
bash collect_corescan.sh <out_dir> [iters] [mode] [isMte3]
# 单 msprof 配置（须已按 <db|sb> 预编 .so；cfg∈{db_rw,db_r,sb_rw,sb_r}）
bash collect_single.sh <db|sb> <cfg> <isMte3> [iters]
# 解析（注意 parse_matrix.py 参数顺序：iters 在前、matrix_dir 在后）
python3 parse_matrix.py [iters] [matrix_dir]
python3 parse_corescan.py <cfg_dir> <isMte3> [t_nddma_us] [T_max_ms] [iters]
```

> 本机若无 NPU/CANN runtime，以上仅在真实 Ascend 950PR 环境执行；迁移仅做文件搬运 + 路径重指。

## 文件清单（名 → 角色）

| 文件 | 角色 |
|------|------|
| `nddma_broadcast_launch.asc` / `.h` | ASC launch kernel + C-ABI（compute 头在 `../op_kernel/arch35`） |
| `nddma_broadcast_ext.cpp` | pybind11 扩展（`bcast_5d` + `SCENE_*` 常量） |
| `pta_nddma.py` | JIT 构建扩展 + Python API `bcast_5d(...)` |
| `validate_pta.py` | 精度校验（bitwise vs torch broadcast_to；`NDDMA_MODE=db\|sb` 见 S1 警告） |
| `collect_matrix.sh` | **全量四配置扫描**（自动 toggle+重编） |
| `collect_matrix_5d.sh` | 5D 场景全量四配置扫描 |
| `collect_corescan.sh` | 5D 核数扫描（单配置） |
| `collect_single.sh` | 单 msprof 配置（预编 .so，不 toggle） |
| `prof_pta_matrix.py` | 采集驱动（由 collect 经 msprof `--application` 调用，勿直接跑） |
| `parse_matrix.py` | 解析 5D op_summary → `<matrix_dir>/parsed5d.json` + `analysis5d.json` |
| `parse_corescan.py` | 解析核数扫描 → `corescan5d_<cfg>.json` |
| `CMakeLists.txt` | standalone pta/ 工程（`find_package(ASC)` + `project()`） |
