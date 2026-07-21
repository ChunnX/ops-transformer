# FusedFloydAttention A3→A5 待修改点

> 算子：`ops-transformer/attention/fused_floyd_attention`
> 架构：DAV_2201（A3）→ DAV_3510（A5）

## P0 阻塞（不改无法运行）

### 1. 算子定义未注册 ascend950

- **位置**：`op_host/fused_floyd_attention_def.cpp:87-88`
- **现状**：仅注册 `ascend910b`（A2）和 `ascend910_93`（A3），未注册 `ascend950`
- **后果**：框架不会为 A5 生成该算子的编译产物，aclnn 调用直接报不支持
- **改法**：在第 88 行后新增一行，复用同一份 `aicore_config`（动态编译/动态格式/AiCore 等配置均适用）
  ```cpp
  this->AICore().AddConfig("ascend950", aicore_config);
  ```

### 2. Kernel 缺 A5 Cube 分支

- **位置**：`op_kernel/fused_floyd_attention.cpp:22`
- **现状**：用 `#ifdef __DAV_C220_CUBE__` 区分 Cube / Vector 两套实现
- **后果**：A5 编译时 `__DAV_C220_CUBE__` 不定义（A5 对应 `__DAV_C310_CUBE__`），`#ifdef` 为假，Cube 侧 `COPY_TILING_DATA` / `REGIST_MATMUL_OBJ` 的特化逻辑被跳过，直接走 `#else` 的 VECTOR 分支，Cube 计算路径丢失
- **改法**（二选一，取决于 A5 与 A3 的 Cube 侧行为是否一致）：
  - **简单合并**：若 A5 可复用 A3 的 Cube/Vector 分支，改为
    ```cpp
    #if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
    ```
  - **arch35 独立分支**：参考 `flash_attention_score`（`op_kernel/arch35/flash_attention_score_entry_regbase.h:23`），新建 arch35 目录，用 `#elif defined(__DAV_C310_CUBE__)` 单独处理 A5 Cube 侧差异
- **判断依据**：A5 与 A3 在 Cube 侧 tiling data 提取方式、`REGIST_MATMUL_OBJ` 行为是否一致。若一致用合并方案；若 A5 因 L0C 扩容、数据通路变化需不同处理，用 arch35 分支

## 测试

| # | 文件:行 | 修改 |
|---|---|---|
| 1 | `tests/ut/op_host/test_fused_floyd_attention_tiling.cpp:36,56,66,86` | 新增 A5 用例：`SocVersion::ASCEND950`、`aivNum=56/64`、`aicNum=28/32`、`ubSize=253952`、`l0cSize=262144`，重算 `expectTilingKey`/`expectTilingData` |

## 已就绪（无需改）

- `op_host/CMakeLists.txt:18-25` — ascend950 编译分支、`Ascend950PR_9599`、`ENABLE_CV_COMM_VIA_SSBUF` 已预置
- Tiling 侧 `GetPlatformInfo()` — 通过 `PlatformAscendC` 接口运行时获取，已自适应 A5
- 4:2 稀疏取消 — 本算子 `SparseMode` 指 attention mask 压缩模式，非权重稀疏，不受影响

## 架构关键差异速查

| 项 | A3（DAV_2201） | A5（DAV_3510） |
|---|---|---|
| L0C | 128 KB | 256 KB |
| UB | 192 KB | 248 KB |
| BT | 1 KB | 4 KB |
| SSBuffer | — | 256 KB（新增） |
| Cube 宏 | `__DAV_C220_CUBE__` | `__DAV_C310_CUBE__` |
| `L1→GM` / `GM→L0A/L0B` 通路 | 存在 | **已删除** |
| CV 直通（L0C→UB、UB→L1） | 无 | 新增 |
| 同步 | set/wait 强配对 | BufferID |
| 典型 Cube:Vec 核数 | 24:48 | 28:56（PCIE）/ 32:64（Server） |
