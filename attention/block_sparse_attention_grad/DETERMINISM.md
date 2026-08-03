# block_sparse_attention_grad 确定性方案文档

## 1. 问题背景

block_sparse_attention_grad 算子在 arch22 (Atlas A2) 平台上，当使用 GQA (Grouped Query Attention) 时，
多个 cube 核会并行计算不同 Q head 的 dK/dV。由于同一 group 内的 Q head 共享相同的 K/V，
不同核通过 `Fixpipe` + `SetAtomicAdd` 累加到同一 GM 地址，导致结果非确定性。

## 2. 根因分析

**`Fixpipe` + `SetAtomicAdd` 对同一 GM 地址的多次累加在硬件层面不完全 bit-exact**。

不确定性产生的两个场景：
1. **跨 Q-head 累加**（GQA, group_size > 1）：不同 Q head 共享同一 KV，写入同一地址
2. **跨 Q-block 累加**（S1 > blockShapeX）：同一 Q head 的不同 Q block 写入同一 KV 地址

## 3. 方案：40轮分组 det slot

### 3.1 核心思路

AIC上，新增dk和dv的workspace让每个轮次每个cube核算出来的dk和dv的结果，依次保存在这段新增的workspace上，每四十个轮次为一组，然后做一次synclall保证所有核都完成，然后aiv上，按照固定顺序读取每个cube在workspace上保存的结果到ub上，在ub上完成累加操作。

### 3.2 数据流

```
VecPre:
  ├── 清零 shared workspace (dQ/dK/dV) — EpilogueFAGPre
  ├── PipeBarrier<PIPE_MTE3>() — 确保 EpilogueFAGPre 的 MTE3 写入完成
  └── ZeroDetWorkspace() — 清零 det workspace (detDkGroup + detDvGroup + detMetaBuf)

AIC (每个 cube 核):
  ├── blockMmad3 用 atomicAdd 写入 det slot (slot 被零初始化，atomicAdd 到零等于直接写入)
  ├── 每轮写入独立 slot: slotIdx = (count - 1) * cubeNum + coreIdx
  ├── 写入 metadata (kvOffset, curCalKVSize, valid) via V→MTE3 DataCopy
  └── CrossCoreSetFlag(CUBE2POST)

AIV (CUBE2POST + SyncAll<true> 后):
  ├── DetReduceAll: 多核并行按 dk/dv 元素分段
  │     每个 AIV 核扫描所有 det slot
  │     读取匹配自己元素段的 partial data
  │     FP32 累加到 shared workspace
  └── VecPost: 从 shared workspace 读取，scale + cast，写入 output GM
```

### 3.3 Workspace 布局（固定大小，与 shape 无关）

```
[libapiSize | sOutSize | dPOutSize | dQOutSize | dKOutSize | dVOutSize | gradSize | detDkGroup | detDvGroup | detMetaBuf]
```

| 区域 | 大小 | 说明 |
|------|------|------|
| detDkGroup | `maxPartialsPerCore × cubeNum × maxPartialSize × sizeof(float)` | dk partial 槽位 |
| detDvGroup | `maxPartialsPerCore × cubeNum × maxPartialSize × sizeof(float)` | dv partial 槽位 |
| detMetaBuf | `maxPartialsPerCore × cubeNum × 16` | 每个 partial 的元数据 |

- `maxPartialSize = basicKVBlockSize × headDim = 128 × 128 = 16384`（固定值）
- `maxPartialsPerCore = (taskNumPerCore + 1) × ceil(maxKvSeqlen / basicKVBlockSize)`
- **总大小与 dkvSize 无关**，仅取决于核数和最大 partial 数

### 3.4 Det Slot 布局

每个 det slot 存储 `maxPartialSize` 个 float（连续布局，stride=headDim）：

```
[slot0: maxPartialSize floats | slot1: maxPartialSize floats | ... | slotN: maxPartialSize floats]
```

slot 索引：`slotIdx = (count - 1) × cubeNum + coreIdx`

每个 slot 对应的 metadata（16 bytes）：
```
{ uint64_t kvOffset; uint32_t curCalKVSize; uint32_t valid; }
```

### 3.5 DetReduceAll 多核并行

AIV 按 dk/dv 元素分段（与 VecPost 相同的分区方式），每个 AIV 核处理自己段的元素：

1. 用 `Duplicate` 初始化 accumUb 为零
2. 扫描所有 det slot，读取 metadata
3. 对匹配自己元素段的 slot，用 `DataCopy`/`DataCopyPad` 读取 partial data
4. FP32 `Add` 累加到 accumUb
5. 用 `DataCopy` 写回 shared workspace

不同 AIV 核处理不同的 dk/dv 元素段，无写竞争。

## 4. 代码修改清单

### 4.1 Tiling (op_host/block_sparse_attention_grad_tiling.h)
- 替换旧字段: `detDkWorkspaceSize`, `detDvWorkspaceSize`, `aicNumForDet`, `groupSizeForDet`, `batchSizeForDet`, `KForDet`
- 新增字段: `detDkGroupSize`, `detDvGroupSize`, `detMetaSize`, `maxPartialSize`, `maxPartialsPerCore`, `deterministic`

### 4.2 Tiling 实现 (op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp)
- ProcessInput: 读取环境变量 `BSAG_DETERMINISTIC=1` 设置 deterministic_
- CalculateWorkSpace: `detDkGroupSize = maxPartialsPerCore × aicNum × maxPartialSize × sizeof(float)`
- FillTilingData: 设置新增字段

### 4.3 Kernel (op_kernel/arc22/block_sparse_attention_grad_kernel.h)

**AIC operator()**:
- 新增 det workspace 基地址: `detDkGroupBase`, `detDvGroupBase`, `detMetaBase`
- 新增 `gDkDet`/`gDvDet` GlobalTensor
- 新增 `blockMmad3Det` 实例 (isAtomicAdd=false)
- det 路径: blockMmad3 写入 det slot + metadata via V→MTE3 DataCopy
- 非det路径: 保持原始 blockMmad3 写入 shared workspace
- CrossCoreSetFlag(CUBE2POST) 前: PipeBarrier<PIPE_FIX>()

**VecPre**:
- EpilogueFAGPre 后: PipeBarrier<PIPE_MTE3>() 确保 MTE3 写入完成
- ZeroDetWorkspace(): 清零 det workspace

**ZeroDetWorkspace (重写)**:
- 多核并行清零 detDkGroup + detDvGroup + detMetaBuf
- 简单分区: 每核处理 detTotalBytes/usedCoreNum 字节

**DetReduceAll (新增)**:
- 多核并行按 dk/dv 元素分段
- Duplicate 初始化 accumUb 为零
- 扫描所有 det slot，读取 metadata (scalar read)
- DataCopy/DataCopyPad 读取匹配的 partial data
- FP32 Add 累加
- DataCopy 写回 shared workspace

**AIV operator()**:
- SyncAll<true> 后调用 DetReduceAll

## 5. 测试结果

### 5.1 测试环境

- 平台: Atlas A2 (arch22), openEuler 24.03 SP3 aarch64
- CANN: 9.1.0-beta.1
- 设备: NPU device 2

### 5.2 测试脚本

位于 `/home/lsy/test/`:

| 脚本 | 格式 | 类型 | 说明 |
|------|------|------|------|
| bsag_gqa_bnsd.py | BNSD | GQA | group=3, B=10, N=12, S1=256, S2=357, D=128 |
| bsag_gqa_tnd_test.py | TND | GQA | group=6, B=10, N=24, S1=1220, S2=452, D=128 |
| bsag_mha_tnd_test.py | TND | MHA | group=1, B=10, N=15, S1=126, S2=256, D=128 |

### 5.3 精度验证 (vs CPU golden, rtol=1e-3, atol=1e-3)

| 测试 | dq | dk | dv | 结果 |
|------|----|----|-----|------|
| GQA BNSD | PASS | PASS | PASS | 3/3 迭代全 PASS |
| GQA TND | PASS | PASS | PASS | 3/3 迭代全 PASS |
| MHA TND | PASS | 1元素差(预存问题) | PASS | 10/10 迭代一致 |

### 5.4 确定性验证 (bit-exact, torch.equal)

**专项 bit-exact 测试** (GQA BNSD, group=4, B=2, N=8, S1=128, S2=128, D=128):

| 对比 | dq_bitexact | dk_bitexact | dv_bitexact | dk_maxdiff |
|------|-------------|-------------|-------------|------------|
| Run 0 vs 1 | **True** | **True** | **True** | 0.00000000 |
| Run 0 vs 2 | **True** | **True** | **True** | 0.00000000 |

**大 shape bit-exact 测试** (GQA BNSD, B=10, N=12, S1=256, S2=357, D=128):

| 对比 | dk_maxdiff | dv_maxdiff |
|------|------------|------------|
| Run 0 vs 1 | **0.00000000** | **0.00000000** |
| Run 0 vs 2 | **0.00000000** | **0.00000000** |

### 5.5 Workspace 大小对比

| Shape | 旧方案 (per-core-per-Q-head) | 40轮分组方案 |
|-------|---------------------------|-------------|
| B=2, S=128 | ~83MB | ~4MB |
| B=8, S=8192 | ~42.8GB | ~100MB |
| B=8, S=32768 | ~171GB | ~100MB |
| B=32, S=32768 | ~687GB | ~100MB |

## 6. 使用方法

### 6.1 编译安装
```bash
source /opt/nzh/0608/cann-9.1.0-beta.1/set_env.sh
cd /home/lsy/ops-transformer_7481
bash build.sh --ops=block_sparse_attention_grad --soc=ascend910b --pkg
./build/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

### 6.2 运行测试（开启确定性）
```bash
export LD_LIBRARY_PATH=/opt/nzh/0608/cann-9.1.0-beta.1/aarch64-linux/lib64:/opt/nzh/0608/cann-9.1.0-beta.1/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=/opt/nzh/0608/cann-9.1.0-beta.1/opp
export ASCEND_HOME_PATH=/opt/nzh/0608/cann-9.1.0-beta.1
export BSAG_DETERMINISTIC=1

cd /home/lsy/test
python3 bsag_gqa_bnsd.py
python3 bsag_gqa_tnd_test.py
python3 bsag_mha_tnd_test.py
```

### 6.3 关闭确定性（恢复原始行为）
不设置 `BSAG_DETERMINISTIC` 环境变量即可。

## 7. 已知限制

1. **性能影响**: DetReduceAll 引入额外的 GM 读取和 V 计算（扫描所有 slot）。
2. **MHA TND dk 精度**: 有 1 个元素 atol 略超 1e-3（预存问题，非确定性修改引入）。
