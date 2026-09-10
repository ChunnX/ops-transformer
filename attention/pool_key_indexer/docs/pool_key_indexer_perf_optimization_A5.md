# pool_key_indexer A5(950) 性能优化报告

- 日期：2026-09-10
- 范围：arch35（Ascend 950PR/DT）vector 侧 kernel；arch22（A2/A3）未改动
- 模型侧基线：`A5_kernel.csv`（PoolKeyIndexer，MIX_AIC，Duration 1188.686us，aiv_scalar_ratio 92.4%）

## 1. 基线复现（要求 1）

### 1.1 模型场景还原

由 CSV 输入输出推导参数（`2051 = topk + pool_size - 1`，且 2048 的因子必为 2 的幂）：

| 参数 | 值 |
|---|---|
| query | (1, 4096, 32, 128) BF16 BSND |
| pool_key | (1, 1024, 1, 128) BF16 BSND（1024 池） |
| weights | (1, 4096, 32) BF16 |
| pool_tail_k | (1,) INT64 |
| topk / pool_size | 2048 / 4（sparseCount=512 池） |
| 输出 | (1, 4096, 2051) INT32 + FLOAT |
| mask_mode | 3（因果；mask0 亦复现，耗时相近） |

复现脚本：`tests/pytest/perf_repro_baseline.py`（torch.npu.Event 计时）。
本机 NPU 为 950PR（aicNum=28，模型侧为 32），核数差异导致绝对值略高，pipe 形态完全一致。

### 1.2 复现数据（msprof，单核均值口径）

| 指标 | CSV 基线 | 本地复现（优化前） |
|---|---|---|
| Duration | 1188.7us | **1349.6us**（wall 1364us） |
| Block Num | 32 | 28 |
| aiv_scalar_time / ratio | 1097.6us / 92.4% | **1256.3us / 93.1%** |
| aiv_vec_time / ratio | 87.8us / 7.4% | 98.7us / 7.3% |
| aic_mac_time | 66.1us | 74.1us |

瓶颈画像与模型侧一致：**AIV 标量管道占 93%**，AIC 侧（mac 66~77us）大量时间在等 AIV。

## 2. 根因分析

`ExpandAndAppendIndices`（arch35 `pool_key_indexer_service_vector_arch35.h`）将 TopK 选出的
512 个池索引展开为 2048 个 token 索引。**pool_size=4 非 8 的倍数**时，向量写偏移
（`k*poolSize` 等）不满足 32B 操作数对齐（该约束即原代码回退标量的根因），于是走全标量路径：

```cpp
for (k = 0; k < expandRounds; k++) {          // 512 轮/行
    base = poolIndices.GetValue(k) * poolSize; // 标量读 UB
    for (p = 0; p < poolSize; p++)
        tokenIndices.SetValue(off + p, base + p); // 标量写 UB ×4
}
```

- 每查询行：512 次 GetValue + 2048 次 SetValue ≈ 2560 次标量 UB 访问 + VToSSync/SToMTE3Sync 硬同步
- 每 AIV（64 行）：约 16 万次标量访问，实测 ~6.7ns/次 ≈ **1256us**，与 aiv_scalar_time 完全吻合
- pool_size 为 8 倍数（如 16）时走 Duplicate+Add 向量路径，但每轮仍有 1 次 GetValue + 2 次小向量
  + 1 次 PipeBarrier，同样低效

即：该算子从 lightning_indexer 演进时新增的"尾部块追加与展开"落在标量路径，成为瓶颈。

## 3. 优化实现（要求 2，已落地）

### 3.1 方案：64-lane 寄存器 gather 向量化展开

展开的数学形式：`out[m] = ps * idx[m/ps] + (m % ps)`。对 2 的幂 pool_size，利用
`(chunkBase + r)/ps == chunkBase/ps + r/ps`（ps|64 时整除成立）将 64-lane 输出窗口的
gather 索引分解为"常驻模板 + 标量基址"，实现每 64 个输出仅 5 条向量指令：

```
每组: LoadAlign 64 个池索引进寄存器        （每 64 池一次）
每个 64-lane 输出 chunk:
    Adds(gatherIdx, rIdxTpl, j*64/ps)      // r/ps 模板 + 组内基址
    Gather(gathered, poolReg, gatherIdx)   // vselr 寄存器选择，单周期级
    Mul (gathered, gathered, psReg)        // ×pool_size
    Add (gathered, gathered, rOffTpl)      // + r%ps 模板
    StoreAlign(out + c*64, gathered)
```

- `rIdx[r]=r/ps`、`rOff[r]=r%ps` 两个 64 元素模板在 InitBuffers 一次构建常驻（128 次标量写/kernel，
  替代原每行 2560 次）
- gather 表直接复用 TopK 输出的 `indicesOutLocal_`，零额外 UB；模板存于 workLocal_ 头部 512B
- 对齐安全：ps=2/4 时 `validExpand`/`topk` 仅 4/2 对齐，-1 填充改用 ProcessTopK 同款
  **masked Duplicate**（头块 64 元素 + 对齐续刷），规避 32B 对齐异常
- 尾块垃圾覆盖：gather 尾 chunk lane 的越界值由 `[validExpand, topk)` 的 -1 填充与尾区恒清空覆盖，
  暴露面上界分析见代码注释

### 3.2 路径分派（ExpandAndAppendIndices）

| pool_size | 路径 |
|---|---|
| 2/4/8/16/32/64（2 的幂且 ≤64） | **新 gather 向量路径** |
| 128 | 原 Duplicate+Add 对齐路径（64-lane 不跨满 pool 窗口需逐 chunk 偏移修正，多一条 Adds 实测触发寄存器溢出致 aiv_vec 翻倍，且 sc≤64 轮次开销可接受） |
| 8 的倍数（非 2 幂，如 24/40/48） | 原 Duplicate+Add 对齐路径（offsetTpl 移至 workLocal_[128]） |
| 其它（非 8 倍数且非 2 幂，如 3/12） | 原标量回退路径（罕见配置） |

另修复一处存量问题：workLocal_ 第二段 `Align(outputLen+64,8)` slack（最大 ~33KB）无任何消费方，
已删除——sc=1024（topk=8192）场景 UB 贴近上限，冗余分配即触发 aicore exception。

### 3.3 实测演进（模型场景，本地）

| 版本 | 耗时 | 说明 |
|---|---|---|
| 基线（标量展开） | 1364us | aiv_scalar 1256us/93.1% |
| +vgather2（UB 表 gather） | 776us | vgather2 单次 ~600 cycle，仍慢 |
| +vselr（寄存器 gather，最终） | **608.7us** | aiv_scalar 87us/14.7%，aiv_vec 523us/88.2% |

msprof 终态：Duration 1349.6 → **592.3us（2.28x）**；aiv_scalar 1256.3 → 87.4us。
换算模型侧（32 核）：预期 1188.7 → **约 530~560us（约 2.1~2.2x）**。

隔离实验证实展开开销已近消除：ps=1（无展开路径）地板 537.6us vs ps=4 的 608.7us，
展开净开销 ~70us（原 ~800us）；ps=16 从原路径同获收益（569us）。

### 3.4 回归验证

全量 362 用例通过（run.sh 分套件执行）：
smoke 5 + precision 133 + pabs 12 + ge 68 + aclgraph 138 + aclgraph_pa 6。
另抽查 topk=8192×ps=8/16、topk=2040×ps=12（非 2 幂回退）均正常。
（注：`test_zz_aclgraph_coverage_summary` 在多文件合跑单 pytest 进程时失败为存量调用顺序问题，
旧内核同样复现，与本改动无关。）

## 4. 后续优化方向（剩余瓶颈：Vec1 的 V-pipe 带宽）

优化后瓶颈转移为 `MulWeightAndReduceSum`（aiv_vec 523us/88%）：每 (s1 行, s2 块) 按 gSize=32
循环做 ReLU+加权累加，全量 536MB fp32 qk 中间数据流经 V-pipe，属结构性带宽瓶颈：

1. **权重广播降指令（低风险，预期 ~8-10%）**：非量化路径逐 g 用
   `Duplicate(brcGatherIndex,i)+Gather` 两条指令广播权重（量化路径已是 `BroadcastLane` 单指令）。
   将 weights 预 cast 成 fp32 UB 后统一改 BroadcastLane，每调用 256 条指令可省 ~26 条。
2. **Fixpipe 输出降精度（中风险，预期再 ~30%）**：mm1 结果由 fp32 Fixpipe 改 bf16/fp16 输出，
   Vec1 读带宽减半。需评估分数精度（topk 可比性、returnValue 数值），涉及 cube 侧改动。
3. **跨核 CV 同步合并（需设计）**：每 (gS1 块, s2 块) 一次 CrossCore Set/Wait 乒乓，
   AIC 侧 ~180us 等待中或有可压缩空间。
4. 长期：TopK 直方图与展开的融合（选池后直接流式展开，省 indicesOutLocal 中转）。

## 5. 变更清单

| 文件 | 变更 |
|---|---|
| `op_kernel/arch35/vf/pool_key_indexer_vf_expand.h` | 新增：vselr 向量化展开 helper |
| `op_kernel/arch35/pool_key_indexer_service_vector_arch35.h` | InitBuffers：模板构建 + workBuf 布局/瘦身；ExpandAndAppendIndices：pow2≤64 gather 路径 + masked Duplicate 对齐安全填充 |
| `tests/pytest/perf_repro_baseline.py` | 新增：模型场景基线复现脚本 |

构建安装：`bash build.sh --pkg --ops=pool_key_indexer --soc=ascend950 -j32`，安装 `.run` 后
将 `vendors/custom_transformer` 的 kernel/config 同步至 `opp/vendors/custom_transformer`。
