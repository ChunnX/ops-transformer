# M-RoPE MX 验证证据索引

> 状态：当前默认 Vector RowBatch16 源码身份的证据索引。本文不是合同、Golden、tiling 或实现的事实源；这些事实分别以 `01_contract.md`、测试 Golden、Host tiling 与 Kernel 源码为准。旧实现结果只保留为历史证据，不得反向覆盖当前结论。

## 0. 当前结论与边界

| 项目 | 当前判定 | 证明边界 |
|---|---|---|
| 当前实现身份 | 默认 Vector key `302059776`；所有合法Q配置使用同一个运行时RowBatch16 VF，K使用自己的逐行融合VF。 | VF SHA256 `d43e6bba...7323`；layout SHA256 `89d11e1d...754b`；4a3 object SHA256 `c419b316...9d2b`。 |
| 干净构建 | CANN 9.1.0、Ascend950 opkernel构建通过，三个kernel对象与实测OPP逐字节一致。 | 构建日志和七项SHA256见第1节。 |
| 当前设备精度 | ACLNN `T=768,Nq=16/24/32/64,D=128` 共4条用例通过，每条五输出均为 `100%`。 | 固定seed 42、Golden开启、`--run=1`；设备 `Ascend950PR_9589`。 |
| 当前正式性能 | 固定seed 42、`--run=50`、`--proc-no-reuse`、`reuse_hbm=false`，四组分别为 `14.474/20.901/21.517/32.669 us`。 | TTK ACLNN task-prof；每条总AiCore时间除以49次计时。 |
| 相对上一版MX | `Nq=16/24/32/64` 分别为 `-3.66%/-1.40%/-6.89%/-14.51%`。 | 使用相同shape、slot mapping、Golden插件和TTK计时口径的已冻结基线。 |
| 相对非MX | 四组分别为 `+5.04%/+44.12%/+31.08%/+35.71%`。 | `10%`是优化方向，不是硬性验收线；当前只有`Nq=16`达到该量级。 |
| 当前未闭环 | `Nq=8/40/48/56`、priority-0全矩阵、最大T、Torch E2E、详细msopprof、wheel与Host/device layout exact-equality UT。 | 四组代表性用例不能替代完整矩阵。 |
| 设计确认门 | D11、D25、D26仍有效。动态scratch使D11需要比较`Nq`对应的所有offset和size。 | 单点精度和性能不会自动消除设计漂移风险。 |

## 1. 当前源码与产物身份

| 实体 | SHA256 / 身份 |
|---|---|
| 隔离源码树 | `/tmp/qkv_mrope_simd_h16_20260814`，基线commit `4efa682af807043e5639ae969b8d0a7fe5799e9f` 上的未提交工作树 |
| RowBatch16 VF `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vf.h` | `d43e6bba1351cfa6248a08f7ffbda63ab241ceec8089bd57c6fec5ae4a497323` |
| UB layout `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h` | `89d11e1d518927eb4e96e06b8ff4b7906013503ec6d23c8b6f97c3d97eec754b` |
| 当前4a3 release object | `c419b316b097e0018a9c9f9ba33d4914cc913481220be57053346b6bf3039d2b` |
| 当前4a3 JSON | `9fa56c6aac62dd85977b896cd24141fe49da31379356a201994081cdb8f66445` |
| 当前70f object / JSON | `fa52f9cd3dfda67400e24dfa6a1c5091fd1354d89d4c493d6ff00c3c19a1f86c` / `50948b1670755b479baf9c9403ef50fe3243e33bb3315be4265470dcf6288ff7` |
| 当前78c object / JSON | `32d2ad913fde153b3e1df11564e16f778922edde795fbad8319eb24513b10060` / `a0ed108eb73b386a25b5a91c90d3ce3ac4d16dece15d00ae12bc1aa94771bd35` |
| 当前Host tiling库 | `675293da08c8afd95acd23c34909c388f97aba89c31663d882d8a05764d9711f` |
| Kernel 名 | `QkvRmsNormRopeCacheWithKScale_4a3c0825b271def4133e00ce232fe74c` |
| tiling key / blockDim | `302059776 (0x12011100)` / `64` |
| 冻结 OPP | `/tmp/qkv_mrope_rowgroup32_20260819/opp_all_full_batches/vendors/custom_transformer` |
| CANN / SoC | CANN 9.1.0 / `Ascend950PR_9589` |
| ops-test-kit | commit `c1bac9c6298a649ec18cc7e02951a6fd234765ee`；ACLNN性能使用内部warmup与 `--run=50` |
| Golden插件 | `qkv_rms_norm_rope_cache_with_k_scale_golden.py`，SHA256 `470b80ebfe3687cac113b078f1b355622463e515ee779c471d54f1f8c94ac66d` |

最终干净构建的三个object、三个JSON和Host tiling库与冻结OPP逐字节一致。构建日志为`/tmp/qkv_mrope_rowgroup32_20260819/candidate_opkernel_clean_final.log`，SHA256为`f9ede020052de8b6c0949e544846129b234c64e8e1af8fbe503f0fcabfa37173`。

## 2. 当前精度、路由与性能

### 2.1 ACLNN 精度

当前代表性shape：

```text
T=768, Nq/Nk/Nv=16/2/2, 24/3/3, 32/4/4, 64/8/8, D=128
qQuantMode=MxQuant, kQuantMode=MxQuant
mropeSection=[22,12,10]
```

四条ACLNN用例均为：

```text
qOut/qScale/kCache/vCache/kScaleCache = 100.0%/100.0%/100.0%/100.0%/100.0%
```

执行条件为固定seed 42、唯一slot mapping、Golden开启、`--run=1`、`--proc-no-reuse`且不启用`reuse_hbm`。证据为：

```text
/tmp/qkv_mrope_rowgroup32_20260819/candidate_precision_result.csv
/tmp/qkv_mrope_rowgroup32_20260819/candidate_precision_run.log
```

文件SHA256分别为`e568f4e003c5b0fd1d617c9d9d999e5c2eebe169f66442fc055d0f43580b3948`和`cf86cc5011126c2e5f6fb8ee34786c5e139e1dcd82a2a3773fa7e8803ac57622`。

### 2.2 产物绑定

当前Vector scene对应：

```text
Kernel Name = QkvRmsNormRopeCacheWithKScale_4a3c0825b271def4133e00ce232fe74c
Tiling Key  = 302059776
BlockDim    = 64
Workspace   = 33554432 bytes
Memory      = OK
```

实测OPP内的4a3 object/JSON和Host tiling库与最终干净构建完全一致。性能轮关闭Golden，`PRECISION_STATUS=PASS`只表示Golden被显式抑制；五输出数值精度只引用2.1的ACLNN结果。

### 2.3 当前正式性能

执行条件为固定seed 42、唯一slot mapping、`--run=50`、`--proc-no-reuse`、不启用`reuse_hbm`、Golden关闭。TTK在单次调用内完成预热，task-prof记录49次AiCore计时：

| Nq/Nk/Nv | AiCore总时间 / 次数 | 当前耗时 | 上一版MX | 当前相对上一版 | 非MX | 当前MX相对非MX |
|---|---:|---:|---:|---:|---:|---:|
| 16/2/2 | `709.25/49` | `14.474 us` | `15.025 us` | `-3.66%` | `13.781 us` | `+5.04%` |
| 24/3/3 | `1024.15/49` | `20.901 us` | `21.197 us` | `-1.40%` | `14.502 us` | `+44.12%` |
| 32/4/4 | `1054.33/49` | `21.517 us` | `23.109 us` | `-6.89%` | `16.416 us` | `+31.08%` |
| 64/8/8 | `1600.76/49` | `32.669 us` | `38.213 us` | `-14.51%` | `24.072 us` | `+35.71%` |

正式结果和日志为：

```text
/tmp/qkv_mrope_rowgroup32_20260819/candidate_all_perf_result.csv
/tmp/qkv_mrope_rowgroup32_20260819/candidate_all_perf_run.log
```

文件SHA256分别为`433569db32ac3b9516d9d628542b916f6cbee85b897eb4456be41ac8fc3bf395`和`a022ddd8c00bd235803c030d6cd05a3c2267cb39bc8f5969e64daac1f773746e`。

### 2.4 测量边界

固定seed只固定输入值，不固定HBM虚拟地址、L2/TLB状态、频率、温度或其他设备负载。本轮只把单次TTK调用的内部预热和重复结果作为正式值，不合并其他invocation，也不使用`reuse_hbm=true`结果。

## 3. 当前 SIMD 性能分解

当前源码的matched结论：

- 基线CANNsim显示，Q每个H16批次都独立执行store→load屏障。每批第一次读取max约等待101 cycles，第一次读取reciprocal约等待49～50 cycles；`Nq=64`每token重复四次。
- PostLiteral候选删除一部分地址scalar指令后，`Nq=64`从相邻基线`38.244 us`变为`38.316 us`，慢`0.19%`，说明这些scalar指令不在主关键路径。
- 当前实现先生成同一token的全部完整H16批次，再统一执行scale和data阶段，把三次屏障从“每批一次”改为“每token一次”。`Nq=64`相对上一版MX降低`14.51%`，与屏障固定成本被四批共享的机制一致。
- `Nq=16`只有一个完整批次，收益`3.66%`；`Nq=24`还有8-row逐行尾批，收益只有`1.40%`。这说明下一项高价值工作应分解Q尾批和K逐行量化，而不是继续删Q完整批次中的地址scalar。
- 当前MX相对非MX在`Nq=24/32/64`仍高`44.12%/31.08%/35.71%`，尚未达到约10%的方向目标；不能把本轮Q完整批次收益解释为全部MX成本已经消除。

完整探针、恢复hash与解释见：

```text
/tmp/qkv_mrope_rowgroup32_20260819
/tmp/qkv_mrope_simd_opt_20260818/cannsim_addrreg_t65/record_out_allvf_bound/
  cannsim_20260818185939_mrope_mx_k_rowbatch16_t65_h64_8_kernel.py/report/trace_reports/trace_core0.json
```

本轮没有改变数值公式、E8M0特殊值处理、FP32数据暂存类型或FP8转换方式。收益来自阶段组织和屏障摊销。

## 4. 历史 SIMT/SIMT-VK 证据

以下结果属于当前RowBatch16 Vector之前的独立实现身份：

| 历史实现 | 结果 | 当前用途 |
|---|---:|---|
| 原始SIMT | `81.05 us` | 旧调度基线 |
| 优化SIMT | `60.44 us` | 证明旧SIMT优化收益，不代表当前Vector |
| 同包SIMT | `53.87 us` | 旧内部A/B控制 |
| 同包SIMT-VK | `52.50 us` | 相对旧SIMT约 `-2.55%` |
| 更早Vector/SIMT | `87.11/69.36 us` | 更早源码身份，仅历史参考 |

历史SIMT曾完成priority-0 `9/9 PASS`和 `T=262144` `1/1 PASS`；SIMT-VK完成目标shape精度与路由。这些证据仍可用于各自实现的回溯，但不能证明当前RowBatch16 Vector的全矩阵、最大T、Torch或详细pipeline。

历史详细身份与路径保留在：

```text
/tmp/qkv_mrope_mx_simt_opt12_vpair_kpack_20260814
/tmp/qkv_mrope_mx_simt_vk_isolated_20260814
```

## 5. 证据失效规则与下一步

以下任一变化都要求生成新身份并重跑受影响checkpoint：公开合同或Golden、tiling key/TilingData ABI、Host selector、UB layout、RowBatch16阶段组织、Kernel数值实现、ACLNN ABI、Torch schema/bridge、package object。

当前下一步按价值排序：

1. 分别测量Q的8-row尾批和K逐行量化，确认`Nq=24`剩余差距的主要归属，再决定下一项SIMD改动；
2. 为`MropeMxUbLayout`增加Host/device逐字段exact-equality UT，覆盖八组`Nq`对应的动态scratch；
3. 用当前RowBatch16源码身份补`Nq=8/40/48/56`、priority-0全矩阵和最大T；
4. 补Torch eager E2E、clean package/wheel身份；只有在需要阶段级stall证据时，再采当前object的详细msopprof。

交付时继续分别报告BUILD、LAUNCH、VERIFY、repeat stability与PERF；不得用build替代设备结果，不得用旧SIMT/SIMT-VK或 `reuse_hbm=true` 数据包装成当前正式性能。
