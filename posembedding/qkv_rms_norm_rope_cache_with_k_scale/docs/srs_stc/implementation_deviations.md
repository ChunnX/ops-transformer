# M-RoPE MX 当前实现与验收差距

本文只记录当前源码尚未关闭的验收项。已经完成的 V1 映射和测试生成器迁移不再列为设计偏差。

## 当前事实

- ACLNN 直接修改 V1 workspace 接口并增加 `kQuantMode`；当前 1200 条 ACLNN 正例全部调用 V1。
- Host 为 `Nq=8,16,...,64` 固定生成公开 key `302059776`；Kernel 从 `qHeadNum` 选择 `ROW_BATCH16`、`TAIL_TILE_WAVE` 或 `ALL_Q_TILE_WAVE`。
- K 在 `Nk=2/4/8` 时分别按 8/4/2 个 token 组成 16 行批次；`Nk=1/3/5/6/7` 和批次余数进入通用路径。
- 单一生成器按当前源码生成 200/650/50/250/50 五类文件，总数固定为 1200。
- 2026-08-23 的 Host tiling UT 为 134/134 PASS；生成器和 Golden 离线检查通过。

## 尚未关闭的项目

| ID | 目标 | 当前事实 | 关闭条件 |
|---|---|---|---|
| DEV-005 | 八组合法 head 在当前 package 上完成五输出设备精度 | 生成矩阵和 Host 路由已经覆盖八组；本轮未执行真实 NPU | 使用同一 source/package identity 跑完对应 ACLNN、Kernel 和 Torch 正例，逐 case 保存五输出结果 |
| DEV-008 | 单一公开 Q key 下的三类内部 schedule、K 四类路由、偶数完整批和奇数/批次余数都有设备证据 | L0 与 Kernel 都含 16 个显式白盒锚点；目前只有离线生成和 Host 断言 | 执行 16 个锚点，确认实际 tiling key 都是 `302059776`，并核对 blockDim、五输出和 cache 未命中区域 |
| DEV-010 | 新数值边界在设备上符合 Golden | Golden 已增加 `scale_edge_qk` 和 `max_finite_qk`，并保留 zero/tiny/saturation/NaN/Inf | 在当前 package 上分别执行这些 profile；E8M0 用原始字节比较，NaN/Inf 单独报告 |
| DEV-011 | runner 保留 padded view 的物理 allocation、offset 和 stride | 生成集覆盖 data/scale 的 token/head padding、mixed stride 和 nonzero offset；生成器只能证明参数存在 | 用已修复的物理 storage runner 执行相关 ACLNN/Torch case，确认未命中 cache 字节不变 |
| DEV-012 | 当前 V1 身份完成全部 1200 条分层验收 | 历史 1200/1200 来自合并 V2 诊断身份，不能迁移到当前 V1 | 保存 current source/package/OPP/runner identity，依次完成 P0、P1、P2；分别统计 PASS、FAIL、FAILURE |
| DEV-013 | 为普通 RoPE 的 zero、NaN 和 Inf 定义独立数值合同 | 2026-08-24 的 Kernel 修复集剩余 14 条数值 FAIL：`nan_qk=7`、`inf_qk=6`、`zero_qk=1`。设备原始字节只能说明当前实现行为，不能用来修改 Golden 模拟 DAV3510 cast | 先在公开合同中定义支持域和输出语义，再由独立 Golden 实现该语义；合同要求支持时修复 Kernel，合同不要求支持时把用例改成负例或健壮性测试 |
| DEV-014 | Torch functional 支持带物理 padding 的 E8M0 `k_scale_cache` | functional C++ wrapper 在调用 ACLNN 前依次复制三份 cache；非连续 E4M3 K/V copy 已完成，`k_scale_cache.clone()` 触发 `aclnnInplaceCopy/ViewCopy` 561103，错误明确为不支持 `DT_FLOAT8_E8M0` | functional bridge 使用支持 E8M0 的复制方案，保持逻辑 shape、stride、offset 和 backing padding；11 条 padded-head、padded-token、mixed-stride 定向用例全部通过 |

## 2026-08-24 未决问题

### 普通 RoPE 特殊数值不能用设备行为修正 Golden

本轮先修复用例生成、输出绑定、runner restore/capture 和环境失败。合并结果中，Kernel 共有 250 条：236 条 PASS，14 条 FAIL，0 条 FAILURE。14 条 FAIL 全部属于普通 RoPE 特殊数值：7 条 NaN、6 条 Inf、1 条全零 Q/K。

Golden 负责计算独立的算子语义。设备 dump 负责说明当前 Kernel 实际写出了什么。两者不同时，测试应报告实现偏差，不能因为 DAV3510 的 `SAT cast` 写出某个字节，就在 Golden 中加入同样的硬件分支。只有公开合同明确规定该输出字节时，Golden 才能按合同实现它。

因此，本轮没有用 DAV3510 非有限值行为修改 Golden。尚未执行的非有限值 Golden v4 已作废。关闭 DEV-013 前必须先回答：

1. 接口是否支持 zero、NaN 和 Inf 输入；
2. 支持时，scale 和量化数据分别采用什么数学语义和编码；
3. 不支持时，用例应检查参数拒绝、执行稳定性还是仅检查无越界。

### Torch functional 为什么触发非连续 E8M0 ViewCopy

测试生成器故意构造 cache 的物理 padding，用它检查 Host、ACLNN 和 Torch 是否保留真实 storage。以失败用例 `tor_mrope_mx_0012_t1025_h56_tnd_tnd_kscale_padded_head_position_1024_slot_cross_block_functional` 为例：

```text
k_scale_cache dtype          = E8M0
当时 logical shape          = [66, 7, 16, 2, 2]
当时 physical storage shape = [66, 8, 16, 2, 2]
当时 logical stride         = [512, 64, 4, 2, 1]
当时 compact logical stride = [448, 64, 4, 2, 1]
新合同 logical shape        = [66, 7, 16, 4]
新合同 physical shape       = [66, 8, 16, 4]
新合同 logical stride       = [512, 64, 4, 1]
```

生成器给 head 维增加一个物理 padding head。这个 padding 使相邻 block 之间保留 64 个 E8M0 元素的间隔，所以逻辑 view 不是 contiguous。runner 复制完整 backing storage，再用 `as_strided` 重建该逻辑 view；它没有把输入偷偷压紧。2026-08-24 后合同把四个线性 scale 从尾部 `[2,2]` 扁平为 `[4]`，元素数和物理间隔不变；下述失败证据仍属于旧 rank-5 身份，新 rank-4 合同必须独立复测，不能直接宣称已经失败或已经通过。

functional API 不能修改调用者传入的三份 cache，因此 C++ bridge 在调用 ACLNN 前执行：

```text
k_cache.clone()
-> v_cache.clone()
-> k_scale_cache.clone()
-> aclnnQkvRmsNormRopeCacheWithKScale(...cloned caches...)
```

隔离重放在每一步前后打印阶段标记。K cache 和 V cache 的 clone 均完成；程序在 `k_scale_cache.clone()` 内进入 `aclnnInplaceCopy/ViewCopy`，返回 561103：

```text
ViewCopy on aicpu do not support dtype: DT_FLOAT8_E8M0
Convert tensor to contiguous failed
```

程序没有进入主 ACLNN。因此该失败属于 Torch functional bridge，不属于 Kernel、Host tiling 或 ACLNN 主接口。

不能把结论扩大为“所有非连续 E8M0 tensor 都不能 clone”。Python 直接构造的小 slice，以及部分直接 NPU allocation + `as_strided` 探针可以走另一条 copy 路由并通过；`empty_strided` 和当前 C++ functional bridge 则稳定进入不支持 E8M0 的 ViewCopy 路由。当前需要修复的是 production bridge 的确定路径，不是 runner 的 stride 测试。

其他 dtype 没有暴露同一问题，原因分为两层：

- 当前失败用例中的 BF16、FP32 和 INT32 tensor 是只读输入，functional bridge 不复制它们；
- bridge 先复制 E4M3 K/V cache，再复制 E8M0 scale cache。`mixed_stride` 用例证明非连续 E4M3 clone 已完成，随后才在 E8M0 clone 失败；最小真机探针中的 BF16、FP32、INT32 和 E4M3 copy 也通过。错误来自当前 ViewCopy 路由的 dtype 支持表，而不是“一字节 dtype”或 stride 公式本身。

修复不能通过删除 padded view 用例或让 runner 将输入 densify 来规避。functional bridge 必须复制 cache，同时保持输入不变；复制方案还必须明确是只保留逻辑值，还是连 backing padding、offset 和 stride 一起保留。DEV-014 关闭前，11 条 E8M0 padded-head、padded-token 和 mixed-stride functional 用例必须全部执行，并确认五个返回值及三份原输入 cache。

本节证据：

```text
/tmp/qkv_mrope_nonop_fix3_retry1_20260824_TFaSg3/merged_closure.json
/tmp/qkv_mrope_nonop_fix3_20260824_L5e75B/inputs/torch_noncontiguous_operator_11.csv
/tmp/qkv_mrope_e8m0_viewcopy_rca_20260824/probe_result_v3_runner_materialization.json
/tmp/qkv_mrope_e8m0_viewcopy_rca_20260824/probe_result_v4_storage_origin.json
/tmp/qkv_mrope_e8m0_viewcopy_rca_20260824/instrumented_result/result.csv
```

## 不能混用的证据

- Host UT 证明 Host checker、tiling key 和 UB 选择，不证明设备 VF 数值。
- Golden 离线执行证明参考算法和输入构造可运行，不证明硬件输出匹配。
- 历史 V2 补跑可以帮助定位 runner、Golden 和环境问题，但不能替代当前 V1 package 的验收。
- CSV 生成成功只证明设计覆盖和格式一致；设备测试仍必须逐行检查结果文件。
