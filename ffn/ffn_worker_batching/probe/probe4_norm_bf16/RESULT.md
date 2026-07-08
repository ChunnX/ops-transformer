# probe4_norm_bf16 —— 模板穿刺结果

**任务**：迭代一 穿刺4（`probe4_norm_bf16`）
**TilingKey**：100（NORM）　**Dtype**：bf16（P1，token_dtype=1）　**排序模式**：单核 sort
**穿刺验证目标**：P1 bf16 字节 gather 可行性，确认 y=bf16 二进制一致（8 路输出对拍 CPU golden）

---

## 状态：✅ 成功

## 运行环境：NPU（**真实实跑，非 Mock**）

| 项 | 值 |
|----|----|
| SoC | **Ascend950DT_9584** |
| 架构 | arch35 / DAV_3510 |
| CANN | 9.1.0 |
| 运行时 aivNum | 64（`GetCoreNumAiv()`） |
| 运行时 UB | 253952 B = 248.0 KB（`GetCoreMemSize(UB)`，**未写死**） |
| launch blockDim | 2（core0 gather / core1 group_listing，单核 sort 在 core0） |

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | ✅ 通过 | `--npu-arch=dav-3510`，A2 头文件直编 arch35 无 error（仅 1 条 kernel-type 提示，见下） |
| NPU 运行 | ✅ 通过 | 真实 Ascend950DT_9584 实跑，exit 0，8 路输出全部产出 |
| 精度（二进制一致） | ✅ 通过 | 8 路输出 `np.array_equal`（rtol=atol=0）全 PASS |
| 确定性 | ✅ 通过 | 同输入连跑 3 次输出 SHA256 完全一致（bitwise_reproducible） |

**8 路输出对拍结果（全 PASS）**：`y`(bf16/384) · `group_list`(int64/16) · `session_ids`(int32/24) · `micro_batch_ids`(int32/24) · `token_ids`(int32/24) · `expert_offsets`(int32/24) · `dynamic_scale`(fp32/24，bf16 恒 0) · `actual_token_num`(int64/1=18)

## 穿刺形状

`A=4, M=2, BS=2, K(topK+1)=3, H=16, expert_num=8, token_dtype=1(bf16)`
→ `totalLength = outNum*BS*K = 24`，`Y = A*BS*K = 24`，`HS = H*2 = 32B/token`。
含 mask：24 token 中 6 个被 mask（expert_id=1000000 排末尾丢弃）→ `actual_token_num=18`；
有效专家 {0,1,3,4,6}（5<expert_num，触发 group_list `[0,0]` 终止符），存在同专家并列（测稳定排序）。

## 关键结论（对迭代一/二 arch35 移植的降风险）

1. **bf16 字节 gather 可行、二进制一致**：`token_dtype=1` 时 `KernelFfnWBGatherOutAll` 走
   `tokenDtypeSize_ = sizeof(half) = 2B` 分支，`DataCopyPad` 逐字节搬运；y(bf16，以 uint16 原始 bit
   校验，覆盖任意 bit-pattern) 与 CPU golden 完全一致。bf16 与主线 fp16 的 gather 路径除 dtype 标记外
   完全同构（同 2 字节步长），**主线 fp16 打通后 bf16 可直接复用，无需额外分支**。
2. **A2 经典 SIMD API 全量可在 arch35/DAV_3510 编译并正确运行**：本穿刺直接把主线 A2 头文件
   （`ffn_wb_sort_one_core.h` 的 `Sort<float,true>`/`Concat`/`Extract`/`GatherMask`/`CompareScalar`/`Cast`/`Muls`/`Duplicate`/`ArithProgression`、
   `ffn_wb_gather_out_all.h` 的 `TQueBind`/`DataCopyPad`、`ffn_wb_group_listing.h` 的 `SetAtomicAdd`/`Duplicate`/`DataCopyPad`、
   `ValidGatherIdxLengthCompute` 的 `DataCacheCleanAndInvalid`）以 `--npu-arch=dav-3510` 编译，
   NPU 实跑结果二进制正确。**证实 DESIGN §3.1/§6 结论：主路线经典 SIMD/MemBase 在 arch35 可用，无需 RegBase**。
   （注：DESIGN §3.4 曾标 `GatherMask` 未在 moe arch35 蓝本出现——本穿刺补齐实证：`GatherMask` 在 DAV_3510 编译+运行均正常。）
3. **单核 sort → gather + group_listing 三步链在 arch35 端到端正确**：单核稳定排序（含 mask 置极大值排末尾）、
   核间 gather/group_listing 分工、`SyncAll` 栅栏、压缩 group_list（升序 [expert,count]+[0,0]）全部与 CPU golden 一致。
4. **UB/核数运行时取值、无 arch 常量写死**：`ubSize`/`aivNum` 经 `PlatformAscendCManager::GetInstance(socName)`
   运行时查询（248KB/64 核），`sortLoopMaxElement` 由 ubSize 派生（A2 §3.6.3 公式），对齐 MEMORY「UB size no hardcode」。

## 已知小观察（非阻塞）

- 编译期提示 `WARNING: kernel type of __global__ func: probe_ffn_wb_norm is not marked`：Kernel 直调工程无 def/config
  标记 kernel 类型（AIV），编译器自动推导。本算子为纯 Vector（AIV）算子，`if(g_coreType==AIC) return;` 已守卫，
  实跑推导正确、结果无误。**正式 arch35 路径经 `_apt.cpp` + `def.cpp`/`config/ascend950` 由算子框架标记 kernel 类型，此提示不复现**。

## 与主线/正式工程的关系（供迭代整合）

- 本穿刺工程直接复用真实 A2 头文件作为 arch35 移植候选，仅绕过 1024B `schedule_context` 逐字节解析
  （该项为**主线首验点**，非本穿刺关注），改由 host 传标量 + GM 指针组装 `ScheduleContextInfo`。
  gather/sort/group_listing 三步逻辑与 A2 **逐字一致**，结论可直接支撑迭代一/二 arch35 kernel 整合。
- `arch35/*.h` 除 `ffn_wb_gather_out_all.h` 移除对 `ffn_wb_get_schedule_context.h` 的 include（穿刺不解析 1024B 结构体）外，
  与 A2 主线头文件内容一致。

## 复现

```bash
cd probe/probe4_norm_bf16
bash run.sh            # 编译(dav-3510) -> gen_data+golden -> NPU 运行 -> 8 路二进制对拍
bash run.sh --skip-build   # 复用已编译产物
```

## 工程文件

```
probe/probe4_norm_bf16/
├── op_kernel/
│   ├── probe_ffn_wb_tiling.h          # 扁平 ProbeTilingData（标量参数，无 arch 常量）
│   ├── probe_ffn_wb_kernel.asc        # 穿刺 kernel 入口（镜像 NORM 路径三步链）
│   └── arch35/                        # A2 头文件（arch35 移植候选，直编 dav-3510）
│       ├── ffn_wb_common.h ffn_wb_sort_base.h ffn_wb_sort_one_core.h
│       ├── ffn_wb_gather_out_all.h    # 移除 get_schedule_context include
│       └── ffn_wb_group_listing.h
├── op_host/probe_ffn_wb.asc           # main + KernelCall（运行时查 UB/核数）
├── scripts/{gen_data,golden,verify_result}.py
├── CMakeLists.txt run.sh RESULT.md
```
