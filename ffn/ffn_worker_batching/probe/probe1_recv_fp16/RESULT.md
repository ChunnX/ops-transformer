# probe1_recv_fp16 —— 模板穿刺结果

**任务**：迭代一 穿刺1（`probe1_recv_fp16`）
**TilingKey**：101（RECV）　**Dtype**：fp16（token_dtype=0）　**排序模式**：单核 sort（scan_token_info 前置）
**穿刺验证目标**：RECV 路径「扫描 token_info 等待 micro batch ready → 单核稳定 sort → gather + group_listing」骨架在 arch35/DAV_3510 上可行，8 路输出与 CPU golden `np.array_equal` 二进制一致。

> **本次为新 CANN 环境复核**：CANN 已统一切换到 `/home/zl/Ascend`（`cann-9.1.0`）。此前该工程曾在旧环境 `/root/Ascend` 上跑到 NPU 二进制一致；本次在新环境重建 + 重跑，确认结论一致。

---

## 状态：✅ 成功

## 运行环境：NPU（**真实实跑，非 Mock**）

| 项 | 值 |
|----|----|
| CANN 根 | **/home/zl/Ascend**（`ascend-toolkit → cann-9.1.0`） |
| SoC | Ascend950DT_9584 |
| 架构 | arch35 / DAV_3510（`--npu-arch=dav-3510` 编译） |
| CANN | 9.1.0 |
| 编译器 | `bisheng`（`/home/zl/Ascend/cann-9.1.0/bin/bisheng`） |
| host 架构 | x86_64 |
| NPU 设备 | `/dev/davinci0` |
| 运行时 vectorCoreNum | 64（`ACL_DEV_ATTR_VECTOR_CORE_NUM`） |
| 运行时 ubPerCore | 221184 B = 216 KB（`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`，**未写死**） |
| launch blockDim | 32（RECV 且 Y≤200000 限核 `min(coreNum,32)`；末 1 核 group_listing，其余 gather） |

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | ✅ 通过 | 新环境 `/home/zl/Ascend` 下 `--npu-arch=dav-3510` 无 error（仅 12 条 `cce_global attribute ignored` 提示，见下） |
| NPU 运行 | ✅ 通过 | 真实 Ascend950 实跑，`aclrtSynchronizeStream` 返回 0，`kernel finished on NPU`，8 路输出全部产出 |
| 精度（二进制一致） | ✅ 通过 | 8 路输出 `np.array_equal`（rtol=atol=0）全 PASS |
| 确定性 | ✅ 通过 | 同输入连跑 3 次，8 路输出 SHA256 完全一致（bitwise reproducible） |

**8 路输出对拍结果（全 PASS）**：`y`(fp16/128) · `group_list`(int64/16) · `session_ids`(int32/16) · `micro_batch_ids`(int32/16) · `token_ids`(int32/16) · `expert_offsets`(int32/16) · `dynamic_scale`(fp32/16，fp16 恒 0) · `actual_token_num`(int64/1=7)

**连跑 3 次 SHA256（设备输出，节选）**：
- `y.bin`：`2451f0e2…89e25b`（3 次一致）
- `group_list.bin`：`ff8eda32…44240a`（3 次一致）
- `actual_token_num.bin`：`aae89fc0…b1a8534`（3 次一致）

## 穿刺形状

`A=4, M=2, BS=2, K(topK+1)=2, H=8, expert_num=8, token_dtype=0(fp16), curMicroBatchID=0`
→ `Y = A*BS*K = 16`，`HS = H*2 = 16B/token`，`BSK=4 → 32B 对齐补 pad=4 → BSK_PAD=8`。
micro batch0 每 session 的 expert 分配 ~50% mask（`expert_id=2000000 ≥ 1e6` 视无效丢弃）；16 个 slot 中 7 个有效
→ `actual_token_num=7`；有效专家 {0,2,5,7}（含 e2 三并列、e5 两并列，测稳定排序），group_list 升序 `[expert,count]`
`[0,1][2,3][5,2][7,1]` + `[0,0]` 终止符。

## 关键结论（对迭代二 arch35 整合的降风险）

1. **RECV 路径 scan_token_info 前置 + 单核 sort 在 arch35 可行、二进制一致**：RECV 与 NORM 的差异在于 RECV 需先
   `KernelScanTokenInfo` 扫描 `token_info`（等待当前 micro batch 的 `sum(flags)==A` ready），再进入单核稳定 sort +
   gather + group_listing 三步链。本穿刺全链在 DAV_3510 端到端跑通，8 路输出与 CPU golden 逐字节一致，
   证实 RECV 前置扫描不破坏 arch35 移植结论。
2. **Kernel 直调地址注入机制在新环境依旧成立**：RECV 输入为 1024B `schedule_context`，其中内嵌 device 侧
   `token_info_buf`(off 384) / `token_data_buf`(off 400) / `polling_index`(off 416) 二级指针。host 侧 `aclrtMalloc`
   设备缓冲后把地址 `memcpy` 进 ctx 再整体拷到 device，`<<<blockNum,nullptr,stream>>>` 直调，kernel 内
   `ScheduleContextInfoCompute` 逐字段解析。**该机制在新 CANN `/home/zl/Ascend` 下无需改动即复现**——印证
   任务前提「TTK kernel 模式无地址注入能力、必须 Kernel 直调」。
3. **Kernel 直调 workspace 手工配置在 3510 正确**：`SetSysWorkspaceForce(workspace)`（内部 `__set_kfc_workspace_addr`）
   + `GetUserWorkspace`（偏移 16MB `RESERVED_WORKSPACE`）在新环境依旧生效；普通 `SetSysWorkspace` 在 3510 不生效的
   结论未变。
4. **UB/核数运行时取值、无 arch 常量写死**：`vectorCoreNum`/`ubPerCore` 经 `aclrtGetDeviceInfo` 运行时查询
   （64 核 / 216KB），`ubSize=ubPerCore-16KB`、`sortLoopMaxElement` 由 ubSize 派生，对齐 MEMORY「UB size no hardcode」。

## 已知小观察（非阻塞）

- 编译期 12 条 `warning: 'cce_global' attribute ignored when parsing type`：出现在 host 侧把 `void*` device 指针
  强转为 `GM_ADDR`(`__gm__ uint8_t*`) 传入 `<<<>>>` 的位置。`__gm__` 的 `cce_global` 属性在 host 端解析类型时被忽略，
  属 Kernel 直调工程 host/device 混编的正常现象，不影响实际地址传递（NPU 实跑结果二进制正确）。
- 无 kernel-type 标记提示（本工程核函数 `extern "C" __global__ __aicore__` 且 `if(g_coreType==AIC) return;` 守卫）。

## 与旧环境的一致性（复核结论）

- 旧环境 `/root/Ascend`（cann-9.1.0，`build/run.log` 存档）：`vectorCoreNum=64 ubPerCore=221184`，
  `tiling: Y=16 coreNum=32 ubSize=204800 sortLoopMaxElement=6400`，`kernel finished on NPU`。
- 新环境 `/home/zl/Ascend`（本次）：同样 `vectorCoreNum=64 ubPerCore=221184 coreNum=32 ubSize=204800
  sortLoopMaxElement=6400`，8 路输出全 PASS。**新旧环境 tiling 参数与验证结论完全一致，复核通过**。

## 复现

```bash
cd probe/probe1_recv_fp16
bash run.sh              # source /home/zl/Ascend/set_env.sh -> 编译(dav-3510) -> gen_data+golden -> NPU 直调 -> 8 路二进制对拍
bash run.sh --skip-build # 复用已编译产物
```

## 工程文件

```
probe/probe1_recv_fp16/
├── op_kernel/
│   ├── ffn_wb_probe_tiling.h              # 扁平 TilingData（8×int64，无 arch 常量）
│   ├── ffn_wb_recv_probe_kernel.asc       # 穿刺 kernel 入口（RECV scan+sort+gather+group_listing）
│   ├── ffn_wb_common.h ffn_wb_sort_base.h
│   ├── ffn_wb_get_schedule_context.h      # 1024B schedule_context 解析
│   ├── ffn_wb_scan_token_info.h           # RECV 专属：扫描等待 micro batch ready
│   ├── ffn_wb_scan_sort_one_core.h        # 单核稳定 sort
│   ├── ffn_wb_gather_out_all.h            # gather 8 路输出
│   └── ffn_wb_scan_group_listing_one_core.h  # 末核 group_listing
├── op_host/
│   ├── ffn_wb_recv_probe.asc              # main + KernelCall（构造 1024B ctx 内嵌 GM 地址，运行时查 UB/核数）
│   └── data_utils.h
├── scripts/{gen_data,verify_result}.py    # 生成输入+CPU golden / 8 路 np.array_equal 对拍
├── CMakeLists.txt run.sh RESULT.md
```
