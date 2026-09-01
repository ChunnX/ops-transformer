# CCU AllToAll 代码检视报告（可读性 / 结构合理性）

> 检视时间：阶段1 编码完成后（用户阅读代码期间）
> 检视方：子 agent 独立检视（通读全部 8 文件 + AG 基线 5 源文件对照 + CANN `ccu_kernel_all_to_all_mesh1d.h` 核对）
> 检视口径：**只看可读性与结构合理性，不看功能正确性与性能**（由阶段2 精度验证负责）
> 状态：**已全部修复**（2025-08-29，用户指示"全都一起修复"；修复不编译，待用户服务器验证）
> 总评：**7/10**（修复前）
>
> **修复记录（2025-08-29）**：
> - 高：D-1+C-1+E-1 → `BridgeSlot` 枚举 SSOT + `WriteBridgeSlot` helper（AIC 写/CCU 读/host 分配四方按枚举索引，"10 个标量"过时注释随重构消灭）；G-1 → 数据重设计为溯源编码时已同步（见下方状态更新）
> - 中：B-1 `PrepareAllToAll` 拆为 Validate/Acquire/Alloc/ComputeAndCopy 四个 helper；B-2 `Sample` 拆为 GenerateInputData（原名 FillSendBuffer，2025-08-29 用户检视改名）/RunA2aOnce（后更名 InvokeAllToAll）/VerifyAndDumpRecv；C-2 差异③注释改为如实描述（独立累加 localSrc）；C-3 位域工具 6 函数归位+逐函数注释恢复；E-2 `A2aGoSize` 结构体贯穿（CalGoSize 返回值/kernel 签名/ctx 字段，4 参数穿透消除）；E2-1 恒假容量检查删除；H-1 退出码透传（Sample→ThreadContext.ret→main 聚合，run.sh 显式 PASS/FAIL）；H-2 usage+try/catch+非法 mode 报错
> - 低：A-1 常量重命名 `CCU_MEMSLICES_PER_LOOP`/`CCU_LOCAL_COPY_LOOP_COUNT`+注释；A-2 拼写说明入文件头；A-3 channelIdx/j 统一；A-4 `myRank` 别名；C-4/C-5 注释修正；D-2 依赖前置符号清单；D-3 条件 include 注释归属修正；D-5 Notify 常量挪至使用处；E-3 `GetParallelParamForHost` 删除（与 GetParallelParam 重复）；E-4 `MakeLocalCopyConfig` 单一装配出处；E-5 `ReleaseA2aContext` 与分配配对；E2-4 默认参数单一出处（ThreadContext）；F-1 kernelArg null 防护；F-2 SetBits 语义注释恢复；F-6 `1u<<` 显式转换+溢出说明；H-3 首错打印去重；H-5 sleep 注释说明
>
> **状态更新（2025-08-29 数据重设计后）**：应用户要求，输入数据从"整块常量双因子"重设计为
> **溯源编码**（每元素 `srcRank<<12 | dstBlock<<8 | (t^(t>>8))&0xFF`，每 slice/每元素不同）。
> 该变更顺带解决了：G-1（§6.2 数据模型三方矛盾）、F-3（首错反解魔法数字——位域反解取代）、
> P6 相关校验盲区（块内偏移现可检出，消除 Δ≡0 mod 256 类盲区）。
>
> 一致性自检已过：槽位四方一致（枚举/写侧/读侧/分配侧）/ A2aGoSize 全链路 / 花括号平衡 /
> 旧符号零残留（channelIndex/rankIdx/GetParallelParamForHost/goSize[0] 等均清零）。

---

## 待修复清单（用户读完后勾选）

### 高严重度（3 项）

| # | 问题 | 位置 | 改法 | 修 |
|---|---|---|---|---|
| D-1 | 11 槽参数桥无单一事实源：槽位语义裸数字×2 处代码 + 注释副本×7 处，人肉对齐 | CCU 读侧 `.asc` LoadArgs（`vars[base+0..10]`）；AIC 写侧 `a2a_bridge_kernel.h`（`args0..args10`） | `enum BridgeSlot` + `static_assert(BRIDGE_SLOT_NUM == HCOMM_HOST_TASK_ARG_NUM)` + `WriteBridgeSlot(r, slot, value)` helper，7 处注释收敛为"见 BridgeSlot" | ☐ |
| C-1 | 过时注释"该 round 的 10 个标量参数"——计数错（实际 3 个 per-round 标量从 GM 读，11 槽是 Variable 总数）；AG 10 槽时代话术残留 | `a2a_bridge_kernel.h:89` | 改"该 round 的 3 个 per-round 标量（srcOffset/dstOffset/sliceSize）" | ☐ |
| G-1 | DESIGN_ANALYSIS §6.2 数据模型过时：写 `myRank*1000+j`（早期方案），代码/README 是 `*16`——文档≠README≠代码三方矛盾，且 §6.2 期望公式与代码公式数学上互不相等 | `docs/DESIGN_ANALYSIS.md` §6.2 | 重写为与 `sample_main.cpp:21-31` 文件头一致（`*16` 填充/期望公式/elemsPerBlock 命名） | ☐ |

### 中严重度（8 项）

| # | 问题 | 位置 | 改法 | 修 |
|---|---|---|---|---|
| B-1 | `PrepareAllToAll` 168 行长函数（注释自述 9 步、代码一坨） | `host_launcher.h:95-263` | 拆 ValidateA2AParams / AcquireChannelsAndIns / AllocBridgeVarsAndEvents / ComputeOffsetsAndGoSize / CopyOffsetsToDevice | ☐ |
| B-2 | `Sample` 169 行混杂七件事（初始化/分配/填充/执行/释放/校验/落地） | `sample_main.cpp:73-242` | 拆 FillSendBuffer / RunOnce / VerifyAndDumpRecv | ☐ |
| C-2 | 注释描述"本地块复用循环递推"但实际用独立 localSrc 累加（注释与实现不符）；README:105 同 | `.asc:612-613`、README:105 | 注释改如实描述"独立累加构造，与远端 j 循环互不复用" | ☐ |
| C-3 | 位域工具 banner 说"4 个函数"实际 3 个；GetLoopParam 等 3 个函数无归属；DESIGN 说"6 个"——4/3/6 三口径 | `.asc:362-437` | 3 个函数上移合并进 banner，计数改 6，恢复 AG 逐函数一句话说明 | ☐ |
| E-1 | 11 连写 argsN 声明 + 11 连写 WriteGmByPassDCache 机械重复（D-1 的载体） | `a2a_bridge_kernel.h:95-121` | 随 D-1 的 WriteBridgeSlot helper 一并消除 | ☐ |
| E-2 | goSize 四元组三次机械展开（vector 返回→逐个赋值→4 参数穿透 kernel 签名） | `.asc` CalGoSize / `host_launcher.h:211-215` / `.asc:822-828` | host 侧 POD `GroupOpSizeHost`，`<<<>>>` 按 struct 传（AG MatmulTilingData 先例） | ☐ |
| E2-1 | 恒假容量检查 `inputSize != rankSize*blockSize`（blockSize 定义使然永假），AG 有语义的检查照搬后退化 | `host_launcher.h:254-258` | 删除 | ☐ |
| H-1 | 精度 FAIL 不透传退出码：Sample FAIL 仍 return 0，main 恒 0，run.sh set -e 闭环断裂 | `sample_main.cpp:241,338` | FAIL return 非 0；main 聚合各线程结果；run.sh 末尾报 PASS/FAIL | ☐ |
| H-2 | 命令行脆弱：非法 argv[1] 静默落默认参数；stoull 无 try/catch 直接 terminate；无 usage | `sample_main.cpp:267-279` | else 分支报 usage；try/catch 转 usage 提示 | ☐ |

### 低严重度（摘要，共 15 项）

- A-1 `CCU_LOCAL_COPY_MS_PER_LOOP` vs `CCU_MS_LOCAL_COPY_LOOP_COUNT` 仅词序不同、值同为 8，易混（`.asc:131-134`）
- A-3 channelId/channelIdx/channelIndex 三拼法；同函数 rankIdx/j 混用
- A-4 `device` 兼作 rankId，建议 `const uint32_t myRank = ctx->device;`
- C-4 `.asc:16` 文件名笔误 `cc_kernel_...`（少 u）
- C-5 `.asc:182` CANN 成员名写错（`rankSize_` 应为 `dimSize_/rankId_`）
- D-2 host_launcher.h 对 .asc 前文 ~12 个符号隐式依赖，无清单注释
- D-3 两处重复的条件 include + `.asc:52` 注释归属写错
- D-5 OUTPUT_XN_ID 等 4 常量距使用点 350 行
- E-3 GetParallelParamForHost 与 GetParallelParam 逐字节相同，可删一
- E-4 LoopGroupConfig 默认装配两处且乘法操作数顺序相反
- E2-4 默认参数值双处声明（ThreadContext 默认值 vs main 局部默认值）
- F-2 SetBits(end) 语义差一（AG banner 的说明裁剪时丢失）
- F-3 首错反解 6 处魔法数字（>>10、/16、%16），与 Fp16BitsFromSmallInt 无配对 helper
- F-6 `1 << rankSize` int 移位截断风格
- H-3 首错打印 block/from rank 同值重复；H-5 sleep 错峰 R=16 时等 15s

### 检视确认的正面结论（无需动作）

- 参数桥 `commTurn×11` 与 Event 布局在 4 处文件完全一致（README ↔ DESIGN ↔ .asc ↔ bridge.h ↔ host_launcher.h）
- DESIGN 引用的 AG 行号抽查 4 处全命中；配额验算 `16*11=176<400` 正确
- README 差异速览表逐行与代码对得上
- 相对 AG 的实质改进：508 行桥 kernel→155 行、消灭 commTurn 硬编码阴影常量、新增进程内断言+首错反解（AG 无任何断言）、Fp16BitsFromSmallInt 取代裸魔法常量
- `ExecuteAllToAllTransfer`（76 行）全工程最清晰函数；host/device 边界段落划分与 banner 完全对齐

---

## 与 AG 基线对比：照搬导致的别扭（检视原文摘录）

1. 恒假容量检查（E2-1）——AG 的 `totalRecvSize > recvBufSize` 有真实语义（AG recvBuf 膨胀 R 倍），照搬时未随语义重构
2. "10 个标量参数"注释残留（C-1）——AG 10 槽时代话术，槽位改 11 后没跟上
3. goSize/裸标量穿透 kernel 签名（15 参数）——AG 24 参数签名的缩小版，但 AG 已示范 struct 按值传参可行
4. 位域工具区组织混乱被原样继承且加重（C-3）——AG banner 说 4 个列 6 个；本工程裁注释后说 4 个剩 3 个
5. `schd={1,0,0x01}`、`system(mkdir)`、sleep 错峰、C 风格强转等习惯逐行照搬

## TOP3 修复建议（检视方优先级，待用户确认）

1. **BridgeSlot 枚举 SSOT**（D-1+C-1+E-1 一揽子，~60 行）——防未来槽位错位的根治手段
2. **文档-代码同步**（G-1 + C-2 + C-3 + C-4）
3. **拆长函数 + FAIL 退出码透传**（B-1/B-2/H-1）——让 run.sh 一键验证真正闭环


---

# 第二轮检视记录（2025-08-29 用户驱动：命名与结构）

> 背景：阶段2 精度验证 PASS 后，用户逐条检视代码提出 8 项意见；agent 启动子 agent 全面 review
> （命名一致性×重构联动完整性专项），报告 19 项问题，全部修复后文件名再统一。**19+8 项全闭环**。

## 用户检视 8 项（全部修复）

| # | 意见 | 修复 |
|---|---|---|
| U-1 | Sample 入参 void* 不类型安全 | `ThreadContext &ctx` + `std::ref` 传引用 |
| U-2 | ThreadContext 字段无含义标注 | 逐字段 [来源+用途] 注释 |
| U-3 | 变量定义与使用距离远 | 数据面 lambda 收敛 sendBuf/recvBuf 生命周期（声明到使用 <10 行） |
| U-4 | devCount 命名 | device/devCount → rankId/rankSize 全局统一 |
| U-5 | FillSendBuffer 名不达意 | → GenerateInputData（+定位注释：测试向量生成器，非通信流程） |
| U-6 | RunA2aOnce 命名 | → InvokeAllToAll（用户三选一） |
| U-7 | PrepareAllToAll 混 check/prepare | 拆 ValidateAllToAllParams（含 skip 出参）+ PrepareAllToAll |
| U-8 | AllToAllCustom/alltoall_bridge_kernel 命名 | → AllToAllKernel / alltoallAicoreKernel + 文件改名 |

## 子 agent review 19 项（18 修复 + 1 保留）

高 1：genRet/fillRet 改名联动漏改（编译级，修复）
中 8：注释旧名 ValidateA2AParams / README+DESIGN 的 A2AContext / CODE_GUIDE bridge.h 残留 /
  行号系统性过时（17 函数全量重算）/ alltoallAicoreKernel 命名风格（**保留：用户指定**）/
  Validate 补 rankSize 范围校验（除零+越界防护）/ ReleaseA2aContext→ReleaseAllToAllContext 对称 /
  DESIGN §6.5 dstOffset 语义过时
低 10：步骤编号 1,2,2,3 / waitCke→doneVal / <functional><cstdlib> 显式 include /
  hcclComm→comm 形参 / 注释图示 A2A→A2a / 参数桥首现补注×2 / 重复空指针预检删除 /
  host 缓冲失败路径 FreeHost / run.sh 文案+目录树 / PrepareAllToAll 死参数 sendCount 删除

## 术语规范化（用户指示）

- "后背/后背值" → "GM 中的地址/实际地址"（12 处清零）
- "参数桥" 首现加注 = 共享内存参数区（4 文件；Bridge 符号体系保留：BridgeSlot/WriteBridgeSlot/
  BRIDGE_SLOT_NUM/AllocBridgeResources）
- 术语问答沉淀至 CODE_GUIDE §7.5：Event vs Notify vs CKE / 通道数 C(R,2) vs 句柄数 R(R-1)

## 文件名统一（用户方案）

| 原名 | 新名 | 说明 |
|---|---|---|
| ccu_alltoall_kernel.asc | alltoall_ccu_kernel_impl.asc | 主文件/单翻译单元入口 |
| a2a_bridge_kernel.h | alltoall_aicore_kernel_impl.h | AIC 侧 device 组件 |
| host_launcher.h | alltoall_host_impl.h | host 编排 |
| sample_main.cpp | main.cpp | 入口 |

三个 impl 按执行域命名（ccu/aicore kernel + host）；git mv 历史保留；
include/CMake/\file/文档树全量同步；修复一处全局替换误伤（AG 溯源引用）。
