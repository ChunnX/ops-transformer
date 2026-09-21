# FfnWorkerBatching 代码概要

范围：HEAD~4..HEAD（f2c3c7d5d、b2f11facc、b4c5e09d0、399be5bfc）及其基线。侧别：**混合侧（Kernel + Host Tiling/推导 + Torch/ACLNN 接口）**。以下位置为概要生成时工作树行号，后续新增注释可能移动行号。

## 功能及调用脉络

本算子根据调度上下文整理 FFN token，并按专家分组；不执行 FFN 矩阵乘法，也不把浮点 token 量化为 MX。MXFP8/MXFP4 输入已经量化，kernel 逐字节搬运 token 及紧随其后的 E8M0 scale。

调用关系：`torch_extension.ffn_worker_batching` → 注册 custom op → C++ `op_api::ffn_worker_batching` → `aclnnFfnWorkerBatchingV2GetWorkspaceSize`/执行 → `FfnWorkerBatching` Host 注册 → arch35 tiling → `ffn_worker_batching` kernel → `FfnWorkerBatchingA5<isRecv>`。

GE 路径经 graph converter 创建同名节点，shape/dtype 由 infershape 产生；Python meta 为 FakeTensor 提供静态输出描述。实际 eager 输出内存由 C++ 显式 dtype 分配；meta 不会修改该内存的 dtype。

Kernel：`ScheduleContextParse` → `RunPrepare` → `RunSortPipeline` → `RunEmitAndGather`。Prepare 将 NORM expert_ids 或 RECV 描述符 expert_ids 归一到 flat workspace；段内 Sort32/Sort → 四路 MrgSort 多轮合并 → Extract 分离专家与原始位置 → GroupList 游程编码 → Gather 输出 token/路由索引/scale。

## 分支与数学语义

| 分支 | 行为及约束 |
|---|---|
| tiling key 100 / need_schedule=0 | NORM，输入有效行数 outNum，session/microbatch id 从单独数组读取 |
| tiling key 101 / need_schedule=1, sync_flag=false | 同步 RECV，只等待入口 polling_index 对应 microbatch 的全部 A 个 session |
| need_schedule=1, sync_flag=true | 异步 RECV，循环扫描 M 个 microbatch，选第一个存在 ready session 的项；保存本轮 DMA 观察到的 flags，之后各核仅使用同一快照 |
| 异步 layer_id | 排序键 e_global=layer_id*(expert_num/layer_num)+e_local；无效层/专家改 sentinel；原始 gather 位置不变 |
| totalValid=0 | 仍写 group_list 和 actual_token_num；跳过 Extract/Gather |
| token_dtype 0/1 | FP16/BF16，每行 2H 字节 |
| token_dtype 2 | INT8，每行 H 字节，附加 FP32 scale |
| token_dtype 3/4 | E5M2/E4M3，每行 H 字节，附加 ceil(H/32) 个 E8M0 字节 |
| token_dtype 5 | E2M1，逻辑 H 个值，存储 H/2 字节；H 要求偶数，scale 仍按逻辑 H 计算 |
| 原始位置容量 >2^24 | Gather 使用整数商余数；小容量使用 FP32 倒数乘法并修正 ±1 商误差 |

输出容量 Y=A*BS*K；只有 `[0, actual_token_num)` 为有效 token 前缀。RECV flat 行跨度为 align8(BS*K)，padding 使用 INT32_MAX，不能改变后续 gather 的 session/BS/K 解码基准。group_list 为专家及其 token 数的游程列表。

计算模式为编译期分支分派（`isRecv`/`isScanFlag`），内部包含多步 Vector 和双缓冲搬运；无 AIC-AIV 计算协同，入口 AIC 直接退出。

## 同步契约

- RECV waiter 前 SyncAll：所有核先读取入口 polling_index，之后核 0 才能推进共享游标。
- waiter 后 SyncAll：核 0 已完成 selectedMb/flags workspace 发布；所有核恢复同一本次 microbatch。
- phase 间 SyncAll：flat ID、段排序、每轮归并、Extract 的跨核依赖。
- EnQue/DeQue 管理队列交接；HardEvent 管理手动 MTE2/Vector/Scalar/MTE3 依赖；PipeBarrier 仅控制核内向量流水。
- pipe.Reset 回收本核 UB；它不替代跨核发布屏障。
- Prepare 提前清 descriptor IDs 和 flag，Gather 仍读取 token_data；正确性依赖上游不会在本次 gather 前复用 token 槽。代码注释声明此协议，但本算子没有单独完成标志来自行证明这一点。

## Tiling 业务语义与值域来源

Host 使用运行时 GetCoreNumAiv/GetCoreMemSize；ReserveLocalMemory(32K) 后取可用 UB，不能再次扣同一预留。精确硬件核数/UB 容量由本次平台对象提供，本工作树没有本次实际调用值，不猜固定数值。

| Tiling 字段 | Host 来源/公式 | 约束与使用 |
|---|---|---|
| Y,H,tokenDtype,expertNum | ParseMaxOutShape/ParseOptionalAttrs/GetAttrsInfo → DoOpTiling | Y=A*BS*K；乘法前检查 int64 溢出，Y≤INT32_MAX；H≤UINT32_MAX；dtype 0..5；专家 1..8192 |
| coreNum | min(aivNum,ceil(Y/32))，至少 1 | PostTiling blockDim；kernel 使用实际 GetBlockIdx/GetBlockNum/GetTaskRation 分工 |
| ubSize | GetPlatformInfo 可用 UB | phase 最小缓冲需求检查后用于所有切分 |
| flatElements | RECV:A*align8(BS*K)，NORM:Y | ≤INT32_MAX；实际 activeElements 使用设备 context 的 A/outNum、BS、K |
| expertStart | EXPERT_ID_MASK_START | 排除 mask 专家值 |
| preparePerLoopRows | clamp((UB-selectionBytes)/(align8(BS*K)*4+32),1,A) | 异步实际按单行处理，行大于 UB 时另走分块 |
| preparePerLoopElements | min(rows*bskAlign, floor((UB-selectionBytes-rows*32)/32)*8) | ID 缓冲容量，不要求整行驻留 |
| sortSegNum,sortPerSegElements | 先按 coreNum 均分；超过 segCap 增段 | segCap=floor((UB-160)/36/64)*64；sort 原始下标为 uint32 位模式 |
| sortLenPerSeg | align32(sortPerSegElements)*2 | 每个 proposal 包含 4B key+4B original index |
| mergeRounds | ceil(log4(sortSegNum)) | 每轮最多四路归并，轮间全核同步 |
| mergeCountCacheSegments | min(sortSegNum,128) | 每段计数占 32B，分页缓存而非全部常驻 UB |
| mergeOneLoopElements | floor((UB-cache*32-96)/64/32)*32，至少32 | 四路输入+输出 proposal 的总预算 |
| extractPerLoopElements | floor((UB-96)/16/32)*32，再限制≤255*32与Y（至少32） | 约束 Extract repeatTime |
| glRowsPerLoop | SplitGroupList 按 UB 比例、16B 行和专家数推导 | 核0写专家游程 |
| syncFlag,layerNum | 属性，layerNum 仅异步RECV保留 | 异步必须 layerNum>0 且 expertNum%layerNum=0 |
| wsFlatIds/wsPairA/wsPairB/wsSegCnt/wsSortedIds/wsGatherIdx | LayoutWorkspace 连续累加 int32 word 偏移 | 对应 flat、两份 proposal、每段32B计数、两个Y长数组 |
| wsReady | 将前面结束位置按8 words对齐 | 仅异步RECV追加(A+1)*32B；头块selectedMb，其后A个flag块 |

Workspace 总量为最终 word 数*4+系统 workspace，与偏移在同一次布局累加中产生。

## 跨文件防御与变量溯源

| 变量/契约 | 声明与赋值 | 防御边界 |
|---|---|---|
| schedule_context | 公共 attention_ffn_schedule.h 定义；ContextParse 按 offsetof/sizeof 读取 | Host CheckInputParam 检查 INT8、1D、≥1024B；Torch 检查 contiguous/NPU；Host 无法从输入描述读取二级指针有效性 |
| ctx.A/M/BS/K/HS、bufferPtr | ContextParse 从设备 context 读取 | 不等于 Host 对 max_out_shape 的校验；实际值不超容量、M非零、HS足够、指针有效是调用方协议 |
| curMicroBatchID | polling_index；异步再由 LoadSelectedMicroBatch 更新 | ContextParse ASSERT(curMicroBatchID<M)；选中值由 waiter 模 M 推进 |
| expertsPerLayer | tiling expertNum/layerNum | Host ParseOptionalAttrs 确认非零整除；ReadLayerOffset 和 EncodeLayerExperts 筛无效层及本层专家 |
| tokenRowBytes/scaleRowBytes | Gather Init 从 tokenDtype/H 推导 | FP4 偶数H在 Host/Torch 检查；MX只复制位模式 |
| token源地址 | EmitTokens 根据session/M/BS/K/HS算字节偏移 | 第一项显式提升int64，避免uint32乘法先回绕 |
| FP4 Torch输出 | C++创建Float4_e2m1fn_x2 [Y,H/2]；ACLNN转换为逻辑[ Y,H ] | singleton [1,1]使用stride[2,1]避免公共转换器误判打包轴 |
| uint8 transport | ACLNN完成原生dtype描述后 view(uint8) | view不是cast；GE converter明确拒绝不支持的FP4/uint8输出路径 |

## 原有实现与四次提交关联

基线已有 arch35 分阶段排序/gather流水，原 NORM/RECV 入口及 phase 顺序沿用。新增重点是：异步接收快照和 microbatch 扫描、层编码、MX存储及scale搬运、Torch注册和meta/GE接口、ACLNN V2与文档测试。改动还覆盖旧路径的有效行容量、UB预算、大行分块、段计数分页、整数位置精度等。A2/A3 旧 kernel 仍是独立实现；修改通用属性不意味着旧 kernel 获得 MX/异步支持，必须结合注册支持范围检查。

## 高性能设计

| 阶段 | 分工/Buffer |
|---|---|
| Prepare | session按实际参与AIV数均分；单队列ID块+每行32B清零块；异步另32B snapshot scratch |
| SegSort | 段grid-stride分配到核；输入/原始位置、三个proposal相关缓冲、mask合计约36B/元素+160B |
| Merge | 每组最多4路；两份GM proposal乒乓；UB输入4*loop*8B、输出同宽、计数cache*32B，额外96B |
| Extract | proposal8B/元素+ID4B+index4B；额外96B；单次至多8160元素 |
| GroupList | 核0，10路向量缓冲+2路mask+16B/输出行拼装 |
| Gather | 每轮128行索引；token输入双缓冲，沿H字节分块；输出索引双缓冲；MX scale独立字节流复制 |

## 源码定位索引

以下自动索引用于快速反查所有定义、API与常量；函数签名按源码合并多行。类内同名 Init/Process 必须结合文件和所在类辨识。仅语法抽取，不冒充编译器解析；宏注册及虚函数由框架间接调用，不因文字引用少就判为死代码。

### ffn/ffn_worker_batching/torch_extension/__init__.py


### ffn/ffn_worker_batching/torch_extension/graph_convert_ffn_worker_batching.py


函数清单：

| 行 | 源码 |
|---|---|
| 45 | `def FfnWorkerBatching( schedule_context: Tensor, *, expert_num: int, max_out_shape: List[int], token_dtype: int = 0, need_schedule: int = 0, layer_num: int = 0, sync_flag: bool = False, dependencies=None, node_name=None, ):` |
| 128 | `def convert_ffn_worker_batching( schedule_context: Tensor, expert_num: int, max_out_shape: List[int], *, token_dtype: int = 0, need_schedule: int = 0, layer_num: int = 0, sync_flag: bool = False, mx_output_uint8: bool = False, meta_outputs: Optional[List[TensorSpec]] = None, ):` |
| 172 | `def convert_ffn_worker_batching(*args, **kwargs):` |

### ffn/ffn_worker_batching/torch_extension/ffn_worker_batching.py


函数清单：

| 行 | 源码 |
|---|---|
| 48 | `def _validate( schedule_context: torch.Tensor, expert_num: int, max_out_shape: List[int], token_dtype: int, need_schedule: int, layer_num: int, sync_flag: bool, ) -> None:` |
| 120 | `def _output_specs( expert_num: int, max_out_shape: List[int], token_dtype: int, mx_output_uint8: bool = False, ) -> List[Tuple[Tuple[int, ...], torch.dtype]]:` |
| 152 | `def __init__(self):` |
| 155 | `def sources(self) -> List[str]:` |
| 158 | `def schema(self) -> str:` |
| 167 | `def register_meta(self) -> None:` |
| 206 | `def _ffn_worker_batching( schedule_context, expert_num, max_out_shape, *, token_dtype=0, need_schedule=0, layer_num=0, sync_flag: bool = False, mx_output_uint8=False, ):` |
| 238 | `def ffn_worker_batching( schedule_context: torch.Tensor, expert_num: int, max_out_shape: List[int], *, token_dtype: int = 0, need_schedule: int = 0, layer_num: int = 0, sync_flag: bool = False, mx_output_uint8: bool = False, ) -> _Outputs:` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_scan_sort_multi_core.h


函数清单：

| 行 | 源码 |
|---|---|
| 33 | `__aicore__ inline KernelScanSortMaskMultiCore(){};` |
| 34 | `__aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe);` |
| 36 | `__aicore__ inline void Process();` |
| 39 | `__aicore__ inline void VBSProcess();` |
| 40 | `__aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt);` |
| 41 | `__aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);` |
| 42 | `__aicore__ inline void VMSProcess();` |
| 43 | `__aicore__ inline void SortOutProcess();` |
| 44 | `__aicore__ inline void VBSCopyInAndClear(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt);` |
| 45 | `__aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);` |
| 46 | `__aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);` |
| 47 | `__aicore__ inline void InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset);` |
| 49 | `__aicore__ inline void InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);` |
| 50 | `__aicore__ inline void CopyOutValidCount();` |
| 51 | `__aicore__ inline void ClearTokenInfoFlag();` |
| 99 | `__aicore__ inline void KernelScanSortMaskMultiCore::VBSCopyInAndClear(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt)` |
| 136 | `__aicore__ inline void KernelScanSortMaskMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)` |
| 191 | `__aicore__ inline void KernelScanSortMaskMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)` |
| 223 | `__aicore__ inline void KernelScanSortMaskMultiCore::InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset)` |
| 243 | `__aicore__ inline void KernelScanSortMaskMultiCore::InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)` |
| 266 | `__aicore__ inline void KernelScanSortMaskMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements)` |
| 306 | `__aicore__ inline void KernelScanSortMaskMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt)` |
| 314 | `__aicore__ inline void KernelScanSortMaskMultiCore::VBSProcess()` |
| 334 | `__aicore__ inline void KernelScanSortMaskMultiCore::VMSProcess()` |
| 373 | `__aicore__ inline void KernelScanSortMaskMultiCore::SortOutProcess()` |
| 389 | `__aicore__ inline void KernelScanSortMaskMultiCore::CopyOutValidCount()` |
| 403 | `__aicore__ inline void KernelScanSortMaskMultiCore::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe)` |
| 469 | `__aicore__ inline void KernelScanSortMaskMultiCore::ClearTokenInfoFlag()` |
| 496 | `__aicore__ inline void KernelScanSortMaskMultiCore::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 102 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 111 | `DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);` |
| 117 | `// SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 119 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 122 | `LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 123 | `Duplicate<int32_t>(clearLocal, INT_MAX, size);` |
| 124 | `sortDataCopyOutQueue.EnQue(clearLocal);` |
| 126 | `clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 131 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 132 | `DataCopyPad(expertIdsGm[inOffset], clearLocal, copyoutParams);` |
| 133 | `sortDataCopyOutQueue.FreeTensor(clearLocal);` |
| 138 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 143 | `Cast(expertIdsLocalFp32, expertIdsLocal, RoundMode::CAST_ROUND, size);` |
| 149 | `Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);` |
| 161 | `GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 164 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 175 | `Duplicate(expertIdsLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 181 | `GatherMask(rowIdsLocal, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 184 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 185 | `Sort<float, true>(outLocal, concatLocal, rowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 187 | `sortDataCopyOutQueue.EnQue<float>(outLocal);` |
| 188 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 199 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();` |
| 206 | `DataCopyPad(workspaceGms[1][wkOffset], outLocal, copyParams);` |
| 208 | `DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);` |
| 211 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 217 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 219 | `DataCopyPad(workspaceSortNumGm_[blockIdx_ * tilingData_->sortNumWorkSpacePerCore + progress], tempTensor,` |
| 229 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 230 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 238 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 239 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 240 | `tempBuffer.FreeTensor(outSortNumLocal);` |
| 248 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 249 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 262 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 263 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 331 | `SyncAll();` |
| 369 | `SyncAll();` |
| 386 | `SyncAll();` |
| 391 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 398 | `DataCopyPad(rsvdCntGm[0], outLocal[0], params);` |
| 400 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 463 | `pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize_);` |
| 464 | `pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize_);` |
| 465 | `pipe->InitBuffer(sortedBuffer, bufferSize_);` |
| 466 | `pipe->InitBuffer(tempBuffer, bufferSize_);` |
| 479 | `Duplicate<int32_t>(clearLocal, 0, duplicateNum);` |
| 481 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 491 | `DataCopyPad(expertIdsGmFStart_[idx * perLoopElement * contextInfo_->M * F], clearLocal, copyOutParams);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SCAN_SORT_MULTI_CORE_H` |
| 96 | `static constexpr int64_t MAX_MRGSORT_LIST = 4;` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_common.h


函数清单：

| 行 | 源码 |
|---|---|
| 133 | `__aicore__ inline void SetWaitFlag(HardEvent evt)` |
| 140 | `__aicore__ inline int64_t Ceil(int64_t a, int64_t b)` |
| 148 | `__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)` |
| 157 | `__aicore__ inline T Min(T a, T b)` |
| 163 | `__aicore__ inline T Max(T a, T b)` |
| 169 | `__aicore__ inline T2 CeilDiv(T1 x, T2 y)` |
| 180 | `__aicore__ inline T PowerOfFourCeil(T x)` |
| 192 | `__aicore__ inline void TilingOneCoreSort(SortCustomTilingDataKernel *tilingData, const int64_t totalLength)` |
| 205 | `__aicore__ inline void TilingMultiCoreSort(SortCustomTilingDataKernel *tilingData, const int64_t totalLength, const ScheduleContextInfo *contextInfo)` |
| 257 | `__aicore__ inline void TilingScanMultiCoreSort(SortCustomTilingDataKernel *tilingdataSort, const ScheduleContextInfo *contextInfo)` |
| 285 | `__aicore__ inline void TilingSort(SortCustomTilingDataKernel *tilingdataSort, const ScheduleContextInfo *contextInfo)` |
| 299 | `__aicore__ inline void TilingScanSort(SortCustomTilingDataKernel *tilingdataSort, const ScheduleContextInfo *contextInfo)` |
| 318 | `__aicore__ inline void ValidGatherIdxLengthCompute(GM_ADDR work_space, ScheduleContextInfo &contextInfo, GM_ADDR actual_token_num)` |
| 338 | `__aicore__ inline void Tiling4SrcToDstCompute(const ScheduleContextInfo *tilingData, FfnWBGroupListingTileInfo &tilingInfo)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 133 | `__aicore__ inline void SetWaitFlag(HardEvent evt)` |

常量清单：

| 行 | 源码 |
|---|---|
| 23 | `#define OP_KERNEL_FFN_WB_COMMON_H` |
| 31 | `constexpr int64_t NUM_TWO = 2;` |
| 32 | `constexpr int64_t NUM_THREE = 3;` |
| 33 | `constexpr int64_t NUM_FOUR = 4;` |
| 34 | `constexpr int64_t MRG_LIST_NUM = 4;` |
| 35 | `constexpr int64_t SORT32_ALIGN_ELEMENT = 32;` |
| 36 | `constexpr int64_t MGR_SORT_MAX_ELEMENT = 1024;` |
| 37 | `constexpr int64_t TILINGKEY_ONECORE_SORT = 0;` |
| 38 | `constexpr int64_t TILINGKEY_MULTICORE_SORT = 1;` |
| 39 | `constexpr int64_t NORMAL_BSK = 1; // Bs*K+pad小于等于UB` |
| 42 | `constexpr float MIN_FP32 = -3.4e38;` |
| 43 | `constexpr int64_t ONE_REPEAT_SORT_NUM = 32;` |
| 44 | `constexpr int64_t MERGE_LIST_TWO = 2;` |
| 45 | `constexpr int64_t MERGE_LIST_THREE = 3;` |
| 46 | `constexpr int64_t MERGE_LIST_FOUR = 4;` |
| 48 | `constexpr int64_t MERGE_LIST_IDX_TWO = 2;` |
| 49 | `constexpr int64_t MERGE_LIST_IDX_THREE = 3;` |
| 51 | `constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;` |
| 52 | `constexpr int64_t BLOCK_BYTES = 32;` |
| 54 | `constexpr int64_t MAX_EXPERT_NUM = 5120;` |
| 55 | `constexpr int64_t EXPERT_ID_VALUE_NUM = 2;` |
| 57 | `constexpr int64_t GROUP_LISTING_MULTI_CORE_LENGTH = 512;` |
| 58 | `constexpr uint32_t GROUP_LISTING_MULTI_AIV_NUM = 4;` |
| 59 | `constexpr uint32_t GROUP_LISTING_AIV_NUM = 1;` |
| 60 | `constexpr uint32_t GROUP_LISTING_SCAN_AIV_NUM = 1;` |
| 61 | `constexpr int64_t ASSIST_NUM = 256;` |
| 62 | `constexpr int64_t OFFSET_SORTED_EXPERT_IDS = 128; // workspace偏移128个uint32，开始写排序的结果` |
| 63 | `constexpr int64_t SCAN_BATCHID_GM_OFFSET = 32;` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_group_listing.h


函数清单：

| 行 | 源码 |
|---|---|
| 33 | `__aicore__ inline KernelFfnWBGroupListing(){};` |
| 34 | `__aicore__ inline void Init(GM_ADDR x, GM_ADDR groupList, GM_ADDR groupListTmp, ScheduleContextInfo *scheduleContext, TPipe *tPipe, int64_t groupListingDealFlag);` |
| 36 | `__aicore__ inline void Process(int64_t groupListingDealFlag);` |
| 37 | `__aicore__ inline void ProcessExpertCount();` |
| 40 | `__aicore__ inline void CopyIn(int64_t loop, int64_t curLoopElements);` |
| 41 | `__aicore__ inline void Compute(int64_t curLoopElements);` |
| 42 | `__aicore__ inline void CopyOut();` |
| 44 | `__aicore__ inline void expertCountCopyIn();` |
| 45 | `__aicore__ inline void expertCountCompute();` |
| 46 | `__aicore__ inline void expertCountCopyOut();` |
| 47 | `__aicore__ inline void CopyInOneCore(int64_t inputNum);` |
| 48 | `__aicore__ inline void ComputeOneCore(int64_t inputNum);` |
| 88 | `__aicore__ inline void KernelFfnWBGroupListing::Init(GM_ADDR x, GM_ADDR groupList, GM_ADDR groupListTmp, ScheduleContextInfo *scheduleContext, TPipe *tPipe, int64_t groupListingDealFlag)` |
| 143 | `__aicore__ inline void KernelFfnWBGroupListing::Process(int64_t groupListingDealFlag)` |
| 182 | `__aicore__ inline void KernelFfnWBGroupListing::ProcessExpertCount()` |
| 191 | `__aicore__ inline void KernelFfnWBGroupListing::CopyIn(int64_t loop, int64_t curLoopElements)` |
| 203 | `__aicore__ inline void KernelFfnWBGroupListing::Compute(int64_t curLoopElements)` |
| 232 | `__aicore__ inline void KernelFfnWBGroupListing::CopyOut()` |
| 245 | `__aicore__ inline void KernelFfnWBGroupListing::expertCountCopyIn()` |
| 257 | `__aicore__ inline void KernelFfnWBGroupListing::expertCountCompute()` |
| 285 | `__aicore__ inline void KernelFfnWBGroupListing::expertCountCopyOut()` |
| 295 | `__aicore__ inline void KernelFfnWBGroupListing::CopyInOneCore(int64_t inputNum)` |
| 305 | `__aicore__ inline void KernelFfnWBGroupListing::ComputeOneCore(int64_t inputNum)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 139 | `pipe_->InitBuffer(sortedExpertIdxInQueue, 1, perLoopRows * sizeof(int32_t));` |
| 140 | `pipe_->InitBuffer(expertIdxCountOutQueue, 1, cntSize);` |
| 155 | `groupListOutLocal = expertIdxCountOutQueue.AllocTensor<int64_t>();` |
| 175 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 176 | `DataCopyPad(groupListGm, groupListOutLocal, copyParams);` |
| 178 | `expertIdxCountOutQueue.FreeTensor(groupListOutLocal);` |
| 193 | `LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue.AllocTensor<int32_t>();` |
| 198 | `DataCopyPad(sortedExpertIdxInLocal, expandedExpertIdsGm[sortedexpertIdxOffset], dataCopyParams, dataCopyPadParams);` |
| 200 | `sortedExpertIdxInQueue.EnQue(sortedExpertIdxInLocal);` |
| 205 | `LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue.DeQue<int32_t>();` |
| 206 | `LocalTensor<int32_t> expertCountOutLocal = expertIdxCountOutQueue.AllocTensor<int32_t>();` |
| 207 | `Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), static_cast<int32_t>(expertNum));` |
| 228 | `expertIdxCountOutQueue.EnQue<int32_t>(expertCountOutLocal);` |
| 229 | `sortedExpertIdxInQueue.FreeTensor(sortedExpertIdxInLocal);` |
| 234 | `LocalTensor<int32_t> expertCountOutLocal = expertIdxCountOutQueue.DeQue<int32_t>();` |
| 239 | `DataCopyPad(expertCountTempGm, expertCountOutLocal, copyParams);` |
| 242 | `expertIdxCountOutQueue.FreeTensor(expertCountOutLocal);` |
| 247 | `LocalTensor<int32_t> expertCountTempInLocal = sortedExpertIdxInQueue.AllocTensor<int32_t>();` |
| 252 | `DataCopyPad(expertCountTempInLocal, expertCountTempGm, dataCopyParams, dataCopyPadParams);` |
| 254 | `sortedExpertIdxInQueue.EnQue(expertCountTempInLocal);` |
| 259 | `LocalTensor<int32_t> expertCountTempInLocal = sortedExpertIdxInQueue.DeQue<int32_t>();` |
| 261 | `LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue.AllocTensor<int64_t>();` |
| 281 | `expertIdxCountOutQueue.EnQue<int64_t>(expertCountOutLocal);` |
| 282 | `sortedExpertIdxInQueue.FreeTensor(expertCountTempInLocal);` |
| 287 | `LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue.DeQue<int64_t>();` |
| 290 | `DataCopyPad(groupListGm, expertCountOutLocal, copyParams);` |
| 292 | `expertIdxCountOutQueue.FreeTensor(expertCountOutLocal);` |
| 297 | `LocalTensor<int32_t> inLocal = sortedExpertIdxInQueue.AllocTensor<int32_t>();` |
| 301 | `DataCopyPad(inLocal, expandedExpertIdsGm, dataCopyParams, dataCopyPadParams);` |
| 302 | `sortedExpertIdxInQueue.EnQue<int32_t>(inLocal);` |
| 307 | `LocalTensor<int32_t> inLocal = sortedExpertIdxInQueue.DeQue<int32_t>();` |
| 308 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 325 | `sortedExpertIdxInQueue.FreeTensor(inLocal);` |

常量清单：

| 行 | 源码 |
|---|---|
| 23 | `#define OP_KERNEL_FFN_WB_GROUP_LISTING_H` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_scan_sort_multi_core_bsk.h


函数清单：

| 行 | 源码 |
|---|---|
| 33 | `__aicore__ inline KernelScanSortMaskMultiCoreBsK(){};` |
| 34 | `__aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe);` |
| 36 | `__aicore__ inline void Process();` |
| 39 | `__aicore__ inline void VBSProcess();` |
| 40 | `__aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt);` |
| 41 | `__aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);` |
| 42 | `__aicore__ inline void VMSProcess();` |
| 43 | `__aicore__ inline void SortOutProcess();` |
| 44 | `__aicore__ inline void VBSCopyInAndClear(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt);` |
| 45 | `__aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);` |
| 46 | `__aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);` |
| 47 | `__aicore__ inline void InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset);` |
| 49 | `__aicore__ inline void InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);` |
| 50 | `__aicore__ inline void CopyOutValidCount();` |
| 51 | `__aicore__ inline void ClearTokenInfoFlag();` |
| 95 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VBSCopyInAndClear(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt)` |
| 151 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)` |
| 212 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)` |
| 236 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset)` |
| 257 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)` |
| 280 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements)` |
| 350 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt)` |
| 358 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VBSProcess()` |
| 384 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VMSProcess()` |
| 420 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::SortOutProcess()` |
| 435 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::CopyOutValidCount()` |
| 451 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe)` |
| 533 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::ClearTokenInfoFlag()` |
| 560 | `__aicore__ inline void KernelScanSortMaskMultiCoreBsK::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 107 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 120 | `DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);` |
| 132 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 135 | `LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 136 | `Duplicate<int32_t>(clearLocal, INT_MAX, size);` |
| 138 | `sortDataCopyOutQueue.EnQue(clearLocal);` |
| 140 | `clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 145 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 146 | `DataCopyPad(expertIdsGm[inOffset], clearLocal, copyoutParams);` |
| 148 | `sortDataCopyOutQueue.FreeTensor(clearLocal);` |
| 153 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 158 | `Cast(expertIdsLocalFp32, expertIdsLocal, RoundMode::CAST_ROUND, size);` |
| 164 | `Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);` |
| 178 | `GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 183 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 194 | `Duplicate(expertIdsLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 201 | `GatherMask(rowIdsLocal, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 204 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 206 | `Sort<float, true>(outLocal, concatLocal, rowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 208 | `sortDataCopyOutQueue.EnQue<float>(outLocal);` |
| 209 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 215 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();` |
| 221 | `DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);` |
| 223 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 230 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 232 | `DataCopyPad(workspaceSortNumGm_[blockIdx_ * tilingData_->sortNumWorkSpacePerCore + progress], tempTensor,` |
| 243 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 244 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 252 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 253 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 254 | `tempBuffer.FreeTensor(outSortNumLocal);` |
| 262 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 263 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 276 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 277 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 326 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 329 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 335 | `DataCopyPad(inLocal, srcWsGm, intriParamsIn, intriPadParamsIn);` |
| 338 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 343 | `DataCopyPad(dstWsGm, inLocal, intriParamsOut);` |
| 345 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 381 | `SyncAll();` |
| 416 | `SyncAll();` |
| 432 | `SyncAll();` |
| 437 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 445 | `DataCopyPad(rsvdCntGm[0], outLocal[0], params);` |
| 448 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 527 | `pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize_);` |
| 528 | `pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize_);` |
| 529 | `pipe->InitBuffer(sortedBuffer, bufferSize_);` |
| 530 | `pipe->InitBuffer(tempBuffer, bufferSize_);` |
| 543 | `Duplicate<int32_t>(clearLocal, 0, duplicateNum);` |
| 545 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 555 | `DataCopyPad(expertIdsGmFStart_[idx * perLoopElement * contextInfo_->M * F], clearLocal, copyOutParams);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SCAN_SORT_MULTI_CORE_BSK_H` |
| 92 | `static constexpr int64_t MAX_MRGSORT_LIST = 4;` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_sort_mrgsort_out.h


函数清单：

| 行 | 源码 |
|---|---|
| 32 | `__aicore__ inline SortCustomMrgsortOut(){};` |
| 33 | `__aicore__ inline void Init(SortCustomMrgsortParam *param, TPipe *tPipe);` |
| 34 | `__aicore__ inline void Process();` |
| 35 | `__aicore__ inline void SetInput(GlobalTensor<float> &gmInput, GlobalTensor<int32_t> &gmSortNum, LocalTensor<float> &ubInput);` |
| 37 | `__aicore__ inline void SetOutput(GlobalTensor<int32_t> &gmOutput1, GlobalTensor<int32_t> &gmOutput2, LocalTensor<float> &ubOutput1, LocalTensor<float> &ubOutput2);` |
| 39 | `__aicore__ inline void SetBuffer(LocalTensor<float> &tempBuffer);` |
| 42 | `__aicore__ inline void CopyIn();` |
| 43 | `__aicore__ inline void UpdateMrgParam();` |
| 44 | `__aicore__ inline void MrgsortCompute();` |
| 45 | `__aicore__ inline void UpdateSortInfo();` |
| 46 | `__aicore__ inline void Extract();` |
| 47 | `__aicore__ inline void CopyOut();` |
| 48 | `__aicore__ inline void ClearCache();` |
| 85 | `__aicore__ inline void SortCustomMrgsortOut::ClearCache()` |
| 92 | `__aicore__ inline void SortCustomMrgsortOut::SetInput(GlobalTensor<float> &gmInput, GlobalTensor<int32_t> &gmSortNum, LocalTensor<float> &ubInput)` |
| 103 | `__aicore__ inline void SortCustomMrgsortOut::SetOutput(GlobalTensor<int32_t> &gmOutput1, GlobalTensor<int32_t> &gmOutput2, LocalTensor<float> &ubOutput1, LocalTensor<float> &ubOutput2)` |
| 116 | `__aicore__ inline void SortCustomMrgsortOut::SetBuffer(LocalTensor<float> &tempBuffer)` |
| 121 | `__aicore__ inline void SortCustomMrgsortOut::UpdateMrgParam()` |
| 137 | `__aicore__ inline void SortCustomMrgsortOut::CopyIn()` |
| 154 | `__aicore__ inline void SortCustomMrgsortOut::MrgsortCompute()` |
| 177 | `__aicore__ inline void SortCustomMrgsortOut::UpdateSortInfo()` |
| 194 | `__aicore__ inline void SortCustomMrgsortOut::Extract()` |
| 206 | `__aicore__ inline void SortCustomMrgsortOut::CopyOut()` |
| 219 | `__aicore__ inline void SortCustomMrgsortOut::Init(SortCustomMrgsortParam *param, TPipe *tPipe)` |
| 232 | `__aicore__ inline void SortCustomMrgsortOut::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 46 | `__aicore__ inline void Extract();` |
| 140 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 144 | `DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]],` |
| 156 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 159 | `MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);` |
| 163 | `MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);` |
| 167 | `MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);` |
| 170 | `DataCopy(this->tempBuffer, this->tmpUbInputs[0],` |
| 194 | `__aicore__ inline void SortCustomMrgsortOut::Extract()` |
| 197 | `AscendC::Extract(this->ubOutput1, this->ubOutput2, this->tempBuffer,` |
| 199 | `PipeBarrier<PIPE_V>();` |
| 200 | `Muls(this->ubOutput1, this->ubOutput1, (float)-1, curLoopSortedNum);` |
| 201 | `PipeBarrier<PIPE_V>();` |
| 202 | `Cast(this->ubOutputInt1, this->ubOutput1, RoundMode::CAST_ROUND, curLoopSortedNum);` |
| 212 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 213 | `DataCopyPad(this->gmOutput1[outOffset], this->ubOutputInt1, intriParams);` |
| 214 | `DataCopyPad(this->gmOutput2[outOffset], this->ubOutputInt2, intriParams);` |
| 239 | `Extract();` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SORT_MRGSORT_OUT_H` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_scan_sort_one_core.h


函数清单：

| 行 | 源码 |
|---|---|
| 32 | `__aicore__ inline KernelScanSortMaskOneCore(){};` |
| 33 | `__aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe);` |
| 35 | `__aicore__ inline void Process();` |
| 38 | `__aicore__ inline void CopyInAndClear();` |
| 39 | `__aicore__ inline void SortCompute();` |
| 40 | `__aicore__ inline void ExpertCountCompute();` |
| 41 | `__aicore__ inline void CopyOut();` |
| 51 | `__aicore__ inline void KernelScanSortMaskOneCore::CopyInAndClear()` |
| 84 | `__aicore__ inline void KernelScanSortMaskOneCore::SortCompute()` |
| 159 | `__aicore__ inline void KernelScanSortMaskOneCore::CopyOut()` |
| 174 | `__aicore__ inline void KernelScanSortMaskOneCore::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe)` |
| 210 | `__aicore__ inline void KernelScanSortMaskOneCore::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 53 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 58 | `DataCopyPad(inLocal, expertIdsGm[contextInfo_->curMicroBatchID * F_ + NUM_TWO], dataCopyParams, dataCopyPadParams);` |
| 62 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 65 | `LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 66 | `Duplicate<int32_t>(clearLocal, INT_MAX, this->totalLength);` |
| 67 | `Duplicate<int32_t>(clearLocal[this->sortNum], 0, contextInfo_->A * 8); // 8: block num` |
| 68 | `sortDataCopyOutQueue.EnQue(clearLocal);` |
| 70 | `clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 78 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 79 | `DataCopyPad(expertIdsGm[contextInfo_->curMicroBatchID * F_ + NUM_TWO], clearLocal, copyoutParams);` |
| 80 | `DataCopyPad(expertIdsGm[contextInfo_->curMicroBatchID * F_], clearLocal[this->sortNum], copyOutParamsF);` |
| 81 | `sortDataCopyOutQueue.FreeTensor(clearLocal);` |
| 87 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 89 | `Cast(expertIdsFp32, inLocal, RoundMode::CAST_ROUND, this->totalLength);` |
| 90 | `PipeBarrier<PIPE_V>();` |
| 96 | `Muls(expertIdsFp32, expertIdsFp32, (float)-1, this->totalLength);` |
| 97 | `PipeBarrier<PIPE_V>();` |
| 103 | `PipeBarrier<PIPE_V>();` |
| 110 | `GatherMask(expertIdsFp32, expertIdsFp32, maskLocalUInt32, true, this->totalLength, gatherMaskParams, rsvdCnt);` |
| 111 | `PipeBarrier<PIPE_V>();` |
| 114 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 124 | `Duplicate(expertIdsFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 125 | `PipeBarrier<PIPE_V>();` |
| 133 | `PipeBarrier<PIPE_V>();` |
| 137 | `GatherMask(rowIdsLocal, rowIdsLocal, maskLocalUInt32, true, this->totalLength, gatherMaskParams, rsvdCnt);` |
| 138 | `PipeBarrier<PIPE_V>();` |
| 140 | `Sort<float, true>(sortedLocal, concatLocal, rowIdsLocal, tempTensor, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 141 | `PipeBarrier<PIPE_V>();` |
| 143 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 146 | `Extract(sortedExpertIdsLocal, sortedRowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 147 | `PipeBarrier<PIPE_V>();` |
| 150 | `Muls(sortedExpertIdsLocal, sortedExpertIdsLocal, (float)-1, rsvdCnt);` |
| 151 | `PipeBarrier<PIPE_V>();` |
| 153 | `Cast(sortedExpertIdsLocalInt32, sortedExpertIdsLocal, RoundMode::CAST_ROUND, rsvdCnt);` |
| 154 | `PipeBarrier<PIPE_V>();` |
| 155 | `sortDataCopyOutQueue.EnQue<float>(outLocal);` |
| 156 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 162 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 164 | `DataCopyPad(sortedexpertIdsGm, outLocal[0], copyParams);` |
| 165 | `DataCopyPad(sortedRowIdsGm, outLocal[this->sortNum], copyParams);` |
| 166 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 204 | `pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);` |
| 205 | `pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);` |
| 206 | `pipe->InitBuffer(tempBuffer, buffSize);` |
| 207 | `pipe->InitBuffer(sortedBuffer, buffSize);` |
| 217 | `SyncAll();` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SCAN_SORT_ONE_CORE_H` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_get_schedule_context.h


函数清单：

| 行 | 源码 |
|---|---|
| 34 | `__aicore__ inline void ScheduleContextInfoCompute(GM_ADDR schedule_context, const FfnWorkerBatchingTilingData *tilingData, ScheduleContextInfo &contextInfo, TPipe *pipe)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 43 | `pipe->InitBuffer(buffer, STRUCT_LENGTH); // 检验数据是否正确；` |
| 45 | `DataCopy(valLocal, scheduleContext, STRUCT_LENGTH);` |
| 47 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |

常量清单：

| 行 | 源码 |
|---|---|
| 23 | `#define OP_KERNEL_FFN_WB_GET_SCHEDULE_CONTEXT_H` |
| 32 | `constexpr int64_t STRUCT_LENGTH = 1024;` |
| 87 | `if constexpr (isScanMode == false) {` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_scan_token_info.h


函数清单：

| 行 | 源码 |
|---|---|
| 34 | `__aicore__ inline KernelScanTokenInfo(){};` |
| 35 | `__aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR token_info_gm_addr, GM_ADDR workspace, ScheduleContextInfo *scheduleContextInfo, TPipe *tPipe);` |
| 37 | `__aicore__ inline void Process();` |
| 40 | `__aicore__ inline void CopyInFlag();` |
| 57 | `__aicore__ inline void KernelScanTokenInfo::CopyInFlag()` |
| 68 | `__aicore__ inline void KernelScanTokenInfo::Init(GM_ADDR schedule_context, GM_ADDR token_info_gm_addr, GM_ADDR workspace, ScheduleContextInfo *scheduleContextInfo, TPipe *tPipe)` |
| 110 | `__aicore__ inline void KernelScanTokenInfo::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 60 | `LocalTensor<int32_t> tokenInfoLocalIn = tokenFlagInQueue.AllocTensor<int32_t>();` |
| 64 | `DataCopyPad(tokenInfoLocalIn, tokenInfoGm[offset], dataCopyParams, dataCopyPadParams);` |
| 65 | `tokenFlagInQueue.EnQue(tokenInfoLocalIn);` |
| 93 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 106 | `pipe->InitBuffer(tokenFlagInQueue, 1, A * BLOCK_SIZE);` |
| 107 | `pipe->InitBuffer(workBuf, A * BLOCK_SIZE);` |
| 118 | `LocalTensor<int32_t> tokenFlagLocal = tokenFlagInQueue.DeQue<int32_t>();` |
| 120 | `Cast(workLocal, tokenFlagLocal, RoundMode::CAST_ROUND, A * BLOCK_SIZE / sizeof(int32_t));` |
| 121 | `PipeBarrier<PIPE_V>();` |
| 123 | `ReduceSum<float>(workLocal, workLocal, workLocal, A * BLOCK_SIZE / sizeof(int32_t));` |
| 124 | `PipeBarrier<PIPE_V>();` |
| 127 | `Cast(resultLocal, workLocal, RoundMode::CAST_ROUND, 1);` |
| 128 | `PipeBarrier<PIPE_V>();` |
| 130 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |
| 133 | `tokenFlagInQueue.FreeTensor(tokenFlagLocal);` |
| 134 | `workBuf.FreeTensor(workLocal);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SCAN_TOKEN_INFO_H` |
| 26 | `constexpr uint32_t BLOCK_SIZE = 32;` |
| 27 | `constexpr uint32_t INT32_SIZE = 4;` |
| 28 | `constexpr uint32_t FLAG_SIZE = 4;` |
| 29 | `constexpr uint32_t FLAG_LAYER_LEN = 2;` |
| 30 | `constexpr uint32_t BUFFER_NUM_ONE = 1;` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_sort_multi_core.h


函数清单：

| 行 | 源码 |
|---|---|
| 34 | `__aicore__ inline KernelSortMaskMultiCore(){};` |
| 35 | `__aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe);` |
| 37 | `__aicore__ inline void Process();` |
| 40 | `__aicore__ inline void VBSProcess();` |
| 41 | `__aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);` |
| 42 | `__aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);` |
| 43 | `__aicore__ inline void VMSProcess();` |
| 44 | `__aicore__ inline void SortOutProcess();` |
| 45 | `__aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);` |
| 46 | `__aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);` |
| 47 | `__aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);` |
| 48 | `__aicore__ inline void InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset);` |
| 50 | `__aicore__ inline void InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);` |
| 51 | `__aicore__ inline void CopyOutValidCount();` |
| 86 | `__aicore__ inline void KernelSortMaskMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)` |
| 100 | `__aicore__ inline void KernelSortMaskMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)` |
| 153 | `__aicore__ inline void KernelSortMaskMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)` |
| 173 | `__aicore__ inline void KernelSortMaskMultiCore::InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset)` |
| 193 | `__aicore__ inline void KernelSortMaskMultiCore::InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)` |
| 216 | `__aicore__ inline void KernelSortMaskMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements)` |
| 255 | `__aicore__ inline void KernelSortMaskMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)` |
| 262 | `__aicore__ inline void KernelSortMaskMultiCore::VBSProcess()` |
| 282 | `__aicore__ inline void KernelSortMaskMultiCore::VMSProcess()` |
| 318 | `__aicore__ inline void KernelSortMaskMultiCore::SortOutProcess()` |
| 333 | `__aicore__ inline void KernelSortMaskMultiCore::CopyOutValidCount()` |
| 346 | `__aicore__ inline void KernelSortMaskMultiCore::Init(GM_ADDR expert_ids, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe)` |
| 409 | `__aicore__ inline void KernelSortMaskMultiCore::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 88 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 92 | `DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);` |
| 97 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 102 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 107 | `Cast(expertIdsLocalFp32, expertIdsLocal, RoundMode::CAST_ROUND, size);` |
| 112 | `Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);` |
| 124 | `GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 127 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 138 | `Duplicate(expertIdsLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 143 | `GatherMask(rowIdsLocal, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 146 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 147 | `Sort<float, true>(outLocal, concatLocal, rowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 149 | `sortDataCopyOutQueue.EnQue<float>(outLocal);` |
| 150 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 156 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();` |
| 161 | `DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);` |
| 162 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 167 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 169 | `DataCopyPad(workspaceSortNumGm_[blockIdx_ * tilingData_->sortNumWorkSpacePerCore + progress], tempTensor,` |
| 179 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 180 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 188 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 189 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 190 | `tempBuffer.FreeTensor(outSortNumLocal);` |
| 198 | `LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();` |
| 199 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 212 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 213 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 279 | `SyncAll();` |
| 314 | `SyncAll();` |
| 330 | `SyncAll();` |
| 335 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 341 | `DataCopyPad(rsvdCntGm[0], outLocal[0], params);` |
| 343 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 381 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 383 | `SyncAll();` |
| 403 | `pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);` |
| 404 | `pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);` |
| 405 | `pipe->InitBuffer(sortedBuffer, bufferSize);` |
| 406 | `pipe->InitBuffer(tempBuffer, bufferSize);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H` |
| 83 | `static constexpr int64_t MAX_MRGSORT_LIST = 4;` |

### ffn/ffn_worker_batching/op_kernel/ffn_worker_batching.cpp


函数清单：

| 行 | 源码 |
|---|---|
| 42 | `extern "C" __aicore__ inline void FfnWorkerBatchingProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num, GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)` |
| 110 | `extern "C" __aicore__ inline void FfnWorkerBatchingRecvProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num, GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)` |
| 177 | `extern "C" __aicore__ inline void FfnWorkerBatchingRecvSpecProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num, GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 103 | `SyncAll();` |
| 124 | `SyncAll();` |
| 192 | `SyncAll();` |

常量清单：

| 行 | 源码 |
|---|---|
| 35 | `#define TILING_KEY_NORM 100` |
| 36 | `#define TILING_KEY_RECV 101` |
| 37 | `#define TILING_KEY_RECV_GATHER 102` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_sort_one_core.h


函数清单：

| 行 | 源码 |
|---|---|
| 32 | `__aicore__ inline KernelSortMaskOneCore(){};` |
| 33 | `__aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR workspace, const SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe);` |
| 35 | `__aicore__ inline void Process();` |
| 38 | `__aicore__ inline void CopyIn();` |
| 39 | `__aicore__ inline void SortCompute();` |
| 40 | `__aicore__ inline void ExpertCountCompute();` |
| 41 | `__aicore__ inline void CopyOut();` |
| 50 | `__aicore__ inline void KernelSortMaskOneCore::CopyIn()` |
| 63 | `__aicore__ inline void KernelSortMaskOneCore::SortCompute()` |
| 133 | `__aicore__ inline void KernelSortMaskOneCore::CopyOut()` |
| 148 | `__aicore__ inline void KernelSortMaskOneCore::Init(GM_ADDR expert_ids, GM_ADDR workspace, const SortCustomTilingDataKernel *tilingData, const ScheduleContextInfo *contextInfo, TPipe *tPipe)` |
| 190 | `__aicore__ inline void KernelSortMaskOneCore::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 52 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 56 | `DataCopyPad(inLocal[0], expertIdsGm, dataCopyParams, dataCopyPadParams);` |
| 60 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 65 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 67 | `Cast(expertIdsFp32, inLocal, RoundMode::CAST_ROUND, this->totalLength);` |
| 68 | `PipeBarrier<PIPE_V>();` |
| 73 | `Muls(expertIdsFp32, expertIdsFp32, (float)-1, this->totalLength);` |
| 74 | `PipeBarrier<PIPE_V>();` |
| 80 | `PipeBarrier<PIPE_V>();` |
| 87 | `GatherMask(expertIdsFp32, expertIdsFp32, maskLocalUInt32, true, this->totalLength, gatherMaskParams, rsvdCnt);` |
| 88 | `PipeBarrier<PIPE_V>();` |
| 91 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 101 | `Duplicate(expertIdsFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 102 | `PipeBarrier<PIPE_V>();` |
| 109 | `PipeBarrier<PIPE_V>();` |
| 112 | `GatherMask(rowIdsLocal, rowIdsLocal, maskLocalUInt32, true, this->totalLength, gatherMaskParams, rsvdCnt);` |
| 113 | `PipeBarrier<PIPE_V>();` |
| 115 | `Sort<float, true>(sortedLocal, concatLocal, rowIdsLocal, tempTensor, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 116 | `PipeBarrier<PIPE_V>();` |
| 118 | `LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();` |
| 121 | `Extract(sortedExpertIdsLocal, sortedRowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);` |
| 122 | `PipeBarrier<PIPE_V>();` |
| 124 | `Muls(sortedExpertIdsLocal, sortedExpertIdsLocal, (float)-1, rsvdCnt);` |
| 125 | `PipeBarrier<PIPE_V>();` |
| 127 | `Cast(sortedExpertIdsLocalInt32, sortedExpertIdsLocal, RoundMode::CAST_ROUND, rsvdCnt);` |
| 128 | `PipeBarrier<PIPE_V>();` |
| 129 | `sortDataCopyOutQueue.EnQue<float>(outLocal);` |
| 130 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 136 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 138 | `DataCopyPad(sortedexpertIdsGm, outLocal[0], copyParams);` |
| 139 | `DataCopyPad(sortedRowIdsGm, outLocal[this->sortNum], copyParams);` |
| 140 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 173 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 184 | `pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);` |
| 185 | `pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);` |
| 186 | `pipe->InitBuffer(tempBuffer, buffSize);` |
| 187 | `pipe->InitBuffer(sortedBuffer, buffSize);` |
| 197 | `SyncAll();` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SORT_ONE_CORE_H` |

### ffn/ffn_worker_batching/op_kernel/ffn_worker_batching_apt.cpp


函数清单：

| 行 | 源码 |
|---|---|
| 55 | `__aicore__ inline void WriteActualTokenNum(GM_ADDR actualTokenNum, int64_t value, TPipe *pipe)` |
| 76 | `__aicore__ inline void RunEmitAndGather(GM_ADDR sortedIdsWs, GM_ADDR gatherIdxWs, GM_ADDR group_list, GM_ADDR y, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num, int64_t totalValid, ScheduleContextInfo &ctx, const FfnWorkerBatchingTilingData *tilingData, TPipe &pipe)` |
| 108 | `__aicore__ inline void RunPrepare(GM_ADDR schedule_context, GM_ADDR flatIdsWs, GM_ADDR readyWs, ScheduleContextInfo &ctx, const FfnWorkerBatchingTilingData *tilingData, int64_t activeElements, TPipe &pipe)` |
| 149 | `__aicore__ inline int64_t RunSortPipeline(GM_ADDR flatIdsWs, GM_ADDR pairAWs, GM_ADDR pairBWs, GM_ADDR segCntWs, GM_ADDR sortedIdsWs, GM_ADDR gatherIdxWs, ScheduleContextInfo &ctx, const FfnWorkerBatchingTilingData *tilingData, int64_t activeElements, TPipe &pipe)` |
| 209 | `__aicore__ inline void FfnWorkerBatchingA5(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num, GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 63 | `pipe->InitBuffer(buf, ONE_BLK_SIZE);` |
| 66 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 68 | `DataCopyPad(outGm, local, cp);` |
| 69 | `SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 118 | `SyncAll();` |
| 127 | `SyncAll();` |
| 144 | `SyncAll();` |
| 168 | `SyncAll();` |
| 185 | `SyncAll();` |
| 203 | `SyncAll();` |

常量清单：

| 行 | 源码 |
|---|---|
| 47 | `#define TILING_KEY_NORM 100` |
| 48 | `#define TILING_KEY_RECV 101` |
| 113 | `if constexpr (isRecv) {` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_sort_base.h


函数清单：

| 行 | 源码 |
|---|---|
| 31 | `__aicore__ inline SortMaskBase(){};` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SORT_BASE_H` |
| 54 | `static constexpr int64_t SYNC_GM_NUM = 2;` |
| 55 | `static constexpr int64_t WORK_GM_NUM = 2;` |
| 56 | `static constexpr int64_t DST_BLK_STRIDE = 1;` |
| 57 | `static constexpr int64_t DST_REP_STRIDE = 8;` |
| 58 | `static constexpr int32_t INT_MAX = 2147483647;` |
| 59 | `static constexpr uint32_t BLOCK_SIZE = 32;` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_gather_out_all.h


函数清单：

| 行 | 源码 |
|---|---|
| 37 | `__aicore__ inline KernelFfnWBGatherOutAll() {}` |
| 38 | `__aicore__ inline void Init(GM_ADDR expertid_idx, GM_ADDR y, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, const ScheduleContextInfo *contextInfo, TPipe *pipe, uint32_t usedCoreNum)` |
| 117 | `__aicore__ inline void CopyInIds()` |
| 128 | `__aicore__ inline void CopyInExpertIdx(int32_t expertIdxOffset, int32_t curRows)` |
| 138 | `__aicore__ inline void Process()` |
| 209 | `__aicore__ inline void CopyAllLocalOut(int64_t allLocalOffset, int64_t copyLength)` |
| 235 | `__aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols, int64_t indicesIndex, const LocalTensor<int32_t> &dynamicScaleLocal, bool isLastBlock)` |
| 254 | `__aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 102 | `pipe->InitBuffer(inQueueX_, BUFFER_NUM, blockBufferSize);` |
| 105 | `pipe->InitBuffer(outQueALL_, BUFFER_NUM,` |
| 109 | `pipe->InitBuffer(expertIdxQue_, BUFFER_NUM, PER_LOOP_ROWS * sizeof(int32_t));` |
| 113 | `pipe->InitBuffer(tmpBuffer_, sessionNumBlockAlign_ * sizeof(int32_t) * BUFFER_NUM);` |
| 124 | `DataCopyPad(sessionIdsLocal_, sessionIdsInGm_, copyParams1, padParams1);` |
| 125 | `DataCopyPad(microBatchIdsLocal_, microBatchIdsInGm_, copyParams1, padParams1);` |
| 130 | `LocalTensor<int32_t> expertIdxLocal = expertIdxQue_.AllocTensor<int32_t>();` |
| 134 | `DataCopyPad(expertIdxLocal, expertIdxGm_[expertIdxOffset], copyParams, padParams);` |
| 135 | `expertIdxQue_.EnQue(expertIdxLocal);` |
| 164 | `LocalTensor<int32_t> expertIdxLocal = expertIdxQue_.DeQue<int32_t>();` |
| 165 | `LocalTensor<int32_t> outAllLocal = outQueALL_.AllocTensor<int32_t>();` |
| 166 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 202 | `outQueALL_.EnQue(outAllLocal);` |
| 204 | `expertIdxQue_.FreeTensor(expertIdxLocal);` |
| 211 | `LocalTensor<int32_t> outAllLocal = outQueALL_.DeQue<int32_t>();` |
| 214 | `DataCopyPad(sessionIdsOutGm_[allLocalOffset], outAllLocal, copyParams2);` |
| 215 | `DataCopyPad(microBatchIdsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_MICRO_BATCH_IDX], copyParams2);` |
| 216 | `DataCopyPad(tokenIdsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_TOKEN_IDX], copyParams2);` |
| 217 | `DataCopyPad(expertOffsetsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_EXPERT_OFFSETS_IDX],` |
| 226 | `PipeBarrier<PIPE_V>();` |
| 229 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 230 | `DataCopyPad(dynamicScaleOutGm_[allLocalOffset], dynamicScaleLocalFp32, copyParams2);` |
| 232 | `outQueALL_.FreeTensor(outAllLocal);` |
| 238 | `LocalTensor<int8_t> xLocal = inQueueX_.AllocTensor<int8_t>();` |
| 242 | `DataCopyPad(xLocal, tokenDataBufGm_[xSrcOffset], copyParams0, padParams0);` |
| 248 | `DataCopyPad(dynScaleT, tokenDataBufGm_[xSrcOffset + copySize], copyParams1, padParams0);` |
| 251 | `inQueueX_.EnQue(xLocal);` |
| 256 | `LocalTensor<int8_t> xLocal = inQueueX_.DeQue<int8_t>();` |
| 259 | `DataCopyPad(yOutGm_[xDstOffset], xLocal, copyParams2);` |
| 261 | `inQueueX_.FreeTensor(xLocal);` |

常量清单：

| 行 | 源码 |
|---|---|
| 23 | `#define OP_KERNEL_FFN_WB_GATHER_OUT_ALL_H` |
| 112 | `if constexpr (isScanFlag == false) {` |
| 145 | `if constexpr (isScanFlag == false) {` |
| 175 | `if constexpr (isScanFlag == false) {` |
| 265 | `static constexpr uint32_t BLOCK_SIZE = 32;` |
| 266 | `static constexpr int32_t BUFFER_NUM = 2;      // tensor num for each queue` |
| 267 | `static constexpr int32_t PER_LOOP_ROWS = 128; // tensor num for each queue` |
| 268 | `static constexpr int64_t TOKEN_KIND_TWO = 2;` |
| 270 | `static constexpr int32_t VAR_NUM = 5; // session_ids, micro_batch_ids, token_ids, expert_offsets` |
| 271 | `static constexpr int32_t VAR_SESSION_IDX = 0;` |
| 272 | `static constexpr int32_t VAR_MICRO_BATCH_IDX = 1;` |
| 273 | `static constexpr int32_t VAR_TOKEN_IDX = 2;` |
| 274 | `static constexpr int32_t VAR_EXPERT_OFFSETS_IDX = 3;` |
| 275 | `static constexpr int32_t VAR_DYNAMIC_SCALE = 4;` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_sort_mrgsort.h


函数清单：

| 行 | 源码 |
|---|---|
| 38 | `__aicore__ inline SortCustomMrgsort(){};` |
| 39 | `__aicore__ inline void Init(SortCustomMrgsortParam *param);` |
| 40 | `__aicore__ inline void Process();` |
| 41 | `__aicore__ inline void SetInput(GlobalTensor<float> &gmInput, GlobalTensor<int32_t> &gmSortNum, LocalTensor<float> &ubInput);` |
| 43 | `__aicore__ inline void SetOutput(GlobalTensor<float> &gmOutput, LocalTensor<float> &ubOutput, LocalTensor<int32_t> &outSortNumLocal);` |
| 47 | `__aicore__ inline void CopyOutSortNum();` |
| 48 | `__aicore__ inline void CopyIn();` |
| 49 | `__aicore__ inline void UpdateMrgParam();` |
| 50 | `__aicore__ inline void MrgsortCompute();` |
| 51 | `__aicore__ inline void UpdateSortInfo();` |
| 52 | `__aicore__ inline void CopyOut();` |
| 53 | `__aicore__ inline void ClearCache();` |
| 81 | `__aicore__ inline void SortCustomMrgsort::ClearCache()` |
| 88 | `__aicore__ inline void SortCustomMrgsort::SetInput(GlobalTensor<float> &gmInput, GlobalTensor<int32_t> &gmSortNum, LocalTensor<float> &ubInput)` |
| 97 | `__aicore__ inline void SortCustomMrgsort::SetOutput(GlobalTensor<float> &gmOutput, LocalTensor<float> &ubOutput, LocalTensor<int32_t> &outSortNumLocal)` |
| 105 | `__aicore__ inline void SortCustomMrgsort::UpdateMrgParam()` |
| 121 | `__aicore__ inline void SortCustomMrgsort::CopyIn()` |
| 138 | `__aicore__ inline void SortCustomMrgsort::MrgsortCompute()` |
| 161 | `__aicore__ inline void SortCustomMrgsort::UpdateSortInfo()` |
| 178 | `__aicore__ inline void SortCustomMrgsort::CopyOut()` |
| 190 | `__aicore__ inline void SortCustomMrgsort::CopyOutSortNum()` |
| 198 | `__aicore__ inline void SortCustomMrgsort::Init(SortCustomMrgsortParam *param)` |
| 211 | `__aicore__ inline void SortCustomMrgsort::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 124 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 128 | `DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]],` |
| 140 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 143 | `MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);` |
| 147 | `MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);` |
| 151 | `MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);` |
| 154 | `DataCopy(this->ubOutput, this->tmpUbInputs[0],` |
| 184 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 185 | `DataCopyPad(this->gmOutput[outOffset], this->ubOutput, intriParams);` |
| 193 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 195 | `DataCopyPad(gmActualSortNum_, ubOutSortNum_, copyParams);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SORT_MRGSORT_H` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_scan_group_listing_one_core.h


函数清单：

| 行 | 源码 |
|---|---|
| 33 | `__aicore__ inline KernelFfnWBScanGroupListingOneCore(){};` |
| 34 | `__aicore__ inline void Init(GM_ADDR x, GM_ADDR groupList, ScheduleContextInfo *scheduleContext, TPipe *tPipe)` |
| 59 | `__aicore__ inline void Process()` |
| 97 | `__aicore__ inline void CopyIn(int64_t progress, int64_t inputNum)` |
| 107 | `__aicore__ inline void Compute(int64_t inputNum, int64_t curLoop)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 52 | `pipe->InitBuffer(copyInQueue, 1, perLoopRows * sizeof(int32_t));` |
| 53 | `pipe->InitBuffer(copyOutQueue, 1,` |
| 55 | `pipe->InitBuffer(maskBuf, perLoopRows * sizeof(int32_t));` |
| 56 | `pipe->InitBuffer(expertIdsBuf, perLoopRows * sizeof(int32_t));` |
| 62 | `groupListOutLocal = copyOutQueue.AllocTensor<int64_t>();` |
| 88 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 89 | `DataCopyPad(groupListGm[totalOutExpertIdNum * NUM_TWO], groupListOutLocal, copyParams);` |
| 92 | `copyOutQueue.FreeTensor(groupListOutLocal);` |
| 99 | `LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();` |
| 103 | `DataCopyPad(inLocal, expandedExpertIdsGm[progress * perLoopRows], dataCopyParams, dataCopyPadParams);` |
| 104 | `copyInQueue.EnQue<int32_t>(inLocal);` |
| 113 | `LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();` |
| 116 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 128 | `SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);` |
| 134 | `PipeBarrier<PIPE_V>();` |
| 141 | `GatherMask(expertIdsBufFp32, inLocal, maskLocalUInt32, true, inputNum, gatherMaskParams, rsvdCnt);` |
| 158 | `copyInQueue.FreeTensor(inLocal);` |

常量清单：

| 行 | 源码 |
|---|---|
| 23 | `#define OP_KERNEL_FFN_WB_SCAN_GROUP_LISTING_ONE_CORE_H` |

### ffn/ffn_worker_batching/op_kernel/ffn_wb_scan_get_valid_experts.h


函数清单：

| 行 | 源码 |
|---|---|
| 32 | `__aicore__ inline KernelScanGetValidExperts(){};` |
| 33 | `__aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, GM_ADDR groupList, const ScheduleContextInfo *contextInfo, TPipe *tPipe);` |
| 35 | `__aicore__ inline void Process();` |
| 38 | `__aicore__ inline void VBSProcess();` |
| 39 | `__aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t loopSessionCnt);` |
| 41 | `__aicore__ inline void GatherOutProcess();` |
| 42 | `__aicore__ inline void VBSCopyInAndClear(int64_t progress, int64_t size, int64_t loopSessionCnt);` |
| 43 | `__aicore__ inline void UBSortCompute(int64_t progress, int64_t size);` |
| 44 | `__aicore__ inline void VBSCopyOut(int64_t progress, int64_t size);` |
| 46 | `__aicore__ inline void CopyOutValidCount();` |
| 47 | `__aicore__ inline void ClearTokenInfoFlag();` |
| 81 | `__aicore__ inline void KernelScanGetValidExperts::VBSCopyInAndClear(int64_t progress, int64_t size, int64_t loopSessionCnt)` |
| 117 | `__aicore__ inline void KernelScanGetValidExperts::UBSortCompute(int64_t progress, int64_t size)` |
| 167 | `__aicore__ inline void KernelScanGetValidExperts::VBSCopyOut(int64_t progress, int64_t size)` |
| 181 | `__aicore__ inline void KernelScanGetValidExperts::UBSortProcess(int64_t progress, int64_t size, int64_t loopSessionCnt)` |
| 188 | `__aicore__ inline void KernelScanGetValidExperts::VBSProcess()` |
| 206 | `__aicore__ inline void KernelScanGetValidExperts::GatherOutProcess()` |
| 326 | `__aicore__ inline void KernelScanGetValidExperts::CopyOutValidCount()` |
| 347 | `__aicore__ inline void KernelScanGetValidExperts::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, GM_ADDR groupList, const ScheduleContextInfo *contextInfo, TPipe *tPipe)` |
| 422 | `__aicore__ inline void KernelScanGetValidExperts::ClearTokenInfoFlag()` |
| 449 | `__aicore__ inline void KernelScanGetValidExperts::Process()` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 84 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 92 | `DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);` |
| 100 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 103 | `LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 104 | `Duplicate<int32_t>(clearLocal, INT_MAX, size);` |
| 105 | `sortDataCopyOutQueue.EnQue(clearLocal);` |
| 107 | `clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 112 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 113 | `DataCopyPad(expertIdsGm[inOffset], clearLocal, copyoutParams);` |
| 114 | `sortDataCopyOutQueue.FreeTensor(clearLocal);` |
| 119 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 122 | `Cast(expertIdsLocalFp32, inLocal, RoundMode::CAST_ROUND, size);` |
| 123 | `PipeBarrier<PIPE_V>();` |
| 129 | `Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);` |
| 130 | `PipeBarrier<PIPE_V>();` |
| 136 | `PipeBarrier<PIPE_V>();` |
| 143 | `GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);` |
| 144 | `PipeBarrier<PIPE_V>();` |
| 147 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 151 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 152 | `Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, rsvdCnt);` |
| 153 | `PipeBarrier<PIPE_V>();` |
| 154 | `Cast(outLocal, expertIdsLocalFp32, RoundMode::CAST_ROUND, rsvdCnt);` |
| 161 | `GatherMask(expertIdx, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt1);` |
| 163 | `sortDataCopyOutQueue.EnQue<int32_t>(outLocal);` |
| 164 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 170 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 174 | `DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);` |
| 176 | `DataCopyPad(workspaceGms[1][wkOffset], outLocal[interVal], copyParams);` |
| 177 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 203 | `SyncAll();` |
| 209 | `LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();` |
| 226 | `DataCopyPad(inLocal[inOffset], workspaceGms[0][i * GetSortLen<float>(perCoreSortNum_)], dataCopyParams,` |
| 228 | `DataCopyPad(inLocal[interVal + inOffset], workspaceGms[1][i * GetSortLen<float>(perCoreSortNum_)],` |
| 235 | `sortDataCopyInQueue.EnQue(inLocal);` |
| 236 | `inLocal = sortDataCopyInQueue.DeQue<int32_t>();` |
| 239 | `Cast(expertIdsLocalFp32, inLocal, RoundMode::CAST_ROUND, inOffset);` |
| 240 | `PipeBarrier<PIPE_V>();` |
| 248 | `PipeBarrier<PIPE_V>();` |
| 255 | `GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalUInt32, true, inOffset, gatherMaskParams, rsvdCnt);` |
| 256 | `GatherMask(inLocal[interVal], inLocal[interVal], maskLocalUInt32, true, inOffset, gatherMaskParams, rsvdCnt);` |
| 257 | `PipeBarrier<PIPE_V>();` |
| 261 | `PipeBarrier<PIPE_V>();` |
| 262 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |
| 266 | `Cast(inLocal, expertIdsLocalFp32, RoundMode::CAST_ROUND, rsvdCnt);` |
| 267 | `PipeBarrier<PIPE_V>();` |
| 274 | `Duplicate(inLocal[duplicateIndex], INT_MAX, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 275 | `Duplicate(inLocal[interVal + duplicateIndex], INT_MAX, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 276 | `PipeBarrier<PIPE_V>();` |
| 285 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 289 | `PipeBarrier<PIPE_V>();` |
| 290 | `GatherMask(outLocal, inLocal, maskLocalUInt32, true, rsvdCnt, gatherMaskParams, rsvdCnt2);` |
| 291 | `GatherMask(outLocal[interVal], inLocal[interVal], maskLocalUInt32, true, rsvdCnt, gatherMaskParams,` |
| 293 | `PipeBarrier<PIPE_V>();` |
| 294 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |
| 299 | `sortDataCopyOutQueue.EnQue(outLocal);` |
| 300 | `outLocal = sortDataCopyOutQueue.DeQue<int32_t>();` |
| 304 | `DataCopyPad(sortedexpertIdsGm[outOffset], outLocal, dataCopyParams);` |
| 305 | `DataCopyPad(sortedRowIdsGm[outOffset], outLocal[interVal], dataCopyParams);` |
| 307 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 318 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 319 | `DataCopyPad(groupListGm_, groupListOutLocal, copyParams);` |
| 321 | `sortDataCopyInQueue.FreeTensor(inLocal);` |
| 323 | `SyncAll();` |
| 331 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 333 | `DataCopyPad(workspaceSortNumGm_[blockIdx_ * sortNumWorkSpacePerCore_], tempTensor, copyParams1);` |
| 335 | `LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();` |
| 340 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 342 | `DataCopyPad(rsvdCntGm[0], outLocal[0], params);` |
| 344 | `sortDataCopyOutQueue.FreeTensor(outLocal);` |
| 416 | `pipe->InitBuffer(sortDataCopyInQueue, 1, bufferSize_);` |
| 417 | `pipe->InitBuffer(sortDataCopyOutQueue, 1, bufferSize_);` |
| 418 | `pipe->InitBuffer(sortedBuffer, bufferSize_);` |
| 419 | `pipe->InitBuffer(tempBuffer, bufferSize_);` |
| 432 | `Duplicate<int32_t>(clearLocal, 0, duplicateNum);` |
| 434 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 444 | `DataCopyPad(expertIdsGmFStart_[idx * perLoopANum * contextInfo_->M * F_], clearLocal, copyOutParams);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_FFN_WB_SCAN_GET_VALID_EXPERTS_H` |

### ffn/ffn_worker_batching/op_graph/ffn_worker_batching_proto.h


常量清单：

| 行 | 源码 |
|---|---|
| 16 | `#define OP_GRAPH_FFN_WORKER_BATCHING_PROTO_H` |
| 57 | `#define OPS_PROTO_DEF_FFNWORKERBATCHING` |

### ffn/ffn_worker_batching/op_host/ffn_worker_batching_infershape.cpp


函数清单：

| 行 | 源码 |
|---|---|
| 52 | `static graphStatus InferShape4FfnWorkerBatching(gert::InferShapeContext *context)` |
| 145 | `static graphStatus InferDataType4FfnWorkerBatching(gert::InferDataTypeContext *context)` |

常量清单：

| 行 | 源码 |
|---|---|
| 23 | `static constexpr int64_t EXPERT_NUM_ATTR = 0;` |
| 24 | `static constexpr int64_t MAX_OUT_SHAPE_ATTR = 1;` |
| 25 | `static constexpr int64_t TOKEN_DTYPE_ATTR = 2;` |
| 27 | `static constexpr int64_t Y_OUT = 0;` |
| 28 | `static constexpr int64_t GROUP_LIST_OUT = 1;` |
| 29 | `static constexpr int64_t SESSION_IDS_OUT = 2;` |
| 30 | `static constexpr int64_t MICRO_BATCH_IDS_OUT = 3;` |
| 31 | `static constexpr int64_t TOKEN_IDS_OUT = 4;` |
| 32 | `static constexpr int64_t EXPERT_OFFSETS_OUT = 5;` |
| 33 | `static constexpr int64_t DYNAMIC_SCALE_OUT = 6;` |
| 34 | `static constexpr int64_t ACTUAL_TOKEN_NUM_OUT = 7;` |
| 36 | `static constexpr int64_t TOKEN_KIND_ZERO = 0;` |
| 37 | `static constexpr int64_t TOKEN_KIND_ONE = 1;` |
| 38 | `static constexpr int64_t TOKEN_KIND_TWO = 2;` |
| 40 | `static constexpr int64_t NUM_FOUR = 4;` |
| 41 | `static constexpr uint32_t INDEX_ZERO = 0;` |
| 42 | `static constexpr uint32_t INDEX_ONE = 1;` |
| 43 | `static constexpr uint32_t INDEX_TWO = 2;` |
| 44 | `static constexpr uint32_t INDEX_THREE = 3;` |
| 46 | `static constexpr int64_t EVEN_ALIGN = 2;` |
| 47 | `static constexpr int64_t TOKEN_DTYPE_E5M2 = 3;` |
| 48 | `static constexpr int64_t TOKEN_DTYPE_E4M3 = 4;` |
| 49 | `static constexpr int64_t TOKEN_DTYPE_E2M1 = 5;` |
| 50 | `static constexpr int64_t MXFP_SCALE_GROUP = 32;` |

### ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling.cpp


函数清单：

| 行 | 源码 |
|---|---|
| 46 | `ge::graphStatus FfnWorkerBatchingTiling::SetBatchingWorkspaceSize(int64_t expertNum)` |
| 63 | `ge::graphStatus FfnWorkerBatchingTiling::GetPlatformInfo()` |
| 85 | `ge::graphStatus FfnWorkerBatchingTiling::CheckInputParam()` |
| 104 | `ge::graphStatus FfnWorkerBatchingTiling::GetAttrsInfo()` |
| 173 | `void FfnWorkerBatchingTiling::SetBatchingTilingKey(int64_t sortLoopMaxElement)` |
| 184 | `ge::graphStatus FfnWorkerBatchingTiling::RunFfnWorkerBatchingTiling()` |

常量清单：

| 行 | 源码 |
|---|---|
| 20 | `constexpr uint32_t EXPERT_NUM_ATTR = 0;` |
| 21 | `constexpr uint32_t MAX_OUT_SHAPE_ATTR = 1;` |
| 22 | `constexpr uint32_t TOKEN_DTYPE_ATTR = 2;` |
| 23 | `constexpr uint32_t NEED_SCHEDULE_ATTR = 3;` |
| 24 | `constexpr uint32_t LAY_NUM_ATTR = 4;` |
| 25 | `constexpr int64_t X_SHAPE_VALUE = 1024;` |
| 27 | `constexpr uint32_t INDEX_ZERO = 0;` |
| 28 | `constexpr uint32_t INDEX_ONE = 1;` |
| 29 | `constexpr uint32_t INDEX_TWO = 2;` |
| 30 | `constexpr uint32_t INDEX_THREE = 3;` |
| 31 | `constexpr uint32_t BATCH_MODE = 1;` |
| 32 | `constexpr int64_t ONE_REPEAT_SORT_NUM = 32;` |
| 33 | `constexpr int64_t MAX_RESERVE_WK_NUM = 128;` |
| 35 | `constexpr int64_t TILING_KEY_NORM = 100;` |
| 36 | `constexpr int64_t TILING_KEY_RECV = 101;` |
| 38 | `constexpr int64_t NUM_TWO = 2;` |
| 39 | `constexpr int64_t NUM_FOUR = 4;` |
| 40 | `constexpr int64_t EXPERT_IDX_MAX = 8192;` |
| 41 | `constexpr int64_t MAX_SESSION_NUM = 1024;` |
| 42 | `constexpr int64_t MAX_K_NUM = 64;` |
| 43 | `constexpr int64_t TH_RECV_CORE_NUM = 32;` |
| 44 | `constexpr int64_t TH_RECV_Y_NUM = 200000;` |

### ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling.h


常量清单：

| 行 | 源码 |
|---|---|
| 16 | `#define OP_HOST_FFN_WORKER_BATCHING_TILING_H` |

### ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling_arch35.cpp


函数清单：

| 行 | 源码 |
|---|---|
| 192 | `ge::graphStatus FfnWorkerBatchingTilingArch35::GetPlatformInfo()` |
| 211 | `ge::graphStatus FfnWorkerBatchingTilingArch35::CheckInputParam()` |
| 234 | `ge::graphStatus FfnWorkerBatchingTilingArch35::GetAttrsInfo()` |
| 256 | `ge::graphStatus FfnWorkerBatchingTilingArch35::ParseMaxOutShape(const gert::RuntimeAttrs *attrs)` |
| 296 | `ge::graphStatus FfnWorkerBatchingTilingArch35::ParseOptionalAttrs(const gert::RuntimeAttrs *attrs, int64_t expertNum)` |
| 365 | `ge::graphStatus FfnWorkerBatchingTilingArch35::GetShapeAttrsInfo()` |
| 374 | `ge::graphStatus FfnWorkerBatchingTilingArch35::DoOpTiling()` |
| 443 | `void FfnWorkerBatchingTilingArch35::SplitPrepare()` |
| 469 | `void FfnWorkerBatchingTilingArch35::SplitSortAndMerge()` |
| 535 | `void FfnWorkerBatchingTilingArch35::SplitGroupList()` |
| 547 | `void FfnWorkerBatchingTilingArch35::LayoutWorkspace()` |
| 583 | `ge::graphStatus FfnWorkerBatchingTilingArch35::DoLibApiTiling()` |
| 588 | `uint64_t FfnWorkerBatchingTilingArch35::GetTilingKey() const` |
| 594 | `ge::graphStatus FfnWorkerBatchingTilingArch35::GetWorkspaceSize()` |
| 602 | `ge::graphStatus FfnWorkerBatchingTilingArch35::PostTiling()` |

常量清单：

| 行 | 源码 |
|---|---|
| 33 | `constexpr int64_t COMPARE_FP32_ELEMENTS = 256 / static_cast<int64_t>(sizeof(float)); // 一个向量 repeat 的 fp32 元素数` |
| 34 | `constexpr int64_t TOKEN_STRIDE_ALIGNMENT = 512;` |
| 35 | `constexpr int64_t MX_SCALE_BLOCK_ELEMENTS = 32;` |
| 36 | `constexpr int64_t FP4_VALUES_PER_BYTE = 2;` |
| 37 | `constexpr int64_t EXTRACT_MAX_REPEATS = 255; // Extract API 的 repeatTime 上限` |
| 40 | `constexpr int64_t TOKEN_FP16 = 0;` |
| 41 | `constexpr int64_t TOKEN_BF16 = 1;` |
| 42 | `constexpr int64_t TOKEN_INT8 = 2;` |
| 43 | `constexpr int64_t TOKEN_FP8_E5M2 = 3;` |
| 44 | `constexpr int64_t TOKEN_FP8_E4M3 = 4;` |
| 45 | `constexpr int64_t TOKEN_FP4_E2M1 = 5;` |
| 48 | `constexpr int64_t GATHER_QUEUE_DEPTH = 2;` |
| 49 | `constexpr int64_t GATHER_ROWS_PER_LOOP = 128;` |
| 50 | `constexpr int64_t GATHER_OUTPUT_LANES = 5;      // 四路索引 + dynamic scale` |
| 51 | `constexpr int64_t GATHER_FLOAT_INDEX_LANES = 7; // src, qa, a, remainder, qb, b, k` |
| 52 | `constexpr int64_t GATHER_INT_INDEX_LANES = 3;   // a, b, k` |
| 54 | `constexpr int64_t GROUP_VECTOR_BUFFERS = 10; // cur/prev/next/idx/diff/diffF 与四路紧凑数组` |
| 55 | `constexpr int64_t GROUP_MASK_BUFFERS = 2;    // run 起止` |
| 56 | `constexpr int64_t GROUP_ROW_FIELDS = 2;      // expert ID 与 token 数(int64)` |
| 57 | `constexpr int64_t GROUP_MIN_ROWS = 2;` |
| 58 | `constexpr int64_t BITS_PER_BYTE = 8;` |
| 59 | `constexpr int64_t RECV_WAITER_FLAG_BUFFERS = 2; // int32 flag 与 FP32 归约暂存` |
| 62 | `constexpr int64_t SORT_BUFFER_COUNT = 5;            // 输入、concat、临时、排序结果、掩码` |
| 63 | `constexpr int64_t MERGE_BUFFER_COUNT = 3;           // 输入、输出、计数缓存` |
| 64 | `constexpr int64_t EXTRACT_BUFFER_COUNT = 3;         // proposal、expert ID、源下标` |
| 65 | `constexpr int64_t MERGE_COUNT_CACHE_SEGMENTS = 128; // 有界缓存策略,非硬件上限` |
| 68 | `constexpr uint32_t EXPERT_NUM_ATTR = 0;` |
| 69 | `constexpr uint32_t MAX_OUT_SHAPE_ATTR = 1;` |
| 70 | `constexpr uint32_t TOKEN_DTYPE_ATTR = 2;` |
| 71 | `constexpr uint32_t NEED_SCHEDULE_ATTR = 3;` |
| 72 | `constexpr uint32_t LAY_NUM_ATTR = 4;` |
| 74 | `constexpr uint32_t SYNC_FLAG_ATTR = 5;` |
| 76 | `constexpr uint32_t INDEX_ZERO = 0;` |
| 77 | `constexpr uint32_t INDEX_ONE = 1;` |
| 78 | `constexpr uint32_t INDEX_TWO = 2;` |
| 79 | `constexpr uint32_t INDEX_THREE = 3;` |
| 81 | `constexpr int64_t BATCH_MODE = 1;` |
| 82 | `constexpr int64_t ONE_REPEAT_SORT_NUM = 32;` |
| 83 | `constexpr int64_t MAX_RESERVE_WK_NUM = 128;` |
| 85 | `constexpr int64_t TILING_KEY_NORM = 100;` |
| 86 | `constexpr int64_t TILING_KEY_RECV = 101;` |
| 88 | `constexpr int64_t NUM_TWO = 2;` |
| 89 | `constexpr int64_t NUM_FOUR = 4;` |
| 90 | `constexpr int64_t NUM_EIGHT = 8;` |
| 91 | `constexpr int64_t GL_ROW_BYTES = static_cast<int64_t>(sizeof(int64_t)) * NUM_TWO; // 一行 = [expert_id, tokenNum]` |
| 92 | `constexpr int64_t GL_UB_FRACTION = 16;                                            // group_list 拼装区取 UB 的 1/16` |
| 95 | `constexpr int64_t ONE_BLK_BYTES = 32;` |
| 97 | `constexpr int64_t MRG_LIST_NUM = 4;` |
| 100 | `constexpr int64_t EXPERT_ID_MASK_START = 1000000;` |
| 103 | `constexpr int64_t SORT_PAIR_FLOATS =` |
| 107 | `constexpr int64_t SORT_PAIR_REGIONS = 3;` |
| 108 | `constexpr int64_t SORT_UB_BYTES_PER_ELEM = static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO +` |
| 112 | `constexpr int64_t EXTRACT_UB_BYTES_PER_ELEM =` |
| 114 | `constexpr int64_t EXPERT_IDX_MAX = 8192;` |
| 115 | `constexpr int64_t MAX_SESSION_NUM = 1024;` |
| 116 | `constexpr int64_t MAX_K_NUM = 64;` |
| 117 | `constexpr int64_t CONTEXT_BYTES = sizeof(aicpu::ScheduleContext);` |
| 118 | `constexpr int64_t INT32_PER_BLOCK = ONE_BLK_BYTES / sizeof(int32_t);` |
| 119 | `constexpr int64_t SORT_FIXED_BYTES = Arch35Config::SORT_BUFFER_COUNT * ONE_BLK_BYTES;` |
| 120 | `constexpr int64_t MERGE_FIXED_BYTES = Arch35Config::MERGE_BUFFER_COUNT * ONE_BLK_BYTES;` |
| 121 | `constexpr int64_t EXTRACT_FIXED_BYTES = Arch35Config::EXTRACT_BUFFER_COUNT * ONE_BLK_BYTES;` |
| 356 | `constexpr int64_t maxAlignedStride =` |
| 478 | `constexpr int64_t SORT_COMPARE_ELEMENTS = Arch35Config::COMPARE_FP32_ELEMENTS;` |

### ffn/ffn_worker_batching/op_host/ffn_worker_batching_def.cpp


### ffn/ffn_worker_batching/op_host/op_api/aclnn_ffn_worker_batching_v2.h


常量清单：

| 行 | 源码 |
|---|---|
| 12 | `#define OP_API_INC_FFN_WORKER_BATCHING_V2_H_` |

### ffn/ffn_worker_batching/op_host/op_api/aclnn_ffn_worker_batching_v2.cpp


### ffn/ffn_worker_batching/op_host/op_api/aclnn_ffn_worker_batching.cpp


常量清单：

| 行 | 源码 |
|---|---|
| 23 | `constexpr bool SYNC_FLAG_WAIT_ALL = false;` |

### ffn/ffn_worker_batching/op_host/op_api/aclnn_ffn_worker_batching.h


常量清单：

| 行 | 源码 |
|---|---|
| 12 | `#define OP_API_INC_FFN_WORKER_BATCHING_H_` |

### ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_prepare.h


函数清单：

| 行 | 源码 |
|---|---|
| 36 | `__aicore__ inline FfnWbA5RecvWait(){};` |
| 38 | `__aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR tokenInfoBuf, const ScheduleContextInfo *ctx, TPipe *pipe, GM_ADDR readyWs)` |
| 58 | `__aicore__ inline void Process()` |
| 148 | `__aicore__ inline void LoadSelectedMicroBatch(GM_ADDR readyWs, ScheduleContextInfo &ctx, TPipe &pipe)` |
| 172 | `__aicore__ inline FfnWbPrepareArch35(){};` |
| 176 | `__aicore__ inline void Init(GM_ADDR flatIdsWs, const ScheduleContextInfo *contextInfo, TPipe *pipe, int64_t totalLen, int64_t rowsPerLoop, int64_t perLoopElements, int64_t activeRows, GM_ADDR readyWs = nullptr)` |
| 220 | `__aicore__ inline void ProcessNorm(GM_ADDR expertIdsBuf)` |
| 264 | `__aicore__ inline void ProcessRecv(GM_ADDR tokenInfoBuf)` |
| 342 | `__aicore__ inline int32_t ReadLayerOffset(GlobalTensor<int32_t> &info, int64_t descriptorBase)` |
| 359 | `__aicore__ inline void EncodeLayerExperts(const LocalTensor<int32_t> &buf, int64_t count, int32_t offset)` |
| 390 | `__aicore__ inline void ProcessRecvLargeRow(GM_ADDR tokenInfoBuf)` |
| 445 | `__aicore__ inline bool IsSelected(int64_t row)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 52 | `pipe_->InitBuffer(flagQue_, 1, flagWords * sizeof(int32_t));` |
| 53 | `pipe_->InitBuffer(workBuf_, flagWords * sizeof(float));` |
| 54 | `pipe_->InitBuffer(pollBuf_, ONE_BLK_SIZE);` |
| 69 | `LocalTensor<int32_t> flagLocal = flagQue_.AllocTensor<int32_t>();` |
| 76 | `DataCopyPad(flagLocal, tokenInfoGm_[selectedMb * descWords_], cp, pad);` |
| 77 | `flagQue_.EnQue(flagLocal);` |
| 79 | `LocalTensor<int32_t> flags = flagQue_.DeQue<int32_t>();` |
| 81 | `Cast(work, flags, RoundMode::CAST_ROUND, flagElems);` |
| 82 | `PipeBarrier<PIPE_V>();` |
| 83 | `ReduceSum<float>(work, work, work, flagElems);` |
| 84 | `PipeBarrier<PIPE_V>();` |
| 85 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |
| 98 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 99 | `DataCopyPad(readyGm_[ONE_BLK_SIZE / sizeof(int32_t)], flags, snapshot);` |
| 100 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 102 | `flagQue_.FreeTensor(flags);` |
| 119 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 121 | `DataCopyPad(readyGm_, pollLocal.ReinterpretCast<int32_t>(), selected);` |
| 122 | `SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 125 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 127 | `DataCopyPad(ctxGm_[FFN_WB_CTX_OFFSET(ffn.polling_index) / static_cast<int32_t>(sizeof(uint64_t))], pollLocal,` |
| 129 | `SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 156 | `pipe.InitBuffer(selectionBuf, ONE_BLK_SIZE);` |
| 160 | `DataCopyPad(mb, selection, cp, pad);` |
| 161 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 192 | `pipe_->InitBuffer(readyBuf_, ONE_BLK_SIZE);` |
| 214 | `pipe_->InitBuffer(que_, 1, Align(perLoopElements_ * sizeof(int32_t), BLOCK_BYTES));` |
| 215 | `pipe_->InitBuffer(clrBuf_, Align(rowsPerLoop_ * BLOCK_BYTES, BLOCK_BYTES));` |
| 239 | `LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();` |
| 243 | `DataCopyPad(buf, srcGm[off], cp, pad);` |
| 244 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 246 | `Duplicate<int32_t>(buf, MASK_SENTINEL, n);` |
| 247 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 249 | `DataCopyPad(flatIdsGm_[off], buf, cp);` |
| 251 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 252 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 253 | `que_.FreeTensor(buf);` |
| 282 | `LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();` |
| 288 | `Duplicate<int32_t>(buf, MASK_SENTINEL, rowSpan_);` |
| 289 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 291 | `DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, masked);` |
| 292 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 293 | `que_.FreeTensor(buf);` |
| 306 | `DataCopyPad(buf, tokenInfoGm[base + r0 * rowStride + NUM_TWO], inParams, inPad);` |
| 310 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 315 | `DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, outParams);` |
| 318 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 325 | `Duplicate<int32_t>(buf, MASK_SENTINEL, rows * rowSpan_);` |
| 326 | `Duplicate<int32_t>(clr, 0, rows * (BLOCK_BYTES / static_cast<int64_t>(sizeof(int32_t))));` |
| 327 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 332 | `DataCopyPad(tokenInfoGm[base + r0 * rowStride + NUM_TWO], buf, clrIds);` |
| 333 | `DataCopyPad(tokenInfoGm[base + r0 * rowStride], clr, clrFlag);` |
| 334 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 335 | `que_.FreeTensor(buf);` |
| 346 | `SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);` |
| 349 | `DataCopyPad(layer, info[descriptorBase + LAYER_WORD], cp, pad);` |
| 350 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 364 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 377 | `Reg::DataCopy(ids, addr + i * LANES);` |
| 380 | `Reg::Duplicate(masked, static_cast<uint32_t>(MASK_SENTINEL), active);` |
| 382 | `Reg::DataCopy(addr + i * LANES, result, active);` |
| 385 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 404 | `LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();` |
| 408 | `DataCopyPad(buf, info[base + NUM_TWO + off], cp, pad);` |
| 412 | `SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);` |
| 415 | `Duplicate<int32_t>(buf, MASK_SENTINEL, n);` |
| 416 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 419 | `DataCopyPad(flatIdsGm_[row * rowSpan_ + off], buf, out);` |
| 420 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 422 | `Duplicate<int32_t>(buf, MASK_SENTINEL, n);` |
| 423 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 425 | `DataCopyPad(info[base + NUM_TWO + off], buf, clear);` |
| 427 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 428 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 429 | `que_.FreeTensor(buf);` |
| 433 | `Duplicate<int32_t>(flag, 0, BLOCK_BYTES / sizeof(int32_t));` |
| 434 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 436 | `DataCopyPad(info[base], flag, clearFlag);` |
| 437 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 450 | `DataCopyPad(flag, readyGm_[(row + 1) * (ONE_BLK_SIZE / sizeof(int32_t))], cp, pad);` |
| 451 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H` |
| 168 | `constexpr int32_t MASK_SENTINEL = INT32_MAX;` |
| 344 | `constexpr int64_t LAYER_WORD = offsetof(aicpu::FfnDataDesc, layer_id) / sizeof(int32_t);` |
| 365 | `constexpr uint32_t LANES = GetVecLen() / sizeof(uint32_t);` |

### ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_context.h


函数清单：

| 行 | 源码 |
|---|---|
| 90 | `__aicore__ inline void SetWaitFlag(HardEvent evt)` |
| 97 | `__aicore__ inline int64_t Ceil(int64_t a, int64_t b)` |
| 103 | `__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)` |
| 115 | `__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)` |
| 121 | `__aicore__ inline T Min(T a, T b)` |
| 127 | `__aicore__ inline T Max(T a, T b)` |
| 140 | `__aicore__ inline int64_t GetFlatAivIdx()` |
| 148 | `__aicore__ inline int64_t GetFlatAivNum()` |
| 156 | `__aicore__ inline int64_t GetSortLen(int64_t count)` |
| 166 | `__aicore__ inline void ScheduleContextParse(GM_ADDR schedule_context, const TilingT *tilingData, ScheduleContextInfo &ctx, TPipe *pipe)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 90 | `__aicore__ inline void SetWaitFlag(HardEvent evt)` |
| 172 | `pipe->InitBuffer(buffer, SCHEDULE_CONTEXT_BYTES);` |
| 174 | `DataCopy(val, ctxGm, SCHEDULE_CONTEXT_BYTES);` |
| 175 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |

常量清单：

| 行 | 源码 |
|---|---|
| 32 | `#define OP_KERNEL_ARCH35_FFN_WB_A5_CONTEXT_H` |
| 40 | `#define FFN_WB_CTX_OFFSET(field) static_cast<int32_t>(__builtin_offsetof(aicpu::ScheduleContext, field))` |
| 42 | `constexpr int64_t SCHEDULE_CONTEXT_BYTES = static_cast<int64_t>(sizeof(aicpu::ScheduleContext));` |
| 45 | `constexpr int64_t SORT_PAIR_BYTES = static_cast<int64_t>(sizeof(float) + sizeof(uint32_t));` |
| 47 | `constexpr float SORT_FILL_VALUE = -__FLT_MAX__;` |
| 53 | `constexpr int64_t ONE_REPEAT_SORT_NUM = 32;` |
| 54 | `constexpr int64_t MRG_LIST_NUM = 4;` |
| 109 | `constexpr int64_t BLOCK_BYTES = static_cast<int64_t>(ONE_BLK_SIZE);` |
| 110 | `constexpr int64_t NUM_TWO = 2;` |
| 111 | `constexpr int64_t NUM_THREE = 3;` |
| 112 | `constexpr int64_t NUM_FOUR = 4;` |
| 200 | `if constexpr (isRecv) {` |

### ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_group_list.h


函数清单：

| 行 | 源码 |
|---|---|
| 77 | `__aicore__ inline FfnWbA5GroupList(){};` |
| 80 | `__aicore__ inline void Init(GM_ADDR sortedIdsWs, GM_ADDR groupList, const A5GroupListParam &param, const ScheduleContextInfo *ctx, TPipe *pipe)` |
| 132 | `__aicore__ inline void Process()` |
| 162 | `__aicore__ inline A5RunLanes MakeLanes()` |
| 182 | `__aicore__ inline void LoadChunk(A5RunLanes &ln, int64_t begin, int64_t len)` |
| 224 | `__aicore__ inline int64_t BuildAndCompact(A5RunLanes &ln, int64_t len, uint64_t &startNum, uint64_t &endNum)` |
| 271 | `__aicore__ inline void EncodeRuns(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill)` |
| 334 | `__aicore__ inline void AppendRow(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill, int64_t expertId, int64_t tokenNum)` |
| 348 | `__aicore__ inline void FlushRows(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill)` |
| 365 | `__aicore__ inline void ClearRows(const LocalTensor<int32_t> &rows32)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 112 | `pipe_->InitBuffer(curBuf_, vecBytes);` |
| 113 | `pipe_->InitBuffer(prevBuf_, vecBytes);` |
| 114 | `pipe_->InitBuffer(nextBuf_, vecBytes);` |
| 115 | `pipe_->InitBuffer(idxBuf_, vecBytes);` |
| 116 | `pipe_->InitBuffer(diffBuf_, vecBytes);` |
| 117 | `pipe_->InitBuffer(diffFBuf_, vecBytes);` |
| 118 | `pipe_->InitBuffer(outStartBuf_, vecBytes);` |
| 119 | `pipe_->InitBuffer(outEndBuf_, vecBytes);` |
| 120 | `pipe_->InitBuffer(outIdBuf_, vecBytes);` |
| 121 | `pipe_->InitBuffer(cntBuf_, vecBytes);` |
| 123 | `pipe_->InitBuffer(maskSBuf_, maskBytes);` |
| 124 | `pipe_->InitBuffer(maskEBuf_, maskBytes);` |
| 125 | `pipe_->InitBuffer(rowBuf_, rowBytes + ONE_BLK_SIZE);` |
| 129 | `PipeBarrier<PIPE_V>();` |
| 150 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 153 | `DataCopyPad(groupListGm_[outIdx * NUM_TWO], rows, cpOut);` |
| 154 | `SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 191 | `Duplicate<int32_t>(ln.cur, SENTINEL_ID, static_cast<int32_t>(cmpLen));` |
| 192 | `Duplicate<int32_t>(ln.prev, SENTINEL_ID, static_cast<int32_t>(cmpLen));` |
| 193 | `Duplicate<int32_t>(ln.next, SENTINEL_ID, static_cast<int32_t>(cmpLen));` |
| 199 | `SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);` |
| 204 | `DataCopyPad(ln.cur, idsGm_[begin], cpLen, curPad);` |
| 205 | `DataCopyPad(ln.prev, idsGm_[begin - 1], cpLen, curPad);` |
| 212 | `DataCopyPad(ln.next, idsGm_[begin + 1], cpNext, nextPad);` |
| 214 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 220 | `SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);` |
| 232 | `Sub(ln.diff, ln.cur, ln.prev, static_cast<int32_t>(cmpLen));` |
| 233 | `PipeBarrier<PIPE_V>();` |
| 234 | `Cast(ln.diffF, ln.diff, RoundMode::CAST_ROUND, static_cast<int32_t>(cmpLen));` |
| 235 | `PipeBarrier<PIPE_V>();` |
| 238 | `PipeBarrier<PIPE_V>();` |
| 240 | `Sub(ln.diff, ln.next, ln.cur, static_cast<int32_t>(cmpLen));` |
| 241 | `PipeBarrier<PIPE_V>();` |
| 242 | `Cast(ln.diffF, ln.diff, RoundMode::CAST_ROUND, static_cast<int32_t>(cmpLen));` |
| 243 | `PipeBarrier<PIPE_V>();` |
| 246 | `PipeBarrier<PIPE_V>();` |
| 254 | `GatherMask(ln.outStart, ln.idx, ln.maskS, true, static_cast<uint32_t>(len), gmp, startNum);` |
| 255 | `GatherMask(ln.outEnd, ln.idx, ln.maskE, true, static_cast<uint32_t>(len), gmp, endNum);` |
| 256 | `GatherMask(ln.outId, ln.cur, ln.maskE, true, static_cast<uint32_t>(len), gmp, idNum);` |
| 257 | `PipeBarrier<PIPE_V>();` |
| 262 | `Sub(ln.cnt, ln.outEnd, ln.outStart, static_cast<int32_t>(runs));` |
| 263 | `PipeBarrier<PIPE_V>();` |
| 265 | `PipeBarrier<PIPE_V>();` |
| 267 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |
| 283 | `DataCopyPad(cur, idsGm_, cpOne, pad);` |
| 284 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 353 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 356 | `DataCopyPad(groupListGm_[outIdx * NUM_TWO], rows, cp);` |
| 357 | `SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 368 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 369 | `SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);` |
| 370 | `Duplicate<int32_t>(rows32, 0, static_cast<int32_t>(param_.rowsPerLoop * NUM_FOUR));` |
| 371 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |

常量清单：

| 行 | 源码 |
|---|---|
| 38 | `#define OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H` |
| 44 | `constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64; // CompareScalar 的 repeat 粒度(ISA)` |
| 45 | `constexpr int64_t ONE_REPEAT_BLOCKS = 8;       // 一个 repeat 覆盖 8 个 32B 块(ISA)` |
| 46 | `constexpr int64_t ONE_BYTE_BITS = 8;` |
| 47 | `constexpr int32_t SENTINEL_ID = -1; // 哨兵:有效 expert_id 恒 >= 0,故它与谁都不等` |
| 51 | `constexpr int64_t CHUNK_VEC_NUM = 10;` |
| 52 | `constexpr int64_t CHUNK_MASK_NUM = 2;` |
| 53 | `constexpr int64_t CHUNK_BUF_NUM = CHUNK_VEC_NUM + CHUNK_MASK_NUM + 1; // +1:行拼装区` |

### ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_gather.h


函数清单：

| 行 | 源码 |
|---|---|
| 51 | `__aicore__ inline FfnWbA5Gather() {}` |
| 52 | `__aicore__ inline void Init(GM_ADDR expertid_idx, GM_ADDR y, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale, const ScheduleContextInfo *contextInfo, TPipe *pipe, uint32_t usedCoreNum)` |
| 132 | `__aicore__ inline void InitBuffers(TPipe *pipe)` |
| 153 | `__aicore__ inline void CopyInIds()` |
| 164 | `__aicore__ inline void CopyInExpertIdx(int32_t expertIdxOffset, int32_t curRows)` |
| 174 | `__aicore__ inline void Process()` |
| 223 | `__aicore__ inline void EmitTokens(const LocalTensor<int32_t> &outAllLocal, int64_t curLoopElements, int64_t currentOuterStart, int64_t strideSession, int64_t strideMicroBatch, int64_t strideBs, int64_t strideK, const LocalTensor<int32_t> &aIdxAll, const LocalTensor<int32_t> &bsIdxAll, const LocalTensor<int32_t> &kIdxAll)` |
| 266 | `__aicore__ inline void DecomposeIndices(const LocalTensor<int32_t> &expertIdxLocal, int64_t bskProduct, int64_t curLoopElements)` |
| 348 | `__aicore__ inline void CorrectQuotient(const LocalTensor<float> &quot, const LocalTensor<float> &rem, const LocalTensor<float> &t0, const LocalTensor<float> &t1, float divisor, int64_t count)` |
| 374 | `__aicore__ inline void CopyAllLocalOut(int64_t allLocalOffset, int64_t copyLength)` |
| 400 | `__aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols, int64_t indicesIndex, const LocalTensor<int32_t> &dynamicScaleLocal, bool isLastBlock)` |
| 420 | `__aicore__ inline void CopyMxScale(int64_t srcOffset, int64_t dstOffset)` |
| 436 | `__aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 135 | `pipe->InitBuffer(inQueueX_, BUFFER_NUM, blockBufferSize);` |
| 138 | `pipe->InitBuffer(outQueALL_, BUFFER_NUM,` |
| 142 | `pipe->InitBuffer(expertIdxQue_, BUFFER_NUM, PER_LOOP_ROWS * sizeof(int32_t));` |
| 144 | `pipe->InitBuffer(idxCalcBuf_, PER_LOOP_ROWS * sizeof(float) * IDX_F_LANES);` |
| 145 | `pipe->InitBuffer(idxIntBuf_, PER_LOOP_ROWS * sizeof(int32_t) * IDX_I_LANES);` |
| 149 | `pipe->InitBuffer(tmpBuffer_, sessionNumBlockAlign_ * sizeof(int32_t) * BUFFER_NUM);` |
| 160 | `DataCopyPad(sessionIdsLocal_, sessionIdsInGm_, copyParams1, padParams1);` |
| 161 | `DataCopyPad(microBatchIdsLocal_, microBatchIdsInGm_, copyParams1, padParams1);` |
| 166 | `LocalTensor<int32_t> expertIdxLocal = expertIdxQue_.AllocTensor<int32_t>();` |
| 170 | `DataCopyPad(expertIdxLocal, expertIdxGm_[expertIdxOffset], copyParams, padParams);` |
| 171 | `expertIdxQue_.EnQue(expertIdxLocal);` |
| 203 | `LocalTensor<int32_t> expertIdxLocal = expertIdxQue_.DeQue<int32_t>();` |
| 204 | `LocalTensor<int32_t> outAllLocal = outQueALL_.AllocTensor<int32_t>();` |
| 205 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 216 | `outQueALL_.EnQue(outAllLocal);` |
| 218 | `expertIdxQue_.FreeTensor(expertIdxLocal);` |
| 291 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 313 | `Cast(srcF, expertIdxLocal, RoundMode::CAST_ROUND, curLoopElements);` |
| 314 | `PipeBarrier<PIPE_V>();` |
| 315 | `Muls(qaF, srcF, static_cast<float>(1.0f / bskF), curLoopElements);` |
| 316 | `PipeBarrier<PIPE_V>();` |
| 317 | `Cast(aIdxAll, qaF, RoundMode::CAST_FLOOR, curLoopElements);` |
| 318 | `PipeBarrier<PIPE_V>();` |
| 319 | `Cast(aF, aIdxAll, RoundMode::CAST_ROUND, curLoopElements);` |
| 320 | `PipeBarrier<PIPE_V>();` |
| 321 | `Muls(remF, aF, -bskF, curLoopElements);` |
| 322 | `PipeBarrier<PIPE_V>();` |
| 323 | `Add(remF, remF, srcF, curLoopElements); // rem = gatherIdx - aIdx*bskProduct` |
| 324 | `PipeBarrier<PIPE_V>();` |
| 326 | `Cast(aIdxAll, aF, RoundMode::CAST_RINT, curLoopElements);` |
| 327 | `PipeBarrier<PIPE_V>();` |
| 329 | `Muls(qbF, remF, static_cast<float>(1.0f / kNumF), curLoopElements);` |
| 330 | `PipeBarrier<PIPE_V>();` |
| 331 | `Cast(bsIdxAll, qbF, RoundMode::CAST_FLOOR, curLoopElements);` |
| 332 | `PipeBarrier<PIPE_V>();` |
| 333 | `Cast(bF, bsIdxAll, RoundMode::CAST_ROUND, curLoopElements);` |
| 334 | `PipeBarrier<PIPE_V>();` |
| 335 | `Muls(kF, bF, -kNumF, curLoopElements);` |
| 336 | `PipeBarrier<PIPE_V>();` |
| 337 | `Add(kF, remF, kF, curLoopElements); // kIdx = rem - bsIdx*K` |
| 338 | `PipeBarrier<PIPE_V>();` |
| 340 | `Cast(bsIdxAll, bF, RoundMode::CAST_RINT, curLoopElements);` |
| 341 | `PipeBarrier<PIPE_V>();` |
| 342 | `Cast(kIdxAll, kF, RoundMode::CAST_RINT, curLoopElements);` |
| 343 | `SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);` |
| 353 | `PipeBarrier<PIPE_V>();` |
| 354 | `Maxs(t0, t0, 0.0f, count);` |
| 355 | `PipeBarrier<PIPE_V>();` |
| 356 | `Mins(t0, t0, 1.0f, count); // hi = (rem >= divisor)` |
| 357 | `PipeBarrier<PIPE_V>();` |
| 358 | `Muls(t1, rem, -1.0f, count);` |
| 359 | `PipeBarrier<PIPE_V>();` |
| 360 | `Maxs(t1, t1, 0.0f, count);` |
| 361 | `PipeBarrier<PIPE_V>();` |
| 362 | `Mins(t1, t1, 1.0f, count); // lo = (rem < 0)` |
| 363 | `PipeBarrier<PIPE_V>();` |
| 364 | `Sub(t0, t0, t1, count); // corr = hi - lo` |
| 365 | `PipeBarrier<PIPE_V>();` |
| 366 | `Add(quot, t0, quot, count);` |
| 367 | `PipeBarrier<PIPE_V>();` |
| 368 | `Muls(t1, t0, -divisor, count);` |
| 369 | `PipeBarrier<PIPE_V>();` |
| 370 | `Add(rem, t1, rem, count);` |
| 371 | `PipeBarrier<PIPE_V>();` |
| 376 | `LocalTensor<int32_t> outAllLocal = outQueALL_.DeQue<int32_t>();` |
| 379 | `DataCopyPad(sessionIdsOutGm_[allLocalOffset], outAllLocal, copyParams2);` |
| 380 | `DataCopyPad(microBatchIdsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_MICRO_BATCH_IDX], copyParams2);` |
| 381 | `DataCopyPad(tokenIdsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_TOKEN_IDX], copyParams2);` |
| 382 | `DataCopyPad(expertOffsetsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_EXPERT_OFFSETS_IDX],` |
| 391 | `PipeBarrier<PIPE_V>();` |
| 394 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 395 | `DataCopyPad(dynamicScaleOutGm_[allLocalOffset], dynamicScaleLocalFp32, copyParams2);` |
| 397 | `outQueALL_.FreeTensor(outAllLocal);` |
| 403 | `LocalTensor<int8_t> xLocal = inQueueX_.AllocTensor<int8_t>();` |
| 407 | `DataCopyPad(xLocal, tokenDataBufGm_[xSrcOffset], copyParams0, padParams0);` |
| 413 | `DataCopyPad(dynScaleT, tokenDataBufGm_[xSrcOffset + copySize], copyParams1, padParams0);` |
| 416 | `inQueueX_.EnQue(xLocal);` |
| 425 | `LocalTensor<int8_t> local = inQueueX_.AllocTensor<int8_t>();` |
| 428 | `DataCopyPad(local, tokenDataBufGm_[srcOffset + offset], params, padding);` |
| 429 | `inQueueX_.EnQue(local);` |
| 430 | `local = inQueueX_.DeQue<int8_t>();` |
| 431 | `DataCopyPad(mxScaleOutGm_[dstOffset + offset], local, params);` |
| 432 | `inQueueX_.FreeTensor(local);` |
| 438 | `LocalTensor<int8_t> xLocal = inQueueX_.DeQue<int8_t>();` |
| 441 | `DataCopyPad(yOutGm_[xDstOffset], xLocal, copyParams2);` |
| 443 | `inQueueX_.FreeTensor(xLocal);` |

常量清单：

| 行 | 源码 |
|---|---|
| 27 | `#define OP_KERNEL_ARCH35_FFN_WB_A5_GATHER_H` |
| 34 | `constexpr int64_t IDX_LANE_SRC = 0; // gatherIdx(fp32)` |
| 35 | `constexpr int64_t IDX_LANE_QA = 1;  // gatherIdx / bskProduct` |
| 36 | `constexpr int64_t IDX_LANE_AF = 2;  // float(aIdx)` |
| 37 | `constexpr int64_t IDX_LANE_REM = 3; // gatherIdx - aIdx*bskProduct` |
| 38 | `constexpr int64_t IDX_LANE_QB = 4;  // rem / K` |
| 39 | `constexpr int64_t IDX_LANE_BF = 5;  // float(bsIdx)` |
| 40 | `constexpr int64_t IDX_LANE_KF = 6;  // rem - bsIdx*K` |
| 41 | `constexpr int64_t IDX_LANE_AI = 0;  // aIdx(int32,独立缓冲)` |
| 42 | `constexpr int64_t IDX_LANE_BI = 1;  // bsIdx` |
| 43 | `constexpr int64_t IDX_LANE_KI = 2;  // kIdx` |
| 44 | `constexpr int64_t IDX_F_LANES = 7;  // srcF/qaF/aF/remF/qbF/bF/kF` |
| 45 | `constexpr int64_t IDX_I_LANES = 3;  // aIdx/bsIdx/kIdx` |
| 148 | `if constexpr (isScanFlag == false) {` |
| 181 | `if constexpr (isScanFlag == false) {` |
| 234 | `if constexpr (isScanFlag == false) {` |
| 285 | `constexpr int64_t FP32_EXACT_POSITION_CAPACITY = 1LL << 24;` |
| 447 | `static constexpr uint32_t BLOCK_SIZE = 32;` |
| 448 | `static constexpr int32_t BUFFER_NUM = 2;      // tensor num for each queue` |
| 449 | `static constexpr int32_t PER_LOOP_ROWS = 128; // tensor num for each queue` |
| 450 | `static constexpr int64_t TOKEN_KIND_TWO = 2;` |
| 452 | `static constexpr int32_t VAR_NUM = 5; // session_ids, micro_batch_ids, token_ids, expert_offsets` |
| 453 | `static constexpr int32_t VAR_SESSION_IDX = 0;` |
| 454 | `static constexpr int32_t VAR_MICRO_BATCH_IDX = 1;` |
| 455 | `static constexpr int32_t VAR_TOKEN_IDX = 2;` |
| 456 | `static constexpr int32_t VAR_EXPERT_OFFSETS_IDX = 3;` |
| 457 | `static constexpr int32_t VAR_DYNAMIC_SCALE = 4;` |

### ffn/ffn_worker_batching/op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h


常量清单：

| 行 | 源码 |
|---|---|
| 27 | `#define FFN_WB_ARCH35_TILING_DEF_H` |

### ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_sort.h


函数清单：

| 行 | 源码 |
|---|---|
| 39 | `__aicore__ inline FfnWbA5SegSort(){};` |
| 44 | `__aicore__ inline void Init(GM_ADDR flatIdsWs, GM_ADDR pairWs, GM_ADDR cntWs, const A5SortSegParam &param, const ScheduleContextInfo *ctx, TPipe *pipe)` |
| 70 | `__aicore__ inline void Process()` |
| 80 | `__aicore__ inline void PrepareKeys(const LocalTensor<float> &keys, const LocalTensor<uint32_t> &maskBits, int64_t count)` |
| 130 | `__aicore__ inline void ProcessOneSeg(int64_t seg)` |
| 154 | `__aicore__ inline int64_t SortOneSegBody(const LocalTensor<int32_t> &inLocal, int64_t seg, int64_t begin, int64_t len)` |
| 263 | `__aicore__ inline FfnWbA5MrgSort(){};` |
| 266 | `__aicore__ inline void Init(GM_ADDR wsA, GM_ADDR wsB, GM_ADDR cntWs, const A5MergeParam &param, TPipe *pipe)` |
| 288 | `__aicore__ inline int64_t ProcessRound(int64_t round, int64_t srcIdx)` |
| 301 | `__aicore__ inline int64_t TotalValid()` |
| 311 | `__aicore__ inline int64_t Pow4(int64_t e)` |
| 320 | `__aicore__ inline void LoadSegCounts(int64_t seg)` |
| 340 | `__aicore__ inline int64_t SegCount(int64_t seg)` |
| 350 | `__aicore__ inline int64_t ListLength(int64_t firstSeg, int64_t listStride)` |
| 359 | `__aicore__ inline void MergeOneGroup(int64_t group, int64_t groupStride, int64_t listStride, int64_t srcIdx)` |
| 398 | `__aicore__ inline int64_t FillLists(const LocalTensor<float> &inLocal, int64_t listNum, int64_t loopFloats, int64_t srcIdx, const int64_t (&offsets)[MRG_LIST_NUM],` |
| 422 | `__aicore__ inline void MergeStream(const LocalTensor<float> &inLocal, const LocalTensor<float> &outLocal, int64_t listNum, int64_t loopFloats, int64_t &outOffset, int64_t &allRemain, int64_t (&offsets)[MRG_LIST_NUM], int64_t (&remains)[MRG_LIST_NUM],` |
| 504 | `__aicore__ inline FfnWbA5Extract(){};` |
| 509 | `__aicore__ inline void Init(GM_ADDR pairWs, GM_ADDR idsWs, GM_ADDR idxWs, const A5ExtractParam &param, TPipe *pipe)` |
| 528 | `__aicore__ inline void Process()` |
| 542 | `__aicore__ inline void ProcessOneChunk(int64_t begin, int64_t len)` |

API 调用索引：

| 行 | 源码 |
|---|---|
| 63 | `pipe_->InitBuffer(inQue_, 1, alignSeg * NUM_TWO * sizeof(int32_t) + ONE_BLK_SIZE);` |
| 64 | `pipe_->InitBuffer(concatBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);` |
| 65 | `pipe_->InitBuffer(tmpBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);` |
| 66 | `pipe_->InitBuffer(sortedBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);` |
| 67 | `pipe_->InitBuffer(maskBuf_, alignSeg * sizeof(uint32_t) + ONE_BLK_SIZE);` |
| 102 | `Duplicate(keys[repeatBegin], SORT_FILL_VALUE, tailMask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 103 | `PipeBarrier<PIPE_V>();` |
| 117 | `Reg::DataCopy(keyReg, keyAddr + i * FLOAT_REG_ELEMENTS);` |
| 118 | `Reg::Muls(keyReg, keyReg, negOne, loopMask);` |
| 119 | `Reg::DataCopy(keyAddr + i * FLOAT_REG_ELEMENTS, keyReg, loopMask);` |
| 122 | `PipeBarrier<PIPE_V>();` |
| 127 | `PipeBarrier<PIPE_V>();` |
| 136 | `LocalTensor<int32_t> inLocal = inQue_.AllocTensor<int32_t>();` |
| 143 | `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);` |
| 145 | `DataCopyPad(cntGm_[seg * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t)))], cntLocal, cpCnt);` |
| 147 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 148 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 149 | `inQue_.FreeTensor(inLocal);` |
| 161 | `DataCopyPad(inLocal, flatIdsGm_[begin], cp, pad);` |
| 166 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 168 | `PipeBarrier<PIPE_V>();` |
| 171 | `Cast(keys, inLocal, RoundMode::CAST_ROUND, len);` |
| 172 | `PipeBarrier<PIPE_V>();` |
| 184 | `GatherMask(keys, keys, maskBits, true, static_cast<uint32_t>(len), gp, rsvd);` |
| 185 | `PipeBarrier<PIPE_V>();` |
| 187 | `GatherMask(idxU32, idxU32, maskBits, true, static_cast<uint32_t>(len), gp, rsvd);` |
| 188 | `PipeBarrier<PIPE_V>();` |
| 202 | `Duplicate(keys[validCnt - tailNum], SORT_FILL_VALUE, dupMask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 203 | `PipeBarrier<PIPE_V>();` |
| 210 | `PipeBarrier<PIPE_V>();` |
| 211 | `Sort<float, true>(sortedLocal, concatLocal, idxU32, tmpLocal, alignCnt / ONE_REPEAT_SORT_NUM);` |
| 212 | `PipeBarrier<PIPE_V>();` |
| 214 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 217 | `DataCopyPad(pairGm_[seg * param_.sortLenPerSeg], sortedLocal, cpOut);` |
| 218 | `SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);` |
| 281 | `pipe_->InitBuffer(inQue_, 1, loopFloats * MRG_LIST_NUM * sizeof(float) + ONE_BLK_SIZE);` |
| 282 | `pipe_->InitBuffer(outQue_, 1, loopFloats * MRG_LIST_NUM * sizeof(float) + ONE_BLK_SIZE);` |
| 283 | `pipe_->InitBuffer(cntBuf_, param_.countCacheSegments * ONE_BLK_SIZE + ONE_BLK_SIZE);` |
| 328 | `SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);` |
| 336 | `DataCopyPad(cntLocal_, cntGm_[cachedBegin_ * (ONE_BLK_SIZE / sizeof(int32_t))], cp, pad);` |
| 337 | `SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);` |
| 379 | `LocalTensor<float> inLocal = inQue_.AllocTensor<float>();` |
| 380 | `LocalTensor<float> outLocal = outQue_.AllocTensor<float>();` |
| 391 | `inQue_.FreeTensor(inLocal);` |
| 392 | `outQue_.FreeTensor(outLocal);` |
| 413 | `DataCopyPad(inLocal[liveNum * loopFloats], wsGm_[srcIdx][offsets[i]], cp, pad);` |
| 437 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 442 | `DataCopy(outLocal, lists[0], Align(GetSortLen<float>(counts[0]), sizeof(float)));` |
| 449 | `MrgSort<float, true>(outLocal, srcList, counts, sortedNums, validBit, 1);` |
| 451 | `PipeBarrier<PIPE_V>();` |
| 467 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 470 | `DataCopyPad(wsGm_[1 - srcIdx][outOffset], outLocal, cpOut);` |
| 471 | `// 下一轮要用 MTE2 载入输入、用 MrgSort(向量)重写输出缓冲,两者都要等本次搬出读完。` |
| 472 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 473 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 523 | `pipe_->InitBuffer(pairQue_, 1, GetSortLen<float>(alignLoop) * sizeof(float) + ONE_BLK_SIZE);` |
| 524 | `pipe_->InitBuffer(idsBuf_, alignLoop * sizeof(int32_t) + ONE_BLK_SIZE);` |
| 525 | `pipe_->InitBuffer(idxBuf_, alignLoop * sizeof(int32_t) + ONE_BLK_SIZE);` |
| 547 | `LocalTensor<float> pairLocal = pairQue_.AllocTensor<float>();` |
| 551 | `DataCopyPad(pairLocal, pairGm_[GetSortLen<float>(begin)], cpIn, pad);` |
| 552 | `SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);` |
| 569 | `Duplicate(pairLocal[repeatBegin], 0.0f, tailMask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);` |
| 570 | `PipeBarrier<PIPE_V>();` |
| 578 | `Extract(keysLocal, idxU32, pairLocal, static_cast<int32_t>(alignLen / ONE_REPEAT_SORT_NUM));` |
| 579 | `PipeBarrier<PIPE_V>();` |
| 581 | `Muls(keysLocal, keysLocal, static_cast<float>(-1), len);` |
| 582 | `PipeBarrier<PIPE_V>();` |
| 583 | `Cast(idsLocal, keysLocal, RoundMode::CAST_ROUND, len);` |
| 584 | `PipeBarrier<PIPE_V>();` |
| 586 | `SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);` |
| 588 | `DataCopyPad(idsGm_[begin], idsLocal, cpOut);` |
| 589 | `DataCopyPad(idxGm_[begin], idxLocal, cpOut);` |
| 590 | `// 下一块会用 Extract(向量)重写这两块缓冲:必须等本块搬出读完。` |
| 591 | `SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);` |
| 592 | `SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);` |
| 593 | `pairQue_.FreeTensor(pairLocal);` |

常量清单：

| 行 | 源码 |
|---|---|
| 22 | `#define OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H` |
| 226 | `static constexpr int64_t FLOAT_REG_ELEMENTS = 64;` |
| 227 | `static constexpr int64_t BLOCKS_PER_REPEAT = 8;` |
| 228 | `static constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;` |
| 230 | `static constexpr int64_t DST_BLK_STRIDE = 1;` |
| 231 | `static constexpr int64_t DST_REP_STRIDE = 8;` |
| 597 | `static constexpr int64_t FLOAT_REG_ELEMENTS = 64;` |
| 598 | `static constexpr int64_t DST_BLK_STRIDE = 1;` |
| 599 | `static constexpr int64_t DST_REP_STRIDE = 8;` |

### ffn/ffn_worker_batching/torch_extension/csrc/ffn_worker_batching.cpp


函数清单：

| 行 | 源码 |
|---|---|
| 38 | `Outputs ffn_worker_batching(const at::Tensor &schedule_context, int64_t expert_num, const std::vector<int64_t> &max_out_shape, int64_t token_dtype, int64_t need_schedule, int64_t layer_num, bool sync_flag, bool mx_output_uint8)` |

常量清单：

| 行 | 源码 |
|---|---|
| 18 | `constexpr int64_t CONTEXT_BYTES = 1024;` |
| 19 | `constexpr int64_t SHAPE_DIMS = 4;` |
| 20 | `constexpr int64_t GROUP_LIST_COLUMNS = 2;` |
| 21 | `constexpr int64_t MAX_SESSIONS = 1024;` |
| 22 | `constexpr int64_t MAX_EXPERTS = 8192;` |
| 23 | `constexpr int64_t MAX_K = 64;` |
| 24 | `constexpr int64_t MX_BLOCK_SIZE = 32;` |
| 25 | `constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;` |
