# 代码检视报告

## 检视概览
- 代码文件：ffn/ffn_worker_batching；HEAD~4..HEAD（d967eeb08..399be5bfc）及关联原有实现
- 代码侧别：混合（Kernel、Host、Torch/GE、Build、测试与示例）
- 检视文档：ascendc-red-line、ascendc-topk、cpp-secure、ascendc-api、ascendc-op-conventions、regbase-review-checks、ascendc-perf、cpp-general、python-secure、compile-secure；另含 cpp-style 与 doc-style
- 总条例数：126（另有 19 条风格检查；按条款计数，不等于独立功能缺陷数量，根因归并见 docs/FOUR_COMMIT_REVIEW.md）
- 设计文档来源：ffn/ffn_worker_batching/docs
- 检视时间：2026-09-17

## 检视统计
| 状态 | 条例数 | 占比 |
|------|------|------|
| PASS | 105 | 83.3% |
| FAIL（发现问题）| 11 | 8.7% |
| SUSPICIOUS（需关注）| 10 | 7.9% |

## 设计一致性检查

- 文档来源：ffn/ffn_worker_batching/docs

- 总体评级：不一致

| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |
|------|------|---------|---------|------|
| S1 | 架构匹配 | DESIGN §1/4/6：AIV分阶段prepare/sort/merge/extract/group/gather，UB复用，阶段间跨核同步。 | 概要及apt入口一致，跳过AIC，key100/101分别为NORM/RECV；未声明Cube计算。 | ✅ |
| S2 | 分支覆盖 | REQUIREMENTS §4.4：NORM、同步全部ready、异步首个非空快照；非法层/ID过滤；六dtype；N=0；大位置整数路径。 | 概要分支表与prepare/gather入口一致。无ready时持续等待和生产者不得提前覆盖payload是明确外部协议，不作为缺失分支。 | ✅ |
| S3 | API清单 | DESIGN §2/10：V1补false，V2 bool，八输出；FP4/E8M0原生描述后返回uint8视图；GE拒绝不支持组合。 | 概要/API预研确认上述分配、schema、converter链路；文档不承诺GE全dtype已实测。RegBase与Buffer API按函数边界使用；本项为设计映射，不替代专项API核验。 | ✅ |
| S4 | 数据流追踪 | DESIGN §3/6–9：真实二级地址→ready快照/flatIds→key和position排序→group/gather；token及scale原样搬运。 | 代码概要与源码入口一致。prepare清flag早于payload搬运，文档已要求外部生命周期保护，不能据此单独判定实现有数据竞争。 | ✅ |
| S5 | 参数语义 | 文档集应对expertOffsets给出一致语义；V1参数表称为专家分组内偏移，详细设计和V2称为源路由槽。 | EmitTokens实际写kIndices，为源K维槽位；V1旧描述错误，应修文档而不是改变kernel。 | ❌ |
| S6 | 伪代码映射 | DESIGN §6/7/9伪代码及导读算例可映射；Torch文档承诺独立完整示例脚本可运行。 | 调度/层编码/位置解码映射一致；两个被引用的Python示例脚本均不存在，文档运行入口不可直接复现。页内代码可手动提取，但不能代替被承诺的仓内文件。 | ❌ |
| S7 | 约束合规 | 详细设计/需求限定Y与padded flatElements≤INT32_MAX；两个ACLNN文档仍写BS和Y无硬上限（受内存限制）。 | Host在计算Y后明确拒绝超int32容量；约束文档冲突，修正文档。实际M、指针和HS由调用方遵守且已明确，不把未对设备载荷全面校验作为bug。 | ❌ |
| D8 | 文档格式 |  |  | ❌ |

### 设计一致性偏差（❌ 项详情）

**S5**：V1文档expertOffsets写为所属专家分组内偏移，kernel却输出源路由槽k。应将文档改为原始topK/共享专家选择位置，与V2、需求和导读一致；这是接口说明错误。
- 代码位置（校对后行号）：ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_gather.h:245
- 设计依据：aclnnFfnWorkerBatching.md 参数说明第231行：“每个输出token在其所属专家分组内的偏移。”DESIGN.md §9.6明确写的是kIndices。

**S6**：文档引用的独立eager脚本不存在；本页同时宣称从仓库根目录运行却未提供运行命令。补充实际脚本和命令，或明确示例仅为页内代码。
- 代码位置（校对后行号）：ffn/ffn_worker_batching/docs/torchapi_ffn_worker_batching.md:78
- 设计依据：torchapi_ffn_worker_batching.md 调用示例：“完整脚本：[torch_ffn_worker_batching_async.py](../examples/torch_ffn_worker_batching_async.py)。不依赖ops-test-kit或测试辅助模块。”

**S6**：TorchAir示例引用的独立GE脚本不存在，无法按所给链接直接复现。不要据此声称GE验证完成；补脚本或改为明确的页内示例。
- 代码位置（校对后行号）：ffn/ffn_worker_batching/docs/torchapi_ffn_worker_batching.md:266
- 设计依据：torchapi_ffn_worker_batching.md TorchAir GE调用方式：“完整脚本：[torch_ffn_worker_batching_async_ge.py](../examples/torch_ffn_worker_batching_async_ge.py)。”

**S7**：V1“BS和Y无硬上限”与arch35 Host的Y≤INT32_MAX及RECV对齐后flat容量限制矛盾；应分架构说明具体表示上限。
- 代码位置（校对后行号）：ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling_arch35.cpp:290
- 设计依据：aclnnFfnWorkerBatching.md 约束说明第360行：“参数BS（micro batch size）和Y支持泛化，无硬上限（受内存限制）。”

**S7**：V2同样宣称Y无硬上限，而Host在内存分配容量以外还强制int32索引上限；应同步补齐该约束。
- 代码位置（校对后行号）：ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling_arch35.cpp:290
- 设计依据：aclnnFfnWorkerBatchingV2.md 约束说明第362行：“参数BS（micro batch size）和Y支持泛化，无硬上限（受内存限制）。”

### D8 文档格式违规

| 文档名 | 违规位置 | 违规描述 | 修复建议 |
|-------|---------|---------|---------|
| DESIGN.md | ffn/ffn_worker_batching/docs/DESIGN.md:1 | D1：标题“FfnWorkerBatching 四提交详细设计”包含中英文之间空格，正文有同类情况。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |
| REQUIREMENTS.md | ffn/ffn_worker_batching/docs/REQUIREMENTS.md:1 | D1：标题“FfnWorkerBatching 四提交需求分析”包含中英文之间空格，正文有同类情况。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |
| ARCH35_OPERATOR_GUIDE.md | ffn/ffn_worker_batching/docs/ARCH35_OPERATOR_GUIDE.md:1 | D1：标题“arch35 算子介绍”包含中英文之间空格，正文有同类情况。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |
| torchapi_ffn_worker_batching.md | ffn/ffn_worker_batching/docs/torchapi_ffn_worker_batching.md:8 | D1：正文“重排 token、scale 和来源索引”包含中英文之间空格。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |
| aclnnFfnWorkerBatching.md | ffn/ffn_worker_batching/docs/aclnnFfnWorkerBatching.md:357 | D1：正文“支持 ≤ 1024”包含中文、符号和数字之间多余空格。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |
| aclnnFfnWorkerBatchingV2.md | ffn/ffn_worker_batching/docs/aclnnFfnWorkerBatchingV2.md:359 | D1：正文“支持 ≤ 1024”包含中文、符号和数字之间多余空格。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |
| aclnnFfnWorkerBatching.md | ffn/ffn_worker_batching/docs/aclnnFfnWorkerBatching.md:366 | D2：同一Ascend950约束子列表前两项以句号结尾，scale项没有结尾句号。 | 按doc-style规则统一格式；代码、数学公式和产品官方名称保留必要字符与空格。 |

## 发现问题（HIGH 置信度）

### [TOPK-1] 必须校验函数返回值
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：新增V2示例对aclCreateIntArray返回值未判空，直接传给GetWorkspaceSize；无法区分数组创建失败与正常输入。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/examples/test_aclnn_ffn_worker_batching_v2.cpp
- **起始行号**：341
- **中止行号**：352
- **代码片段**：
  ```cpp
      ret = CreateAclTensorNoData(actualTokenNumShape, &actualTokenNumAddr, aclDataType::ACL_INT64, &actualTokenNumRef);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 创建maxOutShape aclIntArray
      aclIntArray *maxOutShapeArray = aclCreateIntArray(maxOutShapeValue.data(), maxOutShapeValue.size());

      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // V2在layerNum之后显式增加sync_flag；本例传false且needSchedule=0，只演示
      // 新符号的参数顺序及NORM兼容性，不会进入异步RECV。不能在V1的实参表中
      // 直接插入bool调用旧符号，否则后续输出指针参数的ABI位置会整体错位。
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 新增V2示例对aclCreateIntArray返回值未判空，直接传给GetWorkspaceSize；无法区分数组创建失败与正常输入。 |
| 上下文防御缺失 | +30% | 检查同函数与上游路径未发现覆盖此操作的校验 |
| 数据流风险 | +15% | 失败返回值或外部属性直接流入下一操作 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：创建后立即检查nullptr并清理已分配资源。

### [4.1] 资源申请成功判断
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：原有GE示例附录问题：WriteDataToFile未检查fopen返回值，工作目录不可写/不存在时将nullptr传给fwrite导致进程崩溃。已抽取原函数用g++编译，对/proc/ffn_wb_review_nonexistent/out.bin调用，进程返回-11(SIGSEGV)。不属于新增kernel缺陷。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/examples/test_geir_ffn_worker_batching.cpp
- **起始行号**：176
- **中止行号**：187
- **代码片段**：
  ```cpp

  int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
  {
      FILE *fp;
      fp = fopen(bin_file.c_str(), "w");
      fwrite(inputData, sizeof(uint8_t), data_size, fp);
      fclose(fp);
      return SUCCESS;
  }

  int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs, std::vector<Operator> &outputs,
                       Graph &graph, ExampleResources &resources)
  {
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 申请FILE句柄后未判空 |
| 上下文防御缺失 | +30% | fopen与fwrite之间没有任何检查 |
| 调用链风险 | +15% | main直接调用WriteDataToFile；抽取原函数探针在无效路径确定SIGSEGV |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：检查fopen/fwrite/fclose结果并返回失败，调用者传播失败；保持资源清理路径。

### [TOPK-7] 融合规则/InferShape/Tiling外部输入校验
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：InferShape仅检查H>0；MX路径H+31可能有符号溢出，例如A=BS=K=1、H=INT64_MAX、token_dtype=3。Tiling中的uint32上界发生在后续阶段，不能保护先执行的InferShape。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_host/ffn_worker_batching_infershape.cpp
- **起始行号**：132
- **中止行号**：143
- **代码片段**：
  ```cpp
      if (tokenDtype >= TOKEN_DTYPE_E5M2 && tokenDtype <= TOKEN_DTYPE_E2M1) {
          // MX 的数据和 scale 成对搬运：每 32 个逻辑元素对应一个 E8M0 字节。
          // FP4 的分组仍按逻辑 H 计算，不按打包后的 H/2 字节计算；尾组不足 32 个
          // 元素也保留一个 scale。输出是紧凑二维布局，不包含输入记录的 512B padding。
          *dynamicScale = {Y, (H + MXFP_SCALE_GROUP - 1) / MXFP_SCALE_GROUP};
      } else {
          *dynamicScale = {Y};
      }
      *actualTokenNum = {1};

      return GRAPH_SUCCESS;
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | InferShape仅检查H>0；MX路径H+31可能有符号溢出，例如A=BS=K=1、H=INT64_MAX、token_dtype=3。Tiling中的uint32上界发生在后续阶段，不能保护先执行的InferShape。 |
| 上下文防御缺失 | +30% | 检查同函数与上游路径未发现覆盖此操作的校验 |
| 数据流风险 | +15% | 失败返回值或外部属性直接流入下一操作 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：推导阶段校验H范围，或采用H/32+(H%32!=0)并统一入口支持范围。

### [1.1] 外部数据合法性
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：与secure-host3.1和红线相同原有根因，主报告合并：A2/A3 CheckInputParam只检查rank/dtype，不检查context的1024字节最小容量；旧kernel仍固定读取1024字节。不是新增arch35缺陷。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling.cpp
- **起始行号**：92
- **中止行号**：104
- **代码片段**：
  ```cpp
                  return ge::GRAPH_FAILED);

      auto inputX = context_->GetInputShape(0);
      OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
      auto xShape = Ops::Transformer::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
      OP_CHECK_IF(xShape.GetDimNum() != 1,
                  OP_LOGE(context_->GetNodeName(), "x shape %s dim num not 1", Ops::Base::ToString(xShape).c_str()),
                  return ge::GRAPH_FAILED);

      return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus FfnWorkerBatchingTiling::GetAttrsInfo()
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款与所引用代码模式不一致 |
| 上下文防御缺失 | +30% | 当前调用路径未发现保护所述输入的有效检查 |
| 调用链风险 | +15% | 旧kernel固定搬入1024字节，旧Host长度无保护 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：沿用已记录修复建议，为旧平台Host加最小context长度拒绝；不把有效context内部由生产者保证的协议另报为错误。

### [3.1] 外部输入合法性
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：原有A2/A3路径输入防御缺口，低于本次新增缺陷优先级：CheckInputParam检查INT8/rank1后直接成功，未要求context至少1024字节；ScheduleContextInfoCompute固定DataCopy 1024字节，短context可导致读取越界。arch35新增检查不保护旧分支。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_host/ffn_worker_batching_tiling.cpp
- **起始行号**：92
- **中止行号**：104
- **代码片段**：
  ```cpp
                  return ge::GRAPH_FAILED);

      auto inputX = context_->GetInputShape(0);
      OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
      auto xShape = Ops::Transformer::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
      OP_CHECK_IF(xShape.GetDimNum() != 1,
                  OP_LOGE(context_->GetNodeName(), "x shape %s dim num not 1", Ops::Base::ToString(xShape).c_str()),
                  return ge::GRAPH_FAILED);

      return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus FfnWorkerBatchingTiling::GetAttrsInfo()
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 直接外部Tensor容量不足仍通过Tiling |
| 上下文防御缺失 | +30% | CheckInputParam在rank1检查后返回成功 |
| 调用链风险 | +15% | op_kernel/ffn_wb_get_schedule_context.h:45 固定DataCopy(valLocal,scheduleContext,STRUCT_LENGTH)，STRUCT_LENGTH=1024 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：在旧路径下发前也检查context长度至少sizeof(ScheduleContext)，并增加旧SoC短输入拒绝用例；与红线同根因去重。

### [API-1] 禁止生产代码GlobalTensor逐元素GetValue/SetValue
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：原 A2/A3 生产路径继续使用 GlobalTensor GetValue/SetValue 读写段计数、实际token数量和polling_index，违反该专项条款。示例ValidGatherIdxLengthCompute读取workspace标量并写actual_token_num。已有DCCI维持缓存可见性，故此结果仅为API约束/性能建议，不宣称数值错误，也不是近期arch35新增回归。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/ffn_wb_common.h
- **起始行号**：324
- **中止行号**：336
- **代码片段**：
  ```cpp
      workSpace.SetGlobalBuffer((__gm__ int32_t *)work_space, OFFSET_SORTED_EXPERT_IDS);
      actualTokenNumGm.SetGlobalBuffer((__gm__ int64_t *)actual_token_num);

      DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
          workSpace);
      contextInfo.validGatherIdxLength = workSpace.GetValue(0);

      if (GetBlockIdx() == 0) {
          actualTokenNumGm.SetValue(0, contextInfo.validGatherIdxLength);
          DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
              actualTokenNumGm);
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | API-1禁止生产GlobalTensor SetValue/GetValue，当前是实际调度数据而非调试打印。 |
| 上下文防御缺失 | +30% | 该生产分支未改为UB短块DMA；DataCacheCleanAndInvalid解决可见性但不消除此API用法。 |
| 领域关联 | +10% | 同类标量GM访问还在scan_token_info、sort_one_core等原路径。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：后续优化原平台路径时以32B UB暂存和DataCopyPad读写少量有效字节，保持原缓存/同步语义；该项不得混作已确认性能瓶颈，应先测量。

### [RL-4] 确保无符号整数运算不回绕
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：原 A2/A3 gather 的 strideSession 在 uint32 中连乘，再赋 int64；M=BS=K=64、HS=16384 时乘积 2^32 回绕为零，session>0 读错 token 行。四提交中的 arch35 路径已经先提升 int64，但原路径仍存在该问题。本结果是基线缺陷，不是新增回归。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/ffn_wb_gather_out_all.h
- **起始行号**：147
- **中止行号**：159
- **代码片段**：
  ```cpp
          } else {
              bskProduct = contextInfo_->BS * contextInfo_->K + BsKPaddingCount;
          }

          int64_t curLoopElements = PER_LOOP_ROWS;
          int64_t strideSession = contextInfo_->M * contextInfo_->BS * contextInfo_->K * contextInfo_->HS;
          int64_t strideMicroBatch = contextInfo_->BS * contextInfo_->K * contextInfo_->HS;
          int64_t strideBs = contextInfo_->K * contextInfo_->HS;
          int64_t strideK = contextInfo_->HS;

          for (int64_t i = 0; i < rowLoops_; i++) {
              int64_t currentOuterStart = i * PER_LOOP_ROWS;
              if (i == rowLoops_ - 1) {
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | ffn_wb_common.h:75-78 的 M/BS/K/HS 都是 uint32_t；clause.check_bounds 固定反例得 4294967296，超过 uint32。 |
| 上下文防御缺失 | +30% | 原 tiling 对 BS/H 仅正数检查，没有该 stride 的32位乘积限制；M/K=64 符合上下文和文档约束。 |
| 数据流风险 | +15% | strideSession 后续乘 sessionIndices 计算 tokenDataBufGm_ 读取地址，回绕直接改变读取行。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：与 arch35 一致，在 M*BS*K*HS 的第一次乘法前提升为 int64，并覆盖 strideMicroBatch/strideBs；加入大于4GiB逻辑行跨度的地址计算测试。

### [RL-9] GM内存偏移或大小必须用int64表示
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：与 RL-4 是同一基线缺陷，请合并计数：原 A2/A3 gather 的 GM stride 虽以 int64_t 接收，但 RHS 多维连乘先按 uint32_t 计算。M=BS=K=64、HS=16384 时 session stride 从4GiB变为0；arch35 已提升首个操作数，原路径尚未同步。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/ffn_wb_gather_out_all.h
- **起始行号**：147
- **中止行号**：159
- **代码片段**：
  ```cpp
          } else {
              bskProduct = contextInfo_->BS * contextInfo_->K + BsKPaddingCount;
          }

          int64_t curLoopElements = PER_LOOP_ROWS;
          int64_t strideSession = contextInfo_->M * contextInfo_->BS * contextInfo_->K * contextInfo_->HS;
          int64_t strideMicroBatch = contextInfo_->BS * contextInfo_->K * contextInfo_->HS;
          int64_t strideBs = contextInfo_->K * contextInfo_->HS;
          int64_t strideK = contextInfo_->HS;

          for (int64_t i = 0; i < rowLoops_; i++) {
              int64_t currentOuterStart = i * PER_LOOP_ROWS;
              if (i == rowLoops_ - 1) {
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款9明确包含“最终int64但右侧uint32先溢出”的模式，当前strideSession完全匹配。 |
| 上下文防御缺失 | +30% | 原Host没有32位stride上限，文档允许M<=64且该反例的BS/K/H均满足输入约束。 |
| 数据流风险 | +15% | 回绕stride参与tokenDataBufGm_的实际GM偏移。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：合并 RL-4 修复，在所有多维GM stride第一次乘法前提升int64，避免仅给左侧变量使用int64。

### [OP-2] Kernel禁止TILING_KEY_IS宏
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：arch35入口仍按TILING_KEY_IS(TILING_KEY_NORM/RECV)分派，原A2/A3入口也使用该宏，违反该工程约定。属于已有结构性规范问题，不能由此推断本次功能错误。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/ffn_worker_batching_apt.cpp
- **起始行号**：268
- **中止行号**：279
- **代码片段**：
  ```cpp
      REGISTER_TILING_DEFAULT(FfnWorkerBatchingTilingData);
      GET_TILING_DATA_WITH_STRUCT(FfnWorkerBatchingTilingData, tilingDataIn, tiling);
      const FfnWorkerBatchingTilingData *tilingData = &tilingDataIn;

      if (TILING_KEY_IS(TILING_KEY_NORM)) {
          FfnWorkerBatchingA5<false>(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                     expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
      } else if (TILING_KEY_IS(TILING_KEY_RECV)) {
          FfnWorkerBatchingA5<true>(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                    expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | op-conventions2明确禁止Kernel入口TILING_KEY_IS，宏参数为数值常量别名。 |
| 上下文防御缺失 | +30% | 外层未迁移模板入口，内部FfnWorkerBatchingA5<bool>不消除入口宏分派。 |
| 领域关联 | +10% | Host与Kernel以100/101约定耦合。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：后续独立迁移Host模板key注册与Kernel模板入口，保持原100/101契约映射并重新全量构建验证；本轮注释任务不必直接改分派。

### [RL-3] 确保有符号整数运算不溢出
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：新增 C++ wrapper 在调用 ACLNN 校验前计算 a*bs*k；BS 只有正数限制，直接扩展入口可传 a=2,BS=INT64_MAX,K=1，使表达式先发生 signed overflow。另一个独立位置是 op_host/ffn_worker_batching_infershape.cpp 的 MX scale (H+32-1)/32，H 只校验正数，H=INT64_MAX 也先溢出；后续 tiling 的 UINT32 上界无法保护此前的推导。原 A2/A3 tiling 的 Y=A*BS*K 同样保留未保护乘法。均属参数拒绝前触发的未定义行为，非正常尺寸计算错误。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/torch_extension/csrc/ffn_worker_batching.cpp
- **起始行号**：77
- **中止行号**：89
- **代码片段**：
  ```cpp
                                                at::ScalarType::Float8_e4m3fn,
                                                at::ScalarType::Float4_e2m1fn_x2};
          // 输出按最大容量分配，不能在 host 读取 context.out_num/ready 来缩小：
          // 那会等待设备、破坏图捕获，而且随后生产者仍可能改变本轮可消费集合。
          // kernel 最终只定义 actual_token_num 指出的前缀，尾部无需初始化或对外使用。
          const int64_t y_rows = a * bs * k;
          const bool mx = token_dtype >= FP8_E5M2;
          const int64_t y_cols = token_dtype == FP4_E2M1 ? h / FP4_ELEMENTS_PER_BYTE : h;
          // 以逻辑 H 分组：FP4 也是每 32 个四位元素一组，不是每 32 个存储字节一组。
          // 使用商+非零余数实现 ceil，避免 (h+31)/32 在属性上界附近发生加法溢出。
          const int64_t scales = h / MX_BLOCK_SIZE + (h % MX_BLOCK_SIZE != 0);
          auto options = schedule_context.options();
          // Native FP4 x2 counts packed bytes. The common converter expands the
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 乘法与 MX 向上取整加法存在 int64 溢出；clause.check_bounds 对具体可输入反例给出 VIOLATION。 |
| 上下文防御缺失 | +30% | 已检查 wrapper、Python 检查和 infer：BS/H 没有保护这些表达式的上限；Host 后置检查执行太晚。 |
| 调用链风险 | +15% | pybind 直接暴露该函数；y_rows 先于 at::empty 和 ACLNN_CMD 计算。父级独立 UBSan 表达式 probe 已确认 H/y 两处溢出；不是完整算子上板复现。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：C++ 分配前按 INT64_MAX/操作数逐步检查乘法，并同步后端 Y 的 int32 容量限制；MX scale 用 H/32+(H%32!=0)，或先限制 H 再加；补非法极值的拒绝测试。

### [1.3] 禁止使用未定义行为
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：与红线已确认整数溢出同一根因，请主报告合并：新增Torch封装在ACLNN/Host校验之前用有符号int64计算a*bs*k，仅校验各维为正及A/K上界，巨大BS可先触发未定义行为。Meta使用Python整数也不会替代C++运算防护。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/torch_extension/csrc/ffn_worker_batching.cpp
- **起始行号**：78
- **中止行号**：90
- **代码片段**：
  ```cpp
                                                at::ScalarType::Float4_e2m1fn_x2};
          // 输出按最大容量分配，不能在 host 读取 context.out_num/ready 来缩小：
          // 那会等待设备、破坏图捕获，而且随后生产者仍可能改变本轮可消费集合。
          // kernel 最终只定义 actual_token_num 指出的前缀，尾部无需初始化或对外使用。
          const int64_t y_rows = a * bs * k;
          const bool mx = token_dtype >= FP8_E5M2;
          const int64_t y_cols = token_dtype == FP4_E2M1 ? h / FP4_ELEMENTS_PER_BYTE : h;
          // 以逻辑 H 分组：FP4 也是每 32 个四位元素一组，不是每 32 个存储字节一组。
          // 使用商+非零余数实现 ceil，避免 (h+31)/32 在属性上界附近发生加法溢出。
          const int64_t scales = h / MX_BLOCK_SIZE + (h % MX_BLOCK_SIZE != 0);
          auto options = schedule_context.options();
          // Native FP4 x2 counts packed bytes. The common converter expands the
          // final dimension and strides to logical FP4 elements for aclTensor.
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | signed int64乘积无前置溢出检查 |
| 上下文防御缺失 | +30% | 当前C++仅约束bs>0，a/k的局部上界不能限制乘积 |
| 调用链风险 | +15% | at::empty及ACLNN_CMD均在乘积之后，后续Host校验不能撤销先前UB |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：在乘法前逐项检查乘积可表示性并统一Y的int32容量限制；与已有红线修复合并，不重复计数。

## 疑似（LOW 置信度）

### [5.1] 安全函数使用
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：仅编码规范建议，非已证实内存缺陷：GE示例使用std::memcpy而非安全函数。此处目标std::array大小与ScheduleContext完全相同且有static_assert，因此不能据此宣称缓冲区越界。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/examples/test_geir_ffn_worker_batching.cpp
- **起始行号**：247
- **中止行号**：259
- **代码片段**：
  ```cpp
      schedule_context.ffn.micro_batch_ids_buf_size = micro_batch_ids.size() * sizeof(int32_t);
      schedule_context.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expert_ids_device);
      schedule_context.ffn.expert_ids_buf_size = expert_ids.size() * sizeof(int32_t);
      schedule_context.ffn.out_num = session_num;
      std::memcpy(resources.schedule_context.data(), &schedule_context, sizeof(schedule_context));

      auto schedule_context_data = op::Data("schedule_context").set_attr_index(0);
      TensorDesc schedule_context_desc(ge::Shape(schedule_context_shape), FORMAT_ND, DT_INT8);
      schedule_context_desc.SetPlacement(ge::kPlacementHost);
      schedule_context_desc.SetFormat(FORMAT_ND);
      schedule_context_desc.SetRealDimCnt(schedule_context_shape.size());
      schedule_context_data.update_input_desc_x(schedule_context_desc);
      Tensor schedule_context_tensor(schedule_context_desc, resources.schedule_context.data(),
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | cpp-secure5.1要求安全函数替换memcpy |
| 领域关联 | +10% | Host示例直接复制设备上下文ABI |

自信值 = Σ正向 + Σ负向 = 50% ≥ 70% → 未达违规阈值
- **修复建议**：按仓库规范可改为有返回值检查的安全拷贝或等价类型安全复制；不应当作核心功能问题。

### [4.3] 常量单一职责
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：维护性建议，不是当前计算错误：BLOCK_BYTES来自硬件搬运块字节数，同时被用于MX每32个逻辑元素的scale分组。当前二者都为32，计算正确，但未来调整硬件常量会隐式改变量化存储协议。Host已有独立MX_SCALE_BLOCK_ELEMENTS命名。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_gather.h
- **起始行号**：69
- **中止行号**：80
- **代码片段**：
  ```cpp

          // 此处 32 的业务含义是“每组逻辑元素数”，恰与 BLOCK_BYTES=32 相等。
          // E8M0 每组仅一个字节，尾组也占一个字节；它不是 INT8 的逐 token FP32 scale。
          // 本算子只复制编码，不把 E8M0 解码成 float，也不把 FP4 nibble 拆成浮点数。
          scaleRowBytes_ = contextInfo_->tokenDtype >= 3 ? CeilDiv(contextInfo_->H, BLOCK_BYTES) : 0;

          // NORM 的路由表只消费 outNum 前缀；RECV 不读取这两张路由表。
          activeSessionNum_ = isScanFlag ? contextInfo_->A : contextInfo_->outNum;
          sessionNumBlockAlign_ = Align(activeSessionNum_, sizeof(int32_t));
          int64_t validGatherIdxLength = contextInfo_->validGatherIdxLength;

          // ⚠️ 逐项扣除本类**所有**会 InitBuffer 的缓冲。少扣一项会让 maxBlockSize_ 算大,
          // inQueueX_ 随即申请越界、分配失败,输出成为垃圾。新增缓冲必须同步加进这个式子。
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款与所引用代码模式不一致 |
| 领域关联 | +10% | 已检查实际调用与构建继承；本项仍仅是规范/构建疑点，不构成新增功能故障证据 |

自信值 = Σ正向 + Σ负向 = 50% ≥ 70% → 未达违规阈值
- **修复建议**：可为kernel增加独立MX_SCALE_BLOCK_ELEMENTS=32并用于scale行宽，保持数据搬运对齐常量职责独立；属于附录建议。

### [PERF-1] 循环内逐元素操作
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：容量超过2^24后，gather逐token标量GetValue/整数除法/SetValue，是潜在性能测量点。该分支为了避免FP32原始索引丢位而新增；正常容量仍走向量路径，未采集性能数据，不能宣称速度回退，也不能恢复旧浮点算法来消除该循环。原A2/A3 gather同样为逐token整数处理。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_gather.h
- **起始行号**：301
- **中止行号**：314
- **代码片段**：
  ```cpp
              // FP32 无法表示该范围内的奇数，更换舍入方式/修正商余数均不能恢复丢位。
              // 仅大容量使用标量整数除法，保留常规形状的原有向量计算路径。
              SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
              for (int64_t i = 0; i < curLoopElements; ++i) {
                  const int64_t position = expertIdxLocal.GetValue(i);
                  const int64_t session = position / bskProduct;
                  const int64_t remainder = position - session * bskProduct;
                  const int64_t token = remainder / contextInfo_->K;
                  aIdxAll.SetValue(i, static_cast<int32_t>(session));
                  bsIdxAll.SetValue(i, static_cast<int32_t>(token));
                  kIdxAll.SetValue(i, static_cast<int32_t>(remainder - token * contextInfo_->K));
              }
              // 这三路结果由后续 EmitTokens 的标量 GetValue 消费，无向量消费者。
              return;
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | PERF-1触发模式是循环内GetValue/SetValue，当前大容量分支确实存在。 |
| 数据流风险 | +15% | 每个token两次标量整数除法，开销随有效token数增长。 |
| 领域关联 | +10% | token gather阶段可能受标量索引解码开销影响，尚无实测。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_gather.h:299 外层capacity>2^24才进入整数分支，普通容量保留向量快路径，整数解码保护原始位置精度。 |

自信值 = Σ正向 + Σ负向 = 45% ≥ 70% → 未达违规阈值
- **修复建议**：仅列待测项：覆盖大量有效token且capacity>2^24的输入，对比索引解码和DMA占比；若成为瓶颈再考虑有精度证明的整数向量/SIMT实现，保留748165720次FP32边界验证的适用范围，勿把它用于大容量。

### [RB-3] 寄存器级计算边界
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：异步EncodeLayerExperts对末轮使用无mask的整寄存器Reg::DataCopy，active只作用于后续计算/写回；que_按32B对齐分配而非向量256B对齐。小行如BS*K=1时rowSpan=8，仅32B缓冲却发整寄存器加载。输出有效lane已掩蔽，尚未证明数值错误或物理UB越界，需检查编译器/检查工具是否允许跨分配对象读未使用lane。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_prepare.h
- **起始行号**：378
- **中止行号**：390
- **代码片段**：
  ```cpp
          {
              Reg::RegTensor<uint32_t> ids, encoded, masked, result;
              Reg::MaskReg active, valid;
              for (uint16_t i = 0; i < repeats; ++i) {
                  active = Reg::UpdateMask<uint32_t>(remain);
                  Reg::DataCopy(ids, addr + i * LANES);
                  Reg::Compares<uint32_t, CMPMODE::LT>(valid, ids, limit, active);
                  Reg::Adds(encoded, ids, bias, valid);
                  Reg::Duplicate(masked, static_cast<uint32_t>(MASK_SENTINEL), active);
                  Reg::Select(result, encoded, masked, valid);
                  Reg::DataCopy(addr + i * LANES, result, active);
              }
          }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | RegBase边界条款要求load边界明确；SDK reg_compute/kernel_reg_compute_datacopy_intf_impl.h:44确认所用DataCopy重载无mask参数。 |
| 上下文防御缺失 | +30% | Init仅Align(perLoopElements*sizeof(int32),32)，缺少完整向量尾部物理容量。 |
| 数据流风险 | +15% | async固定单行，合法BS=K=1对应count8；末轮加载宽于这段UB对象。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_wb_a5_prepare.h:388 写回Reg::DataCopy(addr+i*LANES,result,active)使用active，非有效lane不写回，因此不能直接认定输出错误。 |

自信值 = Σ正向 + Σ负向 = 65% ≥ 70% → 未达违规阈值
- **修复建议**：优先用目标SDK仿真/检查工具确认整寄存器load对小UB对象的约束；若必须覆盖完整加载宽度，把prepare缓冲容量向256B对齐并同步Host UB预算，或使用官方允许的尾部加载方案。

### [SEC-7.4] 对外结构体新增字段在末尾添加
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：syncFlag 和 wsReady 插在原有 wsGatherIdx 之前，改变旧字段offset，与条款要求TilingData尾部追加不符。不过此结构作为同版Host/kernel内部载荷，当前注释明确要求配套构建，未证实存在支持的新旧组件混用场景；故列兼容性注意项，不宣称当前配套构建错误。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h
- **起始行号**：61
- **中止行号**：73
- **代码片段**：
  ```cpp
      int64_t wsPairB{0};     // 排序/归并工作区 B（与 A 乒乓）
      int64_t wsSegCnt{0};    // 各段有效元素数：每段一个 32B 槽
      int64_t wsSortedIds{0}; // 有序 expert_id：Y 个 int32
      // 与 tokenDtype 正交，不增加 tiling key；kernel 仅在 RECV 路径解释此值。
      bool syncFlag{false}; // false: wait all; true: consume ready (RECV only)
      // 单位为 int32 word，相对 userWorkspace；仅异步 RECV 使用。
      // 头块存 uint64 selectedMb；后续每块存一个 int32 flag，其余字节为对齐填充。
      int64_t wsReady{0};     // selected micro batch block + A padded flag blocks
      int64_t wsGatherIdx{0}; // gather_idx：Y 个 int32
      // 以下容量字段在本版尾部追加，保持本版之前字段的相对顺序；这不是跨版本序列化格式。
      // struct 还包含 bool 后的编译器对齐空隙，不能把成员大小简单求和当作总大小，也不能
      // 手写字节偏移。Host 直写与 kernel 读取必须使用同一版头文件和配套构建产物。
      int64_t preparePerLoopElements{0};  // ID 缓冲的元素容量；大行按此值分块
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | HEAD~4结构末尾为wsSortedIds、wsGatherIdx；当前在两者之间新增syncFlag/wsReady。 |
| 调用链风险 | +15% | Host写结构体与Kernel GET_TILING_DATA读取依赖同一内存布局，旧二进制混搭将解释不同offset。 |
| 上下文防御缺失 | +30% | 结构中没有版本tag；旧字段offset未保持。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h:71 当前注释明确不是跨版本序列化格式，Host/kernel必须同版头文件与配套构建。 |

自信值 = Σ正向 + Σ负向 = 65% ≥ 70% → 未达违规阈值
- **修复建议**：若需要维持旧字段offset，将syncFlag/wsReady同样追加到wsGatherIdx之后；若只支持配套更新，明确记录该内部tiling ABI不保证跨版本混搭，不把它当作已复现运行缺陷。

### [5.2] 外部路径规范化和校验
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：测试工具规范建议：FFN_WB_SAVE_ARCHIVE/FFN_WB_REPLAY_ARCHIVE环境变量直接传入open/np.load，未规范化路径或提前校验普通文件/父目录。该工具本来允许调用者选择归档位置，没有确定的受限目录或不可信调用方，不能把它宣称为已证实路径穿越漏洞。
- **代码文件**：/home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py
- **起始行号**：275
- **中止行号**：287
- **代码片段**：
  ```python
          )
          if ret:
              raise RuntimeError(f"archive D2H failed: {ret}")
          records[f"buffer_{offset}"] = payload
          struct.pack_into("<Q", header, offset, 0)
      records["header"] = header
      with open(path, "wb") as stream:
          np.savez(stream, **records)


  def restore_context_archive(path, expected_attributes=None):
      """Allocate fresh HBM and relocate the header; never reuse saved addresses."""
      # 文件只提供字节与元数据，旧进程虚拟地址不可继承。先检查所有 payload，
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | python-secure5.2要求外部路径规范化后再使用 |
| 数据流风险 | +15% | npu_preprocess从环境变量直接传至save_context_archive/restore_context_archive |

自信值 = Σ正向 + Σ负向 = 55% ≥ 70% → 未达违规阈值
- **修复建议**：如落实工具规范，可统一Path.expanduser/resolve，校验输入是文件、输出父目录存在并给出清楚错误；不要凭空新增仓库根目录限制或把用户指定路径当攻击。

### [COMPILE-3] RELRO
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：同一共享builder配置疑点，四个编译条例应合并成一条附录建议：cxx_args包含-Wl,-z,relro,-z,now,-z,noexecstack及-s，但load将它们只传入extra_cflags；extra_ldflags仅提供-L/-l。编译阶段-c不会执行链接，配置不能证明JIT最终so启用了这些选项。OPP的intf_pub继承配置已正确设置，不能泛化为本算子所有构建都缺失硬化；未取得本次最终JIT链接命令/ELF，故仅SUSPICIOUS。
- **代码文件**：/home/chenpengyi/ops-transformer/torch_extension/cann_ops_transformer/op_builder/builder.py
- **起始行号**：175
- **中止行号**：185
- **代码片段**：
  ```python
          op_module = load(
              name=self.name,
              sources=self.get_absolute_paths(self.sources()),
              extra_include_paths=self.get_absolute_paths(self.include_paths()),
              extra_cflags=self.cxx_args(),
              extra_ldflags=self.extra_ldflags(),
              verbose=verbose,
          )
          OpBuilder._loaded_ops[self.name] = op_module

          return op_module
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款与所引用代码模式不一致 |
| 领域关联 | +10% | 已检查实际调用与构建继承；本项仍仅是规范/构建疑点，不构成新增功能故障证据 |

自信值 = Σ正向 + Σ负向 = 50% ≥ 70% → 未达违规阈值
- **修复建议**：在共享builder的链接参数通道添加适用链接选项，并用实际link命令/readelf检查最终so；属于共享既有配置，不计入最近四提交新增功能缺陷。

### [COMPILE-4] 立即绑定
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：同一共享builder配置疑点，四个编译条例应合并成一条附录建议：cxx_args包含-Wl,-z,relro,-z,now,-z,noexecstack及-s，但load将它们只传入extra_cflags；extra_ldflags仅提供-L/-l。编译阶段-c不会执行链接，配置不能证明JIT最终so启用了这些选项。OPP的intf_pub继承配置已正确设置，不能泛化为本算子所有构建都缺失硬化；未取得本次最终JIT链接命令/ELF，故仅SUSPICIOUS。
- **代码文件**：/home/chenpengyi/ops-transformer/torch_extension/cann_ops_transformer/op_builder/builder.py
- **起始行号**：175
- **中止行号**：185
- **代码片段**：
  ```python
          op_module = load(
              name=self.name,
              sources=self.get_absolute_paths(self.sources()),
              extra_include_paths=self.get_absolute_paths(self.include_paths()),
              extra_cflags=self.cxx_args(),
              extra_ldflags=self.extra_ldflags(),
              verbose=verbose,
          )
          OpBuilder._loaded_ops[self.name] = op_module

          return op_module
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款与所引用代码模式不一致 |
| 领域关联 | +10% | 已检查实际调用与构建继承；本项仍仅是规范/构建疑点，不构成新增功能故障证据 |

自信值 = Σ正向 + Σ负向 = 50% ≥ 70% → 未达违规阈值
- **修复建议**：在共享builder的链接参数通道添加适用链接选项，并用实际link命令/readelf检查最终so；属于共享既有配置，不计入最近四提交新增功能缺陷。

### [COMPILE-5] 不可执行栈
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：同一共享builder配置疑点，四个编译条例应合并成一条附录建议：cxx_args包含-Wl,-z,relro,-z,now,-z,noexecstack及-s，但load将它们只传入extra_cflags；extra_ldflags仅提供-L/-l。编译阶段-c不会执行链接，配置不能证明JIT最终so启用了这些选项。OPP的intf_pub继承配置已正确设置，不能泛化为本算子所有构建都缺失硬化；未取得本次最终JIT链接命令/ELF，故仅SUSPICIOUS。
- **代码文件**：/home/chenpengyi/ops-transformer/torch_extension/cann_ops_transformer/op_builder/builder.py
- **起始行号**：175
- **中止行号**：185
- **代码片段**：
  ```python
          op_module = load(
              name=self.name,
              sources=self.get_absolute_paths(self.sources()),
              extra_include_paths=self.get_absolute_paths(self.include_paths()),
              extra_cflags=self.cxx_args(),
              extra_ldflags=self.extra_ldflags(),
              verbose=verbose,
          )
          OpBuilder._loaded_ops[self.name] = op_module

          return op_module
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款与所引用代码模式不一致 |
| 领域关联 | +10% | 已检查实际调用与构建继承；本项仍仅是规范/构建疑点，不构成新增功能故障证据 |

自信值 = Σ正向 + Σ负向 = 50% ≥ 70% → 未达违规阈值
- **修复建议**：在共享builder的链接参数通道添加适用链接选项，并用实际link命令/readelf检查最终so；属于共享既有配置，不计入最近四提交新增功能缺陷。

### [COMPILE-6] 发布符号strip
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：同一共享builder配置疑点，四个编译条例应合并成一条附录建议：cxx_args包含-Wl,-z,relro,-z,now,-z,noexecstack及-s，但load将它们只传入extra_cflags；extra_ldflags仅提供-L/-l。编译阶段-c不会执行链接，配置不能证明JIT最终so启用了这些选项。OPP的intf_pub继承配置已正确设置，不能泛化为本算子所有构建都缺失硬化；未取得本次最终JIT链接命令/ELF，故仅SUSPICIOUS。
- **代码文件**：/home/chenpengyi/ops-transformer/torch_extension/cann_ops_transformer/op_builder/builder.py
- **起始行号**：175
- **中止行号**：185
- **代码片段**：
  ```python
          op_module = load(
              name=self.name,
              sources=self.get_absolute_paths(self.sources()),
              extra_include_paths=self.get_absolute_paths(self.include_paths()),
              extra_cflags=self.cxx_args(),
              extra_ldflags=self.extra_ldflags(),
              verbose=verbose,
          )
          OpBuilder._loaded_ops[self.name] = op_module

          return op_module
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 条款与所引用代码模式不一致 |
| 领域关联 | +10% | 已检查实际调用与构建继承；本项仍仅是规范/构建疑点，不构成新增功能故障证据 |

自信值 = Σ正向 + Σ负向 = 50% ≥ 70% → 未达违规阈值
- **修复建议**：在共享builder的链接参数通道添加适用链接选项，并用实际link命令/readelf检查最终so；属于共享既有配置，不计入最近四提交新增功能缺陷。

## 代码风格

> 来自 cpp-style 检视，不走假设检验，违反即 FAIL。不并入上方统计表。

| 条例 | 严重级别 | 问题描述 | 代码位置（校对后行号） | 修复建议 |
|------|---------|---------|----------------------|---------|
| STYLE-1.4 变量命名采用小驼峰风格 | 中 | 新增Torch C++封装内部变量y_rows、y_cols、scale_shape等采用snake_case，与小驼峰规范不一致。 | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/torch_extension/csrc/ffn_worker_batching.cpp:78-89 | 内部局部变量改为yRows、yCols、scaleShape；对外schema名称保持兼容。 |
| STYLE-2.1 行宽不超过120字符 | 低 | 新增MX类型的DataType初始化列表超过120字符且不是规范豁免的URL/include行。 | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/op_host/ffn_worker_batching_def.cpp:100-111 | 按类型列表语义分行并保持对齐。 |
| STYLE-2.8 多个变量定义不写在一行 | 低 | a、bs、k、h四个局部变量在同一语句定义初始化，违反单行单变量风格。 | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/torch_extension/csrc/ffn_worker_batching.cpp:48-59 | 拆成四个独立定义，便于逐变量注释与审查。 |
| STYLE-3.3 禁止遗留开发阶段注释 | 中 | UT注释仍包含“迭代一”开发阶段标记，与最终功能无关；不将有实际同步语义的算法phase描述混同于临时标记。 | /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/ut/op_host/test_ffn_worker_batching_tiling.cpp:73-84 | 删除“迭代一”并保留arch35核心路径的功能说明。 |
