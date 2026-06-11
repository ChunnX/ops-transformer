# 代码检视报告

## 检视概览
- 代码文件：/mnt/workspace/transformer-issue/attention/lightning_indexer/op_host/lightning_indexer_tiling.cpp
- 代码侧别：Tiling
- 检视文档：cpp-secure.md, cpp-general.md, ascendc-topk.md
- 总条例数：85
- 检视时间：2026-06-11

## 检视统计
| 状态 | 条例数 | 占比 |
|------|------|------|
| PASS | 72 | 84.7% |
| FAIL（发现问题）| 7 | 8.2% |
| SUSPICIOUS（需关注）| 6 | 7.1% |

## 发现问题（HIGH 置信度）

### [SEC-3.5] 指针使用前判空 / [SEC-1.2] 保证内存安全 / [SEC-10.1] 禁止从空指针创建std::string / [TOPK-#1] 必须校验函数返回值 / [TOPK-#7] Tiling外部输入校验

> 以上 5 条条例指向同一组缺陷，合并描述。

- **问题描述**：`GetAndCheckAttrParaInfo()` 函数中，7 个属性指针（`layOut`、`layOutKey`、`sparseCount`、`sparseMode`、`preTokens`、`nextTokens`、`returnValue`）通过 `attrs->GetStr()` / `attrs->GetAttrPointer<T>()` 获取后，在 lines 232-263 被直接解引用或传入 `std::string()` 构造函数，但**之前没有空指针校验**。其中 `layOut`/`layOutKey`/`sparseCount`/`sparseMode` 的空指针校验在 `CheckRequiredAttrExistence()` 中，但该函数在 `ParseAndCheck()` 调用链中位于 `GetAndCheckAttrParaInfo()` **之后**（line 853 vs line 854），形成"先用后验"的无效防御。`preTokens`、`nextTokens`、`returnValue` 则**在整个代码中都没有空指针校验**。

  同一函数内 lines 210-230 的日志代码显式检查了这些指针是否为 null（如 `if (opParamInfo_.layOutKey != nullptr)`），证明开发者意识到这些指针可能为 null，但 lines 232-263 的校验代码未做同样保护。

- **代码片段**（lightning_indexer_tiling.cpp 行 197-266）：
  ```cpp
  197: ge::graphStatus LIInfoParser::GetAndCheckAttrParaInfo()
  198: {
  199:     auto attrs = context_->GetAttrs();
  200:     OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(...), return ge::GRAPH_FAILED);
  201:     OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
  202:     opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
  203:     opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
  204:     opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
  205:     opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
  206:     opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
  207:     opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
  208:     opParamInfo_.returnValue = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
  209:     // ... 条件日志(有nullptr判断，仅控制日志) ...
  232:     OP_CHECK_IF(
  233:         ((std::string(opParamInfo_.layOutKey) != "PA_BSND")            // layOutKey 可能 nullptr → UB
  234:         && (std::string(opParamInfo_.layOut) != std::string(opParamInfo_.layOutKey))),  // layOut 可能 nullptr → UB
  235:         ...
  247:     OP_CHECK_IF((!((*opParamInfo_.sparseCount > 0) && ...)),           // sparseCount 可能 nullptr
  252:     OP_CHECK_IF(!((*opParamInfo_.sparseMode == 0) || ...),             // sparseMode 可能 nullptr
  255:     OP_CHECK_IF(*opParamInfo_.preTokens != INT64_MAX,                  // preTokens 全局无 nullptr 校验
  258:     OP_CHECK_IF(*opParamInfo_.nextTokens != INT64_MAX,                 // nextTokens 全局无 nullptr 校验
  261:     OP_CHECK_IF(*opParamInfo_.returnValue && ...,                      // returnValue 全局无 nullptr 校验
  ```

- **调用链证据**（lightning_indexer_tiling.cpp 行 850-856）：
  ```cpp
  850: ge::graphStatus LIInfoParser::ParseAndCheck(LITilingInfo &liInfo)
  851: {
  852:     if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() ||
  853:         ge::GRAPH_SUCCESS != GetOpParaInfo() ||              // ← 先调用，内部解引用属性指针
  854:         ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) { // ← 后调用，内部才做 nullptr 校验
  855:         return ge::GRAPH_FAILED;
  856:     }
  ```

- **假设检验证据**：

  正向证据：
  | 证据类型 | 分值 | 证据描述 |
  |---------|------|---------|
  | 规范违反 | +40% | 7 个属性指针在 lines 232-263 直接解引用，无 nullptr 校验 |
  | 上下文防御缺失 | +30% | CheckRequiredAttrExistence() 在 GetAndCheckAttrParaInfo() 之后才执行；preTokens/nextTokens/returnValue 全局无校验 |
  | 领域关联 | +15% | 代码为 Tiling 函数，完全在 TOPK-#7 适用范围 [适用: Host] 内 |

  负向证据：
  | 证据类型 | 分值 | 证据描述 |
  |---------|------|---------|
  | 条件日志 | -0% | lines 210-230 有 nullptr 判断但仅控制日志输出，不阻止后续解引用 |

  自信值 = 40% + 30% + 15% = **85%** ≥ 70% → 判定违规

- **修复建议**：
  1. 在 `CheckRequiredAttrExistence()` 中补充 `preTokens`/`nextTokens`/`returnValue` 的 nullptr 校验
  2. 将 `CheckRequiredParaExistence()` 的调用移到 `GetOpParaInfo()` **之前**，或在 `GetAndCheckAttrParaInfo()` 内部、解引用之前增加 nullptr 校验

  ```cpp
  // 方案: 在 GetAndCheckAttrParaInfo() 中，lines 209 之后增加校验
  OP_CHECK_IF(opParamInfo_.layOut == nullptr || opParamInfo_.layOutKey == nullptr ||
              opParamInfo_.sparseCount == nullptr || opParamInfo_.sparseMode == nullptr ||
              opParamInfo_.preTokens == nullptr || opParamInfo_.nextTokens == nullptr ||
              opParamInfo_.returnValue == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Required attr pointer is nullptr"),
              return ge::GRAPH_FAILED);
  ```

---

### [SEC-11.2] LOG API参数必须与格式化占位符逐位一致

- **问题描述**：多处 LOG 调用的格式化占位符与参数类型不匹配。`int64_t` 类型使用 `%d`（仅读取 4 字节，64 位值被截断），`uint32_t` 类型使用 `%d`（应为 `%u`）。

- **代码片段**（lightning_indexer_tiling.cpp 行 222-227）：
  ```cpp
  222:     if (opParamInfo_.preTokens != nullptr) {
  223:         OP_LOGI(context_->GetNodeName(), "pre tokens is:%d", *opParamInfo_.preTokens);   // %d for int64_t → 截断
  224:     }
  225:     if (opParamInfo_.nextTokens != nullptr) {
  226:         OP_LOGI(context_->GetNodeName(), "next tokens is:%d", *opParamInfo_.nextTokens); // %d for int64_t → 截断
  227:     }
  ```

- **代码片段**（lightning_indexer_tiling.cpp 行 601-604）：
  ```cpp
  601:     maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
  602:     s2Size_ = maxBlockNumPerBatch_ * blockSize_;
  603:     OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
  604:               maxBlockNumPerBatch_, blockSize_, s2Size_);
  //               %d for uint32_t    %d for int32_t  %d for int64_t
  ```

- **代码片段**（lightning_indexer_tiling.cpp 行 461, 504）：
  ```cpp
  461:     OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);  // n1Size_ 为 uint32_t，%d 应为 %u
  504:     OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_); // n2Size_ 为 uint32_t，%d 应为 %u
  ```

- **假设检验证据**：

  正向证据：
  | 证据类型 | 分值 | 证据描述 |
  |---------|------|---------|
  | 规范违反 | +40% | 3 处 int64_t 误用 %d（lines 223, 226, 603），2 处 uint32_t 误用 %d（lines 461, 504, 603） |
  | 重复模式 | +40% | 多处独立证据叠加 |

  自信值 = **80%** ≥ 70% → 判定违规

- **修复建议**：

  | 行号 | 当前 | 修复 |
  |------|------|------|
  | 223 | `"pre tokens is:%d"` | `"pre tokens is:%lld"` |
  | 226 | `"next tokens is:%d"` | `"next tokens is:%lld"` |
  | 603 | `"%d, %d, %d"` | `"%u, %d, %lld"` |
  | 461 | `"n1Size is %d"` | `"n1Size is %u"` |
  | 504 | `"n2Size_ is %d"` | `"n2Size_ is %u"` |

---

### [TOPK-#5] 属性获取类型需与ir原型一致

- **问题描述**：`sparse_count` 和 `sparse_mode` 两个属性使用 `GetAttrPointer<int32_t>` 获取，但 ir 原型（`lightning_indexer_def.cpp`）中定义为 `.Int()` 类型，对应 C++ 类型应为 `int64_t`。类型不匹配会导致以错误类型重新解释底层内存。单元测试也以 `int64_t` 创建这两个属性（`test_lightning_indexer_tiling.cpp:52-53`），进一步证实类型不一致。

- **代码片段**（lightning_indexer_tiling.cpp 行 197-209）：
  ```cpp
  197: ge::graphStatus LIInfoParser::GetAndCheckAttrParaInfo()
  198: {
  199:     auto attrs = context_->GetAttrs();
  200:     OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(...), return ge::GRAPH_FAILED);
  201:     OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
  202:     opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
  203:     opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
  204:     opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);  // 应为 int64_t
  205:     opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);   // 应为 int64_t
  206:     opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);     // 正确
  207:     opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);   // 正确
  208:     opParamInfo_.returnValue = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);    // 正确
  209: }
  ```

- **假设检验证据**：

  正向证据：
  | 证据类型 | 分值 | 证据描述 |
  |---------|------|---------|
  | 规范违反 | +40% | GetAttrPointer<int32_t> 与 IR .Int() 类型不匹配 |
  | 交叉验证 | +30% | 单元测试以 int64_t 创建属性，证实 IR 层以 int64_t 存储 |
  | 头文件同步错误 | +15% | lightning_indexer_tiling.h:118-119 声明为 const int32_t*，应为 const int64_t* |

  自信值 = **85%** ≥ 70% → 判定违规

- **修复建议**：
  1. `lightning_indexer_tiling.h:118-119`：将 `sparseMode` 和 `sparseCount` 类型改为 `const int64_t *`
  2. `lightning_indexer_tiling.cpp:205-206`：将 `GetAttrPointer<int32_t>` 改为 `GetAttrPointer<int64_t>`
  3. 级联检查所有使用 `*opParamInfo_.sparseCount` 和 `*opParamInfo_.sparseMode` 的位置，确保赋值目标变量类型兼容

## 需关注（MED 置信度）

### [SEC-2.2] 无符号整数运算不回绕

- **问题描述**：`s2Size_ = maxBlockNumPerBatch_ * blockSize_`（line 602），`uint32_t * int32_t` 乘法在 uint32_t 空间计算，结果可能超过 UINT32_MAX 后回绕，再赋值给 int64_t。`blockSize_` 已约束 ≤ 1024，将回绕临界值推高至 maxBlockNumPerBatch_ > 4,194,303，实际风险低但代码无显式防护。

- **代码片段**（lightning_indexer_tiling.cpp 行 593-606）：
  ```cpp
  593: ge::graphStatus LIInfoParser::GetS2SizeForPageAttention()
  594: {
  595:     if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
  596:         return ge::GRAPH_FAILED;
  597:     }
  598:     if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
  599:         return ge::GRAPH_FAILED;
  600:     }
  601:     maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
  602:     s2Size_ = maxBlockNumPerBatch_ * blockSize_;   // uint32_t * int32_t → uint32_t 回绕风险
  603:     OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
  604:               maxBlockNumPerBatch_, blockSize_, s2Size_);
  605:     return ge::GRAPH_SUCCESS;
  606: }
  ```

- **修复建议**：乘法前提升为 uint64_t：`s2Size_ = static_cast<uint64_t>(maxBlockNumPerBatch_) * static_cast<uint64_t>(blockSize_);`

---

### [SEC-4.1] 外部输入合法性校验 / [GEN-1.1] 外部数据合法性检查

> 与 SEC-3.5/TOPK-#7 指向同一组缺陷，已在「发现问题」中详细描述。

- **问题描述**：`preTokens`/`nextTokens`/`returnValue` 三个外部属性指针在解引用前缺少空指针校验。

---

### [SEC-11.4] LOG消息英语行文语法正确

- **问题描述**：多处 LOG 消息存在语法错误、拼写错误或 copy-paste 错误。

- **问题清单**：

  | 行号 | 问题 | 修复 |
  |------|------|------|
  | 69 | `"datatype %d not support"` | `"datatype %d is not supported"` |
  | 155 | `"SOC Version[%d] is not support."` | `"SOC Version[%d] is not supported."` |
  | 217 | `"selscted count is:%d"` | `"selected count is:%d"` |
  | 254 | `"sparse_count must be 0 or 3."` (校验的是 sparse_mode) | `"sparse_mode must be 0 or 3."` |
  | 488 | `liActualSeqLenName + "should"` (缺少空格) | `liActualSeqLenName + " should"` |
  | 553 | `"head_dim only support 128"` (主谓不一致) | `"head_dim only supports 128"` |
  | 20+处 | `"must be same"` (缺少冠词 "the") | `"must be the same"` |

---

### [GEN-4.2] 禁止使用魔鬼数字/字符串

- **问题描述**：多处业务校验逻辑使用未命名的数字字面量，而头文件已定义大量 constexpr 常量。

- **具体问题**：
  - Line 459: `GetDim(1)` — 应使用 `DIM_IDX_ONE`（头文件已定义）
  - Line 577: `blockSize_ % 16` 和 `blockSize_ > 1024` — 应定义 `BLOCK_SIZE_ALIGNMENT = 16` 和 `BLOCK_SIZE_UPPER_LIMIT = 1024`
  - Line 248: `sparseCount % 1024` — 应定义 `SPARSE_COUNT_ALIGNMENT = 1024`

---

### [GEN-10.6] 指针/引用形参不修改用const

- **问题描述**：`LightningIndexerTiling::DoTiling(LITilingInfo *tilingInfo)`（line 922）的指针形参在函数体内仅被读取（20+ 处 `tilingInfo->xxx` 均为只读），从未被修改，应声明为 `const LITilingInfo *tilingInfo`。

- **代码片段**（lightning_indexer_tiling.cpp 行 922-948）：
  ```cpp
  922: ge::graphStatus LightningIndexerTiling::DoTiling(LITilingInfo *tilingInfo)
  923: {
  924:     auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);  // 只读
  925:     uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
  926:     uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
  927:     uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
  928:     context_->SetBlockDim(blockDim);
  929:     uint64_t workspaceSize = LiCalcWorkspaceSize(ascendcPlatform, tilingInfo->s2Size, aicNum);  // 只读
  930:     size_t *workSpaces = context_->GetWorkspaceSizes(1);
  931:     workSpaces[0] = workspaceSize;
  932:     tilingData_.set_bSize(tilingInfo->bSize);           // 只读
  933:     tilingData_.set_s2Size(tilingInfo->s2Size);         // 只读
  934:     tilingData_.set_s1Size(tilingInfo->s1Size);         // 只读
  935:     tilingData_.set_sparseCount(tilingInfo->sparseCount); // 只读
  ```

- **修复建议**：声明和定义中参数改为 `const LITilingInfo *`

## 疑似（LOW 置信度）

无

## 通过条例

SEC-1.1 保证静态类型安全
SEC-1.3 禁止使用未定义行为
SEC-2.1 有符号整数运算不溢出
SEC-2.3 除法/余数运算除零保护
SEC-3.1 禁止使用未初始化的变量
SEC-3.2 资源释放后指针置新值
SEC-3.3 数组索引校验
SEC-3.4 禁止sizeof指针
SEC-3.6 字符串存储有足够空间
SEC-4.2 内存操作长度校验
SEC-5.1 资源申请后判断是否成功
SEC-5.2 资源泄露防护
SEC-5.3 new/delete配对使用
SEC-5.4 new操作符错误处理
SEC-8.1 使用安全函数替代危险函数
SEC-8.2 正确设置安全函数destMax参数
SEC-8.3 检查安全函数返回值
SEC-9.1 禁止逐位操作非trivially copyable对象
SEC-10.2 不要保存c_str/data指针
SEC-10.3 敏感信息使用后清零
SEC-10.4 结构体字段末尾添加
SEC-10.5 接口变更考虑兼容性
SEC-11.1 LOG API禁止传入空指针
SEC-11.3 LOG API禁止传入已释放内存的指针
GEN-1.3 清理无效冗余代码
GEN-2.2 禁止头文件循环依赖
GEN-2.3 避免包含用不到的头文件
GEN-2.4 禁止extern声明引用外部接口
GEN-2.5 禁止在extern "C"中包含头文件
GEN-2.6 避免在头文件中使用using导入命名空间
GEN-3.1 避免滥用typedef/#define类型别名
GEN-3.2 使用using而非typedef定义别名
GEN-4.3 每个常量保证单一职责
GEN-5.2 避免全局变量谨慎使用单例
GEN-5.3 禁止变量自增/自减表达式中再次引用
GEN-6.1 表达式比较左变右不变
GEN-6.2 使用括号明确操作符优先级
GEN-7.1 使用C++类型转换而非C风格
GEN-8.1 switch语句要有default分支
GEN-9.1 禁止用memcpy_s/memset_s初始化非POD
GEN-10.2 优先使用unique_ptr而非shared_ptr
GEN-10.3 使用make_shared而非new创建shared_ptr
GEN-10.4 使用智能指针管理对象
GEN-10.5 禁止使用auto_ptr
GEN-11.1 字符串存储确保有'\0'结束符
GEN-12.1 断言不能用于运行期错误处理
GEN-13.1 delete/delete[]配对使用
GEN-13.2 禁止std::move操作const对象
GEN-13.3 严格使用virtual/override/final
GEN-14.1 使用RAII追踪动态分配
GEN-14.2 非局部lambda避免按引用捕获
TOPK-#2 使用GetInputDesc获取Dtype
TOPK-#3 生命周期内使用局部变量指针避免野指针
TOPK-#4 属性从context获取禁止CompileInfo传递
TOPK-#6 特殊值和边界值处理
TOPK-#8 gm内存偏移或大小必须用int64表示
TOPK-#10 可整数计算时不允许转浮点数计算
TOPK-#12 宏定义中临时变量命名不能和外部变量冲突
TOPK-#13 dlopen管理的so禁用thread_local
