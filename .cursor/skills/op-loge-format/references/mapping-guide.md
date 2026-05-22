# OP_LOGE 整改对照指南（ops-transformer）

工作目录：`/home/xxp/ops-transformer`。参考算子：`attention/nsa_selected_attention_infer`。

## 0. 两层 OP_LOGE 第一参数（必读）

| 层级 | 文件位置 | 头文件 | 第一参数 | 格式串中 op 名 |
|------|---------|--------|---------|---------------|
| op_host | `*_tiling*.cpp`、`*_infershape.cpp` | `log/log.h` | `GetNodeName()` / `context_->opName` | 同上 |
| op_api | `op_api/aclnn_*.cpp`、`op_api/*.cpp` | `opdev/op_log.h` | `ACLNN_ERR_PARAM_INVALID` 等 | `kOpName` 常量 |

```cpp
// op_host — 错误：传 context 指针
OP_LOGE(context, "The inputLayout should be BSND...");

// op_host — 正确
OP_LOGE(context->GetNodeName(), "In op [%s], [%s] is not supported, got [%s]",
        context->GetNodeName(), "input_layout", inputLayoutStr.c_str());
```

```cpp
// op_api — 错误：传算子名（编译报 to_string 错误）
OP_LOGE("NsaSelectedAttentionInfer", "The op [%s] received bad params...");

// op_api — 正确
static const char kOpName[] = "NsaSelectedAttentionInfer";
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The op [%s] received bad params, the reason is: [%s]",
        kOpName, "NsaSelectedAttentionInfer InferShape failed");
```

## 1. op_host — missing parameter

```cpp
// Before
OP_CHECK_IF(context_->query.shape == nullptr,
    OP_LOGE(context_->opName, "Shape of tensor query is nullptr"), return ge::GRAPH_FAILED);

// After
const char *op_name = context_->opName;
OP_CHECK_IF(context_->query.shape == nullptr,
    OP_LOGE(op_name, "The op [%s] missing [%s] parameter", op_name, "query"),
    return ge::GRAPH_FAILED);
```

## 2. op_host — missing attribute

```cpp
// Before
OP_CHECK_IF(context_->numHeads == nullptr,
    OP_LOGE(context_->opName, "attr numHeads is nullptr"), return ge::GRAPH_FAILED);

// After
OP_LOGE(op_name, "The op [%s] missing [%s] attribute", op_name, "num_heads");
```

## 3. op_host — received bad params（含 snprintf）

```cpp
// Before
OP_LOGE(context_->opName, "blockSize=%ld, it need align to 16 when kv dtype is fp16/bf16.", blockSize_);

// After
char reason[512] = {0};
OP_LOGE(op_name, "The op [%s] received bad params, the reason is: [%s]", op_name,
        (snprintf(reason, sizeof(reason),
                  "blockSize=%ld should be 16 aligned when kv dtype is fp16/bf16", blockSize_) > 0)
            ? reason
            : "blockSize should be 16 aligned when kv dtype is fp16/bf16");
```

## 4. op_host — shape not supported

```cpp
// Before
OP_LOGE(context_->opName, "QK Head dim must == 192, but got: %u.", headDim_);

// After
OP_LOGE(op_name, "In op [%s], the shape of [%s] is not supported, got [%s]. Constraint: [%s]", op_name,
        "query",
        (snprintf(reason, sizeof(reason), "head dim %u", headDim_) > 0) ? reason : "invalid head dim",
        "QK head dim must be 192");
```

## 5. op_host — tensor shapes / dtypes mismatched

```cpp
// Before
OP_LOGE(context_->opName, "when input Q type %d is not equal to input KV type %d", inputQType_, inputKvType_);

// After
OP_LOGE(op_name, "In op [%s], the data types of [%s..] are mismatched, the reason is: [%s]", op_name,
        "query, key, value", "query dtype is not equal to kv dtype");
```

```cpp
// Before — layout 不支持
OP_LOGE(context_->opName, "Only support input_layout(BSH, BSND, TND), actually is %s", layout.c_str());

// After
OP_LOGE(op_name, "In op [%s], [%s] is not supported, got [%s]", op_name, "input_layout", layout.c_str());
```

## 6. op_host — 硬编码 `[NsaSelectInfer]` → context_->opName

```cpp
// Before
OP_LOGE("[NsaSelectInfer]", "When input layout is BSH, the dimension should be 3, dimOfQ: %lu...", dimOfQ);

// After
OP_LOGE(op_name, "In op [%s], the shape of [%s] is not supported, got [%s]. Constraint: [%s]", op_name,
        "query, key, output",
        (snprintf(reason, sizeof(reason), "dimOfQ=%lu dimOfK=%lu dimOfOut=%lu", dimOfQ, dimOfK, dimOfOut) > 0)
            ? reason : "invalid dim num",
        "dim num should be 3 when input layout is BSH");
```

## 7. op_api — DtypeCheck

```cpp
// Before
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type does not meet requirements.");
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "topk dtype does not match with required dtype int32.");

// After
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], the data type of [%s] is not supported, got [%s]",
        kOpName, "query", op::ToString(DataType(qDtype)).GetString());
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], the data type of [%s] is not supported, got [%s]",
        kOpName, "topk_indices", op::ToString(DataType(topkDtype)).GetString());
```

## 8. op_api — layout / shape 校验

```cpp
// Before
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayout, shapeInfo.dimNum);

// After
char reason[512] = {0};
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], [%s] is not supported, got [%s]", kOpName, "input_layout",
        (snprintf(reason, sizeof(reason), "%s with dim_num %lu", inputLayout, shapeInfo.dimNum) > 0)
            ? reason : inputLayout);
```

```cpp
// Before
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", ...);

// After
OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "In op [%s], the tensor shapes of [%s..] are mismatched, the reason is: [%s]", kOpName,
        "query, key",
        (snprintf(reason, sizeof(reason), "qD=%ld kD=%ld", shapeInfo.axes.d, shapeInfo.axes.dk) > 0)
            ? reason : "qD and kD mismatch");
```

## 9. 展开 OP_LOGE_FOR_* / OP_LOGE_WITH_*

与 posembedding / attention 算子相同，见 SKILL.md 包装宏表。展开后仍遵守 op_host / op_api 第一参数规则。

## 10. 不宜机械替换

| 场景 | 处理 |
|------|------|
| `OPS_REPORT_VECTOR_INNER_ERR(...)` | 不整改（非 OP_LOGE 格式串） |
| `tests/` 下 UT | 默认跳过 |
| op_api 无 `GetNodeName()` 的 static 函数 | 用 `kOpName` 填格式串中的 `[%s]` |
| op_api 编译 `to_string(const char*)` 报错 | 第一参数改回 `ACLNN_ERR_*` |

## 11. 容器内验证命令

```bash
docker exec -u root zqj_9.0.0 bash -lc '
  cd /home/xxp/ops-transformer
  OP=nsa_selected_attention_infer
  grep -rn OP_LOGE attention/$OP/op_host --include="*.cpp" \
    | grep -v "The op \[\\|In op \[" || echo "format OK"
  bash build.sh -n $OP --ophost && bash build.sh -n $OP --opapi
'
```
