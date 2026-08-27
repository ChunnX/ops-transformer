# PR #10056 检视报告 — 新增 AttentionToFFNV2、FFNToAttentionV2 算子

**PR 概述**：59 文件，+6611/-368 行。新增 V2 算子（arch35 + MX FP8/FP4 量化），将 V1 tiling 重构为共享 base。

**注意**：PR 描述中嵌入的 CANNBot 审查报告针对较早 commit，以下已修复的问题不再列出：
- SEC-3.5（判空前解引用）：已修复，`CheckAndSetAttrs` 中判空已前置
- SEC-11.2（LOG 格式串）：已修复，`"%."` → `"%u."`，`HS is %u` → `HS is %lu` 等

## 中 — Bug

### 1. `ffn_to_attention_v2_tilling.cpp:69-70` 缺少 tilingData 判空

```cpp
FFNToAttentionV2TilingData *tilingData = context->GetTilingData<FFNToAttentionV2TilingData>();
bool rankTableMode = tilingData->ffnToAttentionV2Info.isInputRankTable;  // 未判空即解引用
```

`GetTilingData` 可能返回 nullptr。同仓库的 `attention_to_ffn_v2_tiling.cpp:166` 在相同调用后正确执行了 `tilingData == nullptr` 检查。应保持一致，在解引用前添加 `OP_TILING_CHECK(tilingData == nullptr, ...)`。

### 2. 两个 V2 op_api 文件的 `CheckNullStatus` 遗漏 `context` 参数

`aclnn_attention_to_ffn_v2.cpp:34` 和 `aclnn_ffn_to_attention_v2.cpp:33`：

`context` 是必需输入，直接传入 Inner 函数，但 `CheckNullStatus` 未对其校验。`workspaceSize` 和 `executor` 输出参数也未检查。如果调用方传入 `context == nullptr`，会未经验证地到达 `aclnnInnerAttentionToFfnV2GetWorkspaceSize`。

应添加 `OP_CHECK_NULL(context, return false);` 及输出指针检查。

## 低 — 代码质量

### 3. 校验错误信息缺少实际值

`attention_to_ffn_v2_tiling.cpp:251-257` 中三条 `OP_TILING_CHECK` 失败消息不包含具体数值：

- `"expert_ids shape is invalid"` — 未说明哪个维度不匹配及实际值
- `"expert count is invalid"` — 未包含 `moeExpertNum` 或 `expertNum`
- `"communication attribute is invalid"` — 未指出 `worldSize`/`syncFlag`/`ffnStartRankId`/`cclBufferSize` 哪个失败

用户无法从这些消息定位根因。V1 tiling 使用 `OP_LOGE_FOR_INVALID_VALUE` 带实际值，应保持一致。

### 4. Python 死代码

- `attention_to_ffn_v2.py:153` 和 `ffn_to_attention_v2.py:123`：`_get_npu_backend` 已定义但从未调用
- `ffn_to_attention_v2.py:200-205`：存在注释掉的代码块

应移除。

### 5. 缺少换行

`attention_to_ffn_v2_tiling.cpp:~390` 函数末尾闭括号与 return 语句同行：

```cpp
                          info.syncFlag == 1U, info.isActiveMask);}
```

闭括号应另起一行。
