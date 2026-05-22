---
name: op-loge-format
description: >-
  按 CANN 规范整改 ops-transformer 源码中 OP_LOGE / OP_LOGE_IF / OP_CHECK_IF 及 OP_LOGE_FOR_* 包装宏的错误日志文案。
  在用户说「  」「统一错误日志格式」「ErrMessage 规范」「BAD_PARAM / NOT_SUPPORTED 日志」、
  修改 op_host / infershape / tiling / op_api 错误日志，或扫描某算子目录下非标准 OP_LOGE 时使用。
disable-model-invocation: false
---

# OP_LOGE 错误日志格式规范（ops-transformer）

## 固定上下文

| 项目 | 值 |
|------|-----|
| **工作仓库** | `/home/xxp/ops-transformer` |
| **算子目录模式** | `{attention,mc2,moe,ffn,gmm,posembedding,mhc,experimental}/<op_name>/` |
| **主要整改范围** | `op_host/`（tiling / infershape / def）、`op_host/op_api/`、`framework/` |
| **默认跳过** | `tests/`、`build/`、`output/`、`3rdparty/`、`third_party/`、`.cursor/` |
| **容器验证** | `docker exec -u root zqj_9.0.0 bash -lc 'cd /home/xxp/ops-transformer && ...'` |
| **编译验证** | `bash build.sh -n <op_name> --ophost` 与 `bash build.sh -n <op_name> --opapi`（有 op_api 时两者都要过） |

> 参考算子：`attention/nsa_selected_attention_infer`（已完成整改并通过容器编译）。

## 适用范围

整改以下调用中的**错误日志 format 字符串**（及为适配 format 所需的最小实参调整）：

- `OP_LOGE` / `OP_LOGE_IF` / `OP_CHECK_IF(..., OP_LOGE(...), ...)`
- `OP_TILING_CHECK(..., OP_LOGE(...), ...)`
- `OPS_LOG_E`、`CUBE_INNER_ERR_REPORT`、`VECTOR_INNER_ERR_REPORT_*` 等最终落到 `OP_LOGE` 的宏
- CANN 包装宏（应**展开替换**为标准 `OP_LOGE`）：
  - `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON`
  - `OP_LOGE_FOR_INVALID_SHAPEDIM` / `OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON`
  - `OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON`
  - `OP_LOGE_FOR_INVALID_DTYPE` / `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON`
  - `OP_LOGE_WITH_INVALID_ATTR`

**不改**：`OP_LOGD` / `OP_LOGI` / `OP_LOGW`；`OPS_REPORT_VECTOR_INNER_ERR` / `REPORT_INNER_ERR_MSG`（另有规范）；kernel 侧 debug。

## 标准 ErrMessage 模板（verbatim）

| 错误分类 | 错误定义 | ErrMessage |
|:---|:---|:---|
| BAD_PARAM | 输入参数或属性不正确 | The op [%s] received bad params, the reason is: [%s] |
| BAD_PARAM | 缺失输入张量 | The op [%s] missing [%s] parameter |
| BAD_PARAM | 缺失输入属性 | The op [%s] missing [%s] attribute |
| BAD_PARAM | 输入的多个张量形状互相不匹配，比如需要一样形状的两个张量实际输入形状不一致 | In op [%s], the tensor shapes of [%s..] are mismatched, the reason is: [%s] |
| BAD_PARAM | 输入的多个张量数据类型互相不匹配，比如需要一样数据类型的两个张量实际输入类型不一致 | In op [%s], the data types of [%s..] are mismatched, the reason is: [%s] |
| NOT_SUPPORTED | 需要的功能不支持 | In op [%s], [%s] is not supported, got [%s] |
| NOT_SUPPORTED | 不支持的数据排布 | In op [%s], the format of [%s] is not supported, got [%s] |
| NOT_SUPPORTED | 不支持的数据类型 | In op [%s], the data type of [%s] is not supported, got [%s] |
| NOT_SUPPORTED | 输入的某个张量形状不支持，比如某个维度是0，或者输入形状太大超过存储限制 | In op [%s], the shape of [%s] is not supported, got [%s]. Constraint: [%s] |

模板中的 `[%s]` 为 `printf` 占位符；`[%s..]` 表示多个张量/参数名用逗号拼接成单个字符串填入一个 `%s`。

## 两层调用约定（关键）

ops-transformer 存在**两套 `OP_LOGE` 宏**，第一参数含义不同；**格式串统一用上表 9 种模板**，但第一参数按层区分。

### A. op_host 层（tiling / infershape / def）

头文件：`log/log.h`。签名：`OP_LOGE(op_name, fmt, ...)`。

```cpp
const char *op_name = context->GetNodeName();  // 或 context_->opName（已由 ConvertContext 赋值）

OP_LOGE(op_name, "The op [%s] missing [%s] parameter", op_name, "query");
OP_LOGE(op_name, "In op [%s], the data type of [%s] is not supported, got [%s]",
        op_name, "dy", dtypeStr.c_str());
```

**规则：**

- 第一参数必须是 **node name 字符串**，禁止传 `context` / `context_` 指针。
- 禁止硬编码 `"[NsaSelectInfer]"` 等缩写；用 `GetNodeName()` / `context_->opName`。
- `GetNodeName()` 为 `nullptr` 时，可用 PascalCase 算子名常量（如 `"NsaSelectedAttentionInfer"`）作 fallback。

### B. op_api 层（aclnn / l0op）

头文件：`opdev/op_log.h`（经 `opdev/op_dfx.h` 等间接引入）。签名：`OP_LOGE(errno, fmt, ...)`。

```cpp
namespace {
static const char kOpName[] = "NsaSelectedAttentionInfer";  // PascalCase，与 OpDef 一致
}

// 参数非法
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The op [%s] received bad params, the reason is: [%s]",
        kOpName, "QK head dim must be 192");

// 内部空指针 / 分配失败
OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The op [%s] missing [%s] parameter", kOpName, "attention_out");
```

**规则：**

- 第一参数必须是 **`ACLNN_ERR_*` 错误码**，**不能**传算子名字符串（会导致 `to_string` 编译错误）。
- 格式串中的 `The op [%s]` / `In op [%s]` 用 `kOpName` 或 `GetOpName().c_str()` 填充。
- 错误码选用：`ACLNN_ERR_PARAM_INVALID`（参数/shape/dtype 校验失败）、`ACLNN_ERR_INNER_NULLPTR`（内部 nullptr / 分配失败）。

### 通用规则

1. **格式串必须与上表 ErrMessage 完全一致**；动态细节放进 `reason` / `got` / `Constraint` 实参。
2. **parameter / attribute 名**用 CANN 注册名（`query`、`layout`、`num_heads`），不用 `"get xxx desc fail"` 句式。
3. **动态 reason** 用 `char reason[512]` + `snprintf`，format 仍用标准模板。
4. **包装宏必须展开**为标准 `OP_LOGE(...)`，不新增自定义包装宏。
5. **仅改日志字符串**（及最小实参）；不改校验逻辑、返回码、宏结构。

## 包装宏 → 标准模板

| 旧包装宏 | 对应标准模板 |
|---------|-------------|
| `OP_LOGE_FOR_INVALID_DTYPE(op, tensor, got, supported)` | `the data type of [%s] is not supported, got [%s]` |
| `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op, names, reason, ...)` | `the data types of [%s..] are mismatched, the reason is: [%s]` |
| `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(op, tensor, got, constraint)` | `the shape of [%s] is not supported, got [%s]. Constraint: [%s]` |
| `OP_LOGE_FOR_INVALID_SHAPEDIM(op, tensor, got, constraint)` | 同上 |
| `OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(op, "a and b", reason, ...)` | `the tensor shapes of [%s..] are mismatched, the reason is: [%s]` |
| `OP_LOGE_WITH_INVALID_ATTR(op, attr, got, expected)` | `received bad params, the reason is: [%s]`（reason 含 attr/got/expected） |

## 分类决策（先选模板再写文案）

```
nullptr / 未提供输入 tensor desc 或 shape？
  └─ YES → missing [%s] parameter

nullptr / 未提供 attr？
  └─ YES → missing [%s] attribute

多个 tensor 的 shape 必须相同但不相同？
  └─ YES → tensor shapes of [%s..] are mismatched

多个 tensor 的 dtype 必须相同但不相同？
  └─ YES → data types of [%s..] are mismatched

单个 tensor dtype 不在支持列表？
  └─ YES → the data type of [%s] is not supported, got [%s]

单个 tensor format 不在支持列表？
  └─ YES → the format of [%s] is not supported, got [%s]

单个 tensor shape 非法（dim=0、维数不对、超限等）？
  └─ YES → the shape of [%s] is not supported, got [%s]. Constraint: [%s]

某功能/模式/枚举值不支持？
  └─ YES → [%s] is not supported, got [%s]

属性值越界、组合非法、内部步骤失败、其它参数错误？
  └─ YES → received bad params, the reason is: [%s]
```

## 整改工作流

### 1. 定位算子目录

```bash
cd /home/xxp/ops-transformer
OP=nsa_selected_attention_infer
find . -type d -name "$OP" -not -path './build/*' -not -path './.git/*'
```

### 2. 扫描待整改日志

```bash
grep -rn 'OP_LOGE' attention/$OP/op_host --include='*.cpp' --include='*.h' \
  | grep -v 'The op \[\\|In op \['
```

### 3. 逐条改写

1. 判断文件属于 **op_host 层** 还是 **op_api 层**，选用对应第一参数规则；
2. 按「分类决策」选定模板；
3. 展开 `OP_LOGE_FOR_*` / `OP_LOGE_WITH_*`；
4. op_host：修正 `context` 指针为第一参数的常见错误；
5. 保持 `OP_CHECK_IF` / `OP_TILING_CHECK` 宏结构不变。

### 4. 自检

- [ ] 每条 format 串匹配上表 9 种之一
- [ ] op_host：`OP_LOGE` 第一参数为 node name，非 context 指针
- [ ] op_api：`OP_LOGE` 第一参数为 `ACLNN_ERR_*`，非算子名字符串
- [ ] 无 `OP_LOGE_FOR_*` / `OP_LOGE_WITH_*` 残留
- [ ] `missing` 类区分 `parameter` vs `attribute`
- [ ] 未改动非日志逻辑

### 5. 容器内编译验证

```bash
docker exec -u root zqj_9.0.0 bash -lc '
  cd /home/xxp/ops-transformer
  OP=nsa_selected_attention_infer
  echo "=== scan ==="
  grep -rn OP_LOGE attention/$OP/op_host --include="*.cpp" \
    | grep -v "The op \[\\|In op \[" || echo "(none)"
  echo "=== ophost ==="
  bash build.sh -n $OP --ophost
  echo "=== opapi ==="
  bash build.sh -n $OP --opapi
'
```

opapi 若报 `no matching function for call to 'to_string(const char*')'`，说明把算子名误传为 `OP_LOGE` 第一参数，改回 `ACLNN_ERR_*`。

## 常见旧式 → 新式对照

详细示例见 [references/mapping-guide.md](references/mapping-guide.md)。

| 旧式特征 | 新模板 | 备注 |
|---------|--------|------|
| `OP_LOGE(context, "xxx fail")` | 按语义选 9 种之一 | op_host：第一参数改 `GetNodeName()` |
| `OP_LOGE(ACLNN_ERR_*, "invalid dtype")` | data type not supported | op_api：保留 `ACLNN_ERR_*` 第一参数 |
| `OP_LOGE("[NsaSelectInfer]", ...)` | 标准模板 + `context_->opName` | 禁止硬编码缩写 |
| `OP_LOGE_FOR_INVALID_*` | 展开为标准 OP_LOGE | 见包装宏表 |
| shape/dim 带 `%ld`/`%u` 的旧式整句 | shape not supported 或 bad params | 数值放进 got/reason，用 snprintf |
