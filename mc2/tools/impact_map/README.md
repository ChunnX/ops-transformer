# mc2 common/3rd 文件影响范围映射（impact_map）

## 1. 背景与用途

mc2 算子的公共代码集中在 `mc2/common` 与 `mc2/3rd`，改动这些目录时无法用算子目录级
依赖（CMake `*_depends`）评估影响面。本目录提供**文件级**的反向影响映射：

> 输入：PR 修改的 common/3rd 文件路径 + 目标芯片
> 输出：需要回归测试的 mc2 算子清单

**每个文件恰好归入两类之一：测全量 / 测部分算子**（不存在不测的情况）。

## 2. 目录结构

```
mc2/tools/
├── gen_common_impact_yaml.py    # 生成器（本表唯一来源，重新生成会覆盖yaml）
└── impact_map/
    ├── common/                  # mc2/common 头文件的映射（仅含"测部分算子"的头）
    │   ├── A2.yaml / A3.yaml / A5.yaml
    ├── 3rd/                     # mc2/3rd 头文件的映射（仅含"测部分算子"的头）
    │   ├── A2.yaml / A3.yaml / A5.yaml
    └── file_lists.yaml             # 全量匹配清单（按目录分放，common/与3rd/各一份）
```

芯片对应关系：A2=ascend910b，A3=ascend910_93，A5=ascend950。
A2/A3/A5 三张芯片表**键集合完全一致**（同为"测部分算子"的头，按芯片取算子子集）。

## 3. 生成方法

```bash
# 默认：扫描本仓，输出到 mc2/tools/impact_map/
python3 mc2/tools/gen_common_impact_yaml.py

# 对任意时期的仓检出重新生成（yaml头部记录commit，便于跨时期对比）
python3 mc2/tools/gen_common_impact_yaml.py /path/to/another_checkout [输出目录]
```

纯本地扫描（无需联网）。

## 4. 消费规则（每个文件恰好命中一类）

**第一步：先查本目录的 `file_lists.yaml`，命中任一条 → 全量：**
| 命中 | 条件 |
|---|---|
| `full_regression.non_header_extensions` | 文件扩展名为 .cpp/.py/.json/.ini/.sh/.csv/.txt/.md 等非头文件 |
| `full_regression.headers_used_by_all_operators` | .h 被全部算子使用 |
| `full_regression.headers_no_reference` | .h 无任何文件引用（列出仅为防漏查，删除前需人工确认） |

**第二步：未命中 → .h 属"测部分算子"类，查芯片映射表：**
| 文件 | 判定 |
|---|---|
| .h 在芯片映射表中（A2/A3/A5.yaml） | 测试条目 ops 列出的算子（按芯片过滤；`ops: []` 表示该芯片无匹配算子） |

**最后一步（消费侧负责）**：得到的算子集还需**叠加 CMake `*_depends`
反向闭包**后去重，作为最终测试清单。

## 5. 判定规则（生成器侧）

1. **只对 .h 建模**：影响面 = 直接+间接 include 该头的算子（裸文件名匹配，
   重名取并集，传递闭包用不动点迭代，天然处理环）。
2. **二分归类**：影响面 = 全部注册算子 → 全量；影响面为空且无任何引用 → 全量
   （防漏查）；其余 → 部分算子，进芯片映射表。
3. **芯片过滤**：部分算子头的 ops 按 `*_def.cpp` 的 `AICore().AddConfig()`
   注册与当前芯片取交集，三张表键集合保持一致。
4. **重名取并集**：同名头文件以 `note:` 标注，宁多勿漏。

## 6. yaml 格式

```yaml
# 芯片映射表(部分算子类)
common/op_kernel/mc2_tiling_struct.h:
  ops:
    - all_gather_matmul        # 改动该头需测试的算子，显式全列表
  note: 同名文件2份, 映射取并集  # 仅重名文件有此字段
common/op_host/op_tiling/xxx.h:
  ops: []                      # 该头影响的算子均未在此芯片注册(策略后续再定)

# file_lists.yaml(全量匹配清单, 按目录分放)
full_regression:
  non_header_extensions:        # 非.h文件命中即全量
    - ".cpp"
    - ...
  headers_used_by_all_operators: # 被全部算子使用的头
    - 3rd/mat_mul_v3/...
  headers_no_reference:          # 无任何引用的头(防漏查)
    - 3rd/batch_mat_mul_v3/...
```

## 7. 维护约定

- common/3rd 下**头文件新增/删除、include 关系变化**后，重新运行生成器刷新 yaml。
- **对齐上游（rebase/合入上游新基线）后**，需重新运行生成器刷新 yaml 并作为一笔
  提交带入（yaml 头部基线 commit 会同步更新）。
- 本目录不参与编译构建（无 CMakeLists.txt，tools 的构建遍历自动跳过）。
- 映射表由生成器产出，请勿手工编辑（重新生成会覆盖）；如需长期例外，请调整
  生成器规则后在 PR 中说明。
