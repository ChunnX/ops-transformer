# mc2 common/3rd 头文件影响范围映射表（impact_map）

## 1. 背景与用途

mc2 算子的公共代码集中在 `mc2/common` 与 `mc2/3rd`，改动这些目录时无法用算子目录级
依赖（CMake `*_depends`）评估影响面。本目录提供**头文件级**的反向影响映射：

> 输入：PR 修改的文件路径 + 目标芯片
> 输出：需要回归测试的 mc2 算子清单

**只对 .h 建模**；其余文件类型一律按全量回归处理（见第 4 节）。

## 2. 目录结构

```
mc2/tools/
├── gen_common_impact_yaml.py    # 生成器（本表唯一来源，重新生成会覆盖yaml）
└── impact_map/
    ├── common/                  # mc2/common 头文件的映射
    │   ├── A2.yaml / A3.yaml / A5.yaml
    ├── 3rd/                     # mc2/3rd 头文件的映射
    │   ├── A2.yaml / A3.yaml / A5.yaml
    └── full_regression.yaml     # 全量回归算子清单（按芯片）
```

芯片对应关系：A2=ascend910b，A3=ascend910_93，A5=ascend950。

## 3. 生成方法

```bash
# 默认：扫描本仓，输出到 mc2/tools/impact_map/
python3 mc2/tools/gen_common_impact_yaml.py

# 对任意时期的仓检出重新生成（yaml头部记录commit，便于跨时期对比）
python3 mc2/tools/gen_common_impact_yaml.py /path/to/another_checkout [输出目录]
```

纯本地扫描（无需联网）。

## 4. 消费规则（极简，纯文件类型）

| 文件 | 判定 |
|---|---|
| .h 在芯片映射表中 | 测试条目 ops 列出的算子；`ops: []` = 该芯片无引用算子，不用测 |
| .h 不在芯片映射表中 | 全量（无明确 include 到算子关系，含冗余/仅测试引用头） |
| 其他一切文件（.cpp/.py/.json/.ini/.sh/.csv/.txt/.md 等，不限路径） | 全量（算子清单见 `full_regression.yaml`） |

**最后一步（消费侧负责）**：得到的算子集还需**叠加 CMake `*_depends`
反向闭包**后去重，作为最终测试清单。

## 5. 判定规则（生成器侧）

1. **只对 .h 建模**：影响面 = 直接+间接 include 该头的算子（裸文件名匹配，
   重名取并集，传递闭包用不动点迭代，天然处理环）。
2. **芯片过滤**：算子集合按 `*_def.cpp` 的 `AICore().AddConfig()` 注册与
   当前芯片取交集。
3. **全量清单**：`full_regression.yaml` 按芯片列出全部注册算子，凡命中
   上表"全量"情况的改动按该清单执行。

## 6. yaml 格式

```yaml
common/op_kernel/mc2_tiling_struct.h:
  ops:
    - all_gather_matmul        # 改动该头需测试的算子，显式全列表
    - ...
  note: 同名文件2份, 映射取并集  # 仅重名文件有此字段
common/op_host/op_tiling/xxx.h:
  ops: []                      # 有引用但引用算子不支持该芯片
```

## 7. 维护约定

- common/3rd 下**头文件新增/删除、include 关系变化**后，重新运行生成器刷新 yaml。
- **对齐上游（rebase/合入上游新基线）后**，需重新运行生成器刷新 yaml 并作为一笔
  提交带入（yaml 头部基线 commit 会同步更新）。
- 本目录不参与编译构建（无 CMakeLists.txt，tools 的构建遍历自动跳过）。
- 映射表由生成器产出，请勿手工编辑（重新生成会覆盖）；如需长期例外，请调整
  生成器规则后在 PR 中说明。
