# mc2 common/3rd 头文件影响范围映射表（impact_map）

## 1. 背景与用途

mc2 算子的公共代码集中在 `mc2/common` 与 `mc2/3rd`，改动这些目录时无法用算子目录级
依赖（CMake `*_depends`）评估影响面。本目录提供**文件级**的反向影响映射：

> 输入：PR 修改的 common/3rd 文件路径 + 目标芯片
> 输出：需要回归测试的 mc2 算子清单

**本表只保存头文件（.h/.hpp/.cuh）条目**，其余文件类型由消费脚本按第 4 节规则处理。
cpp 的同名头约定为 `.h`（实测仓内 86 个非测试 cpp：75 个有同名 .h，11 个无同名头，
不存在只有 .hpp 的 case）。

## 2. 目录结构

```
mc2/tools/
├── gen_common_impact_yaml.py    # 生成器（本表唯一来源，重新生成会覆盖yaml）
└── impact_map/
    ├── common/                  # mc2/common 头文件的映射
    │   ├── A2.yaml / A3.yaml / A5.yaml
    │   └── zero_reference.yaml           # 零引用头文件清单
    └── 3rd/                     # mc2/3rd 头文件的映射
        ├── A2.yaml / A3.yaml / A5.yaml
        └── zero_reference.yaml
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

## 4. 消费规则（完整覆盖）

消费脚本输入修改文件列表 + 芯片，逐文件判定：

| 文件与条件 | 处理 |
|---|---|
| 非 .h/.hpp/.cuh/.cpp 的一切（.py/.json/.ini/.sh/.csv/.txt/.md 等，不限路径） | 全量 |
| .cpp 且同目录无同名 .h | 全量 |
| .cpp 且同目录同名 .h 在 zero_reference.yaml | 不触发（备注：疑似死代码，建议人工复核） |
| .cpp 且同目录同名 .h 在本表 | 测试同名 .h 条目列出的算子 |
| .h 在 zero_reference.yaml | 人工确认（备注：该头当前无任何引用） |
| .h 在本表 | 测试条目 ops 列出的算子；`ops: []` 表示该芯片无引用算子，不用测 |

**最后一步（消费侧负责）**：以上得到的算子集还需**叠加 CMake `*_depends`
反向闭包**（如 yaml 传递出 `moe_distribute_dispatch_v2`，则依赖它的
`moe_distribute_combine_v2` 等一并纳入），再去重后作为最终测试清单。

## 5. 判定规则（生成器侧）

1. **裸文件名匹配**：扫描 mc2 内全部 `#include "..."`，取最后一段文件名匹配
   （文件挪目录不影响解析）。
2. **重名取并集**：同名头文件以 `note:` 标注，映射取并集，宁多勿漏。
3. **间接 include 传递闭包**：算子 → common 头A → 3rd 头B 的链路中，改动 B
   同样算入算子影响（不动点迭代，天然处理环）。
4. **芯片过滤**：算子集合按 `*_def.cpp` 的 `AICore().AddConfig()` 注册与
   当前芯片取交集。
5. **cpp 不入表**：.cpp 的接口影响等价于其同名 .h（接口反推原则），消费侧按
   第 4 节规则处理，yaml 不保存 .cpp 条目。

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
