# mc2 common/3rd 影响范围映射表（impact_map）

## 1. 背景与用途

mc2 算子的公共代码集中在 `mc2/common` 与 `mc2/3rd`，改动这些目录时无法用算子目录级
依赖（CMake `*_depends`）评估影响面。本目录提供**文件级**的反向影响映射：

> 输入：PR 修改的 common/3rd 文件路径
> 输出：需要回归测试的 mc2 算子清单（按芯片 A2/A3/A5 区分）

配套外部脚本（仓外维护）：
`pr_files.sh`（按 PR 号拉取修改文件列表）+ `pr_impact.py`（算子目录改动 +
本映射表查表 + CMake depends 反向闭包 = 最终测试清单）。

## 2. 目录结构

```
mc2/tools/
├── gen_common_impact_yaml.py    # 生成器（本表唯一来源，重新生成会覆盖yaml）
└── impact_map/
    ├── common/                  # mc2/common 文件的映射
    │   ├── A2.yaml / A3.yaml / A5.yaml
    │   └── unref.yaml           # 零引用文件清单
    └── 3rd/                     # mc2/3rd 文件的映射
        ├── A2.yaml / A3.yaml / A5.yaml
        └── unref.yaml
```

芯片对应关系：A2=ascend910b，A3=ascend910_93，A5=ascend950。

## 3. 生成方法

```bash
# 默认：扫描本仓，输出到 mc2/tools/impact_map/
python3 mc2/tools/gen_common_impact_yaml.py

# 对任意时期的仓检出重新生成（yaml头部记录commit，便于跨时期对比）
python3 mc2/tools/gen_common_impact_yaml.py /path/to/another_checkout [输出目录]
```

纯本地扫描（无需联网），依赖 python3；可选 pyyaml 用于下游查表脚本。

## 4. 判定规则

1. **裸文件名匹配**：扫描 mc2 内全部 `#include "..."`，取最后一段文件名与
   common/3rd 文件名匹配（文件挪目录不影响解析）。
2. **重名取并集**：仓内存在少量同名头文件（yaml 中以 `note:` 标注），映射取
   并集，方向为"宁多勿漏"。
3. **间接 include 传递闭包**：算子 → common 头A → 3rd 头B 的链路中，改动 B
   同样算入算子影响（递归直至引用者落到算子目录为止，带环保护与结果缓存）。
4. **编译级引用**：common/3rd 的非测试 .cpp 由 CMake `file(GLOB)`/
   `add_modules_sources(DIR)` 按目录整包编译进全局目标，无需被 include，改动
   影响全部算子（yaml 中以 `note: 编译级引用` 标注，ops 显式列出）。
5. **零引用文件**：既无 include 引用、也非编译单元的文件单独记录在
   `unref.yaml`（改动当前不影响 mc2 算子编译，但需人工确认是否有仓外引用或
   引用关系尚未建立）。
5. **芯片过滤**：算子集合按 `*_def.cpp` 的 `AICore().AddConfig()` 注册与
   当前芯片取交集。

## 5. yaml 格式

```yaml
common/op_kernel/mc2_tiling_struct.h:
  ops:
    - all_gather_matmul        # 改动该文件需测试的算子，显式全列表
    - ...
  note: 同名文件2份, 映射取并集  # 仅重名文件有此字段
3rd/xxx/orphan_impl.cpp:
  ops: []                      # include不可见（孤立实现文件），需人工评估
```

## 6. 维护约定

- common/3rd 下**新增文件、新增/删除 include 关系**后，重新运行生成器刷新 yaml。
- 本目录不参与编译构建（无 CMakeLists.txt，tools 的构建遍历自动跳过）。
- 映射表由生成器产出，请勿手工编辑（重新生成会覆盖）；如需长期例外，请调整
  生成器规则后在 PR 中说明。
