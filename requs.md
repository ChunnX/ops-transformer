# 需求
## 需求一
1、attention/sparse_flash_attention/tests/pytest这是sfa算子的pytest框架，我目前要上库，但attention/sparse_flash_attention/tests/pytest下面的文件编码规范不满足要求，请参照attention/kv_quant_sparse_flash_attention/tests/pytest对attention/sparse_flash_attention/tests/pytest进行编程规范整改，必要的地方加上注释。还有attention/sparse_flash_attention/tests/pytest/README.md文档也要改
2、attention/kv_quant_sparse_flash_attention/tests/pytest中batch模式存在一个bug，就是excel表block_num如果既有空也有整数时，读取到的block_num会为浮点，进而生成用例报错，这一点可以参照attention/sparse_flash_attention/tests/pytest进行修复
## 实现分析与整改记录
### 需求一-实现分析
1. 先阅读 `attention/sparse_flash_attention/tests/pytest` 与 `attention/kv_quant_sparse_flash_attention/tests/pytest` 两套 pytest 框架，确认本次不是新增功能，而是两类工作并行：
   - 对 SFA pytest 目录做编程规范整改；
   - 修复 QSFA batch 读 Excel 时 `block_num` 因空值与整数混用被 pandas 提升成浮点的问题。
2. 对比后发现 SFA 目录的主要问题集中在：
   - 多个 python 文件缺少和 QSFA 一致的完整版权头；
   - `single` / `batch` / `pt_save` 三个测试入口的线程执行逻辑、异常提示、路径收集方式不统一；
   - `README.md` 仍保留阶段性描述，和当前真实脚本能力、字段约束不完全一致；
   - `utils.py`、`batch/sparse_flash_attention_process.py`、`check_valid_param.py`、`sparse_flash_attention_golden.py` 缺少足够的过程注释，不利于后续维护。
3. QSFA bug 的根因也比较明确：Excel 中某列如果既有空值又有整数，`pandas.read_excel` 往往会把整列提升为 `float`；`block_num` 原本期望是整数，但在 `combin_params -> qsfa -> golden` 这条链路里如果保持成 `256.0` 这类值，后续构造 shape 或命名时就可能报错。

### 需求一-实现方案
1. 对 SFA pytest 目录按 QSFA 风格做代码整理：
   - 为核心 python 文件补齐统一版权头；
   - 为关键函数增加简短注释，说明该函数在 batch/single 链路中的职责；
   - 将 `locals()["testcase_files"]` 这种不直观的写法改成显式的 `_collect_testcase_files()`；
   - 统一线程执行结束后的异常透传方式，失败时保留具体报错内容；
   - 将 `utils.py` 中的结果列定义抽成常量，避免结果表结构漂移。
2. 对 SFA 文档同步整改：
   - README 改成面向实际可执行流程的说明；
   - 补齐 Excel 字段说明、约束说明、结果文件说明；
   - 明确说明当前 CPU golden 仍未补齐，因此现阶段结果主要关注 NPU 连通性与用例落盘。
3. 对 QSFA 的 `block_num` 浮点 bug 做定点修复：
   - 在 `attention/kv_quant_sparse_flash_attention/tests/pytest/utils.py` 中补充 `_normalize_numeric_value()`；
   - 对普通单元格、列表字符串、tuple 字符串统一做递归数值归一化；
   - 当值是 `256.0`、`1.0` 这类“整数语义的浮点”时自动回收成 `int`；
   - 这样既修复 `block_num`，也顺带避免 `Testcase_Number`、`B/S1/S2/...`、`actual_seq_*` 等 Excel 字段在同类场景下被浮点污染。

### 需求一-改动点
1. `attention/sparse_flash_attention/tests/pytest/utils.py`
   - 统一 Excel 解析、结果写回、参数展开、NPU 回放入口；
   - 增加整数归一化逻辑和结果列常量；
   - 补充注释，方便定位 single/batch 公共入口。
2. `attention/sparse_flash_attention/tests/pytest/batch/sparse_flash_attention_process.py`
   - 按 QSFA 同风格整理 CI/graph 两条执行路径；
   - 补充 NPU 搬运、单算子直调、graph 编译回放说明。
3. `attention/sparse_flash_attention/tests/pytest/test_sparse_flash_attention_single.py`
   - 统一 single 模式线程入口和异常处理。
4. `attention/sparse_flash_attention/tests/pytest/test_sparse_flash_attention_batch.py`
   - 改为显式收集 `.pt` 文件列表；
   - 增加目录不存在/目录为空时的清晰提示。
5. `attention/sparse_flash_attention/tests/pytest/batch/test_sparse_flash_attention_pt_save.py`
   - 统一 pt 落盘入口写法和异常处理。
6. `attention/sparse_flash_attention/tests/pytest/check_valid_param.py`
   - 补齐版权头；
   - 将校验逻辑里容易混淆的变量名改得更直观；
   - 给 `actual_seq`、`PA_BSND` 等关键分支补充说明。
7. `attention/sparse_flash_attention/tests/pytest/sparse_flash_attention_golden.py`
   - 补齐版权头和关键注释；
   - 对 sparse index、block table、TND 变换逻辑做更清晰的命名整理。
8. `attention/sparse_flash_attention/tests/pytest/README.md`
   - 按当前真实能力重写说明，避免继续沿用拷贝时期的阶段性文字。
9. `attention/kv_quant_sparse_flash_attention/tests/pytest/utils.py`
   - 新增 `_normalize_numeric_value()`，修复 Excel 混合空值/整数导致 `block_num` 读成浮点的问题。

### 需求一-结果判断
1. SFA pytest 目录的主入口、公共工具、数据构造和 README 已完成编程规范整改，风格与 QSFA 框架保持一致。
2. QSFA batch 模式下，`block_num` 即使在 Excel 中与空值混用，也会在解析时自动回收为整数，不再因浮点值参与后续用例构造而报错。
3. 本次整改没有扩展算子语义，只调整测试框架代码组织、注释和 Excel 入参归一化逻辑，因此风险集中在 pytest 脚本层，便于回归验证。
