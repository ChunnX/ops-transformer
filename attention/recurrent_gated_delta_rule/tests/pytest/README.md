# Recurrent_gated_delta_rule算子测试框架

## 功能说明

基于pytest测试框架，实现Recurrent_gated_delta_rule算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持mtp为1~8。
- 支持NK、Nv小于等于256，Nv大于等于Nk且Nv需整除Nk。
- 支持Dk、Dv小于等于512。
- 支持actual_seq_lengths输入，长度为batch_size。
    - actual_seq_lengths中数值需大于等于0且小于等于mtp。
    - 不指定输入时，默认传入长度为batch_size，数值为mtp的数组。
    - T等于actual_seq_lengths所有元素之和。
- 支持ssm_state_indices输入，长度为T。
    - ssm_state_indices中数值需小于block_num。
    - 不指定输入时默认长度为[0,1,...,T-1]。
- 支持block_num手动传入，需大于等于T。
- 支持data_type为BF16。
- 支持query_datarange左区间最小为-1，右区间最大为1。
- 支持key_datarange左区间最小为-1，右区间最大为1。
- 支持gamma_datarange右区间最大为0。
- 支持gamma_k_datarange右区间最大为0。
- 支持beta_datarange左区间最小为0，右区间最大为1。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                               # 执行脚本
- conftest.py                               # pytest钩子：逐用例记录参数/结果/精度/种子，会话结束落CSV
- recurrent_gated_delta_rule_golden.py      # cpu侧算子golden实现以及cpu golden与npu结果精度对比
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记

单用例测试:

- test_recurrent_gated_delta_rule_single.py                 # 测试单用例运行主程序
- recurrent_gated_delta_rule_operator_single.py             # CPU侧算子逻辑实现获取golden与npu算子直调
- test_recurrent_gated_delta_rule_paramset.py               # 单用例入参配置
- test_recurrent_gated_delta_rule_paramset_rdv.py           # RDV测试入参配置

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_recurrent_gated_delta_rule_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### RDV测试

1、手动配置test_recurrent_gated_delta_rule_paramset_rdv.py的参数

2、执行指令：

``` bash
bash test_run.sh rdv
```

#### 随机用例测试

随机生成N条用例并执行（含CPU golden精度对比），可用`RANDOM_SEED`环境变量固定随机种子复现（不指定则自动生成并记录到CSV）：

``` bash
bash test_run.sh random 100
```

#### 随机用例测试（仅NPU）

随机生成N条用例，设置`SKIP_GOLDEN=1`跳过CPU golden计算与精度对比，仅执行NPU算子（输入张量直接在NPU上生成，host内存占用低），加快执行速度：

``` bash
bash test_run.sh random_npu 100
```

#### mssanitizer检测

随机生成N条用例，仅NPU执行（不跑golden），并在mssanitizer下运行，检测设备侧内存越界、非法地址访问等问题（详细列语义见下方「结果输出与复现」）。屏显含`====== ERROR`即判定FAIL（脚本退出码为1）。默认只检测本算子kernel（`--kernel-name=RecurrentGatedDeltaRule`），跳过 ZerosLike/ViewCopy/TensorMove/rand 等旁路 kernel 的检测开销：

``` bash
bash test_run.sh mss 10
```

**前置条件（mssanitizer 安装）**：

- 官方介绍：[mssanitizer 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/quick_start/mssanitizer_quick_start.md)
- CANN 工具包已自带时无需额外安装（如 `/home/developer/Ascend/cann-9.2.0/bin/mssanitizer`，确保其所在目录在 PATH 中）
- 未自带时源码编译安装：

``` bash
git clone https://gitcode.com/Ascend/mssanitizer.git
cd mssanitizer
python3 build.py
# 编译完成后安装 run 包
artifacts/mindstudio-sanitizer_26.0.0_x86_64.run --run --install-path=/home/developer/Ascend/
```

支持环境变量定制：

``` bash
MSS_TOOL=racecheck bash test_run.sh mss 10                     # 换检测工具(memcheck/racecheck/initcheck/synccheck)
MSS_KERNEL='' bash test_run.sh mss 10                          # 关闭kernel过滤，检测全部kernel
MSS_EXTRA_OPTS='--leak-check=yes --full-backtrace=yes' bash test_run.sh mss 10  # 追加mssanitizer参数
MSSANITIZER_BIN=/path/to/mssanitizer bash test_run.sh mss 10   # 指定mssanitizer路径
```

### 结果输出与复现

所有模式执行后均输出到`output/`目录（已gitignore）：

- `run_<时间戳>.log` / `mss_<时间戳>.log`：完整执行日志（tee屏显）
- `result_<时间戳>.csv`：逐用例结果表，每行一条用例，列含义：

| 列 | 说明 |
|----|------|
| random_seed | 随机shape序列种子（random系模式；single/rdv为固定参数集无此值） |
| tensor_seed | 本条用例张量数值种子（每条独立记录） |
| test_mode | single/rdv/random |
| check_type | precision=带golden精度对比 / execution_only=仅NPU执行 / execution_only+mss_\<tool\>=mssanitizer检测 |
| result | pytest执行结果（PASSED/FAILED/SKIPPED） |
| mss_check | mssanitizer检测结论（PASS/FAIL(errors=N)/CRASH），仅mss模式由脚本解析日志回填 |
| out_pct_rlt / state_pct_rlt | 输出与state的精度达标率PctRlt真实值（如99.999982%），仅precision模式记录 |
| batch_size...state_non_contiguous（24列） | 本条用例全部入参 |
| errmsg | 失败详情（截断2000字符） |

失败用例复现（两层种子配合CSV入参）：

``` bash
# 整批复现（同shape序列）：CSV取random_seed
RANDOM_SEED=<random_seed> bash test_run.sh random N

# 单条数值级复现（同shape+同张量数值）：CSV取入参与tensor_seed
TORCH_SEED=<tensor_seed> bash test_run.sh random 1   # 配合该条入参（或random_seed定位到该条）
```

### 环境变量汇总

| 变量 | 作用 | 适用模式 |
|------|------|---------|
| RANDOM_SEED | 固定随机shape序列种子（不设则自动生成并记CSV） | random/random_npu/mss |
| TORCH_SEED | 固定张量数值种子（不设则每条自动生成并记CSV） | 全部 |
| RANDOM_CASE_COUNT | 随机用例条数（test_run.sh已透传） | random系 |
| SKIP_GOLDEN | =1跳过CPU golden与精度对比，仅NPU执行 | random_npu/mss |
| MSS_TOOL | mssanitizer检测工具（memcheck/racecheck/initcheck/synccheck） | mss |
| MSS_KERNEL | kernel过滤名（默认RecurrentGatedDeltaRule，置空检全部） | mss |
| MSS_EXTRA_OPTS | 追加mssanitizer参数（如--leak-check=yes） | mss |
| MSSANITIZER_BIN | mssanitizer可执行文件路径 | mss |
