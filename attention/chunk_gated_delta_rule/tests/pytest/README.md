# Chunk_gated_delta_rule算子测试框架

## 功能说明

基于pytest测试框架，实现Chunk_gated_delta_rule算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持seqlen序列长度。
- 支持NK、NV head数，NV需要为NK倍数。
- 支持DK、DV不超过128。
- 支持data_type为BF16。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                               # 执行脚本
- chunk_gated_delta_rule_golden.py          # cpu侧算子golden实现
- chunk_gated_delta_rule_main.py            # cpu golden与npu结果精度对比及主调测逻辑
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记

单用例测试:

- test_chunk_gated_delta_rule_single.py     # 测试单用例运行主程序
- chunk_gated_delta_rule_operator_single.py # CPU侧算子逻辑实现获取golden与npu算子直调
- test_chunk_gated_delta_rule_paramset.py   # 单用例入参配置
- test_chunk_gated_delta_rule_paramset_rdv.py   # RDV测试入参配置

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_chunk_gated_delta_rule_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### RDV测试

1、手动配置test_chunk_gated_delta_rule_paramset_rdv.py的参数

2、执行指令：

``` bash
bash test_run.sh rdv
```


## 泛化与精度验证（新增）

输入契约以[接口文档](../../docs/aclnnChunkGatedDeltaRule.md)为准。新增测试与原RDV性能入口分开：**不受`ENABLE_PROF=true`影响，不会跳过精度**。使用公开ACLNN两段式接口直接调用算子，可测试调用者提供的输出描述符与非法输入第一段返回值。

2026-09-08 在 Ascend950 上实测新增 616 个用例：614 通过、2 失败。544 个合法输入与语义用例全部通过，两个失败为 Nv=0/Dv=0 未被拒绝；详见[验证报告与逐用例结果](../../docs/generalization/README.md)。

### 运行

先完成CANN环境配置，并安装当前源码构建的custom包。`CGDR_TEST_VENDOR`指向具体vendor目录；同时在启动Python前配置`ASCEND_CUSTOM_OPP_PATH`，两者必须一致。

```bash
export CGDR_TEST_VENDOR=/path/to/opp/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$CGDR_TEST_VENDOR"
export LD_LIBRARY_PATH="$CGDR_TEST_VENDOR/op_api/lib:${LD_LIBRARY_PATH}"
export TORCH_SEED=20260908
export CGDR_TEST_DEVICE=0
mkdir -p output/generalization
export CGDR_TEST_AUDIT="$PWD/output/generalization/runtime.json"

python3 -m pytest -v \
  test_chunk_gated_delta_rule_generalization.py \
  test_chunk_gated_delta_rule_invariants.py \
  test_chunk_gated_delta_rule_validation.py \
  --junitxml=output/generalization/results.xml
```

仅运行较快的边界/语义用例，追加`-k 'not network_precision'`；仅测长网络用例，使用`-k network_precision`。CPU FP64逐token参考处理长序列耗时明显高于设备算子，这是精度验证而非性能测量。不设置`CGDR_TEST_VENDOR`时通过`libopapi.so`调用环境默认实现，**不能据此声称测到了当前优化包**。

### 覆盖范围

| 契约/风险 | 新增覆盖 |
|---|---|
| Dk/Dv在1～128 | 1、2、5、8、9、15/16/17、31/32/33、63/64/65、95/96/97、127/128；对角及非对称维度 |
| chunk与分流边界 | 1/2、15/16/17、31/32/33、63/64/65、127/128/129、191/192/193、255/256/257、4095/4096/4097；小状态同形状跨长度测试 |
| Nk/Nv及核组调度 | 1/1、1/64、64/64、14/28、16/32、31/31、32/32、33/33、63/63、3/6、5/15 |
| batch与有效长度 | 等长/变长、多batch、多波head、开头/中间/结尾零长度序列；所有合法case仍满足B>0、T>0、sum(lengths)=T |
| gate与数值范围 | g缺省、全0、全-1、接近0、交替、范围内随机；正负混合归一化Q/K、one-hot及交替符号；beta严格位于(0,1)，包含接近两端的BF16可表示值 |
| scale与状态 | 默认、自定义、0、负数、小值；BF16/FP32状态、零状态、零value、较大value/state幅值 |
| 非连续输入 | Q/K/V/beta/g/lengths各轴带孔切片、非零storage_offset、零stride广播；state仅使用文档承诺的0/1轴非连续，并测试两轴同时有孔 |
| 输出与只读输入 | shape、dtype、连续性、NaN/Inf；输入内容/padding保持不变；连续输出view两端哨兵不变且每个有效位置被写入 |
| 数学语义 | 单token闭式解、确定性、g=None与g=0等价、scale零/反号、batch独立性、head置换、布局等价、因果性、零状态零value保持零 |
| 多seed与核心网络 | 16组固定随机seed组合；T=3500/7000/8000/11264与Nk/Nv=16/32或14/28，BF16/FP32两种状态 |
| 非法元信息 | required nullptr、dtype、rank、shape关联轴、空tensor、维度上下界、Nv不能整除Nk；仅第一段校验，不发射非法kernel |

### 独立参考和判定标准

`cgdr_generalization_utils.py`按文档的逐token递推公式，在CPU FP64中计算attention和final state。它使用BF16/FP32输入的实际量化值，不重新归一化Q/K，不覆盖调用者scale，不按候选实现分流或做BF16中间量舍入。单token闭式解还会独立核对这个参考函数。

精度采用预先固定的双重阈值：`RMS(error)/max(RMS(reference),2e-5) <= 0.02`，以及每元素`abs(error)/max(abs(reference_i),RMS(reference),2e-5) <= 0.10`。同时必须shape/dtype正确且全部有限。两项分别限制整体误差与局部离群误差；分母下限用于接近零的输出，**不是任意放宽容差，也不根据测试失败自动调整阈值**。这是本套BF16计算的测试标准，不代表接口文档给出了同等误差保证。

语义等价关系另做逐元素相等检查，输出边界哨兵、输入内容也必须完全相等。XML中保存case参数、attention/final state误差以及非法输入ACLNN返回码。已有golden会重新设置默认scale，已有benchmark对状态按BF16处理，因此新套件不依赖这两者。

### 明确的边界

- 文档将Q/K归一化、g∈[-1,0]和beta∈(0,1)作为调用者责任；非法数值、负序列长度或sum(lengths)不等于T不送入kernel，不假设第一段接口会逐元素检查设备tensor。
- 非法测试首先断言返回非零，并原样记录具体错误码；这不能等同于所有返回码均符合文档。文档与当前实现的错误码差异在验证报告中单独列出。
- 只覆盖文档支持的state外层stride，不将state最后两维非连续或非连续输出视为必需契约。
- 当前实测设备为Ascend950；FP32状态在其他芯片上不是接口承诺的能力。
- 这是有代表性的组合集，不是全部维度、任意长序列、所有数值分布的笛卡尔积，也未覆盖图模式、并发stream、别名/in-place等未承诺行为。
