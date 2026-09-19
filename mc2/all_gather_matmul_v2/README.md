# AllGatherMatmulV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term>   | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | x |
| <term>Atlas推理系列产品</term> | x |
| <term>Atlas训练系列产品</term> | x |

## 功能说明

- **算子功能**：
  完成AllGather通信与MatMul计算融合。在支持x1和x2输入类型为FLOAT16/BFLOAT16的基础上，同时也支持低精度数据类型，此时算子在Matmul计算后会做对应的反量化计算，支持的低精度数据类型与量化方式如下：

  - <term>Ascend 950PR&950DT系列产品</term>：

    新增了对低精度数据类型FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1的支持。支持pertensor、perblock、mx[量化方式](../../docs/zh/context/quant_mode_introduction.md)。

  - <term>Atlas A3系列产品</term>  、<term>Atlas A2系列产品</term>：

    新增了对低精度数据类型INT8/INT4的支持。支持pertoken/perchannel[量化方式](../../docs/zh/context/quant_mode_introduction.md)。

- **计算公式**：
    - 情形1：如果x1和x2数据类型为FLOAT16/BFLOAT16时，入参x1进行AllGather后，对x1、x2进行MatMul计算。

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - 情形2：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的pertensor场景，或者x1和x2数据类型为INT8/INT4的perchannel、pertoken场景，且不输出amaxOut，入参x1进行AllGather后，对x1、x2进行MatMul计算，然后进行dequant操作。

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - 情形3：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的perblock场景，且不输出amaxOut,当x1为(m, k)、x2为(k, n)时， x1Scale为(ceildiv(m, 128)， ceildiv(k, 128))、x2Scale为(ceildiv(k, 128), ceildiv(n, 128))时，入参x1和x1Scale进行AllGather后，对x1、x2进行perblock量化MatMul计算，然后进行dequant操作。

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

    - 情形4：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1的mx量化场景，x1为(m, k)、x2非转置场景下为(k, n)，转置场景下为(n, k)，且x1Scale为(m, ceilDiv(k, 64), 2)、x2Scale非转置场景时(ceilDiv(k, 64), n, 2)，转置场景为(n, ceilDiv(k, 64), 2)，入参x1和x1Scale进行AllGather后，对x1、x2进行MatMul计算，然后进行dequant操作；

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
    </colgroup>
    <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1 </td>
        <td>输入</td>
        <td>MM左矩阵，即计算公式中的x1。</td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、INT8、INT4</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x2 </td>
        <td>输入</td>
        <td>MM右矩阵，即计算公式中的x2。</td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、INT8、INT4</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>bias </td>
        <td>可选输入</td>
        <td>即计算公式中的bias。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x1Scale </td>
        <td>可选输入</td>
        <td>mm左矩阵反量化参数。</td>
        <td>FLOAT、FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x2Scale </td>
        <td>可选输入</td>
        <td>mm右矩阵反量化参数。</td>
        <td>FLOAT、INT64、FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>quantScale </td>
        <td>可选输入</td>
        <td>即计算公式中的bias。</td>
        <td>FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>output </td>
        <td>输出</td>
        <td>AllGather通信与MatMul计算的结果，即计算公式中的output。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gatherOut </td>
        <td>输出</td>
        <td>仅输出all_gather通信后的结果。即公式中的gatherOut。</td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、INT8、INT4</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>amaxOut </td>
        <td>可选输出</td>
        <td>MM计算的最大值结果，即公式中的amaxOut。</td>
        <td>FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>属性</td>
        <td>用于表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>属性</td>
        <td>通信域名称。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherIndex</td>
        <td>属性</td>
        <td>标识gather目标。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>属性</td>
        <td>通信数据切分数，即总数据量/单次通信量。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>属性</td>
        <td>流模式的枚举。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>可选属性</td>
        <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>属性</td>
        <td>通信模式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    </tbody></table>

## 约束说明

- 确定性计算：
  - 该算子默认确定性实现。
- <term>Ascend 950PR&950DT系列产品</term>：
    - 输入x1为2维，其维度为\(m, k\)。x2必须是2维，其维度为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出output为2维，其维度为\(m*rank\_size, n\)，rank\_size为卡数。
    - 输出gatherout为2维，其维度为\(m*rank\_size, k\)，rank\_size为卡数。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16时，output计算输出数据类型和x1、x2保持一致。
    - 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8时，output输出数据类型支持FLOAT16、BFLOAT16、FLOAT。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16/HIFLOAT8时，x1和x2数据类型需要保持一致。
    - 当x1、x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，x1和x2数据类型可以为其中一种。
    - 当x1、x2数据类型为FLOAT16/BFLOAT16/HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2时，x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景。
    - 当x1、x2数据类型为FLOAT4_E2M1时，x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景，k轴需要为偶数，且当x2矩阵非转置时，n轴也需要为偶数。
    - 当groupSize取值为549764202624，bias必须为空。
    - 支持2、4、8、16、32、64卡。
    - 支持CCU通信引擎和AICPU通信引擎，CCU仅支持单机UB域内互联，AICPU可支持跨机UB域内互联。
    - allgather(x1)集合通信数据总量不能超过63*256MB，集合通信数据总量计算方式为：m* k *sizeof(x1_dtype)* 卡数。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。

- <term>Atlas A3系列产品</term>  、<term>Atlas A2系列产品</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1必须是2维，其shape为\(m, k\)。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias仅支持输入nullptr。
    - 输出为2维，其shape为\(m*rank\_size, n\), rank\_size为卡数。
    - 不支持空tensor。
    - x1和x2的数据类型需要保持一致。
    - x1和x2数据类型为INT4时，k与n必须为偶数。
    - 支持2、4、8卡。

## 实现说明

非量化 FLOAT16/BFLOAT16 路径当前存在两套内核实现**并存**，接口行为完全一致（接口冻结）：

- **v1 实现（默认路径）**：HCCL windows + mat_mul_v3 计算栈，覆盖量化与非量化全部支持域，作为完整兜底路径。
- **APACE 实现**：基于 APACE 框架（hcomm 变体）的非量化融合实现——AIC Cube 承载 Blaze `BlockMmad` 主计算，AIC 侧 `Hccl<CCU/AICPU>` in-kernel API 完成通信编排，`FragmentTensor` 管理 GM fragment（HEAD/MAIN/TAIL 三区）。仅承接非量化 FLOAT16/BFLOAT16 路径，由 `USING_APACE_IMPL` 开关控制。新增实现组件位于 `mc2/common/op_kernel/apace/kernel/`（`fusions/all_gather_matmul`、`matmul/batch_matmul` 两个新目录）以及算子侧 tiling 适配（`op_host/op_tiling/arch35/all_gather_matmul_tiling_v2.{h,cpp}`）与 arch35 入口分支（`op_kernel/arch35/all_gather_matmul_v2_apt.cpp`）。

### 实现切换开关

| 项 | 说明 |
| ---- | ---- |
| 开关环境变量 | `ASCEND_MC2_AGMV2_APACE_IMPL`，取值严格为 `1` 时启用 APACE 实现，其他取值或未设置均视为关闭 |
| 默认状态 | **关闭（初期默认 v1 路径）**；APACE 路径上板验证通过后计划切换为默认 |
| 生效域 | 仅非量化 FLOAT16/BFLOAT16 且 rank_size ≤ 64 时可切换至 APACE 实现；量化路径（FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1/INT8/INT4）不受开关影响，恒走 v1 实现（范围隔离） |
| 接口影响 | 无——两分支共享同一 aclnn 接口签名、入参校验与 shape 推导；v1 分支既有 tiling key 编码值不变，APACE 分支 tiling key 为扩维后的新增取值，与 v1 产物并存互不扰动 |

### 通信编排差异说明

APACE 实现的 AllGather 通信采用**双句柄 repeat 批量下发**（dataHead/dataTail 各一个句柄），句柄数恒为 2、与切分 tile 数解耦，repeat 为 uint8（上限 255）。其中 tileCnt 上限 16 为 **CCU 引擎** fit-balance 切分的候选域口径，**AICPU 引擎**自然切分无该 16 上限，两侧 repeat 裕度均充足。ccu / ai_cpu 双引擎（commMode 入参）在两套实现下均支持。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_all_gather_matmul_v2](./examples/test_aclnn_all_gather_matmul_v2.cpp) | 通过[aclnnAllGatherMatmulV2](./docs/aclnnAllGatherMatmulV2.md)接口方式调用AllGatherMatmulV2算子。 |
| aclnn接口（非量化参数化样例） | [test_aclnn_all_gather_matmul_v2_fp16](./examples/arch35/test_aclnn_all_gather_matmul_v2_fp16.cpp) | 通过[aclnnAllGatherMatmulV2](./docs/aclnnAllGatherMatmulV2.md)接口方式调用AllGatherMatmulV2算子。<term>Ascend 950PR/Ascend 950DT</term> 非量化 FLOAT16/BFLOAT16 场景，支持环境变量参数化，内嵌精度自校验。 |

### 非量化参数化样例（Ascend 950PR/Ascend 950DT）

样例 [test_aclnn_all_gather_matmul_v2_fp16.cpp](./examples/arch35/test_aclnn_all_gather_matmul_v2_fp16.cpp) 通过环境变量参数化执行，未设置的环境变量回退内置默认值（m=256, k=256, n=128，2 卡，ccu，fp16）：

| 环境变量 | 含义 | 取值 / 默认值 |
| -------- | ---- | ------------- |
| `AGM_V2_DEV_NUM` | rank 数 | 默认 2（取值 2/4/8/...，与约束说明的卡数支持域一致） |
| `AGM_V2_COMM_MODE` | 通信模式 | `ccu`（默认）/ `ai_cpu` |
| `AGM_V2_DTYPE` | 数据类型 | `fp16`（默认）/ `bf16` |
| `AGM_V2_M` / `AGM_V2_K` / `AGM_V2_N` | 矩阵形状 m / k / n | 默认 256 / 256 / 128（k 取值范围为 [256, 65535)，与约束说明一致） |

出包安装后以 cust 模式运行：

```bash
# 默认参数（2 卡 / ccu / fp16 / m=256, k=256, n=128）
bash build.sh --run_example all_gather_matmul_v2 eager cust \
    --vendor_name=custom --soc=ascend950 \
    --example_name=test_aclnn_all_gather_matmul_v2_fp16

# 参数化运行示例：4 卡 / ai_cpu / bf16 / m=2048, k=4096, n=128
AGM_V2_DEV_NUM=4 AGM_V2_COMM_MODE=ai_cpu AGM_V2_DTYPE=bf16 \
AGM_V2_M=2048 AGM_V2_K=4096 AGM_V2_N=128 \
bash build.sh --run_example all_gather_matmul_v2 eager cust \
    --vendor_name=custom --soc=ascend950 \
    --example_name=test_aclnn_all_gather_matmul_v2_fp16

# 叠加实现切换开关运行 APACE 实现（默认 v1，见「实现说明」）
ASCEND_MC2_AGMV2_APACE_IMPL=1 \
bash build.sh --run_example all_gather_matmul_v2 eager cust \
    --vendor_name=custom --soc=ascend950 \
    --example_name=test_aclnn_all_gather_matmul_v2_fp16
```

说明：

- ccu 模式下样例自动设置 `HCCL_OP_EXPANSION_MODE=CCU_SCHED`；
- 样例内嵌精度自校验：`gatherOut` 与各 rank 的 x1 做位级（bitwise）比对；`output` 与 host 侧 FP32 累加 golden 做混合容差比对（fp16：atol/rtol=2e-3；bf16：atol/rtol=8e-3）；全 rank 通过时进程退出码为 0；
- 批量用例驱动（L0/L1/L2 用例表、dump 落盘与精度比对管线）见 [tests/sim/README.md](./tests/sim/README.md)。


## 有卡环境验证指引

非量化 FLOAT16/BFLOAT16 路径当前完成的验证为编译（全目标出包）、tiling UT（v1/APACE 双分支）、op_api UT、golden 管线自检与代码走查级；**端到端精度、性能与运行时行为须在真实 NPU 环境（≥2 卡、HCCL 可用）执行**。本节为可执行指引，工具细节与判定口径以 [tests/sim/README.md](./tests/sim/README.md) 与 [tests/sim/perf/README.md](./tests/sim/perf/README.md) 为准。

### 前置条件

1. 构建并安装自定义算子包：`bash build.sh --pkg --soc=ascend950 --ops=all_gather_matmul_v2 --module=mc2`，安装产出的 run 包后以 cust 模式运行（见「调用说明」）；
2. CANN 环境已 source；HCCL 多 rank 通信域可建立（样例自带 `HcclCommInitAll` 建链）；
3. build/ 目录与其他构建任务互斥（UT 构建、样例运行、性能采集共用）；
4. **参数化样例契约补全**（批量用例矩阵执行的前置依赖）：种子样例当前仅消费 `AGM_V2_DEV_NUM/COMM_MODE/DTYPE/M/K/N` 6 个环境变量；用例表 dump 契约中的 `AGM_V2_BIAS/TRANS_B/GATHER_OUT/VALUE_SPEC/REPEAT_LAUNCH/SEED/DUMP_DIR` 与计时行 `AGM_V2_TIMING_US`（契约定义见 tests/sim/README.md）尚未消费——含 bias/transB/gatherOut 因子的用例执行与相应性能对比，须先补全样例契约。

### 验证项与执行方式

| # | 验证项 | 执行方式 |
| - | ------ | -------- |
| 1 | 端到端精度全量（L0+L1 共 35 例，含条件例 L1-142 8 卡） | `bash tests/sim/run_sim_cases.sh`（L0+L1 非条件例 34 例）；追加 `--include-conditional` 补齐 L1-142；单例 `--case L0-01`。判定口径（compare_dump.py）：output 混合容差——fp16 rtol=atol=2^-9 且 max_abs_error≤0.1，bf16 rtol=atol=2^-6 且 max_abs_error≤1.0，matched_ratio≥0.99；gatherOut 逐位字节一致（NaN 位形与 ±0 符号位严格检出） |
| 2 | 双分支分发透明性抽查 | 同一用例分别在实现开关两侧执行并比对：默认（v1）与 `ASCEND_MC2_AGMV2_APACE_IMPL=1`（APACE，进程内 tiling 读取），两侧 output/gatherOut 结果一致；性能维度的双分支对比见第 5 项 |
| 3 | ST 补跑（arch35 非量化 28 例） | `tests/st/aclnnAllGatherMatmulV2/arch35_atk_aclnnAllGatherMatmulV2.json`（TTK 多进程框架，WS 2/4）；executor 非量化扩展依赖（bias 透传、transB 实参视图构造、gatherOut 空 tensor）见 tests/sim/README.md「ST 依赖」节 |
| 4 | 运行时专项确认（随精度用例覆盖或抽查） | ①padding 行运行时行为（重点组合：gatherOut 输出 × 尾块非 16 对齐 × 大 k）；②rank_size=64 时 TAIL 区 64 fragment 边界；③ccu / ai_cpu 双引擎行为差异（ccu 优先，ai_cpu 支持口径以实测为准）；④Blaze 累加链端到端数值（含 bias 用例，随第 1 项覆盖） |
| 5 | 性能采集（v1 / APACE 双分支对比） | `bash tests/sim/perf/collect_perf.sh --case L0-01,L1-133 --rounds 10 --mode msprof`（v1 分支）；`bash tests/sim/perf/collect_perf.sh --case L0-01,L1-133 --rounds 10 --mode msprof --apace on`（APACE 分支，自动导出开关 env）；`python3 tests/sim/perf/perf_report.py --raw-root tests/sim/perf/perf_raw` 输出同 case 双分支（`case[v1]` / `case[apace]`）对比报告。MC2 多 rank 必须使用 msprof（`--aic-mode=task-based`），禁止 msprof op；多轮统计以中位数为准 |
| 6 | BI（可选扩展，非验收门槛） | `bash tests/bi/run_bi_test.sh`（现套件聚焦 MXFP8 量化，非量化为可选扩展项） |
