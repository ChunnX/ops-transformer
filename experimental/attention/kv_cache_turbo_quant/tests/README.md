# 自测（精度 / 性能）

本目录给出 KvCacheTurboQuant 的**可复现自测**，共三套：

| 套件 | 入口 | 内容 |
| --- | --- | --- |
| ① 指定用例自测 | `run_all.sh` | `case.json` 的 5 个 case：计时 + 逐字节对拍 + V3 精度 + 生态标准层 |
| ② 边界用例 | `run_edge_cases.sh` | 10 个边界 shape（head 数上下界、tail 非整块、wave 临界、零/近零行、非有限行）：对拍 |
| ③ 非法输入拒绝 | `invalid_cases.py` | 11 个非法输入（shape/dtype/attr 越界）必须被 tiling 拒绝 |

## 前置条件

自测所需代码 = 本目录 `tests/`（用例表 + 数据生成 + 对拍 + 精度 + 边界/非法套件 + 算子 UT）
+ 算子目录下 `examples/`（aclnn 调用驱动器）；两者随算子目录一并提供，无需改动仓库内其它文件。

前置条件：① 已获取任务包（含 `golden.py`、`case.json`、`op.json`，本包 `tests/` 下为原样副本）；
② 已能构建并安装本算子包（见 §1）或已获得对应的 ops-transformer 分支。

## 用例清单（精度测试 case / 性能测试 case）

全部用例来自任务包 `case.json`（原样复制、未修改）。**同一个 case 同时承担精度与性能判定**：
精度判据 = V2 逐字节对拍 + V3 双口径指标；性能判据 = 相对任务书 §3.5 的 PyTorch eager 基线 ≥10×。

**精度测试 case**（由 `run_all.sh` 逐 case 调 `kctq_compare.py` + `kctq_v3_precision.py`）

| # | case | num_tokens | num_kv_heads | head_dim | mse_bits | 精度判据 | 阈值 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `gqa_decode_b1` | 1 | 8 | 128 | 3 | relMSE(MSE-only) / 内积 p95(归一化含 QJL) | < 0.05 / < 0.1 |
| 2 | `gqa_decode_b64` | 64 | 8 | 128 | 3 | 同上 | 同上 |
| 3 | `gqa_prefill_t2048` | 2048 | 8 | 128 | 3 | 同上 | 同上 |
| 4 | `gqa_mse2_t2048` | 2048 | 8 | 128 | 2 | 同上 | 同上 |
| 5 | `gqa_mse4_t2048` | 2048 | 8 | 128 | 4 | 同上 | 同上 |

附加精度用例：套件② 的 10 个边界 case（逐字节对拍）；V1 接口契约检查（4 个输出 dtype/shape 与 `case.json` per-case 比对）。

**性能测试 case**（`run_all.sh` 内的计时串：warmup 3 + repeat 10 → 中位数；基线为任务书固定表，只读）

| # | case | 基线 (µs) | ≥10× 门槛 (µs) | 说明 |
| --- | --- | ---: | ---: | --- |
| 1 | `gqa_decode_b1` | 1491.676 | 149.2 | GQA decode |
| 2 | `gqa_decode_b64` | 1552.784 | 155.3 | GQA decode batch |
| 3 | `gqa_prefill_t2048` | 3855.422 | 385.5 | GQA prefill（余量最紧） |
| 4 | `gqa_mse2_t2048` | 3607.963 | 360.8 | 2-bit 主编码 + 1-bit QJL |
| 5 | `gqa_mse4_t2048` | 5286.502 | 528.7 | 4-bit 主编码 + 1-bit QJL |

逐 case 结果落在 `<work>/out/<case>_b<bits>/{run.log,compare.log,v3.log}`；汇总见 `run_all.sh` 末尾列表。

## 0. 依赖

- CANN 9.1.0+、Ascend 910B（A2）/`ascend910_93`（A3）
- Python 3.11（`numpy`、`ml_dtypes`）、`torch` + `torch_npu`（对拍与精度用）
- g++（编译 `examples/` 里的 aclnn 驱动器）

## 1. 构建并安装算子

```bash
cd <ops-transformer 根目录>
bash build.sh --pkg --experimental --soc=ascend910b --ops=kv_cache_turbo_quant
cd build
./cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=<CANN>/opp
source <CANN>/opp/vendors/custom_transformer/bin/set_env.bash
```

> 也可用 `--extract=<dir> --noexec` 解包后不安装：此时把 `<dir>/packages` 传给
> `examples/build_and_run.sh`，并让运行环境能找到 `libcust_opapi.so`（加入 `LD_LIBRARY_PATH`）。

## 2. 编译 aclnn 驱动器

```bash
bash examples/build_and_run.sh <算子包>/packages /tmp/kctq_example
```

## 3. 生成测试数据

任务包里**未随附** `data/rotation_matrix.bin`、`data/qjl_matrix.bin`（`case.json` 引用了它们），
本目录用固定种子生成统计等价的矩阵，并把 sha256 写进 `data/meta.json`：

```bash
python tests/gen_test_data.py --out data        # 归一化 Hadamard 旋转矩阵 + 高斯 QJL 矩阵
python tests/gen_case_input.py --out data/kctq_case_b1 --tokens 1    --heads 8
python tests/gen_case_input.py --out data/kctq_case_b64 --tokens 64  --heads 8
python tests/gen_case_input.py --out data/kctq_case_t2048 --tokens 2048 --heads 8
# 生成器按 case 名产出独立输入；官方件优先（见下方 KCTQ_OFFICIAL_DATA）
```

> case 输入以**官方件为准**：`run_all.sh` 优先读 `$KCTQ_OFFICIAL_DATA/<case_name>/`
> 下的 `kv_vectors.bin` / `ip_query.bin`（官方 `generate_data.py` 的产物），
> **三个 t2048 形态 case 各用自己那一份**（不再是共用一份），仅在拿不到官方件时才退回
> `gen_case_input.py` 生成的等价输入。

接口扩展回归（矩阵 bf16 / metadata fp16）与正式回归共用同一份矩阵：
`$KCTQ_MATRIX_DIR`（默认 `tests/_out/data`，缺失时自动由 `gen_test_data.py` 生成）。

输入按 `case.json` 声明的 `value_range` 生成（`kv_vectors` 为 `[-2, 2]` 均匀、BF16 舍入）。

## 4. 一键自测

```bash
export KCTQ_MATERIAL=<任务包目录>          # 含 golden.py 的目录
export KCTQ_GOLDEN=$KCTQ_MATERIAL/golden.py
export KCTQ_BIN=/tmp/kctq_example/test_aclnn_kv_cache_turbo_quant
bash tests/run_all.sh
```

`run_all.sh` 会按 `case.json` 依次跑 5 个 case：算子 → `kctq_compare.py`（逐字节对拍）→
`kctq_v3_precision.py`（V3 双口径）→ `kctq_std_tolerance.py`（生态标准混合容差），并在最后打印汇总。

### 4.1 边界用例套件

在 `case.json` 的 5 个指定 case 之外，补充覆盖边界 shape（`edge_cases.json`），逐 case 与 `golden.py` 对拍：

```bash
export KCTQ_MATERIAL=<任务包目录> KCTQ_GOLDEN=$KCTQ_MATERIAL/golden.py
export KCTQ_BIN=/tmp/kctq_example/test_aclnn_kv_cache_turbo_quant
bash tests/run_edge_cases.sh          # 10 个用例，逐个打印 对拍 PASS/FAIL + 中位数
```

| 用例 | 覆盖点 |
| --- | --- |
| `heads4_t32_b3` / `heads32_t8_b3` | `num_kv_heads` 下界 4 / 上界 32（GQA 广播比 1:8、1:1） |
| `heads4_t1_b2` / `heads32_t1_b4` | 最小 tokens（1）与 bits 2/4 组合 |
| `tail_t3_h8_b3` / `tail_t9_h8_b4` / `tail_t17_h8_b2` | tokens 非核数整数倍（对齐尾块） |
| `wave_t2081_h8_b3` | 2081 行触发多轮/尾轮调度 |
| `edge_rows_t8_h8_b3` | 行内容边界：全 0 行、次正规量级行、正负成对抵消行 |
| `extreme_rows_t8_h8_b3` | 特殊值行：+inf / −inf / nan / 1e4 大值（生态标准 §1.4 的 INF/-INF/NAN 场景；位打包输出在该域未定义，见口径说明） |

### 4.2 非法输入拒绝套件

```bash
python tests/invalid_cases.py --json /tmp/invalid_cases.json
```

用 `examples` 驱动器的 `--neg <mode>` 逐项构造非法输入，要求 `GetWorkspaceSize` 返回非 0
（驱动器退出码非 0）。覆盖 `op_host/kv_cache_turbo_quant_tiling.cpp` 的全部校验点：
`mse_bits ∉ {2,3,4}`（5/1）、`kv_vectors` rank=2、`head_dim≠128`、`num_kv_heads < 4`（2）、
`> 32`（64）、`kv_vectors` dtype≠BF16、`rotation_matrix` 形状≠[128,128]、
`rotation_matrix`/`qjl_matrix` dtype≠FP32、空 tensor（tokens=0）；并含 `valid_base`
合法基准自检（必须被接受），防止"全部拒绝"式的假通过。

### 4.3 算子 UT（op_host，仓标准方式）

```bash
cd <ops-transformer 根目录>
PATH=<python3.11>/bin:$PATH \
  bash build.sh -u --ophost --experimental --ops=kv_cache_turbo_quant --soc=ascend910b
```

覆盖 tiling 校验（合法 7 + 非法 10）与 shape 推导（合法 4 + 非法 3），共 **24 例**；本机实测 `[
 PASSED ] 24 tests`。UT 只跑到 tiling/推导层（不执行 kernel），数值正确性由上面三套件保证。

### 单步命令（等价）

```bash
$KCTQ_BIN --kv data/kctq_case_t2048/kv_vectors.bin --rot data/rotation_matrix.bin \
    --qjl data/qjl_matrix.bin --out out/t2048_b3 --tokens 2048 --heads 8 --bits 3 \
    --warmup 3 --repeat 10
python tests/kctq_compare.py --workdir out/t2048_b3 --rot data/rotation_matrix.bin \
    --qjl data/qjl_matrix.bin --tokens 2048 --heads 8 --bits 3 --material "$KCTQ_MATERIAL"
python tests/kctq_v3_precision.py --workdir out/t2048_b3 --rot data/rotation_matrix.bin \
    --qjl data/qjl_matrix.bin --tokens 2048 --heads 8 --bits 3 --golden "$KCTQ_GOLDEN"
```

## 5. 测量与判定口径

- **计时**：warmup 3 + repeat 10，报告**中位数**；计时环内包含每轮的
  `GetWorkspaceSize` + workspace `aclrtMalloc` + launch + sync（保守口径）。
  若需"预分配 workspace"口径，把分配移出计时环即可（同一实现、同一数据）。
- **对拍**：与任务包 `golden.py` **逐字节**比较四个输出，打印每个输出的不一致数量与匹配率；
  末行 `RESULT: PASS/FAIL` 只统计 `idx`/`qjl` 是否逐字节完全一致（`experimental_standard.md` §0
  不覆盖位打包类输出，该行仅作诊断统计）。
- **实测差异位置**：`quant_gamma` 在 3-bit 档、`quant_qjl` 在 4-bit 档各有 **1 个单点**与 `golden.py`
  不同（分别为 1 ULP / 1 bit 级；其余输出逐位一致）。因此 4-bit 档对拍末行会打印 `RESULT: FAIL`。
  逐 case 明细见各 case 日志；V3 指标与生态标准层判定见交付件 `task_submission/`（精度报告与日志）。
- **非法输入**：判据是"必须被拒绝"，且 `valid_base` 必须被接受；打印的 `error` 字段区分
  拒绝来源（`561002` tiling 校验失败 / `161002` OpDef dtype 不一致）。
- **非有限输入（inf/nan）**：任务书与 `golden.py` 均未规定含 inf/nan 的行内位打包输出
  （`idx`/`qjl`）语义；实测算子与 golden 在这些行上给出不同编码（标量输出 `norm`/`gamma`
  语义一致：`norm=inf`、`gamma=nan`）。因此 `kctq_compare.py` 对**非有限输入行
  不参与 idx/qjl 判定**（单独打印行数与不判定处数），`norm`/`gamma` 则按“逐位一致 **或** 同为 nan”
  判定；有限行（声明域）仍要求逐字节一致。对应用例：`extreme_rows_t8_h8_b3`。
- **V3 精度**：主口径即**任务书 §3.2 字面式** —— relMSE = Σ‖x−x̂‖²/Σ‖x‖²（主量化 + 1-bit QJL
  完整重构）、内积 p95 用**字面分母** `max(|<q,x>|,1e-6)`（q 为官方 `ip_query`，与 x 相关）；
  另给“仅主量化”对照行（2-bit 该行 ≈0.116 = 标量量化理论下界），并给出逐向量 NMSE max/P95
  （2-bit 按官方单列阈值判 max<0.10、P95<0.06）。
- **输入按 case 区分**：`run_all.sh` 优先用 `KCTQ_OFFICIAL_DATA/<case_name>/{kv_vectors,ip_query}.bin`
  （官方件 per-case 输入），其次 `$KCTQ_WORK/inputs/<case_name>/`，再回落旧布局与生成器；
  日志逐行给出 `input=` / `ip_query=` 的 sha256（三个同 shape 的 t2048 档各用各的输入）。
- **生态标准层**（`kctq_std_tolerance.py`）：按《生态算子开源精度标准》
  （`third_party/opbase/docs/zh/ops_precision_standard/experimental_standard.md`，上游文件名
  `mixed_tolerance_standard.md`）的**混合容差**逐元素判定：`|a-g| ≤ atol + rtol·|g|`，
  整体要求 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ limit`。
  `quant_norm`/`quant_gamma`（bf16）用 **BFLOAT16 行**（rtol = atol = 2⁻⁶、limit = 1e0）；
  `quant_idx`/`quant_qjl` 是 uint8 位打包，**标准 §0 明确不覆盖整型/搬运类输出**，只报
  "匹配率"（参照 0.99）并显式声明。该层与任务书 §3.2 的业务指标并列给出，两者判定数据见
  交付件 `task_submission/`。

## 6. 文件说明

| 文件 | 说明 |
| --- | --- |
| `case.json` / `op.json` | **任务包原样复制**（未修改）：5 个自测 case 的 shape/阈值与算子接口 |
| `gen_test_data.py` | 生成旋转/投影矩阵与 `data/meta.json`（含 sha256、来源说明） |
| `gen_case_input.py` | 生成某个 case 的 `kv_vectors.bin`（BF16） |
| `kctq_compare.py` | 与 `golden.py` 逐字节对拍（`--material` 指定任务包目录） |
| `kctq_v3_precision.py` | V3 精度（任务书 §3.2 字面口径：relMSE = Σ‖x−x̂‖²/Σ‖x‖²，主量化 + 1-bit QJL 完整重构；内积 p95 用**字面分母** `max(|<q,x>|,1e-6)`，q 取官方 `ip_query`（`--ip-query`）或缺省自采 `BF16(x+0.5N(0,1))`；另报逐向量 NMSE max/P95。`--golden` 指定 `golden.py`） |
| `kctq_std_tolerance.py` | 生态标准混合容差逐元素判定（bf16 输出按标准行；位打包输出按匹配率） |
| `run_all.sh` | 一键跑 5 个 case 的对拍 + V3 + 计时汇总（套件①） |
| `edge_cases.json` | 边界用例形状表（套件②） |
| `gen_edge_input.py` | 生成边界用例输入；`--pattern edge-rows` 造零行/次正规行/抵消行 |
| `run_edge_cases.sh` | 一键跑 10 个边界用例（套件②） |
| `invalid_cases.py` | 非法输入拒绝套件（套件③，驱动 `examples` 的 `--neg` 模式） |
| `ut/op_host/` | 算子 UT（gtest）：tiling 校验 17 例 + shape 推导 7 例，见 §4.3 |

> 运行产物（`_out/`、`data/`、`inputs/`）不入库，见 `.gitignore`。
