# MXFP8 BSND / BNSD 布局与 metadata 设计和验证

## 结论与范围

本特性将 QuantBlockSparseAttn 的 MXFP8 路径扩展为直接支持 BSND/BNSD Query 布局。
Host 从 shape 获取固定 Sq，Kernel 按布局计算地址与搬运步长，复用矩阵乘法、softmax 和计算流水。

主算子、metadata、Torch 和 GE 统一遵守布局与长度参数契约；
主 Kernel 将 metadata 调度范围限制在物理 Query 行数内，Meta 和真实入口共用 QScale 检查。

支持范围：`quant_mode=2`，Q / QScale / attention_out 扩展到 BSND、BNSD；
K/V 仍为 PA_BNBD，`D=Dv=128`，Q/KV sparse block 均为 64 或均为 128。
`quant_mode=1` 保留 TND/NTD，不新增其 BSND/BNSD 支持；连续 KV 布局也不在此特性范围。

**验证状态：本地 C++ 与 Python 调用边界回归已通过；CANN 构建、真实 Torch 扩展加载、
GE 执行、NPU 精度和性能尚未验证。当前不能宣称已完成上板验收。**

## 1. 输入输出与长度约定

| 布局 | Query | QScale | attention_out | 开启时的 LSE |
| --- | --- | --- | --- | --- |
| TND | `[T,Nq,D]` | `[T,Nq,ceil(D/64),2]` | `[T,Nq,Dv]` | `[T,Nq]` |
| BSND | `[B,Sq,Nq,D]` | `[B,Sq,Nq,ceil(D/64),2]` | `[B,Sq,Nq,Dv]` | `[B,Nq,Sq]` |
| BNSD | `[B,Nq,Sq,D]` | `[B,Nq,Sq,ceil(D/64),2]` | `[B,Nq,Sq,Dv]` | `[B,Nq,Sq]` |

- 主算子的 `layout_out` 必须与 `layout_q` 相同。
- BSND/BNSD 的 Q 和 QScale 必须连续，每个 batch 都使用完整 Sq，不支持同一 Tensor 内变长 Q。
- BSND/BNSD 的 metadata 和主算子调用都传 `cu_seqlens_q=None`、`seqused_q=None`。
  显式空 Tensor 不是 Python/ACLNN 公共入口的 None。
- TND 主算子继续用 `cu_seqlens_q[B+1]`；三种布局的主算子均用 `seqused_kv[B]` 描述 KV 长度。
- 稀疏索引与长度仍为 `[B,Nq,Qb,Kb]` / `[B,Nq,Qb]`，与 Q 的物理轴顺序无关。
  `Qb >= ceil(Sq/q_block_size)`；多分配的 Q 行不能变成 Q 的实际序列长度。
- BSND/BNSD 关闭 LSE 时返回 shape `(0,)`。

## 2. metadata 为什么不需要新增物理 layout 字段

metadata 只记录逻辑 `(batch, query_head, query_block)` 的分核起止位置；
BSND/BNSD 的差异由主 kernel 的编译期布局分支处理。
对相同稀疏输入，TND、BSND、BNSD 的有效 metadata 内容应一致。

metadata 接口没有 Query Tensor 或 Sq 属性，不能从 `Qb * block_size` 反推出准确 Sq，
例如 Sq=129、block=128 时 Qb 至少为 2，而不是说明有 256 个 Q token。
因此保留 metadata 的二进制格式和 API 签名，由已持有准确 Sq 的主 kernel 截断有效范围。
这也保留了 Qb 超额分配的接口能力。

## 3. 接口检查与边界处理

### 3.1 metadata 布局与量化模式检查

metadata 与主算子执行相同的布局约束，拒绝 BSND/BNSD 搭配 FP8、错误的 KV 布局或非空 Q 长度参数。
ACLNN 和 GE 直达 AICPU 的路径均执行检查，避免不同调用入口接受不同的参数组合。

修改文件：

- `quant_block_sparse_attn_metadata/op_kernel_aicpu/quant_block_sparse_attn_metadata.h`：
  增加公共的布局/量化模式判断，供 ACLNN、AICPU 和 Torch metadata 入口复用。
- `quant_block_sparse_attn_metadata/op_host/quant_block_sparse_attn_metadata_check.h`：
  校验 FP8 的 TND/NTD、MXFP8 的 TND/BSND/BNSD 和 PA_BNBD KV；
  新布局要求两个 Q 长度指针为空，稀疏长度为 `[B,Nq,Qb]`，batch/head 与属性一致。
  旧 packed 布局保留 metadata 的 legacy 2D 稀疏长度兼容性。
- `quant_block_sparse_attn_metadata/op_kernel_aicpu/quant_block_sparse_attn_metadata_aicpu.cpp`：
  执行布局校验，使 GE 直接执行也受到约束；检查 sparse dtype、metadata 输出 rank、
  dtype、容量以及固定格式支持的 AIC 核数，避免绕过 ACLNN 后直接写越界。
  GE 的缺省 optional 输入可能表示成零元素占位，因此 AICPU 接受这种内部表示，拒绝有元素的 Q 长度。

### 3.2 Torch metadata 的 optional 空值传递

Python 的 None 需要以空指针传递到 ACLNN；显式零元素 Tensor 仍然是已提供的参数，
不能替代公共入口的 None 契约。

`npu_quant_block_sparse_attn_metadata.cpp` 直接传 optional 参数。
现有 `ops_common.h::ConvertType(optional<Tensor>)` 已支持将 None 转成 `nullptr`。
同时让 NPU 和 Meta 两个入口共用布局/长度/稀疏 shape 检查，防止图追踪阶段接受错误参数。
metadata 的 GE converter 也在构图前拒绝 BSND/BNSD 非法模式或非空 Q 长度。

### 3.3 将 metadata 的 Q 行终点限制在物理 Sq 内

稀疏表可以分配超过物理 Query 所需的行数，分核终点必须与
`ceil(actualS1Size/q_block_size)` 取较小值。

示例：Sq=129、block=128，但稀疏表分配了更多 Q 行。若分核终点为 3，
若不截断，循环会访问 row=2，其 token 起点为 256，随后 `129-256` 的无符号尾块长度发生下溢。
这种情况既影响正常计算，也影响空行清零路径。

在 `arch35/common/util_regbase_mx.h` 增加生产函数 `MxS1LoopEnd()`，
`quant_block_sparse_attn_mx_kernel.h` 使用它把分核终点截断到实际 Q 行数。
行数计算使用除法和余数，避免 `length + block - 1` 的 32 位溢出。
该边界处理同时用于 MXFP8 的 TND 和新布局，不增加 metadata 格式字段。

### 3.4 主算子 NPU / Meta 共用 QScale 检查

`npu_quant_block_sparse_attn.cpp` 的两个入口共用 `CheckPaddedQueryInputs()`：
校验非空四维 Q、连续 Q/QScale、五维 QScale、前三轴与 Q 一致、末两维为 `[ceil(D/64),2]`，
并统一两个 Q 长度参数必须为 None 的约束。

### 3.5 每种布局独立生成 metadata

`QBSA_CHECK_PADDED_LAYOUTS=1` 自检为每种被比较的布局分别生成 metadata，
再运行主算子比较输出和 LSE，同时覆盖布局对应的 metadata 入口和主计算路径。

公开说明 `QuantBlockSparseAttnMx.md` 统一记录 Q/QScale/output/LSE shape、长度参数和 layout 约束。

## 4. 完整链路检查记录

| 层次                   | 复核结果                                                  |
| -------------------- | ----------------------------------------------------- |
| Query parser         | 识别四维 BSND/BNSD，解析 B、Sq、Nq，使用 B*Sq 容量            |
| Host shape/check     | 校验四维 Q/output、五维 QScale、BNS LSE、Qb 容量与量化模式      |
| InferShape / tiling  | 推导四维输出、BNS LSE，并传递 qSeqSize 与布局 tiling key      |
| Cube Q/QScale 搬运     | 通过布局槽位和行步长区分 BSND/BNSD                          |
| Vector output/LSE    | 正常写回与 ClearEmptyQBlock 共用地址规则，LSE 固定 BNS             |
| 主 kernel 长度          | 新布局不读取/预取 cu_seqlens_q；metadata 行终点按物理长度截断       |
| metadata ACLNN/AICPU | 校验布局/长度/shape 契约及 AICPU 输出容量                        |
| Torch NPU/Meta       | metadata 保留 None；主算子统一 QScale 检查                      |
| GE converter         | 透传布局并执行 metadata 构图前契约检查                          |
| Golden / TTK         | 转换公共布局，按布局提取 B/N；golden 为每种布局独立生成 metadata |

## 5. 已完成的本地验证

环境：Windows，Python 3.12；临时下载并校验官方 SHA256 的 Zig 0.14.1 便携编译器。
本机没有 CANN、torch_npu 或 NPU。编译器和产物位于 Git 忽略的 `build/layout-validation/`。

### C++ 生产代码回归

新增 `quant_block_sparse_attn_metadata/tests/standalone/`，使用少量 Tensor/Context 容器替身，
**实际编译**生产 ACLNN 检查头、AICPU scheduler/serializer 和设备地址/边界辅助函数。
不重写一份调度算法作为被测对象；这仍不能代替真实 CANN ABI 或 AscendC 编译。

已通过：

- 1,440 组 padded 调度/覆盖检查：B=2、Nq=4、GQA，block=64/128，
  Sq=1/17/63/64/65/127/128/129/257，AIC=1/2/7/36，Qb 正好够用或额外增加 5 行。
- 稀疏模式包含全空、全有效、固定随机种子、前半空/后半空；
  解码每个核/section 的生产 metadata，验证有效 Q 行恰好被覆盖一次。
- 三种布局的多 section 场景；相同稀疏输入的 padded metadata 与 TND 一致。
- ACLNN/AICPU 的合法布局、非法布局/模式、显式长度/空 Tensor、batch/head 错误、
  dtype 错误、输出不足等正反例；输出缓冲区带哨兵检查。
- 3 batch、8 heads、129-token 的 Q/QScale/output 槽位、BNS LSE 地址，
  超过 32 位的地址偏移及 UINT32_MAX 行数边界。

额外做了两项负向对照：在独立 build 副本中使用基线的 ACLNN 检查头，测试按预期报
`host accepted explicit query lengths`；将终点裁剪恢复为原行为，测试按预期报
`metadata reads beyond physical query rows`。工作目录中的修复未回退。

在仓库根目录运行（任一 C++17 编译器）：

```bash
python experimental/attention/quant_block_sparse_attn_metadata/tests/standalone/run_tests.py --cxx g++
```

也支持 `--zig /path/to/zig`；`--build-dir PATH` 指定产物目录。

### Python 调用边界回归

两项测试已通过：执行生产 converter/golden 函数体，使用记录型后端替身检查
布局属性、None 透传、错误输入在构图前被拒绝，以及 TND/BSND/BNSD 分别生成 metadata。
它们验证调用关系，不运行 GE/NPU。

```bash
python experimental/attention/quant_block_sparse_attn_metadata/tests/standalone/test_converter.py -v
```

另已通过 5 个 Python 文件语法检查、两套 STC 各 124 个用例的列数与名称唯一性检查、
各 12 个 padded 用例的布局/空长度契约检查，以及 `git diff --check`。

## 6. 上板验收步骤（尚未执行）

1. 在支持 MXFP8 的 Ascend 950 环境，按仓库构建流程重新编译安装主算子的 host/kernel、
   metadata host/AICPU 库及当前 `torch_ops_extension`。不能只替换 Python 文件。
   MX tiling-data 新增 qSeqSize 字段，host/kernel 必须同版本部署。

   在仓库根目录、激活 CANN 环境后构建两个算子（参数已对照当前 `build.sh`）：

   ```bash
   bash build.sh --pkg --experimental --soc=ascend950 --ops=quant_block_sparse_attn,quant_block_sparse_attn_metadata --opkernel_aicpu
   ```

   安装 `build_out/` 中本次生成的 run 包并激活安装目录的 `set_env.bash`，
   具体安装路径沿用测试机现有配置。然后在完整仓库内构建并安装扩展：

   ```bash
   (cd experimental/attention/quant_block_sparse_attn/torch_ops_extension && bash build_and_install.sh)
   ```

   注意需要保留完整仓库目录关系：Torch metadata 入口复用了相邻 metadata 算子的布局检查头。

2. 运行 Host UT（包括 `MetadataEndsAreClippedToPhysicalQuery`）。

   ```bash
   bash build.sh --experimental --soc=ascend950 --ops=quant_block_sparse_attn --ophost_test
   ```

3. 运行新增的真实 Torch/NPU 回归，缺少设备或扩展时会报错，不会静默跳过：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/test_layout_metadata.py
```

它检查真实 Meta 输出/拒绝行为及 32 组 padded NPU 对照（两布局、两 block、两 mask、
LSE 开关、精确/超额 Qb）。基准是相同输入的 TND 且 Qb 恰好够用；
BSND/BNSD 分别生成 metadata，超额 Q 行故意设为非零，比较完整 output/LSE。

同一脚本还包括 `test_ge_metadata_matches_eager`：将 metadata 算子直接放入
`torch.compile` 的 GE 图，比较 TND/BSND/BNSD 的 GE 与 eager 有效 metadata 内容。
它使用默认 GE 编译配置；原 golden 的 `--aclgraph` 仅捕获主算子、在图外生成 metadata，
不能替代这个 converter/AICPU 图执行用例。该新增 GE 测试目前仅通过语法检查，尚未在设备执行。

4. 执行新增的 12 条 BSND/BNSD STC，以及 TND 回归；
   eager/graph 都要覆盖。可用已有 golden 的入口，例如：

```bash
cd experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test
python qbsa_mxfp8_golden.py --csv --case_files qbsa_mxfp8_stc.csv --case_name MXSTC003_B1_N3K3_G1_S513x513_SB128_PA128_EQUU_M3R_L1_PVC_BSND --mode all
python qbsa_mxfp8_golden.py --csv --case_files qbsa_mxfp8_stc.csv --case_name MXSTC003_B1_N3K3_G1_S513x513_SB128_PA128_EQUU_M3R_L1_PVC_BSND --mode all --aclgraph
QBSA_CHECK_PADDED_LAYOUTS=1 python qbsa_mxfp8_golden.py --csv --case_files qbsa_mxfp8_stc.csv --case_name MXSTC003_B1_N3K3_G1_S513x513_SB128_PA128_EQUU_M3R_L1_PVC --mode all
```

5. 在同一台设备、同一套输入下检查 TND/BSND/BNSD 精度与性能；
   本地调度和地址测试不能证明硬件性能或实际 DMA 指令行为。
