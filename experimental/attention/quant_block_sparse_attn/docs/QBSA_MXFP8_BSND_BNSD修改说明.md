# QBSA-MXFP8 新增 BSND / BNSD 修改说明

## 基线与范围

基于本地 `master` 提交 `125200e0a`，目标分支为 `layout`。
修改对象为 `experimental/attention/quant_block_sparse_attn`，仅扩展 `quant_mode=2` 的 Q、Q scale 和输出布局。
K/V 继续使用分段 PA_BNBD；现有 FP8 的 TND/NTD 路径保留。

## 设计选择：tiling key 与 tiling data 各司其职

复用现有 `LAYOUT_T` 模板参数，在 tiling key 中增加 BSND=6、BNSD=7。
Kernel 用 `if constexpr` 选择布局，没有新增一套 Cube/Vector 计算模板。
MX tiling data 新增 `qSeqSize`，传递 padded Q 的物理序列长度 Sq。
普通 tiling data 是运行时值，不能直接用于 `if constexpr`。

现有 TND 编译实例不增加布局判断的运行时分支，Q/QScale 的原地址表达式保留。
此设计不能代替性能验证：模板实例数会增加，TND 性能仍需上板对比。

## 输入输出约定

| layout_q / layout_out | Q | Q descale | attention_out | LSE |
| --- | --- | --- | --- | --- |
| TND | [T,Nq,D] | [T,Nq,ceil(D/64),2] | [T,Nq,Dv] | [T,Nq] |
| BSND | [B,Sq,Nq,D] | [B,Sq,Nq,ceil(D/64),2] | [B,Sq,Nq,Dv] | [B,Nq,Sq] |
| BNSD | [B,Nq,Sq,D] | [B,Nq,Sq,ceil(D/64),2] | [B,Nq,Sq,Dv] | [B,Nq,Sq] |

必须显式传入匹配的 `layout_q` 和 `layout_out`。本版不增加两者之间的布局转换。
Q 和 Q descale 要求连续。attention_out 为 BF16，LSE 为 FP32。
LSE 关闭时，新布局返回空 Tensor；开启时固定 BNS。

继续要求 `cu_seqlens_q` 为 int32 [B+1]，起始值为 0，各差值表示有效 Q 长度，且位于 [0,Sq]。
例如有效长度 [129,65]、物理 Sq=129 时，传 [0,129,194]；第二个 batch 的物理起点仍是 129 个 token 容量。
本版不消费 `seqused_q`。仅写有效 token；padding 部分的输出值未定义。
稀疏索引保持 [B,Nq,Qb,Kb]，Qb 至少覆盖 ceil(Sq/q_block_size)，无效 Q 行应令 sparse_seq_len 为零。

## 模块改动

1. **布局与 tiling key**：`op_kernel/quant_block_sparse_attn_const.h` 新增布局及 axis 映射；
   `quant_block_sparse_attn_template_tiling_key.h` 增加 MX 的实例选择，FP8 实例列表保留。
2. **Host parser/check**：`op_host/quant_block_sparse_attn_info_parser.*` 解析四维 Q 和物理 Sq，
   总 token 容量按 B*Sq 计算；`quant_block_sparse_attn_check.cpp` 校验量化模式、匹配布局、
   Q/QScale/输出 shape、连续 stride 和稀疏 Q 行容量。
3. **InferShape/TilingData**：`quant_block_sparse_attn_infershape.cpp` 推导四维输出和 BNS LSE，
   输出 Dv 读取 Value；`quant_block_sparse_attn_tiling.cpp` 写入新增的 MX `qSeqSize` 字段。
4. **Kernel 调度与地址**：`arch35/quant_block_sparse_attn_mx_kernel.h` 将有效 Q 长度和物理 batch 起点分开，
   使用 `b*Sq` 定位 padded batch；`common/util_regbase_mx.h` 统一 Q/scale/输出 token-head 地址计算，使用 64 位偏移。
5. **Cube 搬运**：`quant_block_sparse_attn_mx_block_cube.h` 的 Q ND2NZ、QScale DN2NZ 使用对应行间距：
   BSND 为 Nq*D / Nq*Cscale，BNSD 为 D / Cscale。L1/L0、MX 随路反量化及 Matmul 主计算复用。
6. **Vector 写回**：`quant_block_sparse_attn_mx_block_vec.h` 同步修改正常输出与 ClearEmptyQBlock；
   两种新布局的 LSE 地址均为 `(b*Nq+n)*Sq+s`，目标行间空隙为 0。
7. **Torch extension**：`torch_ops_extension/custom_ops/csrc/npu_quant_block_sparse_attn.cpp` 同步 NPU/Meta 的输出分配，
   新布局四维 attention_out、三维 BNS LSE，检查连续 Q/QScale；converter 原本透传布局属性，可直接复用。
8. **Golden**：`tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py` 放开布局检查，
   在调用边界将 packed CPU 数据打包为 BSND/BNSD，QScale 同步转换；将有效输出恢复 TND/TN 与 CPU golden 比较。
   eager 和 graph 路径共用该适配。额外提供 `QBSA_CHECK_PADDED_LAYOUTS=1` 的 TND 等价回归入口。
9. **TTK**：`tests/assets/bsa_ttk_ops.py` 修正 BNSD 的 batch/head/sequence 识别；
   `tests/assets/impl/inputs.py` 生成对应公共 Q/QScale 布局，`impl/golden.py` 返回对应 output 和 BNS LSE。
10. **Metadata**：已检查独立 metadata 算子，它按 sparse_seq_len 的 batch/head/query-row 调度，
    不使用 Q 的物理地址，本版不改变 metadata 格式；修改有效长度/稀疏行后需重新生成 metadata。

## STC 与单元测试

- pytest STC 新增 12 条（每种布局 6 条）：覆盖 B=1/2/4、GQA、64/128 稀疏块、dense/random、mask=0/3、
  对齐与非对齐尾块、batch 间不同有效长度、LSE 开关。
- TTK STC 新增 12 条匹配布局用例。TTK 全 Tensor 比较不忽略 padding，因此新 TTK 用例采用各 batch 等长 Q；
  batch 间不等长由 pytest STC 覆盖。
- InferShape CSV 新增 8 条：两种布局的 LSE 开/关、输出布局不匹配、FP8 拒绝新布局。
- Host tiling CSV 新增 6 条：两种布局正确输入、QScale shape 错误、LSE shape 错误。
- `test_quant_block_sparse_attn_shape_infershape.cpp`：直接测试生产 C++ 地址函数，覆盖多个 batch/head、129-token 尾块及超 32 位偏移。
- `qbsa_mxfp8_golden.py --check_layout_helpers`：合并后的 CPU 打包/还原自检，覆盖数据、scale、LSE、空 batch 和容量错误；使用现有 golden 依赖环境，不调用 NPU 算子。

测试文件均位于当前算子目录。运行单个新增 STC 的示例：

```bash
cd experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test
python qbsa_mxfp8_golden.py --csv --case_files qbsa_mxfp8_stc.csv --case_name MXSTC003_B1_N3K3_G1_S513x513_SB128_PA128_EQUU_M3R_L1_PVC_BSND --mode all
python qbsa_mxfp8_golden.py --check_layout_helpers
```

## 验证状态与部署注意事项

- 已在本机 MSVC 编译并运行生产 C++ 地址函数测试（使用轻量断言适配替代本机缺失的 gtest）。
- 已通过修改 Python 文件的语法检查、CSV 列数/用例名唯一性检查及 git diff --check。
- 使用 NumPy 字节视图适配器执行生产 golden 打包/还原函数，54 组场景通过；该检查不等同于真实 Torch/NPU 执行。
- CANN Host/Kernel 全量编译、Host UT、CPU Torch 单测、NPU STC 精度与性能尚未执行；本机无 CANN/NPU/Torch 环境。
- MX tiling data 的结构发生变化，必须同时重新编译安装 host 与 kernel，不能混用旧二进制。
- NPU 回归须包含旧 TND 用例，性能对比建议使用相同硬件、数据和测量配置。
