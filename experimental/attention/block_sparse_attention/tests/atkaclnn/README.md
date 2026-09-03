# atk-mxfp4-aclnn

MXFP4 Block Sparse Attention V3 的 ATK 看护工程，**pyaclnn 版本**：DUT 侧直接调用两段式
`aclnnBlockSparseAttentionV3GetWorkspaceSize` / `aclnnBlockSparseAttentionV3`，绕开
`torch_npu` 的 python 封装。与同级的 `atk-mxfp4`（走 `cann_ops_transformer.ops.block_sparse_attention`）
互为补充：同一份用例、同一份 CPU 标杆，只换 DUT 的调用方式。

调用方案参考 `新建文件夹/Gmm_mxa8w4_api/.../function_aclnn_quant_matmul_weight_nz_npu_mxfp4_a8w4.py`
里的 `pyaclnn_aclnn_quant_matmul_weight_nz`。

> **本工程已把 torch 调用全部删除**：不 import `cann_ops_transformer`（import 会触发
> OpBuilder/`torch.utils.cpp_extension.load` 的 JIT 编译），`BlockSparseAttentionV3Api`
> 不再有 `torch_npu` 封装分支，DUT 只经 `PyAclnnBlockSparseAttentionV3Api` 直调 aclnn。
> 数据造数 / CPU 标杆仍用 torch 张量（ATK 输入契约即 torch tensor，绕不开）。
> 保留 torch 调用的完整版本备份在同级 `atk-aclnn-torch/`。

## 文件

| 文件 | 作用 |
|------|------|
| `op_block_sparse_attention.yaml` | 用例定义；比 `atk-mxfp4` 多一行 `aclnn_api_type` |
| `generator_blocksparseattention.py` | 用例生成器，与 `atk-mxfp4` 完全一致 |
| `function_aclnn_blocksparseattention_v3_mxfp4.py` | 造数 + CPU 标杆 + pyaclnn 执行器 |
| `nodes.yaml` | cpu 标杆节点 + pyaclnn 节点 |
| `run-bsa-aclnn.sh` | 生成用例并执行 |

## 两个注册点

- `aclnn_blocksparseattention_v3`（`api_type`）→ `BlockSparseAttentionV3Api(BaseApi)`：
  负责 MXFP4 造数（fp32 → e8m0 scale + nibble-pack）与 CPU 双标杆。只支持 CPU 执行；
  `device == "npu"` 会直接抛错（torch_npu 封装分支已删除），被测请走 pyaclnn 节点。
- `pyaclnn_aclnn_blocksparseattention_v3`（`aclnn_api_type`）→
  `PyAclnnBlockSparseAttentionV3Api(AclnnBaseApi)`：按 C++ 原型逐位组装 `input_args`。

## 为什么自己建 aclTensor 而不用 ATK 的 convert 通道

三个原因，都是 kwargs 的 dtype/shape 表达不出来的：

1. `query/key/value` 的缓冲区是 uint8 nibble-pack `[..., D/2]`，算子要按 `FLOAT4_E2M1`
   看到 `[..., D]`。view dims 与 storage dims 都填逻辑 shape，否则 ACL 会按 0.5B/元素
   再算一遍偏移导致地址错位。
2. `qDequantScale/kDequantScale/vDequantScale` 必须是 `FLOAT8_E8M0`，而 dataset 侧只能
   给 uint8。`float8_e8m0fnu` 的 CPU→NPU 直拷在部分 torch_npu 版本上不支持，所以先按
   uint8 上 NPU 再 view。
3. 原型里的 `pQuantScaleOptional` 在用例 yaml 中不存在，位置只能显式补 nullptr。

另外 `aclIntArray*` 的空指针必须带正确的 ctypes 类型：ATK `before_call` 会拿 `input_args`
的类型逐个和 `get_cpp_func_signature_type()` 返回的原型比对，用 `aclTensor*` 的空指针
冒充会当场判成类型不匹配。

## 数据一致性

cpu 与 pyaclnn 两个节点各自独立造数，靠 `case_derived_seed(用例参数...)` 保证逐位一致
（不能用常量种子，否则所有用例退化成同一份数据）。pyaclnn 侧优先复用 `backend.api_executor`
已造好的结果，省掉一遍量化打包。

## 用法

```bash
bash run-bsa-aclnn.sh
```

节点顺序不能反：pyaclnn 的 `output_info` 依赖标杆节点先跑出输出信息，否则会报
"标杆输出为空"。

## 已知约束

- `maskType`/`blockSize`/`softmaxLseFlag` 固定 0，`preTokens`/`nextTokens` 固定
  2147483647，与 doc 约束一致。
- BNSD/BSND 变长场景 kernel 不写 `[actualSeqLen, S_padded)`，DUT 与标杆两侧都显式置 0
  后再比对。
