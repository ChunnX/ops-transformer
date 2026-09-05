# flash_mla_with_kvcache 编译与单算子调用

本仓 flash_mla_with_kvcache 算子族的编译/安装/调用说明（Ascend 950 / arch35）。

涉及两套产物：

| 产物 | 内容 | 命令入口 |
| --- | --- | --- |
| custom 算子包 `*.run` | kernel bin + aclnn/op_api/op_proto，安装到 CANN vendors | `build_mla.sh` |
| torch extension `*.whl` | torch 前段（JIT 编译 csrc），pip 安装 | `build.sh --torch_extension` |

---

## 1. 编译 custom 算子包（.run）

```bash
cd <repo_root>
bash build_mla.sh                # 全量编译
# 增量编译（复用 build/output）：
# bash build_mla.sh --incremental
```

> ⚠️ **验证/上板前务必用全量**：增量构建多次实测不可信（改源码后 kernel `.o`/json
> 不保证重建，曾出现"kernel not found"假象）。`--incremental` 仅在无源码改动时复用产物。

固定参数：`--soc=ascend950 --ops=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata -j20`
（`flash_mla_with_kvcache_metadata` 为纯 AICPU 伴生算子，负责生成 workspace metadata）。

产物：`build_out/cann-ops-transformer-custom_linux-x86_64.run`（另有 `build/` 下 kernel bin）。

### 1.1 安装 .run

默认安装到 `${ASCEND_HOME_PATH}/opp/vendors`（vendor 名 `custom_transformer`）：

```bash
./build_out/cann-ops-transformer-custom_linux-x86_64.run
```

或指定目录安装（推荐，隔离环境）：

```bash
./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=<dir>
```

指定目录安装后，使用前需 source 环境：

```bash
source <dir>/vendors/custom_transformer/bin/set_env.bash   # 追加 ASCEND_CUSTOM_OPP_PATH
```

校验 vendor 生效（默认安装到 `${ASCEND_HOME_PATH}/opp/vendors` 时）：

```bash
grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini   # 预期含 custom_transformer
```

> 使用 `--install-path` 指定目录安装时无 vendor config.ini，直接 `source <dir>/vendors/custom_transformer/bin/set_env.bash`
> 并以 `echo $ASCEND_CUSTOM_OPP_PATH` 确认即可。

---

## 2. 编译 torch extension（.whl）

### 2.1 整包编译（包名 `cann_ops_transformer`，推荐）

```bash
cd <repo_root>
bash build.sh --torch_extension
```

产物：`build_out/cann_ops_transformer-1.0.0-py3-none-any.whl`（也在 `torch_extension/dist/`）。

### 2.2 单算子编译（包名会变成 `cann_ops_transformer_custom`）

仅编译 flash_mla_with_kvcache 及其 metadata 的 torch 前段；**注意包名不同**：

```bash
bash build.sh --torch_extension --ops=flash_mla_with_kvcache --vendor_name=custom
```

产物：`build_out/cann_ops_transformer_custom-1.0.0-py3-none-any.whl`。

> 包名规则（`torch_extension/setup.py`）：不带 `--ops/--vendor_name` → `cann_ops_transformer`；
> 带 `--ops` → `cann_ops_transformer_<vendor>`。
> 若用单算子包，import 顶层是 `cann_ops_transformer_custom` 而非 `cann_ops_transformer`。

### 2.3 安装 whl

```bash
python3 -m pip install torch_extension/dist/cann_ops_transformer-1.0.0-py3-none-any.whl \
    --force-reinstall --no-deps
```

### 2.4 已知坑（务必注意）

1. **整包与 single-op 包不能同时安装**：两者共用 torch Library namespace
   `cann_ops_transformer`，同 op 名二次 `@impl` 报
   `RuntimeError: already a kernel registered ... PrivateUse1`。
   解决：只保留一个，`pip uninstall cann-ops-transformer-custom`（视安装的哪个包而定）。
2. **卸载 single-op 包会误删共享头文件**：`cann_ops_transformer/common/aclnn_common.h`、
   `hccl_common.h`、`csrc/extension.cpp`，导致 JIT 编译报 `fatal error: aclnn_common.h: No such file`。
   恢复：重新 `pip install --force-reinstall --no-deps` 装整包。
3. **JIT 编译锁残留**：`/root/.cache/torch_extensions/py312_cpu/<op>/lock` 0 字节残留会使
   后续 import 永久卡死（FileBaton 跨进程锁）。定位：`PYTHONFAULTHANDLER=1` + `kill -ABRT`；
   解决：删残留 lock 文件。
4. JIT 编译需要 CANN include（`aclnnop`、`custom_transformer/op_api/include` 等），
   source 过 `setenv.bash` 的环境下编译。

---

## 3. 单算子调用

主算子为两段式：先调 `flash_mla_with_kvcache_metadata` 生成 workspace metadata
（AICPU 算子），再调 `flash_mla_with_kvcache`（cub 主算子），metadata 作为入参传入。

```python
import math

import torch
import torch_npu            # 必须先 import，注册 PrivateUse1 后端
import cann_ops_transformer  # noqa: F401
from cann_ops_transformer.ops import flash_mla_with_kvcache, flash_mla_with_kvcache_metadata

num_heads_q = 8
num_heads_kv = 1            # MLA 约束：kv head 数必须为 1
head_dim_qk = 576           # nope 512 + rope 64
head_dim_v = 512
block_size = 16             # 分页块大小（>=16 且 %16==0）
batch = 2
seqlen_q = 1                # decode：每个 batch 一个 query token
cache_lens = [1024, 4096]

# 1) metadata（AICPU）
cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device="npu")
meta = flash_mla_with_kvcache_metadata(
    cache_seqlens, num_heads_q, num_heads_kv,
    seqused_q=None,
    max_seqlen_q=seqlen_q, max_seqlen_kv=max(cache_lens),
    head_dim_qk=head_dim_qk, head_dim_v=head_dim_v,
    mask_mode=0, layout_q="BSND",
)
# 注意：layout_q 非 TND 时不得传 cu_seqlens_q（aclnn checker 会拒绝）

# 2) PA_BBND 分页 KV：块池 [num_blocks, block_size, 1, 576] + block_table
nb0, nb1 = math.ceil(cache_lens[0] / block_size), math.ceil(cache_lens[1] / block_size)
k_cache = torch.randn(nb0 + nb1, block_size, num_heads_kv, head_dim_qk, dtype=torch.float16).npu()
bt = torch.zeros(batch, max(nb0, nb1), dtype=torch.int32, device="npu")
bt[0, :nb0] = torch.arange(nb0, dtype=torch.int32)
bt[1, :nb1] = torch.arange(nb0, nb0 + nb1, dtype=torch.int32)

# 3) 主算子（cube）：q 为 BSND [B, seq, N, 576]，layout 三处统一
q = torch.randn(batch, seqlen_q, num_heads_q, head_dim_qk, dtype=torch.float16).npu()
attn_out, _lse = flash_mla_with_kvcache(
    q, k_cache,
    block_table=bt, cache_seqlens=cache_seqlens,
    cu_seqlens_q=None, seqused_q=None,
    metadata=meta,
    head_dim_v=head_dim_v,
    softmax_scale=1.0 / math.sqrt(head_dim_qk),
    mask_mode=0,
    max_seqlen_q=seqlen_q, max_seqlen_kv=max(cache_lens),
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",        # 必须显式传，且 == layout_q
)
print("out shape:", attn_out.shape)
```

完整可执行脚本见同目录 `run_flash_mla_op.py`。

### 3.1 参数约束要点（aclnn checker 逐条校验）

- `num_heads_kv` 必须为 1（"MLA kv head num must be 1"）。
- `layout_q` 非 `TND` 时，不可传 `cu_seqlens_q`（"cuSeqlensQOptional should not be provided"）。
- `layout_out` 必须显式传且 == `layout_q`（out==q，不支持转置）。
- **layout_kv 只能是分页布局** `PA_NZ / PA_BBND / PA_BNBD`，连续 KV 不被支持（EZ0026）；分页 KV 必须配 `block_table`。`cu_seqlens_q` 仅 TND 场景需要（且 layout_q 非 TND 时禁止传）。
- 主算子的 `q`/`k_cache` 最后维为 576（nope 512 + rope 64 合并存储）。

### 3.2 运行态排查

- import 卡死 → 检查 2.4 坑 3（JIT 锁残留）。
- 算子内核报错/异常 → 设置 `ASCEND_LAUNCH_BLOCKING=1 python3 run_flash_mla_op.py` 拿准确堆栈；
  仍失败则属于 kernel/AICPU 包或安装版本不一致问题：确认 `.run` 已重装到当前编译版本
  （`build_out/*.run` 改了要重装），并 `source set_env.bash` 后再跑。
