# CCU AllGather + Matmul 融合算子（第二阶段）

2 卡 AllGather + Blaze Matmul 融合（FP16 A/B → BF16 C，M=256, K=256, N=256）。

## 验证记录

- 2025-08-17：第二阶段验证 PASS（2 卡，outputBuf 前 65536 元素=256.0，后 65536=512.0，BF16 精度正确）

## 编译

```bash
cd ~/code/ops-tensor && git submodule update --init --recursive include/tensor_api
source $ASCEND_HOME_PATH/set_env.sh
bash run.sh
```

## 运行环境要求

- `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 必须设置（`run.sh` 已包含，手动运行编译产物时需自行 export）
- `ASCEND_HOME_PATH` 指向 CANN 安装目录

## 已知问题（待解决）

- **CMakeLists.txt 中 ops-tensor 路径硬编码为 `$ENV{HOME}/code/ops-tensor`**：换环境（如 root 用户、非标准路径）时需手动修改 `OPS_TENSOR_ROOT`。后续考虑改为环境变量覆盖或自动查找。
- **tensor_api submodule 权限问题**：部分 WSL 环境下 `.git` 目录不可写，`git submodule update --init` 失败，需 `sudo` 或修复 `.git` 目录权限。
- **链接依赖 `dl` 库**：`dlopen` 等 symbol 需显式链接 `libdl`（已在 CMakeLists.txt 中添加 `dl`）。

## 验证

```python
import numpy as np
raw = np.fromfile("output/0/npu_out.bin", dtype=np.uint16)
c = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
assert np.all(c[:65536] == 256.0)   # rank0 行
assert np.all(c[65536:] == 512.0)   # rank1 行
```
