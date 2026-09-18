# EngramFetch 端到端验证用例

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT</term>                                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |  不支持  |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |  不支持  |

## 功能说明

本用例通过 PTA（`cann_ops_transformer` 的 `ElasticBuffer`）对 engram_fetch 算子做**全链路端到端验证**：
多进程组成 HCCL 通信组 → PTA 构建通信上下文（窗口注册/通道获取）→ aclnn 两段式接口 → kernel 执行 → 与 CPU golden 比对。

engram_fetch 支持两种取数模式（本用例两种均覆盖，可一键运行或分别运行）：

| 模式 | fetch_mode | 全局表布局 | 取数路径 | 支持场景 |
| :--- | :--------: | :--------- | :------- | :------- |
| 切片分存（默认） | `0` (FETCH_MODE_SLICED) | 全局表按 rank 均匀切片，每卡仅持有本 rank 分片 | 跨 NPU 通信（RDMA/a2a），由持有该 entry 的远端 rank 取回 | 推理 + 训练 |
| 全量本 NPU | `1` (FETCH_MODE_LOCAL) | 全局全量表冗余到每个 NPU | 纯本地 gather，无任何跨 NPU 通信 | **仅推理**（`with_grad=True` 不支持，接口会直接报错） |

### 验证项

1. **推理**（`with_grad=False`，两种模式均验证）：`engram_write` 写入本 rank 表 → `engram_fetch` 随机全局 indices（含重复 index）→ 结果与 CPU 全表 embedding gather golden **逐行完全一致**；
2. **训练**（`with_grad=True`，仅 sliced 模式）：
   - 前向 `engram_fetch` 结果与 golden 一致；
   - 反向 `engram_fetch_grad` 输出的 `(grad_unique, unique_local_entry)` 与 CPU 去重 scatter-add 聚合 golden 一致：
     各 rank 得到其所持分片 entry 的**跨 rank 全局聚合**梯度（local entry 为分片内偏移）；
3. **零拷贝**：`with_grad` 模式下 storage 为零拷贝映射，原地更新表数据后再次 fetch 立即可见新数据。

> 说明：local 模式仅支持推理，`run_example.sh --fetch-mode local` 默认只跑推理阶段；显式传 `--fetch-mode local --phase train` 会直接报错退出。

## 环境准备

- Atlas 950DT 机器，CANN Toolkit（需支持 ascend950，且 engram_fetch 内核要求 ASC_DEVKIT > 9.0）；
- 驱动/固件正常（`npu-smi info` 可见设备）；
- Python 环境：`torch`、与 torch 版本匹配的 `torch_npu`、`pip`。

### 1. 编译并安装自定义算子包

```bash
# 仓库根目录
bash build.sh --pkg --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait

# 安装（安装到 $ASCEND_HOME_PATH/opp/vendors）
cd build_out
./cann-ops-transformer-custom_linux-aarch64.run
```

> 若提示 `ASCEND_CUSTOM_OPP_PATH` 含冒号等报错，请先 `unset ASCEND_CUSTOM_OPP_PATH` 再安装。

### 2. 编译并安装 PTA（torch_extension）whl

```bash
# 仓库根目录（整包编译，含 ElasticBuffer）
bash build.sh --torch_extension
python3 -m pip install build_out/cann_ops_transformer-*.whl --force-reinstall --no-deps
```

## 运行方式

### 卡数与设备指定

| 层 | 方法 | 示例 |
| ---- | ---- | ---- |
| HCCL 组大小（几张卡一组） | `--nproc N`（默认 `2`） | `bash run_example.sh --nproc 8 ...` |
| 选用哪些物理卡 | 环境变量 `ASCEND_RT_VISIBLE_DEVICES` | `export ASCEND_RT_VISIBLE_DEVICES=0,1`（选空闲卡，多进程抢卡时常用） |

> 约束：sliced 模式要求 `--num-entries` 能被 `--nproc` 整除；local 模式仅支持推理（`--phase train` 会报错，`both` 自动跳过训练）。

### 小规格功能验证

默认参数（4096 行 × 256 维 × 512 tokens）本身就是小规格，直接跑即可：

```bash
cd examples/mc2/engram_fetch_e2e

# 1. 标准全量功能验证: 2 卡, sliced + local 两模式, 推理+训练+零拷贝
bash run_example.sh

# 2. 叠加 sf(scale factor) 链路验证(推理阶段生效)
bash run_example.sh --with-sf

# 3. dtype 矩阵轮跑(5 种类型 × 2 模式)
bash run_example.sh --dtypes all

# 4. 仅验证 local 模式 / 仅验证训练
bash run_example.sh --fetch-mode local
bash run_example.sh --fetch-mode sliced --phase train

# 5. 更小规格快速冒烟(512 行 × 256 维 × 128 tokens, 直接 torchrun 调起)
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_engram_fetch_e2e.py \
    --fetch-mode sliced --phase infer --with-sf --num-entries 512 --hidden 256 --num-tokens 128
```

### 其他运行示例

```bash
# 4 卡 + 指定数据类型/表规模等（其余参数透传给测试脚本）
bash run_example.sh --fetch-mode sliced --nproc 4 --dtype fp16 --num-entries 8192 --num-tokens 1024

# 也可直接用 torchrun 调起
torchrun --standalone --nproc_per_node=2 test_engram_fetch_e2e.py --fetch-mode local --phase train
```

参数说明（`python3 test_engram_fetch_e2e.py --help` 查看全部）：

| 参数 | 默认值 | 说明 |
| :--- | :----- | :--- |
| `--fetch-mode` | `sliced` | `sliced`(0)=切片跨 NPU 通信；`local`(1)=全量本 NPU |
| `--phase` | `both` | `infer`=仅推理；`train`=仅训练；`both`=全部；`perf`=取数性能测试（见下文） |
| `--dtype` | `bf16` | 表数据类型：`bf16`/`fp16`/`fp32`/`fp8e4m3`/`fp8e5m2` |
| `--num-entries` | `4096` | 全局表总行数（需能被 rank 数整除） |
| `--hidden` | `256` | 隐藏维大小 |
| `--num-tokens` | `512` | 每 rank 取数 token 数（训练模式下作为 `num_max_tokens_per_rank`） |
| `--seed` | `2026` | 随机种子（全量表各 rank 一致，indices/grad 按 rank 区分） |
| `--with-sf` | 关 | 推理/perf 阶段启用 scale factor 链路：`engram_write(storage, sf)` 并校验 `fetched_sf`（fp8 表配 E8M0 sf，其他配 fp32 sf）；训练 kernel 不支持 sf |
| `--iters` | `20` | perf 阶段计时迭代次数 |
| `--warmup` | `5` | perf 阶段预热迭代次数 |

`run_example.sh` 额外选项：

| 选项 | 默认 | 说明 |
| ---- | ---- | ---- |
| `--dtypes LIST\|all` | 关 | 轮跑多种表数据类型（逗号分隔，或 `all`=bf16,fp16,fp32,fp8e4m3,fp8e5m2），与 `--fetch-mode` 组成矩阵；使用后不得再透传 `--dtype` |

## 性能测试（perf 阶段）

```bash
# 客户规格: 51.38万词表 × 576 维, 8192 tokens/fetch, bf16, 2卡
bash run_example.sh --fetch-mode sliced --phase perf --num-entries 513800 --hidden 576 \
    --num-tokens 8192 --iters 20
# local 模式(全量表冗余到每卡)
bash run_example.sh --fetch-mode local --phase perf --num-entries 513800 --hidden 576 \
    --num-tokens 8192 --iters 20
```

客户规格（513800×576）存储占用参考：

| dtype | 全局表 | local 每卡窗口 | sliced 每卡窗口（W 卡） |
| ---- | ---- | ---- | ---- |
| bf16/fp16 | 564.5 MiB | 566.0 MiB | 564.5/W MiB（8卡 72.0 MiB） |
| fp32 | 1129.0 MiB | 1130.0 MiB | 1129/W MiB（8卡 142.0 MiB） |
| fp8e4m3（含 sf 预留） | 282.2 MiB（+E8M0 sf 8.8 MiB NPU 张量） | 330.0 MiB | 330/W MiB（8卡 42.0 MiB） |

> 注：513800 = 2³×5²×7×367，sliced 模式 rank 数需整除（2/4/5/7/8/10 可行，16 不可）；16 卡 local 模式不受限。

## 判定标准

每个 rank 输出各验证项的 `PASS/FAIL`，全部通过时输出：

```
[rank 0] [PASS] [infer/local] fetch 结果与全表 gather golden 一致
[rank 0] [PASS] [train/sliced] 前向 fetched 与全表 gather golden 一致
[rank 0] [PASS] [train/sliced] unique_local_entry 与 golden 去重索引一致
[rank 0] [PASS] [train/sliced] grad_unique 与 golden 梯度聚合一致
[rank 0] [PASS] [train/sliced] 零拷贝: storage 原地更新后 fetch 可见新数据
[rank 0] engram_fetch(local, bf16) 端到端验证全部通过
...
[SUMMARY] 全部模式验证通过: sliced local
```

脚本退出码：`0` 表示全部模式验证通过，非 `0` 表示存在失败项。

## 常见问题

- **非 950 机器（如 910b/A2）**：engram_fetch 仅注册 ascend950，在其他 SoC 上执行会报
  `SoC version xxx verification failed / This SoC is not configured through the AddConfig API` 或算子二进制加载失败，属预期限制；
- **算子未安装**：报 `aclnnEngramFetch` 符号/算子不存在时，请确认自定义算子包已安装且当前 shell 已 source CANN 环境（`set_env.sh`）；
- **HCCL 组初始化失败**（`HcclCommInit failed`）：多进程竞争同一设备时，可通过
  `export ASCEND_RT_VISIBLE_DEVICES=<空闲设备ID列表>` 指定空闲 NPU；
- **whl 未包含 ElasticBuffer**：请使用 `bash build.sh --torch_extension` 整包编译（勿用 `--ops` 过滤，过滤会丢失 mc2/common 下的 ElasticBuffer 实现）。
