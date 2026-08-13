# MegaMoe W4A8 精度与打点测试

该目录从 `moe_test/mega_moe` 迁移了 W4A8 的输入生成、Cascade 参考实现和精度比较链路，并适配本项目的 `timer` 参数。打点数据写入独立的 `int64` 缓冲区，不会覆盖 `y`，因此精度比较和计时可以在同一次执行中完成。

## 1. 环境与构建

在 Ascend 950/A5 环境中加载 CANN，然后从仓库根目录构建并安装 MegaMoe 算子包与 Python 扩展：

```bash
cd /path/to/ops-transformer_haisi
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bash build.sh --pkg --soc=ascend950 --ops=mega_moe --vendor_name=customize -j16
chmod +x build_out/cann-ops-transformer-customize_linux-*.run
build_out/cann-ops-transformer-customize_linux-*.run

bash build.sh --torch_extension_only
python3 -m pip install --force-reinstall build_out/cann_ops_transformer-*.whl
python3 -m pip install -r tests/megamoe_w4a8/requirements.txt
```

还需要与当前 CANN/PyTorch 版本匹配的 `torch_npu` 和 `torchair`。`cann_ops_nn` 为可选依赖：存在 `swiglu_group_quant` 时优先使用；未安装或未注册该算子时，测试会自动使用 SwiGLU 加 `npu_dynamic_mx_quant` 的兼容路径。若算子包使用了自定义安装路径，运行前先执行对应的 `vendors/customize/bin/set_env.bash`。

## 2. 运行

默认用 2 个本地 NPU rank、每 rank 48 个专家，参数为 `bs=1, hidden=4096, intermediate=1024, topk=6`：

```bash
cd /path/to/ops-transformer_haisi
bash tests/megamoe_w4a8/run.sh
```

常用参数通过环境变量覆盖，例如 8 卡：

```bash
MEGAMOE_NPROC_PER_NODE=8 \
MEGAMOE_LOCAL_EXPERTS=48 \
MEGAMOE_BS=1 \
MEGAMOE_HIDDEN=4096 \
MEGAMOE_INTERMEDIATE=1024 \
MEGAMOE_TOPK=6 \
bash tests/megamoe_w4a8/run.sh
```

测试失败时脚本返回非零状态。默认开启打点；只跑精度时可设置：

```bash
MEGAMOE_ENABLE_TIMER=0 bash tests/megamoe_w4a8/run.sh
```

## 3. 解析打点

原始数据默认写入：

```text
tests/megamoe_w4a8/timer/w4a8_ep<EP>_bs<BS>_h<H>_n<N>_topk<K>_local<E>/rank_<rank>_npu_time.csv
```

默认会使用上述打点根目录，并自动选择最近一次产生数据的 case，因此目录参数不是必需的。以默认 2 卡为例：

```bash
python3 tests/megamoe_timer/parse_timer_csv.py \
  --rank-start 0 --rank-end 1

python3 tests/megamoe_timer/trace_parser.py \
  --rank-start 0 --rank-end 1
```

只有需要解析指定 case 时，才传入 `--dir <case目录>` 和 `--timer-dir <case目录>`。若运行测试时设置了 `MEGAMOE_TIMER_DIR` 或 `MEGAMOE_CASE_NAME`，解析脚本也会读取相同的环境变量。

生成的 `all_rank_trace.json` 可用 Perfetto 或 `chrome://tracing` 打开。计时仅接入 Arch35 W4A8 路径，`mega_moe_wave.h` 路径未修改。

## 4. 可配置变量

| 变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `MEGAMOE_NPROC_PER_NODE` | 2 | 本机启动的 NPU rank 数 |
| `MEGAMOE_LOCAL_EXPERTS` | 48 | 每个 EP rank 的本地专家数 |
| `MEGAMOE_BS` | 1 | 每 rank token 数 |
| `MEGAMOE_HIDDEN` | 4096 | hidden size |
| `MEGAMOE_INTERMEDIATE` | 1024 | FFN intermediate size |
| `MEGAMOE_TOPK` | 6 | 每 token 路由专家数 |
| `MEGAMOE_ENABLE_TIMER` | 1 | 是否传入 timer 并导出 CSV |
| `MEGAMOE_TIMER_DIR` | `tests/megamoe_w4a8/timer` | 打点输出根目录 |
| `MEGAMOE_CASE_NAME` | 自动生成 | 输出 case 子目录名 |

首次执行会在测试目录生成 CPU golden cache；后续相同配置会直接复用。
