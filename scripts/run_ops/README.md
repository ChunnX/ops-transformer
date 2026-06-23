# Manual Run Guide For Ops Accuracy

本文档说明如何在 `br_tang_dev` 分支手动运行 `recurrent_gated_delta_rule`（下文简称 RGDR）算子的 ST 精度批量测试。

当前本地 `scripts/st_accuracy` runner 只优先覆盖 `attention/lightning_indexer` 和 `attention/recurrent_gated_delta_rule`。本文只描述 RGDR 的手动执行流程，不能外推为全仓库 ST。

## 1. SoC 参数选择

`build.sh --soc` 和 `run_st_accuracy.py --soc` 必须使用仓库支持的 SoC 字符串。

| 硬件 | 参数 |
| --- | --- |
| Ascend 910B / B3 | `ascend910b` |
| Ascend 950PR / Ascend 950DT | `ascend950` |

如果你的机器是 950，写 `ascend950`，不要写 `950`。

```bash
export SOC=ascend950
```

如果你的机器是 910B/B3：

```bash
export SOC=ascend910b
```

## 2. 进入分支和检查环境

```bash
cd /root/tangxian/packages/ops-transformer
git switch br_tang_dev
git status --short --branch
```

如果要验证远端分支的纯净状态，`git status` 应只显示：

```text
## br_tang_dev...origin/br_tang_dev
```

如果存在本地未提交改动，本次运行会使用本地工作区内容，而不是远端纯净版本。

加载 CANN 环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
which bisheng
python3 --version
gcc --version
cmake --version
```

确认目标算子目录存在：

```bash
OP=recurrent_gated_delta_rule

find attention ffn gmm mc2 moe posembedding -maxdepth 2 -type d -name "${OP}"
find attention ffn gmm mc2 moe posembedding -path "*/${OP}/op_kernel" -type d
```

## 3. 静态发现和覆盖检查

静态发现只确认 runner 能发现 ST case，不代表设备侧精度通过。

```bash
python3 scripts/st_accuracy/run_st_accuracy.py list \
  --ops recurrent_gated_delta_rule \
  --format json
```

当前 RGDR 预期可以发现 8 个 ST case，且 `missing_ops`、`zero_case_ops`、`skipped_assets` 都应为空。

继续检查 adapter coverage：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py coverage \
  --ops recurrent_gated_delta_rule \
  --out build/st_accuracy/rgdr_static_coverage_${SOC}
```

预期输出类似：

```text
coverage operators=1 cases=8 executable=8 unsupported=0 out=build/st_accuracy/rgdr_static_coverage_ascend950
```

再跑一次 no-execute，确认筛选、序列化和报告链路可用：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --no-execute \
  --out build/st_accuracy/rgdr_no_execute_${SOC}
```

查看 no-execute summary：

```bash
cat build/st_accuracy/rgdr_no_execute_${SOC}/summary.json
```

`--no-execute` 的结果通常是 `skipped=8`。它不是设备侧精度通过，只说明没有调用 native runner。

## 4. package gate 说明

设备侧精度运行必须使用当前仓库构建并安装的 `custom_transformer` package。当前 `scripts/st_accuracy run` 在真实设备执行时会自动对 selected case 对应算子执行 full package build/install，避免使用旧 package 做精度判断。

如果要分步定位 package 构建或安装问题，可以先手动执行：

```bash
bash build.sh --pkg \
  --soc=${SOC} \
  --ops=recurrent_gated_delta_rule \
  --vendor_name=custom \
  -j16 -O3
```

手动安装 package：

```bash
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

配置运行时 op_api 库路径：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

如果你的 CANN 安装目录不是 `/usr/local/Ascend/cann-9.0.0`，把上面的路径替换成实际 CANN 版本目录。

## 5. 记录 package/load 证据

这一步用于确认后续设备运行加载的是当前安装产物，而不是旧 package 或系统默认实现。

```bash
sha256sum build_out/cann-ops-transformer-custom_linux-aarch64.run 2>/dev/null || true
sha256sum /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so 2>/dev/null || true
find /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer -path "*${SOC}*" -type f | rg "recurrent_gated_delta_rule|RecurrentGatedDeltaRule|kernel|config" || true
```

如果 CANN 安装目录不同，同样替换 `/usr/local/Ascend/cann-9.0.0`。

## 6. 运行设备侧 ST 精度批量测试

`--device` 是传给 `aclrtSetDevice()` 的 ACL runtime logical device id，不一定等同于 `npu-smi` 看到的物理编号。

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --vendor custom \
  --soc ${SOC} \
  --device 0 \
  --out build/st_accuracy/rgdr_device_${SOC}
```

查看结果：

```bash
cat build/st_accuracy/rgdr_device_${SOC}/summary.json
```

用 Python 做一次严格检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path
import os

soc = os.environ.get("SOC", "ascend950")
summary_path = Path(f"build/st_accuracy/rgdr_device_{soc}/summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print(json.dumps(summary, indent=2, sort_keys=True))

expected = 8
bad = {
    "passed": summary.get("passed"),
    "failed": summary.get("failed"),
    "error": summary.get("error"),
    "unsupported": summary.get("unsupported"),
}
if summary.get("passed") != expected:
    raise SystemExit(f"expected passed={expected}, got {bad}")
if summary.get("failed", 0) or summary.get("error", 0) or summary.get("unsupported", 0):
    raise SystemExit(f"device ST has bad status: {bad}")
PY
```

判定标准：

- `passed=8`
- `failed=0`
- `error=0`
- `unsupported=0`

只有满足以上设备侧 `run` 结果，才可以写成当前 runner-supported RGDR ST 精度通过。

## 7. 常用筛选运行

只跑指定 case：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --case-ids 0,1 \
  --vendor custom \
  --soc ${SOC} \
  --device 0 \
  --out build/st_accuracy/rgdr_case_0_1_${SOC}
```

只跑指定 API：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --apis RecurrentGatedDeltaRule \
  --vendor custom \
  --soc ${SOC} \
  --device 0 \
  --out build/st_accuracy/rgdr_api_${SOC}
```

只做静态链路，不上设备：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --apis RecurrentGatedDeltaRule \
  --no-execute \
  --out build/st_accuracy/rgdr_api_no_execute_${SOC}
```

## 8. 一条命令构建安装并运行

真实设备侧 `run` 默认会统一触发 package 构建、安装和设备运行，仍建议先完成第 3 节静态检查。`--prepare pkg` 仅保留为兼容参数，日常不需要显式指定。

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --vendor custom \
  --soc ${SOC} \
  --device 0 \
  --out build/st_accuracy/rgdr_device_${SOC}
```

为便于定位构建、安装和运行失败，日常排查仍推荐使用第 4 至第 6 节的分步命令。

## 9. 常见问题

### SoC 写成 950

现象：

```text
Currently 950 is not supported, please input a valid soc.
```

处理：

```bash
export SOC=ascend950
```

然后重新执行构建和运行命令。

### 找不到 custom op_api 库

现象可能包含：

```text
libcust_opapi.so
custom_transformer
```

处理：

1. 确认真实设备侧 `run` 已执行到 package gate，或手动执行 `bash build.sh --pkg ... --vendor_name=custom`。
2. 确认 package 已安装，或手动执行 `bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet`。
3. 确认 `LD_LIBRARY_PATH` 包含 `custom_transformer/op_api/lib`。

### coverage 通过但设备 run 失败

`coverage` 只说明 JSON case 能被 adapter 接收，不说明设备侧调用和 golden compare 通过。需要查看：

```bash
cat build/st_accuracy/rgdr_device_${SOC}/summary.json
sed -n '1,20p' build/st_accuracy/rgdr_device_${SOC}/cases.jsonl
```

### no-execute 显示 skipped

这是正常现象。`--no-execute` 不调用设备，只验证筛选、序列化和报告路径，因此不能写成精度通过。

### 本地有未提交改动

`git status --short --branch` 如果显示 `M` 或 `??`，本次运行会使用本地工作区内容。要验证远端纯净 `br_tang_dev`，请先使用干净 worktree 或 clone。

## 10. 最小命令清单

Ascend 950 环境下最短手动流程：

```bash
cd /root/tangxian/packages/ops-transformer
git switch br_tang_dev
source /usr/local/Ascend/cann/set_env.sh

export SOC=ascend950

python3 scripts/st_accuracy/run_st_accuracy.py list \
  --ops recurrent_gated_delta_rule \
  --format json

python3 scripts/st_accuracy/run_st_accuracy.py coverage \
  --ops recurrent_gated_delta_rule \
  --out build/st_accuracy/rgdr_static_coverage_${SOC}

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops recurrent_gated_delta_rule \
  --vendor custom \
  --soc ${SOC} \
  --device 0 \
  --out build/st_accuracy/rgdr_device_${SOC}

cat build/st_accuracy/rgdr_device_${SOC}/summary.json
```
