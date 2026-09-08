# FlashKDA: 串行 Q/K 融合

基于 CANNBotDSL 的 Kimi Delta Attention prefill 算子，面向 Ascend 950。当前版本采用串行
Stage1 调度，并在该阶段融合 Q/K 归一化和跨半块交换。

## 算子介绍

算子按 64-token chunk 计算 Kimi Delta Attention，并在 chunk 间递推状态。实现位于
[`op_kernel/flash_kda.py`](op_kernel/flash_kda.py)。

令 $\gamma_t=\exp(\sum_{i\le t}g_i)$，并定义：

$$
\tilde K=\gamma\odot K,\qquad
\bar K=\gamma^{-1}\odot K,\qquad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q).
$$

序列按 $C=64$ 切分。每个 chunk 内先计算严格下三角矩阵及其逆：

$$
T=\operatorname{stril}\!\left(\operatorname{diag}(\beta)\tilde K\bar K^{\top}\right),
\qquad A^{-1}=(I+T)^{-1},
$$

$$
U=A^{-1}\operatorname{diag}(\beta)V,
\qquad W=A^{-1}\operatorname{diag}(\beta)\tilde K,
\qquad M_{qk}=\operatorname{tril}\!\left(\tilde Q\bar K^{\top}\right).
$$

以 $S_0=$ `initial_state`，第 $c$ 个 chunk 的状态和输出为：

$$
S_c=\operatorname{diag}(\gamma_C)S_{c-1}+(K_c^r)^{\top}(U-WS_{c-1}),
$$

$$
O_c=\tilde Q_cS_{c-1}+M_{qk,c}(U-WS_{c-1}).
$$

| 特性 | 说明 |
|:---|:---|
| 场景 | Prefill |
| Layout | BNSD、BSND、TND；BNSD/BSND 可选 `cu_seqlens` |
| 数据类型 | Q/K/V/g/beta/out: BF16；state/A_log/dt_bias: FP32 |
| GQA | 支持，`Nv % Nk == 0` |
| Head dimension | `Dk = Dv = 128` |
| 序列长度 | 内部按 64 对齐；有效区以 metadata 为准，padding 输出未定义 |

## 快速开始

```python
import math
import torch
import torch_npu

import cann_ops_transformer_custom as ops

B, Nk, Nv, S, D = 1, 3, 3, 8192, 128
q = torch.randn(B, Nk, S, D).bfloat16().npu()
k = torch.randn(B, Nk, S, D).bfloat16().npu()
v = torch.randn(B, Nv, S, D).bfloat16().npu()
g = torch.randn(B, Nv, S, D).bfloat16().npu()
beta = torch.randn(B, Nv, S).bfloat16().npu()
initial_state = torch.zeros(B, Nv, D, D, dtype=torch.float32).npu()
A_log = torch.zeros(Nv, dtype=torch.float32).npu()
dt_bias = torch.zeros(Nv, D, dtype=torch.float32).npu()

metadata = ops.flash_kda_metadata(q, v, initial_state, "BNSD", None)
out, final_state = ops.flash_kda(
    q, k, v, g, beta, 1 / math.sqrt(D), initial_state,
    A_log, dt_bias, -5.0, "BNSD", metadata
)
```

## 两算子调用

FlashKDA 调度和计算是两个独立算子：

- `flash_kda_metadata()` 编译并启动 AICPU，读取 shape/layout、`cu_seqlens` 和设备核数，使用内置实测拟合公式返回 device metadata。
- `flash_kda()` 只启动 AINPU，把 metadata 当作普通输入；它不编译、不缓存、也不启动 AICPU。

整网应在 forward 开始时生成一次 metadata，后续各层复用：

```python
import cann_ops_transformer_custom as ops

metadata = ops.flash_kda_metadata(q0, v0, initial_state, layout_qkv, cu_seqlens)

for layer in layers:
    output, final_state = ops.flash_kda(
        layer.q,
        layer.k,
        layer.v,
        layer.g,
        layer.beta,
        layer.scale,
        layer.initial_state,
        layer.A_log,
        layer.dt_bias,
        layer.lower_bound,
        layout_qkv,
        metadata,
    )
```

只有调度身份相同的层才能复用 metadata。B、物理 batch 数 P、S/T、Nv、layout/packed 模式、`cu_seqlens` 内容、设备 AIC 核数、workspace slots 或内置 cost 公式版本变化时，必须重新调用 `flash_kda_metadata()`。Nqk 和 Q/K/V/g/beta/state 的数值不影响 metadata。

没有 `cu_seqlens` 的 BNSD/BSND 直接按 dense 序列调度，不创建或读取 CU。TND 必须传从 0 开始的标准 int32 `[B + 1]` 前缀和。连续 `[T, N, D]` 在 host 侧零拷贝归一成逻辑 `[1, N, T, D]` view。

## 参数

| 参数 | Shape (BNSD) | 类型 | 说明 |
|:---|:---|:---|:---|
| `q`, `k` | `[B, Nk, S, D]` | BF16 | Query 和 Key。算子内部按 FP32 epsilon 完成行级 L2 normalize。 |
| `v` | `[B, Nv, S, D]` | BF16 | Value。 |
| `g` | `[B, Nv, S, D]` | BF16 | 原始门控输入；内核计算 `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。 |
| `beta` | `[B, Nv, S]` | BF16 | 原始写入 logits；内核计算 `sigmoid(beta)`。 |
| `scale` | scalar | FP32 | 常用值为 `D ** -0.5`。 |
| `initial_state` | `[B, Nv, D, D]` | FP32 | 初始状态，布局为 `[Dv, Dk]`。 |
| `A_log` | `[Nv]` | FP32 | 每个 value head 的 log 时间尺度。 |
| `dt_bias` | `[Nv, D]` | FP32 | 每个 value head、每个 D 通道的 gate bias。 |
| `lower_bound` | scalar | FP32 | gate 下界，取值范围 `[-5, 0]`。 |
| `layout_qkv` | scalar | string | `"BNSD"`、`"BSND"` 或 `"TND"`。 |
| `metadata` | `[M]` | INT32 | `flash_kda_metadata()` 返回的调度 tensor。 |

返回 `(out, final_state)`。`out` 的 layout 跟随 `layout_qkv`，`final_state` 的 shape 为 `[B, Nv, D, D]`。
图捕获兼容性由调用该算子的网络脚本保证；FlashKDA 本身不查询或拒绝 capture 状态。

## Transformer 二进制适配

本仓只在原 FlashKDA/FlashKDA Metadata 的 Python 入口外增加发布和加载适配，不新增
ACLNN 入口，也不复制第二份算子实现。运行时用户界面保持为 Torch 扩展接口：

```python
import cann_ops_transformer_custom as ops

metadata = ops.flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens)
out, final_state = ops.flash_kda(
    q, k, v, g, beta, scale, initial_state,
    A_log, dt_bias, lower_bound, layout_qkv, metadata,
)
```

custom run 包使用仓库已有构建入口：

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_kda,flash_kda_metadata --vendor_name=custom
bash build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path="$PWD/install-custom"
source "$PWD/install-custom/vendors/custom_transformer/op_impl/cannbotdsl/set_env.bash"
```

builtin run 包复用已有 CMake 选择算子的方式，不新增 `--builtin` 之类的 build.sh 选项：

```bash
cmake -S . -B build/builtin -DENABLE_BUILT_IN=ON -DPACKAGE_TYPE=run \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  '-DASCEND_OP_NAME=flash_kda;flash_kda_metadata'
cmake --build build/builtin --target package -j16
bash build_out/cann-950-ops-transformer_9.1.0_linux-x86_64.run \
  --full --quiet --install-path="$PWD/install-builtin"
source "$PWD/install-builtin/cann/opp/built-in/op_impl/cannbotdsl/set_env.bash"
```

custom 导入包名为 `cann_ops_transformer_custom`，builtin 导入包名为
`cann_ops_transformer_builtin`。两种 run 包都直接安装 Python 入口、Native Provider 和
FlashKDA Metadata 的 AICPU `.so`，安装后不依赖算子 wheel，也不允许运行时回退到即时编译。
需要强制二进制命中时，可设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require`。

## 相比原 FlashKDA 新增的适配

相对 `/home/z00947297/run_toolkit/cannbot-arena/samples/flash_kda/flash_kda.py` 的原始
FlashKDA，本仓保留原计算公式、分组调度、双阶段执行结构和公开参数语义，只做以下适配：

- 增加 `@register("flash_kda")` 发布入口，使框架打包阶段可以收集原动态编译入口并生成
  Native Provider。
- Transformer 的 `torch_extension/setup.py` 统一识别发布声明并调用 CANNBotDSL
  `package.native.build`；算子文件不负责选择输出目录或执行打包编排。
- 增加 Torch extension 包装，注册 `flash_kda`、`flash_kda_metadata` 的 schema、NPU 调用和
  Meta 推导，让用户通过 `cann_ops_transformer_custom` 或 `cann_ops_transformer_builtin`
  调用。
- 将原代码中已从最新 CANNBotDSL 移除的视图/向量接口迁移到等价新接口，包括行前缀和、
  FP32 到 BF16 state cast 以及 workspace view 构造；这些迁移只替换 DSL API 形态，不改变
  FlashKDA 的数学计算和任务调度。
- 在 L1 清零后、写入单位矩阵对角线前补充一次 `cube_sync_pipe(PIPE.MTE2)`。该同步用于保证
  后续对角线写入不会与前序 MTE2 清零发生乱序；计算公式不变。
- 框架构建统一导出 Metadata AICPU 二进制，算子侧保留加载适配。metadata 的容量估算、序列解析、
  cost 公式、分组和任务划分仍保留在原 Python/AICPU 实现中。

## 接口切换后的精度异常记录

接口切换后做过 custom 和 builtin 两类 run 包验证，测试均强制使用已安装二进制，并拦截
AIC/AIV 和 AICPU 编译入口，确认没有回退到运行时编译。builtin 包在当前验证样例中 8 个用例
全部通过；custom 包曾出现 3 个非整块/GQA 组合精度失败：

- `BNSD`，长度 `(65, 65)`，GQA 比例 `2:1`；
- `TND`，长度 `(1, 63, 65, 129)`，GQA 比例 `2:1`；
- `TND`，长度 `(257, 64, 7)`，GQA 比例 `2:1`。

失败表现为尾部 token 出现非有限值或 P99.9 相对误差超过 `5e-3`。这些失败并非 README 更新或
PR 清理引入：清理前后 FlashKDA/Metadata 19 个最终文件的 blob 与原验证版本保持一致，custom
路径在不同轮次也出现过相近的非稳定失败，而 builtin 同样二进制路径可通过当前覆盖用例。

当前分析结论是：接口迁移本身没有改写公式，但最新 CANNBotDSL 接口移除了旧 `rebase_view` 等
视图 API，迁移后的等价 view 构造会暴露原算子在 padding、非 64 对齐长度和跨 batch/TND 边界
上的尾块敏感性。已确认并修复的一处必要同步是 L1 清零与单位矩阵对角线写入之间的 MTE2
等待；剩余 custom 路径异常需要继续围绕尾块 padding 写回、有效长度裁剪和 custom/builtin
安装资源差异定位。定位过程中仍应保持“只适配接口和发布层，不改算法”的原则；如果需要引入
来自 DSL 最新样例的 padding 行为变化，应作为单独算法修正评审。

## 精度测试

```bash
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 \
CANNBOTDSL_NPU_ARCH=dav-3510 \
python -m pytest attention/flash_kda/test/flash_kda_test.py -v -s
```

测试默认要求 P99.9 相对误差不超过 `5e-3`。builtin 包测试时增加
`KDA_PACKAGE=cann_ops_transformer_builtin`。

## 性能

以下结果由 Ascend 9.2 `msprof` 采集，代码基于 `qihang` 分支，并融合了 Stage2 的 state decay 与 delta accumulation，让 U-delta VF 直接消费 MTE2 输入，同时让 Stage1 gate cumsum 直接写入奇偶 snapshot，省去两次 UB identity copy；metadata 还会在 workspace 约束内选择精确的最大 group，减少 Stage1/Stage2 全核同步轮数（典型 case 从 8 轮降到 5 轮）。此外包含 Stage1 下三角裁剪、resident zero-fill 清理、NORMQK 后复用 UB 槽提前发射下一任务的 Q/K 整块 DMA，以及 Stage2 K-restored/Mqk 载入的 L2 last-use 提示和 O 写回的 L2 disable 提示。
典型 case 固定为 `B=1, N=32, S=8192, D=128`，Q/K/V/g/beta 使用 BF16，state 使用 FP32：

- BNSD/BSND：Q/K 为 `[B, N, S, D]`，V/g 为 `[B, N, S, D]`；
- TND：Q/K 为 `[T, N, D]`，V/g 为 `[T, N, D]`，`T=8192`，`cu_seqlens=[0, 8192]`。

每个 layout 单独启动进程，metadata 在采样循环前生成并复用；每个进程先执行 2 次 warmup，再采集 10 次有效
AINPU `flash_kda_kernel`。三个 layout 均采集 3 个独立 session，并旋转执行顺序，每个 layout 共 30 个有效样本。表中 `Task Duration min` 是三个 session 最小值的中位数，`Task Duration median` 和三个 ratio 是对应 session 中位数的中位数；AICPU metadata 不计入结果。

| B | N | S/T | D | Layout | Task Duration min (us) | Task Duration median (us) | aiv_mte2_ratio | aiv_vec_ratio | aic_mac_ratio |
|--:|--:|--:|--:|:---|--:|--:|--:|--:|--:|
| 1 | 32 | 8192 | 128 | BNSD | 814.482 | 822.810 | 0.400 | 0.625 | 0.395 |
| 1 | 32 | 8192 | 128 | BSND | 835.793 | 843.975 | 0.452 | 0.613 | 0.385 |
| 1 | 32 | 8192 | 128 | TND | 838.375 | 844.205 | 0.456 | 0.613 | 0.385 |

L2 策略采用 Ascend 9.2 CAPI raw 编码：`1=LAST-USE`、`4=DISABLE`。在 BSND/TND 上进行 3 个旋转顺序的配对 session 后，最终策略相对全部 `0=NORMAL` 的 session-median 分别改善 `0.92%` 和 `0.90%`。TND 的两轮反序 L2Cache 采样中，AIC read hit 提高 `8.10%`，read miss 降低 `17.94%`，read victim 降低 `16.55%`，write victim 降低 `64.84%`。identity 和 W 的 persistent、Q/W 的 last-use/persistent/disable 以及 K/Mqk 的 persistent/disable 均因无收益或退化而未保留。
