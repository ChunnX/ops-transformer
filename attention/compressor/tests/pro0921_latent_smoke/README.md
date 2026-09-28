# Pro0921 正式路径 latent 小冒烟

用于验证 `attention/compressor`、`attention/quant_compressor` 与
`attention/sparse_flash_mla`。本地用例使用当前任务工作树自己的
JIT 包、OPP 安装目录、依赖和用例目录。所有本地编译及仿真必须经 `local-resource-runner`。

## 接口与验收

| 算子 | 输入 / 输出 | 检查内容 |
| --- | --- | --- |
| Compressor | 11 / 4 | cmp_kv、原地 state_cache；grad_enabled 时另查 softmax_score 和 kv |
| QuantCompressor | 14 / 2 | HiFloat8 原始位、三个 descale、cmp_kv、原地 state_cache |
| Sparse MLA | 18 / 2 | TND 张量、真实 metadata、delay0/1 输出；不返回 LSE 时仅分配占位缓冲区 |

Compressor 的输出仍是 RMSNorm/RoPE 之前的值，后处理由模型执行。
非边界调用只更新 x_cache；KV/HCA 写入，LI 使用同层已更新的共享缓存。
`check_result.py` 必须看到 kernel 同步成功、设备输出读回回执和逐输出精度比较；
当前 TTK 的 CAMODEL 耗时解析返回 UNKNOWN，保留该值，不把模拟 ticks 冒充微秒。单纯进程退出 0、
或 TTK 在未执行时显示 PASS，均不能算通过。`check_xcache_readback.py`
独立检查从设备内存读回的缓存，禁止用输入文件冒充执行后结果。

## 固定环境

- CANN：`/home/jinxh/Ascend_0722/cann-9.2.0`。
- 本地平台：`Ascend950PR_9589`，32 Cube / 64 Vector，与模型配置一致；不更改板上 profile。
- 仿真库：`tools/simulator/Ascend950PR_9589/lib`。
- 任务 OPP：`<task>/sim-opp`，vendor 为 `pro0921latent_transformer`。
- `vendors/config.ini` 注册 vendor，取消 `ASCEND_CUSTOM_OPP_PATH`，避免 TTK 错选 `impl.dynamic`。
- `scene.info`、`built-in` 和相邻的 `x86_64-linux/data/platform_config` 指向同一套 CANN。
- 本任务的 `sim-python-deps` 提供 en-dtypes；不借用其它任务的安装树。

`compile_formal_ops.sh` 只构建四个正式算子的 JIT 包，并行度 4。
`install_formal_opp.sh` 每次安装指定的新包，不复用旧 staging 目录。
运行期间禁止覆盖该 OPP；安装与执行共用资源锁。

## 最小用例

| 用例 | B / S / R | 覆盖 |
| --- | --- | --- |
| hca_b1_boundary | 1 / 1 / 128 | 预置历史，仅执行一个边界 token |
| hca_b1_boundary_grad | 1 / 1 / 128 | 四输出接口 |
| hca_b1_update_only | 1 / 1 / 128 | 非边界缓存写入 |
| csa_b1_coff2_boundary | 1 / 1 / 4 | coff2 上一组状态与当前组 |
| csa_li_read_shared | 1 / 1 / 4 | LI 只读 x_cache |
| quant_b1_boundary | 1 / 1 / 128 | 量化码与 descale |
| sparse_mla_b1_q1_delay0/1 | 1 / 1 / 4 | 延迟前后压缩可见长度 |

hidden=1024，HCA/Quant 与 LI head=128，CSA KV head=512。
head512 HCA 边界在 180 秒内未完成，因此功能冒烟改用合法的最小 head128；
原 head512 超时保留为独立证据，不宣称其精度通过。不跑整模型、128 步填充或性能矩阵。

## 脚本

1. `build_tiny_cases.py` / `build_quant_cases.py`：生成小输入和独立 CPU 参考结果。
2. `golden_contract_check.py --cases <case-root>`：检查参考结果和用例区分能力。
3. `aicpu_metadata_host/`：链接 CANN 的宿主 context 库，直接调用原版
   `SparseFlashMlaMetadataCpuKernel::Compute`，为每个 delay 分别生成 metadata。
4. `build_smla_linked_cases.py --out <root> --metadata-dir <root>/meta`：接入真实 metadata。
5. `run_tiny_sim.sh <case-root> <精确用例名>`：一次选一个用例，执行预算 180 秒。
6. `PRO0921_READBACK_XCACHE=1`：启用当前 TTK 进程内的输入读回钩子。

脚本在用例目录生成 32/64 核的模型配置，关闭逐指令日志，保留统计。
所有输出 dump 和读回都写到用例目录。

模型库通过一次性 `LD_PRELOAD` 一起加载，以解析库间依赖；不修改系统库。
Sparse MLA 小用例可用 `PRO0921_TILING_KEYS=2082,34850,67618,100386`
限制本用例的 TND 模板，host tiling 仍负责选择实际 key。

`PRO0921_CORE_NUM=4` 通过 CANN 的 `set_platform_info_res` 为 host tiling
提供 4 Cube / 8 Vector 的合法资源子集；以日志中的 `BlockDim=4` 核对实际生效。
这不修改算子 tiling 数据或模拟器的物理核数。仅设置 `aicore_num` 不足以限制 Cube 核。
CSA KV head512 的 host tiling 要求至少 8 Cube，运行该例时使用
`PRO0921_CORE_NUM=8`；HCA/LI/Quant head128 用例使用 4。
Quant 输入保持 BSH `(1, 1, 1024)`，不能省略 S 维而进入 TH 接口。

首次 JIT 编译可能超过执行预算。`run_case_ttk.py` 支持
`PRO0921_COMPILE_ONLY=1` 单独编译，以及 `PRO0921_REUSE_COMPILE=1`
复用同一用例目录中本轮产生的二进制；源代码或编译参数变化后必须重新编译。
编译阶段可单独设置 `PRO0921_CASE_TIMEOUT_S`，执行阶段仍用小预算。
仅编译模式必须直接调用 Python 入口，不使用要求精度结果的 `run_tiny_sim.sh` 收口。

`boundary_logic_check.py` 默认包含宿主 C++ 编译，也必须放在资源锁中；
`--oracle-only` 与 `--extract-only` 可作轻量检查。

## 证据边界

宿主运行真实 AICPU 算法不代表板上 AICPU runtime 已验证。
CAMODEL 的功能精度结果也不能替代板上运行、性能、模型输出或业务完成。
最终执行记录见本任务的迁移验收文档，不以这些脚本的存在推断验证已通过。
