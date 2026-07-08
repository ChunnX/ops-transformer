# FfnWorkerBatching —— TTK 测试集成指南（A5 / arch35）

本文说明如何用 TTK（`/home/zl/ops-test-kit`）测试本算子，供后续穿刺复核与验收 agent 复用。
覆盖：新环境 source 命令、TTK 运行命令、CSV/插件位置、当前 TTK Kernel 模式的**阻塞点与根因**、
以及为后续分支（RECV / 多核 / int8 / bf16）追加用例的方法。

> 结论速览：本算子的**生产接口**（唯一输入 `schedule_context` 内嵌 4 个 GM 绝对地址）
> **无法用 `ttk kernel` 直接驱动**（根因见 §4）。A5 arch35 NORM 骨架的**真实 NPU 二进制一致复验**
> 已通过 `probe4_norm_bf16`（与主线同一套 arch35 头文件）在 Ascend950DT_9584 上完成（§3）。
> TTK 侧插件/用例资产已就绪并自测通过（§2），待 §5 的“解阻”步骤完成后即可端到端跑 `ttk kernel`。

---

## 1. 环境（已变更，务必使用）

```bash
# CANN 改用 /home/zl/Ascend（不要再用 /root/Ascend）
source /home/zl/Ascend/ascend-toolkit/set_env.sh
#   ASCEND_HOME_PATH = /home/zl/Ascend/cann-9.1.0
#   ASCEND_OPP_PATH  = /home/zl/Ascend/cann-9.1.0/opp

# TTK 命令须在 ops-test-kit 目录下运行
cd /home/zl/ops-test-kit
python3 -m ttk info      # 期望：Ascend950DT_9584（arch35/DAV_3510），64 aiv，UB 248KB
```

`python3 -m ttk info` 已实测认到设备：

```
Found 1 device(s):
   ID  Platform          Ver    ...
    0  Ascend950DT_9584  V120   ...
```

TTK 工具链在本 NPU 上可用（已用内置 `add.csv` 实跑验证：`STATUS: PASS`，`Mode: ASCEND_ONBOARD`）：

```bash
python3 -m ttk kernel -i examples/case_store/kernel/add.csv   # 冒烟：编译+NPU 执行通过
```

---

## 2. TTK 侧测试资产（已就绪 + 已自测）

| 资产 | 路径 | 说明 |
|------|------|------|
| golden 插件 | `tests/assets/golden.py` | numpy 二进制精确 golden，注册 `__golden__ = {"kernel": {"ffn_wb_norm_ttk": ...}}` |
| input 插件 | `tests/assets/input.py` | 确定性输入（与 C++ 参考模型 `ByteVal` 逐位一致），注册 `__input__` |
| Kernel CSV | `tests/st/testcases/ffn_wb_norm_ttk_kernel_cases.csv` | L0 NORM fp16 用例，`--compare binary`（rtol=atol=0） |

**精度口径**：二进制精确（`np.array_equal`，rtol=atol=0），取自 `spec.yaml numerical_tolerance.per_dtype`
（非计算类算子，全 dtype `bitwise_equal`），对应 `ops-precision-standard → 非计算类标准`。

**golden/input 已离线自测通过**（无需 NPU，纯 numpy 校验语义正确性）：

```bash
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/tests/assets
python3 -c "
import importlib.util as u
def L(n,p):
  s=u.spec_from_file_location(n,p); m=u.module_from_spec(s); s.loader.exec_module(m); return m
G=L('g','golden.py'); I=L('i','input.py'); print(G.__golden__, I.__input__)
"
```
自测覆盖：手工可核对小用例（`actual_token_num=6`、`group_list=[0,2,1,2,2,1,3,1]`、id 解码全对）、
determinism（重复计算 bitwise 一致）、int8(token_dtype=2) 的 `dynamic_scale` gather 路径——全 PASS。

**插件与用例对应的接口 = 包装算子 `ffn_wb_norm_ttk`**（4 输入显式暴露），
输入顺序与 `probe4` kernel 入口一致：

```
expert_ids[int32, outNum*BS*K] , token_data[原始字节, A*M*BS*K*HS] ,
session_tbl[int32, A] , microbatch_tbl[int32, A]
属性: expert_num, max_out_shape=[A,BS,K,H], token_dtype(0/1/2), need_schedule(=0), M, mask_stride
```

输出 8 路（与算子 def 输出顺序一致）：
`y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num`。

---

## 3. A5 arch35 NORM 骨架的真实 NPU 复验（已通过）

由于 §4 阻塞点，A5 骨架的 NPU 二进制一致复验通过 `probe4_norm_bf16` 完成——它**直接复用主线
`op_kernel/arch35/*.h`**（`ffn_wb_sort_one_core.h` / `ffn_wb_gather_out_all.h` / `ffn_wb_group_listing.h`
/ `ffn_wb_sort_base.h` / `ffn_wb_common.h`），仅把「解析 1024B schedule_context」替换为「host 传标量 + GM 指针」，
sort→gather→group_listing 三步逻辑与主线逐字一致。

```bash
source /home/zl/Ascend/ascend-toolkit/set_env.sh
export CMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64/cmake:$ASCEND_HOME_PATH/x86_64-linux/tikcpp/ascendc_kernel_cmake:$CMAKE_PREFIX_PATH"
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/probe/probe4_norm_bf16
mkdir -p build_newenv && cd build_newenv && cmake .. && make -j4   # --npu-arch=dav-3510
python3 ../scripts/gen_data.py
./probe_ffn_wb            # 真实 NPU（Ascend950DT_9584），blockDim=2
python3 ../scripts/verify_result.py
```

**实测结果（新环境 /home/zl/Ascend）**：SoC=`Ascend950DT_9584`，runtime aivNum=64，UB=253952B（248KB，运行时查询）；
8 路输出全部 `np.array_equal` PASS（`y`bf16/384 · `group_list`int64/16 · 4×id int32/24 · `dynamic_scale`fp32/24 ·
`actual_token_num`int64/1=18）。即：**A1-Main 的 A5 骨架在 /home/zl/Ascend 新环境下编译+NPU 实跑+二进制一致复验通过**。

---

## 4. ⛔ 阻塞点：为什么 `ttk kernel` 不能直接跑本算子

两条相互独立的根因，均已用证据确认：

### 阻塞 A —— schedule_context 内嵌 GM 绝对地址，TTK 无法注入
生产算子 def 仅 1 个物理输入 `schedule_context`（INT8，1024B）。kernel
（`op_kernel/arch35/ffn_wb_get_schedule_context.h`）从该 buffer 的固定偏移读取 **u64 设备指针**并解引用：

| 偏移 | 字段 |
|------|------|
| 400 | `token_data_buf`（token hidden states 缓冲 GM 地址） |
| 528 / 544 / 560 | `session_ids_buf` / `micro_batch_ids_buf` / `expert_ids_buf` GM 地址 |

TTK Kernel 模式的输入生成是**主机侧 numpy → 各自独立 `rtMalloc` + H2D**，各 buffer 地址在拷贝后才产生，
且**没有任何“把 A buffer 的设备地址写进 B buffer 字节”的注入/回填机制**（已核对 TTK 源码输入路径，
无 pointer patch / addr fixup / workspace-address injection）。因此 TTK 无法构造出合法的 schedule_context。
→ **本算子的生产接口天然不在 `ttk kernel` 可驱动范围内**。（这类“输入内嵌上下文指针”的算子应由能写设备地址的
bespoke C++ ST 工程测试——即 B 任务的 `tests/st/test_aclnn_ffn_worker_batching.cpp` Real 模式，它在 1024B ctx 里
`WriteU64(ctx,400,(uint64_t)dTokenData)` 等，正是为此设计。）

### 阻塞 B —— A5(ascend950) 未注册，TTK 无法发现/编译本算子
TTK **不从仓库 .cpp 源码编译**，而是**导入已安装 opp 包里的 Python impl**
（`ttk/core_modules/operator/op_info_keeper.py:121-146` 构造 `{prefix}impl.dynamic.<opFile>` 并 `__import__`；
op 发现读 `aic-<soc>-ops-info.json`，来源 built-in→vendor→custom，custom 经 `ASCEND_CUSTOM_OPP_PATH`）。
本环境现状：
- built-in `aic-ascend950-ops-info-transformer.json` **不含** FfnWorkerBatching；无 `config/ascend950/ops_transformer` 目录。
- 无 `vendors/` 目录；`ASCEND_CUSTOM_OPP_PATH` 未设置。
- 今天构建的自定义包 `build_out/cann-ops-transformer-custom_linux-x86_64.run` **仅含 ascend910b**（无 ascend950 资产）。

实测佐证（CSV 语法正确、用例可列出，但编译阶段被判 `SOC_NOT_SUPPORT`）：

```
python3 -m ttk kernel -i .../ffn_wb_norm_ttk_kernel_cases.csv --validate
  → Validate Complete: 2 total, 0 valid, 2 invalid
  → SOC_NOT_SUPPORT: 2   (Operator [ffn_wb_norm_ttk] is not supported on current SOC)
```

---

## 5. 解阻步骤：让 `ttk kernel` 端到端跑绿（后续接力项）

要让 `ttk kernel` 真正在 NPU 上编译+执行本算子，需**为 ascend950 注册一个 4 输入的包装算子
`ffn_wb_norm_ttk`** 并安装为 custom opp（绕开阻塞 A 的地址注入问题）：

1. **新增包装 op（测试侧，隔离于主线，不改 A2/A3）**：def 暴露 4 输入 + 8 输出 + 属性；
   kernel 直接复用 `probe4` 入口 `probe_ffn_wb_norm`（组装 `ScheduleContextInfo` + 主线三步链）；
   tiling 复用 `ffn_worker_batching_tiling_arch35.cpp` 的字段派生。
2. **构建 ascend950 custom opp 包**：`cd /home/zl/ops-transformer && bash build.sh --pkg --soc=ascend950 --vendor_name=customize --ops=ffn_wb_norm_ttk`
   （产出含 `config/ascend950/aic-ascend950-ops-info.json` + `impl/dynamic/*.py` + tiling + `kernel/ascend950/`）。
3. **安装 / 指向**：`bash cann-ops-transformer-custom_*.run --install` 或
   `export ASCEND_CUSTOM_OPP_PATH=<pkg>/vendors/customize`。
4. **跑 TTK**：
   ```bash
   cd /home/zl/ops-test-kit
   python3 -m ttk kernel \
     -i /home/zl/ops-transformer/ffn/ffn_worker_batching/tests/st/testcases/ffn_wb_norm_ttk_kernel_cases.csv \
     --plugin /home/zl/ops-transformer/ffn/ffn_worker_batching/tests/assets/ \
     --compare binary
   ```
   `--compare binary` = 二进制精确（rtol=atol=0，逐字节相等）。golden/input 插件已就绪，CSV 已就绪。

> 注意：步骤 1/2/3 属于“构建+安装新算子”，有一定 blast radius（写入 CANN 环境/仓库构建），
> 且需确认 TTK 的 import-compile 能处理 arch35 `_apt.cpp` + `REGISTER_OPS_TILING_TEMPLATE` 约定
> （见 LOG.md 2 迭代一：宏 `_1000` 在本仓 opc 下类型解析问题已手写平铺 struct 规避）。建议由主/开发 agent 接力，
> 完成后回到本 §5.4 即可获得 `ttk kernel` NPU 绿。

---

## 6. 为后续分支追加用例（RECV / 多核 / int8 / bf16）

在 `tests/st/testcases/ffn_wb_norm_ttk_kernel_cases.csv` 追加行即可（golden/input 插件已按属性泛化，无需改插件）：

| 分支 | 追加方式 |
|------|---------|
| **bf16（P1）** | `token_dtype:1`，`input_dtypes` 第 2 位 `bfloat16`，`output_dtypes` 第 1 位 `bfloat16`。golden 自动按 2B 字节 gather。 |
| **int8（P2）** | `token_dtype:2`，token_data 每 token 尾部 +32B（HS=H+32），`output_dtypes` 第 1 位 `int8`；golden 自动填充 `dynamic_scale`（原样字节 gather）。 |
| **多核 sort / gather** | 增大 `max_out_shape` 使 `Y = A*BS*K` 超过单核阈值（触发 `groupListingDealFlag=1` 多核分工）。 |
| **RECV（need_schedule=1）** | 需要 `isScanMode=true` 的 token_info 扫描路径与不同的包装 kernel 入口；本 NORM 包装不覆盖，另建 `ffn_wb_recv_ttk` 包装 + `isScanFlag=true` 分支（参考 `probe1_recv_fp16`）。 |
| **边界/极端** | 按 `docs/TEST.md §4` 逐条构造：`mask_stride=1`（k=0 全 mask）、`mask_stride=0`（k=N 全有效）、大 `expert_num`（same_bin）等。 |

CSV/插件语法参考：`ttk-how-write-case`、`ttk-how-write-plugin`；运行参考 `ttk-how-run-test`。

---

## 7. 可追溯证据

| 项 | 命令 / 位置 | 结果 |
|----|-----------|------|
| NPU 环境识别 | `python3 -m ttk info` | Ascend950DT_9584（arch35） ✅ |
| TTK 工具链冒烟 | `ttk kernel -i examples/case_store/kernel/add.csv` | STATUS: PASS, ASCEND_ONBOARD ✅ |
| A5 NORM 骨架 NPU 复验 | `probe4_norm_bf16/build_newenv` 下 `./probe_ffn_wb` + `verify_result.py` | 8/8 输出 `np.array_equal` PASS ✅ |
| golden/input 插件自测 | `tests/assets/` numpy 自测 | 手工用例 + determinism + int8 全 PASS ✅ |
| TTK CSV 解析 | `ttk list -i ffn_wb_norm_ttk_kernel_cases.csv` | 2 cases 正确解析 ✅ |
| TTK kernel 直跑本算子 | `ttk kernel --validate` | SOC_NOT_SUPPORT（阻塞 B，见 §4）⛔ 待 §5 解阻 |
