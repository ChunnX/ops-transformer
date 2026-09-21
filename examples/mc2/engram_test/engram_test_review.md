# engram_test 用例检视报告

- 检视对象：`examples/mc2/engram_test/`（commit `abb15a00c` 新增框架 + `403ea1680` 补全 engram_test.py）
- 对照基准：原版目录 `/home/qianlong/workspace/cann/engram_test`（gitcode wuchenhao123/engram_test）
- 被测对象：本分支（engram-fetch-local-mode）`mc2/engram_fetch` 算子及其 PTA（`mc2/common/torch_extension/elastic_buffer.py`）
- 检视轮次：第一轮（engram_test.py 为 0 字节空文件时）、第二轮（补全 938 行后）

---

## 一、第一轮检视结论摘要（空文件版本）

commit `abb15a00c` 提交的 `engram_test.py` 为 **0 字节空文件**，`run_engram.sh` 的全部执行归结为 `python engram_test.py ...`，空脚本对任意参数 exit 0 → 13 条用例全部打印 `[SUCCESS]` 却不做任何数据生成、算子调用与 golden 比对，形成"全绿假象"。该问题已由 `403ea1680` 补全解决，以下为第二轮检视结果。

## 二、第二轮检视发现（补全后版本）

### 2.1 致命（影响校验有效性本身）

| # | 位置 | 问题 | 实测证据 |
|---|------|------|---------|
| F1 | engram_test.py:909-938 | **失败不上抛退出码**：`final_result="FAIL"` 仅打印+写 CSV，无 `sys.exit(1)`，Python 自然退出 0 → run_engram.sh:291 的 `exec_ret` 校验永不触发，**校验全挂时 runner 仍报 `[SUCCESS] 用例执行完成`** | 代码走查确认无 exit |
| F2 | engram_test.py:896-898 | **子进程硬崩溃 → 父进程永久挂死**：`result_queue.get()` 无超时，`proc.join()` 不检查 exitcode；OOM-kill/段错误绕过 except 分支（本用例大表场景 OOM 概率高）→ 整批挂死 | 代码走查；原版 grad_test.py:1349 有 `get(timeout=10)` 可回移 |
| F3 | engram_test.py:390 | **shape 语义与数据意图相悖（×4 放大）**：`total = num_entries_per_rank * world_size` 把 `input_tensor_shape[0]` 当每 rank 行数。data.md 的 vocab 阶梯（5138000→82208000）按"总 vocab"解释才物理可行（local 模式 large2 = 45.4GB/卡可容；按现语义需 181.6GB/卡不可能）。与 e2e 例子 `--num-entries`（全局总行数）惯例也不一致 | 见设计文档 M1 定量分析 |
| F4 | engram_test.py:292-385 | **数据生成内存策略炸裂**：fp32 整表落 /dev/shm → `frombuffer().clone()` 再整表复制 → `.to(fp8)`。[[82208000,576]] 在 ×4 语义下：fp32 中间 757.7GB(shm) + 40 worker 各自 18.9GB 临时块(共 757.7GB) + clone 757.7GB → 峰值 >2.2TB；**worker 被 OOM killer 杀死后 `mp.Pool.map` 挂死** —— 即"跑一上午构造不出来"的根因 | 用户实测现象 + 内存公式推演 |
| F5 | engram_test.py:40/864/431 | **forkserver 子进程全局 `world_size`=8 而非 CLI 的 4**：子进程以 `__mp_main__` 重新导入模块，`__main__` 内的 CLI 覆盖不生效；`gen_balanced_indices` 分段用的是默认 8。对 ws=4 侥幸无害（8 可被 4 整除），ws=3/6 等流量失衡 | 本机 forkserver 模拟复现：4 个子进程全部看到 world_size=8 |
| F6 | engram_test.py:312/342/378 + 子进程 | **seed 列基本失效**：大数据路径 storage 用 worker 序号种子 1..N，`--seed` 对所有 perf 用例无效；indices 由子进程未播种的 np.random 生成 → 跨 run 不可复现。golden 与 indices 同进程计算自洽，正确性不受影响，但 data.md 的 seed 列形同虚设 | 代码走查 + forkserver 模拟（各 rank indices 不同且与 seed 无关） |

### 2.2 高

| # | 位置 | 问题 |
|---|------|------|
| H1 | engram_test.py:147 + run_engram.sh:164 | data.md 的 `<br>` 残留现为**硬故障**：`_clean()` 只剥 `\n`/`\r` 不剥 `<br>`，`ast.literal_eval` 抛 SyntaxError → prep 即崩 → run_engram.sh:2 `set -e` 使整批在第一条用例中止（若 md→xlsx 保留 `<br>` 文本） |
| H2 | engram_test.py:18 | `from cann_ops_transformer.ops.mc2.common.elastic_buffer import ...` 仅在本分支新构建 wheel（`torch_extension/build/lib/.../ops/mc2/common/elastic_buffer.py`）下成立；当前环境已装的 CANN site-packages 旧版实测 `ModuleNotFoundError`。目录无 README 说明"须先装本仓 torch_extension"前提（e2e 用 `from cann_ops_transformer import ElasticBuffer` 则两代布局通吃） |
| H3 | engram_test.py:684 | `compare_write` 被注释，`write_pass` 恒 True，"WRITE SUCCESS" 无条件打印（fetch golden 间接兜底，但日志误导） |
| H4 | engram_test.py:638-642 | golden"自检"为永真式（golden vs 自身 clone），`[SUCCESS] Golden verification` 纯装饰 |
| H5 | run_engram.sh:203 vs py:700/739 | `PROF_TMP_DIR` 从未被 py 读取，runner 的"独立临时目录"机制完全无效；实际输出固定 `./profiling_data/{case}_profiling`，且相对路径依赖 qrun 执行 cwd 与本地一致 |
| H6 | run_engram.sh:268/278 | `set -e` 下 prep/qrun 失败立即中止脚本，`prep_ret=$?`/`exec_ret=$?` 及后续校验为死代码；首条失败用例中断整批，与逐用例记录失败的设计相反（本机 bash 模拟已复现） |
| H7 | run_engram.sh:186-190 | `data_output=$(read_data)` 失败同样被 set -e 直接终止，`if [[ $? -ne 0 ]]` 不可达；且 python 错误走 stderr，`echo "$data_output"`(stdout) 为空 |

### 2.3 中/低

| # | 位置 | 问题 |
|---|------|------|
| M1 | py:238-251/426-428 | `num_token_per_rank`（含越界防呆）与 `gen_indices` 死代码，无 num_token > vocab 保护 |
| M2 | py:427/444/709/827 | 16M token 走 Python list 链路（`.tolist()` → `torch.tensor(list)`）极慢且耗数百 MB；golden 构造同样低效 |
| M3 | py:745-748 | CPU golden 生成发生在 profile 区域内，污染 CPU timeline |
| M4 | py:31/673 | `explicitly_destroy=True` 全局从未使用，ElasticBuffer 用默认 False 创建（有显式 destroy()，功能无碍） |
| M5 | py:900-903 | "error: ..." 消息不匹配 write/fetch 前缀被静默丢弃（None 兜底仍判 fail，仅丢日志细节）；except 分支 put 两次相同消息 |
| M6 | py:57 硬编码 IP；py:65 固定端口 50008 | 多机 IP 写死、端口无冲突规避（下一条用例可能撞 TIME_WAIT） |
| M7 | run_engram.sh:144 | `int(float(x)) if x != ""` 转换不如原版健壮（原版 run_engram.sh:141-154 的 `to_int` 处理 NaN/空串/nan/none/空格并报明确错误）——**worktree 相对原版回归** |
| M8 | py 缺失 | **原版的 psutil CPU 亲和性绑定被删除**（原版 engram_test.py:59-72/493-508，rank0-3→CPU 288-383、rank4-7→96-191，NUMA 亲和对 C2C 取数带宽有实际影响）——**worktree 相对原版回归** |
| M9 | data.md | `input_tensor_range` 列在 use_fp8=1 分支被静默忽略（py:394-403），data.md 全部行 use_fp8=1 → 该列全部无效 |

### 2.4 显存 OOM 初步定位（用户实测问题）

- 推理路径 workspace 仅 16MB（engram_fetch_tiling.cpp:630 `else` 分支），大 workspace 只在训练路径（:622 `wsTokenStaging = totalRecv × hiddenBytes`）——**推理 OOM 与 workspace 无关**。
- engram 表在 host 侧窗口（num_cpu_bytes=CPU 字节数），**不占 HBM**。
- HBM 大头：fetched 输出（T×576×1B，16M token = 9.0GiB）+ fetched_sf（288MiB）+ indices（64MiB）；eager 单次 ~9.4GiB，64GB 卡本不应 OOM。
- 主要嫌疑：data.md 全部行 `is_graph=1` → `torch.compile(npugraph_ex, dynamic=False)` 对 21 个 token 档位各捕获一张图，**图私有内存池随图存活累积**（Σ 全档位 ≈ 19GiB 输出 + sf/indices/池碎片），叠加后段大档位新分配 → OOM。与"indice 过大时 OOM"现象吻合。
- 详细定量分析与三层拦截方案见《engram_test_modification_design.md》M5。

## 三、与原版 /home/qianlong/workspace/cann/engram_test 的差异

worktree 版 = 原版 + {fetch_mode(local/sliced)、prep_only/run_only 两段式、qrun 排队、并行数据生成(shm)、fetch_token_num 广播、MASTER_PORT 环境变量、loop_time 100→50}，**并删掉了** {psutil CPU 亲和性(M8)、run_engram.sh 的健壮 to_int(M7)}。

原版独有而 worktree 未移植的资产：`grad_test.py`（1401 行完整训练链路：with_grad/engram_fetch_grad/反向 golden/call_mode 0/1/2、`result_queue.get(timeout=10)`）、`analyse.py`（kernel_details.csv 耗时/带宽分析）、`fp8_test.py`（独立 MXFP8 CPU 参考实现）、`run_grad.sh`/`test.sh`。

原版 `engram_test.py` 同样存在 F1/F2/F3(×4 语义)/F5/F6/M1/M2/M4/M5 等问题（forkserver、无退出码、无超时、永真式、compare_write 注释、`total = per_rank × world_size`），即这些问题继承自原版而非 worktree 引入；worktree 新引入的是 F4（并行生成的临时块/clone 放大）与 H1/H5/H6 的 qrun 两段式相关问题。

## 四、覆盖度评估（对分支功能的校验能力）

**已有效覆盖**（补全后）：推理 sliced/local 双模式 × 6 档 vocab 逐 fetch 位级 `torch.equal` 比对；fp8_e4m3fn + e8m0 sf 全链路（fetched_sf 位级比对）；aclgraph(npugraph_ex fullgraph) 路径（e2e 没有的新增覆盖）；cb260d955 tiling 修复的回归方向（条件若回退为反转形式，全部推理用例 tiling 直接报错，可被发现）。

**仍无法覆盖**：
1. 无训练/with_grad 维度、无负向用例 → cb260d955 修复的正向语义（train+local 必须被拒绝）零验证；fe31f38ca 训练侧（退化 backward ctx、grad local 跳过 a2a）、零拷贝原地更新、memcpy 二次 write 更新可见性全部没测（原版 grad_test.py 具备大部分能力，待移植）；
2. 仅 fp8_e4m3fn 一种 dtype，dtype 矩阵缺失；
3. data.md 无 `<br>` 之外的格式规约（md→xlsx 转换约定未文档化）。

## 五、用户实测三大问题的结论指向

| 现象 | 根因 | 对应设计项 |
|------|------|-----------|
| [[82208000,576]] 随机数一上午构造不出 | F3(×4 语义) + F4(fp32 整表+clone+worker 整块临时 → 峰值>2.2TB，Pool worker 被 OOM kill 后 `pool.map` 挂死) | 设计文档 M1+M2 |
| indice 过大显存 OOM | 表在 host 侧不占 HBM；单次输出 9.4GiB 合法；主因 npugraph_ex 21 档位图池累积（待按 M5 方案验证/拦截） | 设计文档 M5 |
| 82208000 语义 | 原版/现版代码均为"每 rank 行数"（total=×4）；用户意图与 e2e 惯例为"总 vocab" → 需改语义并连带修正扫描序列（16M 档位 > small/medium vocab） | 设计文档 M1 |
