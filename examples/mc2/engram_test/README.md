# engram_test 使用说明

engram_fetch 算子的批跑测试框架：从 `engram_test_cases.xlsx` 读取用例表，逐条完成
数据准备（prep，本地 CPU）与 NPU 执行（qrun 排队），对每个 token 档位做 fetch 结果与
CPU golden 的逐行位级比对（`torch.equal`），并支持 aclgraph（`npugraph_ex`）与
profiling 采集。

- 被测对象：本分支 `mc2/engram_fetch` 算子及其 PTA（`cann_ops_transformer` 的 `ElasticBuffer`）；
- 端到端功能验证请用 `examples/mc2/engram_fetch_e2e/`（含训练链路），本框架侧重批量性能/回归扫描。

## 1. 依赖安装

**必须先构建并安装本仓（engram-fetch-local-mode 分支）的 torch_extension whl**，
本框架的 `from cann_ops_transformer import ElasticBuffer` 依赖其中的 ElasticBuffer 实现
（系统里已装的旧版 CANN site-packages 不含本分支新增的 fetch_mode 等接口）。

```bash
# 仓库根目录（整包编译，含 ElasticBuffer；勿用 --ops 过滤，过滤会丢失 mc2/common 下的实现）
bash build.sh --torch_extension
python3 -m pip install build_out/cann_ops_transformer-*.whl --force-reinstall --no-deps

# 同时需安装自定义算子包（详见 examples/mc2/engram_fetch_e2e/README.md）
bash build.sh --pkg --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait
```

安装后验证（通过才可运行本框架）：

```bash
python3 -c "from cann_ops_transformer import ElasticBuffer"
```

其他依赖：`torch`、匹配的 `torch_npu`、`pandas`、`openpyxl`（读 xlsx 用）；
`psutil` 为可选依赖（仅 `--cpu_map` CPU 亲和性绑定需要，未安装时打 WARNING 跳过）。

## 2. 数据表（engram_test_cases.xlsx）

```bash
# 用法
bash run_engram.sh --xlsx engram_test_cases.xlsx --sheet engram_tensor --num 1      # 单条
bash run_engram.sh --xlsx engram_test_cases.xlsx --sheet perf_vocab   --num 1 11    # 范围
```

- `--sheet engram_tensor`：OOM 分析用例（1x/4x/16x tensor，6 条）；
- `--sheet perf_vocab`：vocab 阶梯性能扫描（data.md 的 11 条）；
- `--num` 为 Excel 数据行序号（1 起，第 2 行对应 1）。

### 格式约定

| 列 | 约定 |
| ---- | ---- |
| `input_tensor_shape` | python 字面量，如 `[[5138000, 576]]`；`shape[0]` 为**总 vocab 行数**（见下节语义） |
| `input_tensor_dtype` | python 字面量，如 `['fp8_e4m3fn']`，支持 bf16/fp16/fp32/int8/int32/int64/fp8_e4m3fn/fp8_e5m2 |
| `input_tensor_range` | python 字面量，如 `[(-5, 5)]`（use_fp8=1 时不生效，走 randn+sf 链路） |
| `fetch_indicies` | **单行填写**，各 rank 的 token 档位列表，如 `[[16, 32], [16, 32]]`；**不要用 `<br>` 换行**（runner 会把 `<br>`/`<br/>`/`<br />` 替换为空格，但新表应直接单行书写）；字段内不要出现 `|`（会被替换为 `;` 防 IFS 截断） |
| `world_size` / `num_cpu_bytes` / `is_graph` / `is_profiling` / `use_fp8` / `seed` | **纯整数**；空/NaN/nan/none 按 0 处理，非法值会报列名明确的错误 |
| `sf_dtype` | `float32` 或 `e8m0`，空/0 表示不传（用 fp32 默认） |
| `fetch_mode` | `sliced`(0) 或 `local`(1)，空/0 默认 sliced |

## 3. shape 语义与模式

### input_tensor_shape[0] = 总 vocab（默认 `--shape_semantics total`）

`[[82208000, 576]]` 中的 82208000 是**全局 vocab 总行数**（与 e2e 的 `--num-entries`
惯例一致）：

- **sliced 模式**（fetch_mode=0，默认）：全局表按 rank 均匀切片分存，fetch 任意全局
  index 由持有该 entry 的远端 rank 跨 NPU 通信取回。**约束：总行数必须能被 world_size
  整除**，否则启动即报错退出（错误信息含 total/world_size/建议值）。本表六档 vocab
  均可被 4 整除（82208000/4=20552000 等）。
- **local 模式**（fetch_mode=1，仅推理）：全量表冗余到每个 NPU，纯本地取数，无跨 NPU
  通信。无整除约束。

### 旧表迁移（breaking change）

旧版代码把 `shape[0]` 当**每 rank 行数**（总行数再 × world_size）。若有按旧语义构造的
表，迁移方式二选一：

1. （推荐）把表里的 shape 改成总行数语义（即把原来的 `shape[0] * world_size` 写进表）；
2. 临时加 `--shape_semantics per_rank` 兼容旧行为（总行数 = shape[0] × world_size）。

### numTokens > vocab 是合法场景

token 档位允许超过 vocab 行数（如 16M 档位 vs small vocab 5.14M）：indices 有放回采样自
`[0, vocab)`，对应 LLM Prefill 长上下文/大 batch 场景（batch×seq_len 超 vocab、token 服从
Zipf 分布天然大量重复）。此时仅打一条
`repeated sampling, prefill long-context/large-batch scenario` WARNING，**不判失败**。

## 4. 运行方式

### 两段式执行

`run_engram.sh` 对每条用例分两阶段：

1. **prep**（本地，不排队）：`python engram_test.py ... --prep_only` 生成 storage/sf 表并
   `torch.save` 到 `./{case_name}_prep_data.pt`；
2. **run**（qrun 排队占 NPU）：`qrun --emp q00983190 -n {case_name} python engram_test.py ... --run_only`
   加载 prep 数据后 fork 多个子进程执行 NPU 测试。

prep/qrun 均为容错模式：单条用例失败会记入失败列表并继续下一条，批跑结束汇总打印全部
用例结果（SUCCESS/FAILED），任一失败则脚本退出码非 0。py 侧失败也会以非零码退出
（`sys.exit(0 if all_success else 1)`）。

### 端口与输出

- **master_port**：runner 自动为每条用例传 `--master_port $((50008 + 用例序号))`，
  避免相邻用例撞 TIME_WAIT；手动单跑时可用 `--master_port` 指定（默认取环境变量
  MASTER_PORT 或 50008）。
- **profiling 输出**：py 读取环境变量 `PROF_TMP_DIR` 作为输出根目录（runner 在
  is_profiling=1 时为每条用例 `mktemp -d` 建独立目录并 export）；未设置时默认
  `./profiling_data/{case_name}_profiling`。两种情况一律绝对路径化，无 qrun cwd 依赖。

### 常用 py 侧参数（`python engram_test.py --help` 查看全部）

| 参数 | 默认 | 说明 |
| ---- | ---- | ---- |
| `--shape_semantics` | `total` | `total`=shape[0] 为全局 vocab 总行数；`per_rank`=旧行为（总数=n×world_size） |
| `--fetch_mode` | `sliced` | `sliced`(0)=切片跨 NPU 通信；`local`(1)=全量本 NPU（仅推理） |
| `--start_method` | `fork` | 子进程启动方式；fork 要求父进程不初始化 NPU（框架已保证，违反即报错拒绝），forkserver 为兜底 |
| `--master_port` | 50008 | HCCL rendezvous 端口 |
| `--loop_time` / `--repeat_time` / `--warmup_time` | 50 / 1 / 0 | 性能采集轮数/批次数/热身轮数 |
| `--cpu_map` | 空=关闭 | CPU 亲和性映射，格式 `0-3:288-383;4-7:96-191`，子进程内按 rank 生效（需 psutil） |
| `--verify_write` | 关 | engram_write 后读取 host 窗口指针与 write golden 比对（依赖私有接口，异常时打 WARNING 跳过） |
| `--prep_only` / `--run_only` | — | 两段式开关 |

NUMA 亲和对 C2C 取数带宽有实际影响，建议正式出数时配置 `--cpu_map`（rank 区间:CPU 区间，
分号分隔）；CPU 编号可用 `lscpu` / `numactl --hardware` 探测本机 NUMA 拓扑后确定。

## 5. 运维提示（历史 OOM/故障事件，详见 oom.md）

- **ninja 编译 OOM（exit code 137）**：JIT 编译 torch_extension 时并行度过高被 OOM
  killer 杀死，且残留 stale lock 导致下次重复触发编译形成死循环。处理：
  1. 清理 stale lock：`rm -f ~/.cache/torch_extensions/py311_cpu/*/lock`；
  2. 低负载下预编译：`MAX_JOBS=1 python -c "import ..."`（或 `MAX_JOBS=1` 重跑触发编译的命令）；
  3. 确认 `.so` 生成后再启动测试。
- **HCCL 残留进程/端口冲突**：上次异常退出残留进程占用 rendezvous 端口，报
  `Acquire HCCL channel failed`。处理：启动前 `pkill -f <test_script>` 清理残留进程；
  本框架已按用例序号递增分配 master_port 规避 TIME_WAIT。
- **多用户 stream 耗尽（error 507899）**：共享 NPU 上多用户/大量图编译导致驱动
  Stream/CQ 资源耗尽（`halSqCqAllocate failed`）。处理：切换空闲卡
  （`export ASCEND_RT_VISIBLE_DEVICES=<空闲卡ID>`）或减少不同 shape 的图编译次数。
- **16x tensor（hidden=9216）大 token 档位显存 OOM**：根因是 npugraph_ex 逐档位捕获
  图且图私有内存池累积，8M token 档单次 `fetched` 输出即 72GiB。显存预检与图池释放
  属后续任务（见路线图 C3），当前请参考 oom.md §3.6 的各档位 `fetched` 大小表裁剪
  token 档位。

## 6. 路线图（后续任务，本框架预留演进）

- **C2（数据生成重构 + 缓存复用）**：替换现有"fp32 整表 /dev/shm + clone + torch.save/.pt"
  链路为分块直写最终 dtype 的 mmap 缓存文件（峰值临时内存从数百 GB 降到 <1GB），
  local/sliced 同 shape 配对用例按缓存键复用同一份数据（只缩短 prep 墙钟，不影响 fetch
  计时口径）；`--seed` 全链路生效（SeedSequence 派生）。
- **C3（显存预检 + 图池释放）**：用例侧按 token 档位预算检查可用 HBM（不足则 SKIP 并
  汇总，不算 FAIL）+ graph 模式逐档位 `torch._dynamo.reset()` 释放图池；PTA（csrc）侧
  在输出分配点加 `TORCH_CHECK(fetched_bytes < free*0.8)` 拦截。

以上两项落地后本 README 将同步更新用法说明。
