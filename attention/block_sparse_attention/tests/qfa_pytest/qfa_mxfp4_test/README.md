# QFA MXFP4 TTK 适配

把 `quant_flash_attn` (MXFP4) 算子的 pytest 测试迁移到 TTK (ops-test-kit) 框架, 用 TTK e2e CSV 批跑替代 pytest.

## 文件结构

```
tests/
├── pytest/qfa_mxfp4_test/                 # 新增: TTK 适配主目录
│   ├── qfa_mxfp4_wrapper.py               # NPU wrapper (调 quant_flash_attn)
│   ├── qfa_mxfp4_paramset_common.py       # 参数展开工具
│   ├── qfa_mxfp4_paramset_func_rdv.py     # func_rdv 用例集 (15 个 case, 来自 test_case_STC.py)
│   ├── gen_csv_case_store.py              # paramset → CSV 转换
│   ├── qfa_mxfp4.csv                      # 生成的 CSV 用例表
│   └── common/
│       ├── qfa_mxfp4_golden.py            # 拆分自 quant_flash_attn_golden.py:
│       │                                  #   generate_data / cpu_mxfp4_golden / npu_mxfp4_fa
│       │                                  #   支持 GRAPH_PATH=0 (eager) / 7 (npugraph_ex)
│       ├── result_compare_method.py       # 精度对比 (复制自 mxfp8)
│       └── __init__.py
├── assets/                                # TTK plugin 目录
│   ├── spec.py                            # TestSpec 注册
│   └── impl/
│       ├── inputs.py                      # 输入生成 + 缓存
│       ├── golden.py                      # CPU golden 适配
│       └── compare.py                     # 精度对比
└── pytest/                                # 原有 pytest 测试 (保持不动)
    ├── quant_flash_attn_golden.py         # 原单体 golden (run_fia_eager)
    ├── test_cases/test_case_STC.py        # 原测试用例
    └── ...
```

## 数据流

```
TTK 读 CSV
    ↓
customize_inputs (assets/impl/inputs.py)
    → 调 golden_mod.generate_data() 生成 BNSD FP32 Q/K/V
    → MXFP4 量化 + scale 转换 + layout 重排
    → 缓存到 golden_mod._cached_data (布局后) + golden_mod._cached_bnsd_inputs (BNSD 原始)
    ↓ (无返回值)
TTK 调 wrapper(占位tensor, attrs)
    ↓
wrapper (qfa_mxfp4_wrapper.py)
    → 注入 golden 全局变量 (B, N_q, D, INPUT_LAYOUT, GRAPH_PATH, ...)
    → 从 golden_mod._cached_data 取真实数据
    → 调 golden_mod.npu_mxfp4_fa(data_dict)
        ├── GRAPH_PATH=0: _call_npu_fa_op (eager 直接调用)
        └── GRAPH_PATH=7: _mxfp4_fa_torch_npu (npugraph_ex 编译 Network)
    ↓
golden (assets/impl/golden.py)
    → 注入 golden 全局变量
    → 调 golden_mod.cpu_mxfp4_golden(cached)
        → 复用 flash_attention_cpu_golden_varlen (blockwise_snap_local, fp16 s_dtype)
    ↓
compare (assets/impl/compare.py)
    → 过滤空 tensor, 配对比较 NPU 与 golden 输出
```

## 运行命令

### 路径约定

以下命令假设：

- `TTK_HOME` = TTK (ops-test-kit) 安装根目录，例如 `/mnt/workspace/gitCode/cann/ttk/ops-test-kit`
- `OP_TEST` = 本算子测试目录，即本 README 所在目录的绝对路径。在你机器上一般是：
  ```
  /home/<user>/code/mxfp4/ops-transformer/experimental/attention/quant_flash_attn/tests
  ```
  下面 3 个子路径会用到：
  - `<OP_TEST>/pytest/qfa_mxfp4_test`     — wrapper / paramset / CSV 所在目录（PYTHONPATH 指向它）
  - `<OP_TEST>/pytest/qfa_mxfp4_test/qfa_mxfp4.csv` — 输入 CSV
  - `<OP_TEST>/assets`                     — TTK plugin 目录（spec.py + impl/）

> 坑1: PYTHONPATH **必须**指向 `qfa_mxfp4_test` 目录，否则 wrapper 模块找不到。
> 坑10: **不要**用 TTK 的 `-c`（静态图）或 `-d`（动态图）编译模式，会触发 torchair `get_npu_backend()` bug。图模式由 wrapper 内部根据 `GRAPH_PATH` 自己控制。

### 单 case 调试

```bash
# 先进 TTK 安装目录 (ttk 是 python -m ttk 启动的，需要在它能找到 ttk 包的地方运行)
cd $TTK_HOME

# 再设置环境 + 跑指定 case
PYTHONPATH=<OP_TEST>/pytest/qfa_mxfp4_test \
python3 -m ttk e2e \
  -i <OP_TEST>/pytest/qfa_mxfp4_test/qfa_mxfp4.csv \
  --plugin <OP_TEST>/assets \
  --backend npu \
  -t FP4_B8_Nq12_Nkv12_D128_S1_1024_S2_1024_BNSD_redline \
  --warmup 0 --run 1
```

### 全量功能批跑 (eager, GRAPH_PATH=0)

```bash
cd $TTK_HOME

PYTHONPATH=<OP_TEST>/pytest/qfa_mxfp4_test \
python3 -m ttk e2e \
  -i <OP_TEST>/pytest/qfa_mxfp4_test/qfa_mxfp4.csv \
  --plugin <OP_TEST>/assets \
  --backend npu \
  -o result.csv \
  --warmup 0 --run 1
```

`-o result.csv` 会把每个 case 的 PASS/FAIL/precision 写到 `result.csv`。

### aclgraph 模式 (GRAPH_PATH=7)

```bash
cd $TTK_HOME

# 方式 A: 用环境变量 (推荐, 优先级高于 CSV 里的 graph_path 字段)
GRAPH_PATH=7 \
PYTHONPATH=<OP_TEST>/pytest/qfa_mxfp4_test \
python3 -m ttk e2e \
  -i <OP_TEST>/pytest/qfa_mxfp4_test/qfa_mxfp4.csv \
  --plugin <OP_TEST>/assets \
  --backend npu \
  -t <case_name> \
  --warmup 0 --run 1

# 方式 B: 修改 CSV 里目标 case 的 'graph_path' 属性为 7
# 直接编辑 qfa_mxfp4.csv, 把 'graph_path': 0 改成 'graph_path': 7
```

> 踩坑 A.2: 若 GRAPH_PATH=7 出现 SIGSEGV，需删除 `quant_flash_attn.py` 源码中 `@allow_in_graph` 装饰器 (两个函数都要)，然后 `pip install --force-reinstall` 重新安装 cann_ops_transformer。

### 重新生成 CSV

```bash
cd <OP_TEST>/pytest/qfa_mxfp4_test
python3 gen_csv_case_store.py \
  --paramset qfa_mxfp4_paramset_func_rdv \
  --output qfa_mxfp4.csv
```

### 命令参数逐项解释

| 参数 | 作用 | 示例 |
|------|------|------|
| `cd $TTK_HOME` | 切到 TTK 安装目录，让 `python3 -m ttk` 能找到 ttk 包 | `cd /mnt/workspace/gitCode/cann/ttk/ops-test-kit` |
| `PYTHONPATH=<OP_TEST>/pytest/qfa_mxfp4_test` | 告诉 Python 去 `qfa_mxfp4_test` 目录下找 wrapper 模块 (`qfa_mxfp4_wrapper.py`)。**坑1**：不设的话 wrapper 找不到 | `PYTHONPATH=/home/user/.../tests/pytest/qfa_mxfp4_test` |
| `python3 -m ttk e2e` | 启动 ttk 包的 e2e 子命令，做端到端 (NPU 跑 + golden 对比) 测试 | 固定写法 |
| `-i <csv>` | 输入用例表 CSV 路径，每行一个 case | `-i .../qfa_mxfp4.csv` |
| `--plugin <dir>` | TTK plugin 目录，里面有 `spec.py` 注册 TestSpec (golden/inputs/compare 三个回调) | `--plugin .../tests/assets` |
| `--backend npu` | 设备后端，固定 `npu` (本算子是 NPU 算子) | 固定写法 |
| `-t <case_name>` | 只跑指定 testcase_name 的 case (不指定则跑 CSV 全部) | `-t FP4_B8_Nq12_Nkv12_D128_S1_1024_S2_1024_BNSD_redline` |
| `-o result.csv` | 输出结果 CSV (每行: case_name, pass, precision, ...) | 不指定则不写文件，只在 stdout 打印 |
| `--warmup 0` | 预热次数 (0 表示不预热，直接跑 1 次) | 功能测试用 0；性能测试用 5-10 |
| `--run 1` | 实际运行次数 (功能测试用 1) | 性能测试用 10+ |
| `GRAPH_PATH=7` (环境变量) | 控制 wrapper 内部走 eager (0) 还是 aclgraph (7)。优先级高于 CSV 里的 `graph_path` 字段 | `GRAPH_PATH=7 python3 -m ttk e2e ...` |

### 完整调用链路

```
TTK e2e 命令
    ↓ 读 CSV，解析 attributes (B, N_q, D, layout, graph_path, ...)
    ↓
customize_inputs (assets/impl/inputs.py)
    → 注入 golden 全局变量 (B, N_q, INPUT_LAYOUT, GRAPH_PATH, ...)
    → 调 golden_mod.generate_data() 生成 BNSD FP32 Q/K/V → MXFP4 量化
    → 缓存到 golden_mod._cached_data (布局后) + golden_mod._cached_bnsd_inputs (BNSD 原始)
    ↓ (TTK 忽略返回值, 坑4)
TTK 调 wrapper(占位tensor, attrs)  (qfa_mxfp4_wrapper.py)
    → 注入 golden 全局变量
    → 从 golden_mod._cached_data 取真实 MXFP4 数据
    → 调 golden_mod.npu_mxfp4_fa(data_dict)
        ├── GRAPH_PATH=0: _call_npu_fa_op (eager 直接调 quant_flash_attn_metadata + quant_flash_attn)
        └── GRAPH_PATH=7: _mxfp4_fa_torch_npu (npugraph_ex 编译 Network, 坑9: 不调 get_npu_backend())
    ↓
golden (assets/impl/golden.py)
    → 注入 golden 全局变量
    → 调 golden_mod.cpu_mxfp4_golden(cached)
        → 复用 flash_attention_cpu_golden_varlen (blockwise_snap_local, fp16 s_dtype)
    ↓
compare (assets/impl/compare.py)
    → 过滤空 tensor (坑6), 配对比较 NPU 与 golden 输出
    → 返回 dict {pass, precision, error_info, metrics}
```

## 当前限制

- 仅支持 `kv_storage_mode='continue'` (不支持 PA)
- 仅支持 BNSD / BSND layout (不支持 TND)
- 仅支持 `mask_mode=0` (无 mask)
- 不支持 RoPE
- 不支持 learnable_sink
