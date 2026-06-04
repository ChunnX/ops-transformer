# FlashAttention pytest 架构重构方案

## 一、现状问题

当前 `test_flash_attn.py` 存在以下架构问题：

| 问题 | 表现 |
|---|---|
| **单块巨石** | `call_flash_attn()` 210 行，混合了数据生成、设备搬运、CPU/GPU/NPU 计算、精度对比 |
| **数据生成无抽象** | `normalize_case` 混合了参数补全 + PA block_table 生成 + TND cu_seqlens 推导，职责不清 |
| **设备逻辑分散** | QKV 永远在 CPU 生成后 `.cuda()` / `.to(device)`，无法利用 NPU 直接生成 |
| **无法复用** | 整段代码耦合 flash_attn，切到 FIA/FusionAttn 等其他算子需要重写全部逻辑 |
| **后端耦合** | GPU 和 NPU 的调用方式完全不同，却在同一个 `if/else` 分支里 |

---

## 二、重构目标

1. **数据在目标设备上生成**：NPU 模式 → 张量直接在 NPU 上 `randn`；GPU 模式 → 直接在 GPU 上
2. **后端抽象**：CPU/GPU/NPU 统一接口，新增算子只需实现一个 Backend 类
3. **分层清晰**：数据层 / 计算层 / 对比层 / 编排层，每层独立可测
4. **向后兼容**：`test_cases/*.py` 的 `TestCases` dict 格式不变

---

## 三、目标架构

```
pytests/
├── test_flash_attn.py              # 薄入口 (~60 行)
│
├── core/
│   ├── __init__.py
│   │
│   ├── data.py                     # [新增] TensorSpec / DataSpec / 数据生成
│   │   └── 职责：根据 DataSpec 在指定 device 上生成 Q/K/V 等输入张量
│   │
│   ├── backends/
│   │   ├── __init__.py             # 后端注册表
│   │   ├── base.py                 # Backend ABC
│   │   ├── cpu.py                  # CPUBackend: 调用 tforward()
│   │   ├── gpu.py                  # GPUBackend: 调用 flash_attn_gpu()
│   │   └── npu.py                  # NPUBackend: 调用 flash_attn_npu() / graph
│   │   └── 职责：接收 inputs dict → 做本端布局转换 → 调 kernel → 返回 outputs
│   │
│   ├── compare.py                  # [从 utils/ 迁移] check_result / 三方对比
│   │
│   ├── case_loader.py              # [从 test_flash_attn 抽取] load_case_modules + normalize
│   │
│   ├── orchestrator.py             # [新增] 编排器：串联 data → backend → compare
│   │
│   └── reporter.py                 # [新增] 终端表格 + CSV 报告
│
├── test_cases/                     # 不改动
├── backends/                       # 保留兼容，渐进废弃
├── utils/                          # 保留 data.py / io.py / perf_runner.py 等
└── docs/
    └── perf_test_proposal.md
```

---

## 四、核心抽象设计

### 4.1 DataSpec（数据规格）

声明式描述一个算子的输入张量集，与具体 device 无关。

```python
@dataclass
class DataSpec:
    """描述一次算子调用的输入张量集合"""
    tensors: Dict[str, TensorSpec]  # 如 {"q": ..., "k": ..., "v": ...}
    layout: LayoutInfo               # layout_q / layout_kv / layout_out
    scalar_params: Dict[str, Any]    # mask_mode, scale, block_size 等
    seed: int = 42


@dataclass
class TensorSpec:
    """描述一个张量的生成规则"""
    shape_fn: Callable[..., Tuple[int, ...]]  # 根据 DataSpec 计算实际 shape
    dtype: torch.dtype
    value_range: Optional[Tuple[float, float]] = None  # None = 固定值 10
    seed: Optional[int] = None
    requires_rope: bool = False  # FlashAttn 特有，可参数化


@dataclass
class LayoutInfo:
    q: str = "BNSD"
    kv: str = "BNSD"
    out: str = "BNSD"
    is_pa: bool = False
    block_size: int = 128
```

**关键设计**：`shape_fn` 是 `Callable`，接收当前 case 的完整参数，返回该张量在实际 layout 下的 shape。这样 BNSD → `(B, N1, S1, D)`、BSND → `(B, S1, N1, D)` 等逻辑内聚在 DataSpec 中。

```python
# 示例：FlashAttention 的 DataSpec
FLASH_ATTN_DATASPEC = DataSpec(
    tensors={
        "q": TensorSpec(
            shape_fn=lambda c: {
                "BNSD": (c["B"], c["N1"], c["S1"], c["D"]),
                "BSND": (c["B"], c["S1"], c["N1"], c["D"]),
                "TND":  (c["S1"], c["N1"], c["D"]),
            }[c["layout_q"]],
            dtype=torch.float16,
        ),
        "k": TensorSpec(
            shape_fn=lambda c: ...,
            dtype=torch.float16,
        ),
        "v": TensorSpec(
            shape_fn=lambda c: ...,
            dtype=torch.float16,
        ),
    },
    layout=LayoutInfo(...),
    scalar_params={...},
)
```

### 4.2 Backend（计算后端）

```python
class Backend(ABC):
    """计算后端抽象基类"""
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def compute(self, inputs: Dict[str, torch.Tensor],
                spec: DataSpec) -> Dict[str, torch.Tensor]:
        """
        inputs:  在 self.device 上生成的张量 (q, k, v, ...)
        spec:    描述这些张量的 DataSpec
        returns: {"out": ..., "lse": ...}  仍然在 self.device 上
        """
        ...

    @property
    def device(self) -> torch.device: ...
```

三个实现：

```python
class CPUBackend(Backend):
    name = "cpu"
    device = torch.device("cpu")

    def compute(self, inputs, spec):
        # BNSD 格式的 qf, kf, v → tforward() → out_cpu
        return {"out": tforward(qf, kf, v, **spec.scalar_params)}


class NPUBackend(Backend):
    name = "npu"
    device = torch.device("npu:0")

    def compute(self, inputs, spec):
        # 1. 布局转换 (BNSD → BSND/TND/PA)
        # 2. npu_flash_attn_metadata()
        # 3. npu_flash_attn()
        # 4. synchronize
        # 返回 {"out": ..., "lse": ...} 在 NPU 上


class GPUBackend(Backend):
    name = "gpu"
    device = torch.device("cuda:0")

    def compute(self, inputs, spec):
        # 1. flash_attn_gpu()
        # 2. synchronize
        # 返回 {"out": ...} 在 GPU 上
```

### 4.3 Orchestrator（编排器）

```python
def run_case(spec: DataSpec,
             primary: Backend,          # 主计算后端 (NPU 或 GPU)
             golden: Backend,           # Golden 后端 (CPU)
             comparators: List[Backend] = None,
             ) -> CompareResult:
    """
                ┌──────────────────────────────────────┐
                │  1. generate_inputs(spec, device=primary.device)
                │     → primary_inputs                 │
                │                                      │
                │  2. golden_inputs = to_cpu(primary_inputs)
                │     → CPUBackend.compute(golden_inputs)
                │     → golden_output                  │
                │                                      │
                │  3. NPUBackend.compute(primary_inputs)
                │     → primary_output                 │
                │                                      │
                │  4. compare(primary_output.cpu(), golden_output)
                │                                      │
                │  5. (optional) 对其他 comparators:    │
                │     compar_inputs = primary.to(other_device)
                │     → OtherBackend.compute(...)       │
                │     → compare(other_output, golden)   │
                └──────────────────────────────────────┘
    """
```

**核心理念**：只在一个 device 上生成一次数据（`torch.Generator` 固定 seed），然后拷贝到其他 device。
不依赖不同 device 的 PRNG 实现一致性——CUDA 用 cuRAND、NPU 用自研实现、CPU 用 stdlib，同样的 seed 产出不同的随机数。

### 4.4 数据生成 — `InputSpec` 构建器

核心理念：**声明什么 → 自动生成什么**。一个算子 = 一个 `InputSpec` 对象，约 15 行代码。

#### 4.4.1 唯一的类：`InputSpec`

```python
import torch
from typing import Any, Callable, Dict, List, Optional, Union


class InputSpec:
    """声明式张量生成规则。

    用法：
        spec = (InputSpec()
            .tensor("q", layout=dict(BNSD="B,N1,S1,D", BSND="B,S1,N1,D"))
            .tensor("k", layout=dict(BNSD="B,N2,S2,D"))
            .tensor("v", layout=dict(BNSD="B,N2,S2,_V"))
        )
        inputs = spec.generate(params, device, layout="BNSD")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._items: List[dict] = []

    # ── 核心 API：.tensor() ──

    def tensor(self, name: str, *,
               shape: str = "",           # 默认 shape 表达式
               layout: dict = None,       # {"BNSD": "B,N1,S1,D", "BSND": "..."}
               dtype=None,                # torch.float16 | torch.int32
               method: str = "randn",     # randn | uniform | fixed | zeros
               when: Callable = None,     # fn(params) → bool，条件生成
               **kwargs                   # method 参数 (如 fixed_value=10)
               ) -> "InputSpec":
        self._items.append(dict(
            name=name, shape=shape, layout=layout or {},
            dtype=dtype, method=method, when=when, kwargs=kwargs,
        ))
        return self  # 链式调用

    # ── 生成入口 ──

    def generate(self, params: dict, device,
                 layout: str = "BNSD", seed: int = 42
                 ) -> Dict[str, torch.Tensor]:
        rng = torch.Generator(device=device).manual_seed(seed)
        tensors = {}
        for it in self._items:
            if it["when"] and not it["when"](params):
                continue
            shape = self._resolve(it, params, layout)
            if not shape:
                continue
            dt = it["dtype"] or _dtype(params.get("Dtype", "fp16"))
            t = _generate(it["method"], shape, dt, device, rng, **it["kwargs"])
            tensors[it["name"]] = t
        return tensors

    # ── 内部：shape 解析 ──

    def _resolve(self, item, params, layout):
        shape_str = item["layout"].get(layout, item["shape"])
        if not shape_str:
            return None
        dims = []
        for token in shape_str.split(","):
            token = token.strip()
            if token in params:
                dims.append(params[token])
            elif token == "_V":
                dims.append(params.get("DV", params["D"]))
            elif token == "total_s1":
                cu = params.get("cu_seqlens_q")
                dims.append(cu[-1] if cu else params["S1"])
            elif token == "max_blocks":
                sv = params.get("seqused_kv", [params.get("S2", 1)])
                bs = params.get("block_size", 128)
                dims.append((max(sv) + bs - 1) // bs if sv else 1)
            else:
                dims.append(int(token))  # literal like "128"
        return tuple(dims)


# ── helpers ──

def _dtype(ds): return torch.float16 if ds == "fp16" else torch.bfloat16

def _generate(method, shape, dtype, device, rng, **kw):
    if method == "randn":
        return torch.randn(shape, dtype=dtype, device=device, generator=rng)
    if method == "uniform":
        lo, hi = kw.get("low", 0.0), kw.get("high", 1.0)
        t = torch.rand(shape, dtype=dtype, device=device, generator=rng)
        return t * (hi - lo) + lo
    if method == "fixed":
        return torch.full(shape, kw.get("value", 10.0), dtype=dtype, device=device)
    if method == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if method == "custom":
        return kw["fn"](shape, dtype, device, rng)
    raise ValueError(f"Unknown method: {method}")
```

#### 4.4.2 预定义 recipe：FlashAttention

```python
# ── 一个算子 = 10 行声明 ──

flash_attn_inputs = (
    InputSpec("flash_attn")
    .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "TND": "total_s1,N1,D"})
    .tensor("k", layout={"BNSD": "B,N2,S2,D", "BSND": "B,S2,N2,D", "TND": "total_s2,N2,D"})
    .tensor("v", layout={"BNSD": "B,N2,S2,_V", "BSND": "B,S2,N2,_V", "TND": "total_s2,N2,_V"})
    .tensor("block_table", shape="B,max_blocks", dtype=torch.int32,
            when=lambda p: p.get("layout_kv","").startswith("PA_"))
)
```

#### 4.4.3 预定义 recipe：FIA（新算子）

```python
fia_inputs = (
    InputSpec("fia")
    .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D"})
    .tensor("k", layout={"BNSD": "B,N2,S2,D"})
    .tensor("v", layout={"BNSD": "B,N2,S2,_V"})
    .tensor("actual_seqlen", shape="B", dtype=torch.int32)
)
```

#### 4.4.4 四种扩展方式

**① 新增 method** — 在 `_generate` 中加一个 elif 分支，或传 `custom`：

```python
spec.tensor("indices", shape="B,S1", method="custom",
            fn=lambda s, d, dev, rng: torch.randint(0, 100, s, device=dev))
```

**② 新增条件** — `when` lambda 一行搞定：

```python
spec.tensor("kv_cache", shape="B,N2,max_blocks,block_size,D",
            when=lambda p: p.get("mask_mode") == 3)
```

**③ 新增布局** — 在 `layout` dict 中加一行：

```python
spec.tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "NEW": "..."})
```

**④ 新增算子** — `InputSpec("new_op").tensor(...).tensor(...)` 创建即可。Backend、Orchestrator、Reporter 全部复用。

#### 4.4.5 完整数据流

```
TestCase params = {"B":4, "N1":8, "S1":128, ...}
    │
    ▼
InputSpec.generate(params, device="npu:0", layout="BNSD")
    │  ├─ when filter    → 跳过 DRope=0 时不生成 q_rope
    │  ├─ layout resolve → BNSD → shape (4, 8, 128, 128)
    │  ├─ method dispatch → randn / fixed / zeros / ...
    │  └─ dtype infer    → params["Dtype"] → torch.float16
    │
    ▼
{"q": Tensor(4,8,128,128), "k": ..., "v": ..., "block_table": ...}
    │
    ▼
Backend.compute(inputs) → outputs
```

---

## 五、数据流对比

### 现状

```
CPU randn  → .cuda() / .to(npu) → GPU/NPU compute
  │
  └──→ CPU compute (golden)
```

各后端用的确实是同一份数据（CPU 生成→分发），但 `normalize_case` 和 `call_flash_attn` 混在一起，逻辑不清晰。

### 重构后

```
NPU PRNG randn (seed=42)                        ← 硬件加速，快 10-100x
  │
  ├── [Q,K,V on NPU] → NPUBackend.compute()  → out_npu
  │
  └── .cpu()          → CPUBackend.compute()  → out_cpu (golden)
       (1x D2H copy)
```

**数据一致性**：NPU 生成后 `.cpu()` 的拷贝和 NPU 上的原张量是同一份数据。golden 的输入 = NPU 的输入，无需跨设备 PRNG 对齐。

**三设备对比**（NPU vs GPU vs CPU）：
```
NPU PRNG randn → [Q,K,V on NPU]
  ├── .cpu()    → CPUBackend.compute() → out_cpu (golden)
  ├── NPUBackend.compute()              → out_npu
  └── .to('cuda') → GPUBackend.compute() → out_gpu
       (1x D2D copy via PCIe)
```

---

## 六、test_flash_attn.py 瘦身效果

```python
# 重构后的 main：~40 行

if __name__ == "__main__":
    args = parse_args()
    cases = load_cases(args.case_files, args.case_id)

    if args.perf_mode:
        run_perf(cases, args)
        sys.exit(0)

    # 选择后端
    primary = GPUBackend() if args.use_gpu else NPUBackend()
    golden = CPUBackend()
    comparators = []

    if args.compare_mode:
        comparators = [GPUBackend()] if not args.use_gpu else [NPUBackend()]

    # 逐个 case 执行
    reporter = Reporter()
    for name, config in cases:
        result = run_case(
            spec=build_spec(config),
            primary=primary,
            golden=golden,
            comparators=comparators,
            dump_dir=args.dump_dir,
        )
        reporter.record(name, result)
        if _int_cnt % args.report_interval == 0:
            reporter.print_interim()

    reporter.print_final()
    sys.exit(0 if reporter.fail_cnt == 0 else 1)
```

---

## 七、迁移到其他算子的成本

以 `FusedInferAttention` 为例：

**现有方式**：复制 590 行 `test_flash_attn.py`，修改内部调用，不可复用。

**重构后**：

```python
# 第1步：定义 InputSpec（6 行）
fia_spec = (
    InputSpec("fia")
    .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D"})
    .tensor("k", layout={"BNSD": "B,N2,S2,D"})
    .tensor("v", layout={"BNSD": "B,N2,S2,_V"})
    .tensor("seqlen", shape="B", dtype=torch.int32)
)

# 第2步：实现 NPUBackend（50 行，复用 Backend 基类）
class FIA_Backend(NPUBackend):
    def compute(self, inputs, spec):
        return npu_ops_transformer.ops.npu_fused_infer_attention(
            inputs["q"], inputs["k"], inputs["v"], inputs["seqlen"],
            ...
        )

# 第3步：入口（20 行，直接复用 main 结构）
if __name__ == "__main__":
    ...
    run_case(spec=fia_spec, primary=FIA_Backend(), golden=CPUBackend())
```

总计：**~80 行** vs 现有 **590 行**。Input / Backend / Orchestrator / Reporter 全部复用。

---

## 八、实现计划

### Phase 1：抽象层（不改现有代码）

| 步骤 | 文件 | 说明 |
|---|---|---|
| 1 | `core/backends/base.py` | Backend ABC |
| 2 | `core/data.py` | DataSpec, TensorSpec, generate_inputs |

### Phase 2：后端实现

| 步骤 | 文件 | 说明 |
|---|---|---|
| 3 | `core/backends/npu.py` | NPUBackend（复用现有 npu_impl.py 逻辑） |
| 4 | `core/backends/gpu.py` | GPUBackend（复用现有 gpu_impl.py 逻辑） |
| 5 | `core/backends/cpu.py` | CPUBackend（复用现有 cpu_impl.py 逻辑） |

### Phase 3：编排层

| 步骤 | 文件 | 说明 |
|---|---|---|
| 6 | `core/orchestrator.py` | run_case 编排逻辑 |
| 7 | `core/case_loader.py` | 从 test_flash_attn 抽取 case 加载逻辑 |
| 8 | `core/reporter.py` | 终端表格 + CSV |

### Phase 4：入口瘦身

| 步骤 | 文件 | 说明 |
|---|---|---|
| 9 | `test_flash_attn.py` | 重写为薄入口，引用 orchestrator |
| 10 | `backends/` | 标记 deprecated，保留兼容 |

---

## 九、与现有 perf_mode 的关系

perf_mode 分支已经独立于精度测试路径，重构后：

- `perf_runner.py`：不改动（它自己生成脚本 + 调 msprof）
- `perf_parser.py`：不改动
- 精度测试的 DataSpec / Backend 抽象不影响 perf 路径

---

## 十、关键设计决策

| 决策 | 理由 |
|---|---|
| `shape_fn` 用 Callable 而非固定 shape | 同算子不同 layout 下 shape 不同，用函数延迟计算 |
| Backend 返回 device 上的 tensor，不做 .cpu() | 让调用方决定何时搬移，避免不必要拷贝 |
| `generate_inputs` 接受 device 参数 | 同一个 spec 可以在任意 device 上生成一致数据 |
| DataSpec 不耦合 flash_attn | `tensors` dict 可放任意 key（Q/K/V 或 input/hidden），适配不同类型算子 |
| 保留现有 `backends/` 目录 | 渐进迁移，不破坏现有脚本 |
