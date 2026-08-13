# MegaMoe timer test helper

The timer input is optional. A normal test can enable tracing without changing
its data generation or precision comparison:

Instrumentation is active only for the Arch35 W4A8 paths. The A8W8
`MegaMoeWave` path is intentionally unchanged.

```python
from cann_ops_transformer.ops.mega_moe import mega_moe
from tests.megamoe_timer.timer_utils import create_mega_moe_timer, save_mega_moe_timer

timer = create_mega_moe_timer()
y, expert_tokens = mega_moe(..., timer=timer)
torch.npu.synchronize()
save_mega_moe_timer(timer, rank)
```

Allocate a fresh zeroed timer for every measured invocation. After all ranks
finish, parse the raw buffers and create a Chrome Trace file:

```bash
python3 tests/megamoe_timer/parse_timer_csv.py --dir timer --rank-start 0 --rank-end 7
python3 tests/megamoe_timer/trace_parser.py --timer-dir timer --rank-start 0 --rank-end 7
```

Open `timer/all_rank_trace.json` with `chrome://tracing` or Perfetto.
The timer uses 96 logical cores and supports up to 1024 local experts.
