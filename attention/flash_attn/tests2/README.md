# FA 精度测试框架

面向昇腾NPU算子的通用精度测试框架，从4个既有FA测试框架提取最优方案移植。

## 快速开始

```bash
# CLI模式
python test.py --op flash_attn --stages gen,cpu,npu,compare

# pytest模式
pytest test.py --op flash_attn --stages gen,cpu,npu,compare

# golden独立CLI
python test.py --op flash_attn --stages gen,cpu
```

## 目录结构

- `test.py` — 统一入口(CLI main + test函数 + fixtures)
- `conftest.py` — 极薄pytest hooks
- `pytest.ini` — marker静态注册
- `core/` — 框架骨架(orchestrator/case_loader/compare/reporter + ABC基类)
- `ops/{op}/` — 算子适配(paramset/gen_data/cpu/npu/gpu/quant)
- `data/` — pt中间产物
- `result/` — 测试结果(compare时自动创建)
- `reports/` — CSV解析工具
- `precision_analyze.py` — 失败分析
- `prof/` — 性能采集虚函数接口

## 新算子适配

1. 创建 `ops/{op}/paramset.py` 定义参数+用例集
2. 实现 `ops/{op}/gen_data.py` 中 GenDataBackend
3. 实现 `ops/{op}/cpu.py` 中 CpuBackend
4. 实现 `ops/{op}/npu.py` 中 NpuBackend
5. (可选) 实现 gpu/quant/forward门面

框架核心无需修改即可接入新算子。

## 设计文档

见 `designs/` 目录。
