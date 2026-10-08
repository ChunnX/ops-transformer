# Flash MLA：DeepSeek CUDA 三方对比

通过 TTK 的 `third_party` 类接口调用 DeepSeek `flash_mla.flash_mla_with_kvcache`。加 `--compare cross_check` 后，对比 NPU、CPU FP32 golden 和 CUDA 三方结果。普通运行保留原有混合精度 golden 和定制 comparator。

## 运行

CUDA 服务端需安装 [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA)，并按 TTK 的 xpu-server 文档启动支持 `torch` provider 的 CUDA 服务。本地配置 `ttk.conf.yaml` 的 `remote.endpoints` 指向该服务；例如：

```yaml
remote:
  endpoints:
    - host: YOUR_GPU_HOST
      port: 9090
```

在 NPU 侧使用已安装主算子和 metadata 算子的环境运行：

```bash
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin attention/flash_mla_with_kvcache/tests/assets \
  --compare cross_check --provider torch --config /path/to/ttk.conf.yaml \
  --warmup false --run 1 --pc 1 \
  -o result.csv
```

可加 `--aclgraph` 测试 NPU 图模式。CUDA 依赖延迟导入，普通 NPU 用例不要求安装 `flash_mla`。

## 语义与限制

- 支持 FP16/BF16，Q 为 TND/BNSD/BSND，KV 为 PA_BBND/PA_BNBD/PA_NZ；D=576、DV=512、单 KV head。
- 支持 mask_mode=0 和右对齐 causal 的 mask_mode=3。mode=0 的非零显式 mask、mode=4、sinks 不支持并报错。
- 按每个 batch 的实际 Q 长度调用 CUDA，不将 padding 加入 causal 对齐。支持 seqused_q、零有效 Q 和空 KV。TND 可输出 NTD，其他布局可输出 BNSD/BSND/NBSD。
- DeepSeek dense decode 要求 KV 页长 64。适配层将 64 的整倍数页拆分，重映射实际使用的 block_table；其他页长明确报错。只检查实际引用的 page，未使用的 -1 不参与寻址。
- NPU metadata 不传给 DeepSeek；每个 batch 用 `get_mla_metadata` 获取 CUDA scheduler。兼容旧式 scheduler/num_splits 元组及新式 FlashMLASchedMeta/None 元组。
- padding、空 KV 和 causal 全遮挡行按本算子契约返回 output=0、LSE=+Inf；只规范这些由长度确定的行，可见行保留 CUDA 原始结果，包括 NaN/Inf。
- 三方模式的 CPU golden 将原始 FP16/BF16 Q/KV 转为 FP32，不重新生成输入，P 和最终输出不再舍入到输入 dtype。CPU 运算及分块仍由当前 golden 实现负责。
- 比较使用 TTK 内置 cross_check，沿用 SparseFlashAttention 的 L0 配置：mare/mere/rmse 比率 10/2/2，small_value=2^-10、small_value_atol=2^-16；LSE 的 float32 输出同样走三方比较。普通模式仍使用原有主输出标准及 LSE atol=0.1。
- cross_check 模式下关闭双路定制 comparator，避免它忽略 CUDA 结果；CUDA 服务或第三方结果缺失时由 TTK 报失败，不回退成普通双路 PASS。
- 逐 batch 调用、布局整理和 scheduler 生成均在适配层执行；`--xpu-perf` 的适配器耗时不能视作原生单次 batched FlashMLA kernel 的性能。

接口核对依据：DeepSeek FlashMLA `ba89a34` 的 `flash_mla/flash_mla_interface.py` 和 `csrc/api/dense_decode.cpp`。GPU 架构和 CUDA 版本需满足所安装 FlashMLA 的要求。

## 本次验证

本地适配及 TTK 路由测试 79 项通过，包含三种 Q/KV 布局、FP16/BF16、64/128 页、变长与空行、普通/三方切换、禁用 LSE，以及内置 cross_check 的成功、超差和缺第三方结果分支。CUDA 调用在这些测试中用独立 CPU attention 替身检查参数和布局，不代表 CUDA kernel 真机验证。

使用 CANN 0916 做 NPU Eager 回归：`Decode_FlashMla_1_96_1_7_38048_576_L0_46` PASS，主输出/LSE 均 100%。本机无 CUDA，尚未完成真实 GPU 服务端执行和端到端三方精度验证。
