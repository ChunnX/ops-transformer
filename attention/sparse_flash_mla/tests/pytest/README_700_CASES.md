# SparseFlashMla 700 Case 泛化测试

本文档说明 SparseFlashMla 的 700 条泛化用例运行方法。用例由脚本动态构造，不依赖外部 Excel 文件。

## 覆盖范围

- 计算路径：SWA、CFA、SCFA。
- Q layout：BSND、TND。
- KV layout：BSND、TND、PA_BBND。
- Head ratio：1、2、4、8、16、32、64、128。
- cmp_ratio：CFA/SCFA 覆盖 1、4、128。
- SCFA topk：512、1024。
- actual length：BSND `seqused_*`、TND `cu_seqlens_*`、多 batch 长短序列混合。
- residual：`cmp_residual_kv` 覆盖 0 和非 0。
- CP 切分模拟：cmp_ratio=4 下构造非 rank0 的 `cmp_sparse_indices`，前几行允许从有效 cmp id 开始。
- stride/越界：默认覆盖 KV 非连续 stride、cmp_kv 末尾 padding，并用 NaN padding 辅助发现越界读取。

## 一键运行

先完成 custom 包安装和 torch_extension 依赖准备，然后在仓库根目录执行：

```bash
DEVICE_ID=0 bash attention/sparse_flash_mla/tests/pytest/run_smla_700_cases.sh
```

结果文件默认写到：

```text
attention/sparse_flash_mla/tests/pytest/result/smla_700_cases.jsonl
attention/sparse_flash_mla/tests/pytest/result/smla_700_cases.summary.json
```

`summary.json` 中 `fail=0` 且 `error=0` 表示本轮通过。

## 常用调试命令

只跑前 20 条并遇到首个失败即退出：

```bash
CASE_END=20 STOP_ON_FAIL=1 DEVICE_ID=0 bash attention/sparse_flash_mla/tests/pytest/run_smla_700_cases.sh
```

只跑指定区间：

```bash
CASE_START=100 CASE_END=140 DEVICE_ID=0 bash attention/sparse_flash_mla/tests/pytest/run_smla_700_cases.sh
```

列出将要运行的用例，不真正执行算子：

```bash
python3 scripts/smla_generalization_test_runner.py \
  --out attention/sparse_flash_mla/tests/pytest/result/list.jsonl \
  --list-cases
```

## 兼容老 CANN 包

若所用 CANN 头文件未提供 stride 查询接口，源码会在编译期自动退回连续 stride 计算。该场景下 custom 包仅验证连续 KV stride：

```bash
CONTINUOUS_KV_ONLY=1 DEVICE_ID=0 bash attention/sparse_flash_mla/tests/pytest/run_smla_700_cases.sh
```

使用支持 stride 查询接口的 CANN 包时，直接运行默认命令即可覆盖非连续 `ori_kv`/`cmp_kv` stride。
