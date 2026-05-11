# QAR/QRS 编译和测试流程

## 编译

```bash
# 1. 激活环境
source ~/env.sh

# 2. 编译（指定算子）
cd ~/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=quant_all_reduce,quant_reduce_scatter

# 3. 安装
cd ~/ops-transformer/build_out
bash cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=$HOME/SUB

# 4. 激活子包环境
source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash
```

## 跑前检查

```bash
# 确认 4 卡空闲，有占用先 kill
npu-smi info | tail -12
```

## 精度测试

```bash
cd ~/mc2_test_0416

# QRS level0（24 用例）
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx \
    --sheet level0 --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D

# QAR level0（24 用例）
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx \
    --sheet level0 --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D
```

## 性能测试

```bash
# 跑前清 csv（mc2_test 是 append 模式）
rm -f aclnnQuantReduceScatter_performance_result.csv \
      aclnnQuantReducescatter_result.csv \
      mc2_aclnnQuantReduceScatter_pro_result.csv

# QRS
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx \
    --sheet level0 --num 1 24 --scene one_hccl --world_size 4 --soc 910D

rm -f aclnnQuantAllReduce_performance_result.csv \
      aclnnQuantAllreduce_result.csv \
      mc2_aclnnQuantAllReduce_pro_result.csv

# QAR
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx \
    --sheet level0 --num 1 24 --scene one_hccl --world_size 4 --soc 910D
```

## scene 说明

| scene | 用途 |
|-------|------|
| `one_hccl_npu` | 仅精度（对比 golden） |
| `one_hccl` | 精度 + 性能采集 |
| `one_hccl_cpu` | 生成 golden |

## 调参

xPerBlock 对齐粒度由 tiling 控制，kernel 自动读取：

```cpp
// quant_all_reduce_tiling.cpp / quant_reduce_scatter_tiling.cpp
constexpr uint32_t TARGET_ITER = 3U;
constexpr uint32_t MIN_BLOCK = 2048U;
constexpr uint32_t ALIGN_BLOCK = 1024U;  // 改这里
```
