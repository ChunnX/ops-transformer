# ops-transformer 仓编译运行指南

## 环境信息

| 项目 | 值 |
|------|-----|
| CANN 版本 | 9.2.0 |
| CANN 路径 | /home/l00947526/Ascend806/cann-9.2.0 |
| NPU 架构 | Ascend 950PR (dav-3510) |
| 编译器 | bisheng (CANN 9.2.0) |
| 运行时库冲突 | /usr/local/Ascend/cann-9.1.0 (系统默认, 不兼容) |

## 编译

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0

cd mc2/common/op_kernel/apace/tests/st/apace_quant_matmul_reduce_scatter
rm -rf build && mkdir build && cd build
cmake -S ../../.. -B .
cmake --build . --target apace_quant_matmul_reduce_scatter --parallel 4

# 拷贝二进制到 build/ 根目录 (main.cpp 用 readlink 定位 input/output)
cp apace_quant_matmul_reduce_scatter/apace_quant_matmul_reduce_scatter quant_matmul_reduce_scatter_st
```

## 运行时环境（关键！）

ST 仓的 `find_package(ASC)` 链接了 `/usr/local/Ascend/cann-9.1.0` 的运行时库（`libplatform.so` 等），
与 CANN 9.2.0 编译的 kernel 不兼容，导致 AICore 507015 错误。

**必须设置 LD_LIBRARY_PATH 指向 CANN 9.2.0**：

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0
export LD_LIBRARY_PATH=/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64:/home/l00947526/Ascend806/cann-9.2.0/lib64:$LD_LIBRARY_PATH
```

## 数据路径

- `main.cpp` 用 `readlink("/proc/self/exe")` 定位 input/output
- 二进制必须在 `build/` 根目录下（非 `build/apace_quant_matmul_reduce_scatter/` 子目录）
- input 拷到 `build/input/{rank}/`
- output 在 `build/output/{rank}/`

## 精度验证

### 方式 1: run.sh

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0
export LD_LIBRARY_PATH=/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64:/home/l00947526/Ascend806/cann-9.2.0/lib64:$LD_LIBRARY_PATH

cd mc2/common/op_kernel/apace/tests/st/apace_quant_matmul_reduce_scatter
bash run.sh --skip-build          # 跑 cases.csv 全部用例
bash run.sh --skip-build --cli 2048 3584 4096 2 512  # 单个用例
```

### 方式 2: 直接命令行

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0
export LD_LIBRARY_PATH=/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64:/home/l00947526/Ascend806/cann-9.2.0/lib64:$LD_LIBRARY_PATH

cd mc2/common/op_kernel/apace/tests/st/apace_quant_matmul_reduce_scatter
rm -rf input output build/input build/output
python3 scripts/gen_data.py 2048 3584 4096 2
cp -r input build/input
for r in 0 1; do mkdir -p build/output/$r output/$r; done
cd build
./quant_matmul_reduce_scatter_st 2048 3584 4096 2 precision 512
cd ..
for r in 0 1; do cp build/output/$r/npu_out.bin output/$r/npu_out.bin; done
python3 scripts/verify_result.py 2048 4096 2 ./output
```

### 方式 3: qrun 提交（NPU 排队）

```bash
qrun -n y00845930-精度验证 -t 600 "export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0 && export LD_LIBRARY_PATH=/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64:/home/l00947526/Ascend806/cann-9.2.0/lib64:\$LD_LIBRARY_PATH && cd /path/to/st_dir && PASS=0; FAIL=0; while IFS=, read m k n r h; do [[ \$m == m ]] && continue; echo \"=== Case M=\$m K=\$k N=\$n R=\$r H=\$h ===\"; rm -rf input output build/input build/output; python3 scripts/gen_data.py \$m \$k \$n \$r 2>&1 | tail -1; cp -r input build/input; for i in \$(seq 0 \$((r-1))); do mkdir -p build/output/\$i output/\$i; done; cd build && timeout 30 ./quant_matmul_reduce_scatter_st \$m \$k \$n \$r precision \$h 2>&1 | grep -E 'Step 13|aclError'; cd ..; for i in \$(seq 0 \$((r-1))); do cp build/output/\$i/npu_out.bin output/\$i/npu_out.bin 2>/dev/null; done; if python3 scripts/verify_result.py \$m \$n \$r ./output 2>&1 | grep -q 'ALL PASS'; then echo 'RESULT: PASS'; PASS=\$((PASS+1)); else echo 'RESULT: FAIL'; FAIL=\$((FAIL+1)); fi; echo ''; done < cases.csv; echo \"=== SUMMARY: PASS=\$PASS FAIL=\$FAIL ===\""
```

## 性能采集 (msprof)

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0
export LD_LIBRARY_PATH=/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64:/home/l00947526/Ascend806/cann-9.2.0/lib64:$LD_LIBRARY_PATH
source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh

cd mc2/common/op_kernel/apace/tests/st/apace_quant_matmul_reduce_scatter
msprof --output=msprof_data \
       --ai-core=on \
       --aic-mode=task-based \
       --aic-metrics=PipeUtilization \
       --task-time=on \
       --ascendcl=on \
       build/quant_matmul_reduce_scatter_st 2048 3584 4096 2 perf 512
```

## 已知问题

### 507015 AICore 错误

**原因**：ST CMakeLists.txt 的 `find_package(ASC)` 从 `/usr/local/Ascend/cann-9.1.0` 链接了运行时库，与 CANN 9.2.0 编译的 kernel 不兼容。

**解决方案**：运行时设置 `LD_LIBRARY_PATH` 指向 CANN 9.2.0 的 lib64 目录。

**验证**：`ldd build/quant_matmul_reduce_scatter_st | grep libplatform` 应显示 CANN 9.2.0 路径。

### run.sh target 名不匹配

ST 的 cmake target 名是 `apace_quant_matmul_reduce_scatter`，但 `run.sh` 期望 `quant_matmul_reduce_scatter_st`。
已在 `run.sh` 中修复：编译后自动拷贝二进制到 `build/quant_matmul_reduce_scatter_st`。

## 验证结果

- **精度**：5/5 代表用例 ALL PASS（M=2048 R=2, M=2048 R=4, M=4096 R=2, M=1024 R=2, M=4096 R=4）
- **性能**：相比旧实现平均 2.01x 加速（自适应 headMSize）
- **日期**：2026-08-18
- **commit**：自适应 headMSize 优化

## 自适应 headMSize 策略

| rankNum | headMSize | commTurn | 说明 |
|---------|-----------|----------|------|
| R<=2    | 512       | 2~4      | 大 tile, 少 SyncAll, 通信效率高 |
| R>=4    | 128       | 4~8      | 小 tile, 多 commTurn, 流水掩盖效果好 |

## 性能数据 (msprof cross_rank_max, us)

| Case | 旧实现 | 自适应 | 加速比 |
|------|--------|--------|--------|
| M=2048 K=3584 N=4096 R=2 H=512 | 680 | 304 | 2.24x |
| M=2048 K=3584 N=4096 R=4 H=512 | 412 | 243 | 1.70x |
| M=4096 K=3584 N=4096 R=2 H=512 | 1381 | 548 | 2.52x |
| M=1024 K=3584 N=4096 R=2 H=512 | 362 | 190 | 1.91x |
| M=4096 K=3584 N=4096 R=4 H=512 | 736 | 438 | 1.68x |
