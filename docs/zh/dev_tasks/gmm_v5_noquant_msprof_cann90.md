# GMM V5 非量化 Ascend950 CANN 9.0 msprof op 采集任务

## 目标

在服务器 B 的 Ascend950PR 上，完全使用同事的 CANN 9.0.0 环境重新编译、安装并执行 GMM，再完成 `msprof op` 上板性能采集：

1. 使用 CANN 9.0.0 完整编译并安装 `grouped_matmul` 自定义算子包；
2. 编译并运行 arch35 的 GMM V5 非量化 aclnn 样例，确认功能正常；
3. 使用 `msprof op` 包裹样例可执行文件进行真机采集；
4. 将 `OPPROF_*`、日志和诊断压缩包保存在本次结果目录；
5. 恢复服务器 B 执行前的分支、tracked 修改和未跟踪文件。

## 固定配置

```text
operator: grouped_matmul
SoC: ascend950
CANN: /home/l00803537/CANN90/cann-9.0.0/set_env.sh
vendor: xwqtju_cann90
example: gmm/grouped_matmul/examples/arch35/test_aclnn_grouped_matmul_v5_noquant.cpp
```

打包阶段固定使用 `-j1`。服务器 B 在 `-j2` 下仍出现内核编译进程 `Killed` 和 `Error 137`；串行编译用于规避整机之外的任务或 cgroup 资源限制。

## 服务器 B 执行

服务器 A 每次交付时会提供完整 40 位 commit。服务器 B 只粘贴一条命令：

```bash
cd /home/x00961093/cann/ops-transformer && bash scripts/dev_workflow/server_b_task.sh --repo /home/x00961093/cann/ops-transformer --commit <完整40位commit> --task scripts/dev_tasks/gmm_v5_noquant_msprof_cann90/task.sh
```

执行器会自动暂存 B 的本地修改、通过 HTTPS fetch、切换到精确 commit、执行任务并恢复现场。

## 成功判据

终端及 `summary.log`、`run.log` 应包含：

```text
[INFO] phase build exit code: 0
[INFO] phase install exit code: 0
run test_aclnn_grouped_matmul, execute samples success
[PROFILE] msprof op result: <result-dir>/profiling/OPPROF_*
[INFO] phase run exit code: 0
[OK] build, install and runner completed
```

`run.log` 应包含：

```text
run test_aclnn_grouped_matmul, execute samples success
```

## 失败时返回服务器 A

命令结束时会打印 `<result-dir>`。复制以下内容：

```bash
cat <result-dir>/summary.log
cat <result-dir>/metadata.txt
tail -300 <result-dir>/build.log 2>/dev/null
tail -300 <result-dir>/install.log 2>/dev/null
tail -300 <result-dir>/run.log 2>/dev/null
find <result-dir>/profiling -maxdepth 2 -type f -print
```

首次执行会使用 CANN 9.0.0 完整编译和安装，预计耗时较长。只有 build 和 install 成功而 run 阶段失败时，才可使用 `--resume-run-dir` 复用本次安装目录：

```bash
bash scripts/dev_workflow/server_b_task.sh --repo /home/x00961093/cann/ops-transformer --commit <完整40位commit> --task scripts/dev_tasks/gmm_v5_noquant_msprof_cann90/task.sh --resume-run-dir <此前成功安装的result-dir>
```

## 相关文件

- CANN 9.0 Profiling 任务定义：`scripts/dev_tasks/gmm_v5_noquant_msprof_cann90/task.sh`
- 通用执行器：`scripts/dev_workflow/server_b_task.sh`
- GMM V5 接口说明：`gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md`
