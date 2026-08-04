# GMM V5 非量化 Ascend950 上板任务

## 目标

在服务器 B 的 Ascend950PR、CANN 9.2.0 环境中完成以下闭环：

1. 编译 `grouped_matmul` 自定义算子包；
2. 安装到本次任务的独立目录；
3. 编译并运行 arch35 的 GMM V5 非量化 aclnn 样例；
4. 保存各阶段日志、退出码、安装包 SHA256 和诊断压缩包；
5. 恢复服务器 B 执行前的分支、tracked 修改和未跟踪文件。

## 固定配置

```text
operator: grouped_matmul
SoC: ascend950
CANN: /usr/local/Ascend/cann-9.2.0/set_env.sh
vendor: xwqtju_dev
example: gmm/grouped_matmul/examples/arch35/test_aclnn_grouped_matmul_v5_noquant.cpp
```

打包阶段固定使用 `-j1`。服务器 B 在 `-j2` 下仍出现内核编译进程 `Killed` 和 `Error 137`；串行编译用于规避整机之外的任务或 cgroup 资源限制。

## 服务器 B 执行

服务器 A 每次交付时会提供完整 40 位 commit。服务器 B 只粘贴一条命令：

```bash
cd /home/x00961093/cann/ops-transformer && bash scripts/dev_workflow/server_b_task.sh --repo /home/x00961093/cann/ops-transformer --commit <完整40位commit> --task scripts/dev_tasks/gmm_v5_noquant/task.sh
```

执行器会自动暂存 B 的本地修改、通过 HTTPS fetch、切换到精确 commit、执行任务并恢复现场。

## 成功判据

终端及 `summary.log` 应包含：

```text
[INFO] phase build exit code: 0
[INFO] phase install exit code: 0
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
tail -200 <result-dir>/build.log
tail -200 <result-dir>/install.log
tail -200 <result-dir>/run.log
```

如果 build 阶段失败，`install.log` 和 `run.log` 不会生成；如果 install 阶段失败，`run.log` 不会生成。

如果 build 和 install 已成功，仅 run 阶段失败，可复用已有安装目录而不重新编译：

```bash
bash scripts/dev_workflow/server_b_task.sh --repo /home/x00961093/cann/ops-transformer --commit <完整40位commit> --task scripts/dev_tasks/gmm_v5_noquant/task.sh --resume-run-dir <此前成功安装的result-dir>
```

## 相关文件

- 任务定义：`scripts/dev_tasks/gmm_v5_noquant/task.sh`
- 通用执行器：`scripts/dev_workflow/server_b_task.sh`
- GMM V5 接口说明：`gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md`
