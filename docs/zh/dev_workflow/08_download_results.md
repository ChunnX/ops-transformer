# 08 本地下载与分析结果

## 目标

把远端的功能日志、profiling 和 Dump 数据拉回本地，保持原始数据与 Git commit 的对应关系。

## 查看远端运行记录

```bash
ssh "$OPS_REMOTE" \
  "find '$OPS_REMOTE_ROOT/results/$OPS_OPERATOR' -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort"
```

选择要下载的运行：

```bash
export OPS_RUN_ID="<run-id>"
export OPS_LOCAL_RESULT="$OPS_REPO/results/$OPS_OPERATOR/$OPS_RUN_ID"
mkdir -p "$OPS_LOCAL_RESULT"
```

## 下载

```bash
rsync -av --progress \
  "$OPS_REMOTE:$OPS_REMOTE_ROOT/results/$OPS_OPERATOR/$OPS_RUN_ID/" \
  "$OPS_LOCAL_RESULT/"
```

较大数据中断后可重复执行 rsync。如果只需日志和元数据，可排除原始数据：

```bash
rsync -av --progress \
  --exclude='profiler/' \
  --exclude='dump/' \
  "$OPS_REMOTE:$OPS_REMOTE_ROOT/results/$OPS_OPERATOR/$OPS_RUN_ID/" \
  "$OPS_LOCAL_RESULT/"
```

## 完整性检查

```bash
find "$OPS_LOCAL_RESULT" -maxdepth 2 -type f -printf '%P\n' | sort
cat "$OPS_LOCAL_RESULT/git-commit.txt"
cat "$OPS_LOCAL_RESULT/run-command.txt"
cat "$OPS_LOCAL_RESULT/exit-code.txt"
```

确认结果 commit 可在本地仓库中解析：

```bash
git -C "$OPS_REPO" cat-file -e \
  "$(cat "$OPS_LOCAL_RESULT/git-commit.txt")^{commit}"
```

## Git 管理边界

适合提交 Git：

- 采集和分析脚本。
- 小型 JSON/CSV 摘要。
- 性能对比表和分析结论。
- 原始数据的存储位置、SHA256 和 commit ID。

不适合提交 Git：

- profiler 原始数据。
- Dump 和 core 文件。
- 大型日志和二进制输出。
- 包含机器或用户敏感信息的环境快照。

若 `results/` 仅用于本地原始数据，将它加入个人排除文件 `.git/info/exclude`，不必修改共享 `.gitignore`。

## 下一步

分析后回到 [02 本地编译算子包](02_local_build_operator.md) 进入下一轮迭代。如有异常，查看 [09 常见问题](09_troubleshooting.md)。
