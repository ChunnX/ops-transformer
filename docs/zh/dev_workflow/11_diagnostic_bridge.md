# 11 服务器 B 报错回传链路

## 目标

两台服务器无法通过 SSH 互访时，使用私有 GitCode 仓库 `xwqtju/ops-transformer-results` 作为诊断中转站：

```text
服务器 B --from-server-b--> GitCode --> 服务器 A
服务器 A --from-server-a--> GitCode --> 服务器 B
```

源码和修复仍通过 `xwqtju/ops-transformer` 传递；诊断仓库只保存小型、已过滤的报错资料。

## 服务器 B 首次认证

为服务器 B 创建独立密钥，不要复制服务器 A 的私钥：

```bash
ssh-keygen -t ed25519 \
  -C "ops-transformer-server-b" \
  -f /root/.ssh/id_ed25519_gitcode_results \
  -N ''

cat /root/.ssh/id_ed25519_gitcode_results.pub
```

把显示的公钥添加到 GitCode 账号，然后测试：

```bash
ssh -i /root/.ssh/id_ed25519_gitcode_results \
  -o IdentitiesOnly=yes \
  -T git@gitcode.com
```

## 服务器 B 准备双向目录

```bash
export OPS_BASE="/home/x00961093/cann"
export OPS_SOURCE="$OPS_BASE/ops-transformer"
export OPS_OUTBOX="$OPS_BASE/ops-transformer-results-from-b"
export OPS_INBOX="$OPS_BASE/ops-transformer-results-from-a"
export OPS_RESULTS_SSH="ssh -i /root/.ssh/id_ed25519_gitcode_results -o IdentitiesOnly=yes"

git -c core.sshCommand="$OPS_RESULTS_SSH" clone \
  --branch from-server-b \
  git@gitcode.com:xwqtju/ops-transformer-results.git \
  "$OPS_OUTBOX"

git -c core.sshCommand="$OPS_RESULTS_SSH" clone \
  --branch from-server-a \
  git@gitcode.com:xwqtju/ops-transformer-results.git \
  "$OPS_INBOX"

git -C "$OPS_OUTBOX" config core.sshCommand "$OPS_RESULTS_SSH"
git -C "$OPS_INBOX" config core.sshCommand "$OPS_RESULTS_SSH"
```

两个目录分别负责发送和接收，避免双方同时写同一分支。

## 服务器 B 一键执行并回传错误

```bash
cd "$OPS_SOURCE"

scripts/dev_workflow/report_error.sh \
  --results-repo "$OPS_OUTBOX" \
  --source-repo "$OPS_SOURCE" \
  --server-id server-b \
  -- <实际命令> <实际参数>
```

编译示例：

```bash
scripts/dev_workflow/report_error.sh \
  --results-repo "$OPS_OUTBOX" \
  --source-repo "$OPS_SOURCE" \
  --server-id server-b \
  -- bash build.sh --pkg --soc=ascend910b --ops=<operator_name>
```

脚本会执行原命令并自动保存：

- stdout、stderr 和真实退出码；
- 源码 commit、分支和工作区状态；
- 命令、时间、CPU 架构和 CANN 路径；
- `npu-smi info`；
- 执行前后的 diff 统计，不包含完整源码 diff。

随后提交并推送到 `from-server-b`，最后输出事件编号。脚本自身返回原命令的退出码，因此失败不会被误报为成功。

## 服务器 A 读取报告

服务器 A 首次创建只读工作目录：

```bash
git -c core.sshCommand='ssh -i /home/user/.ssh/id_ed25519_gitcode -o IdentitiesOnly=yes' clone \
  --branch from-server-b \
  git@gitcode.com:xwqtju/ops-transformer-results.git \
  "/home/user/桌面/workspace/cann/ops-transformer-results-from-b"
```

以后获取最新报告：

```bash
git -C "/home/user/桌面/workspace/cann/ops-transformer-results-from-b" \
  pull --ff-only origin from-server-b
```

## 服务器 A 发布修复指令

修复代码推送到 `ops-transformer` 后，在结果仓库的 `from-server-a` 分支创建：

```text
responses/server-a/<incident-id>.md
```

内容至少写明源码分支、目标 commit、重新执行命令和注意事项，再提交并推送 `from-server-a`。

## 服务器 B 获取处理结果

```bash
cd "$OPS_SOURCE"

scripts/dev_workflow/sync_fix.sh \
  --inbox-repo "$OPS_INBOX" \
  --incident <incident-id>
```

该脚本只显示指令，不会自动丢弃本地修改或切换源码 commit。

## 安全与容量限制

脚本不会保存完整环境变量或完整源码 diff。提交前仍应检查报告目录，确保没有业务数据、Token、密码或私钥。

普通 Git 只用于文本日志和小型摘要。`.run` 包、core、Dump 和大型 profiler 数据必须使用对象存储或 Git LFS；诊断仓库只记录其位置、SHA256、源码 commit 和事件编号。
