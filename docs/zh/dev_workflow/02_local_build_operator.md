# 02 本地编译算子包

## 目标

仅编译本次开发的算子，生成可传输到远端的自定义算子 `.run` 包。

## 前提条件

已完成 [01 本地环境检查](01_local_environment.md)，并设置：

```bash
echo "$OPS_REPO"
echo "$OPS_OPERATOR"
echo "$OPS_SOC"
echo "$OPS_VENDOR"
```

## 操作

确认源码改动：

```bash
cd "$OPS_REPO"
git status --short
git diff
```

编译指定算子：

```bash
source "$OPS_CANN_ENV"
bash build.sh \
  --pkg \
  --soc="$OPS_SOC" \
  --ops="$OPS_OPERATOR" \
  --vendor_name="$OPS_VENDOR" \
  -j16
```

调试构建可增加 `--debug`：

```bash
bash build.sh \
  --pkg \
  --debug \
  --soc="$OPS_SOC" \
  --ops="$OPS_OPERATOR" \
  --vendor_name="$OPS_VENDOR" \
  -j16
```

编译结束后查找产物：

```bash
find build_out -maxdepth 1 -type f -name '*.run' -ls
```

如果编译参数或产物名与文档不同，以当前分支的以下文档和 `build.sh --help` 为准：

- [源码构建](../install/compile.md)
- [build 参数说明](../install/build.md)

## 提交源码

编译产物不提交 Git，但生成产物的源码必须先形成可追溯的 commit：

```bash
git add <source-files> <test-files> <documentation-files>
git diff --cached
git commit -m "feat: update ${OPS_OPERATOR}"
git push -u origin HEAD
```

记录 commit：

```bash
git rev-parse HEAD
```

## 成功判据

- `build_out/` 中存在本次新生成的 `.run` 包。
- 编译日志中的 SoC 与 `OPS_SOC` 一致。
- 源码已提交，`git rev-parse HEAD` 可用于标识产物。
- `git status` 中没有被误加入的编译产物。

## 下一步

继续执行 [03 发布与上传产物](03_upload_artifact.md)。
