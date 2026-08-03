# issue: third_party tensor_api 子模块无法拉取，阻塞 opkernel/pkg 构建

- **日期**: 2026-07-27
- **算子**: last_transpose_nddma_bench
- **严重级**: 阻塞（环境/基础设施，非算子缺陷）
- **触发命令**: `bash build.sh --opkernel --soc=ascend950 --ops=last_transpose_nddma_bench`

## 现象

`prepare_build` 阶段（在任何算子编译之前）失败：

```
fatal: fetch-pack: invalid index-pack output
fatal: clone of 'https://gitcode.com/cann/asc-devkit.git' into submodule path
  '.../third_party/ops-tensor/include/tensor_api' failed
Failed to clone 'include/tensor_api' a second time, aborting
CMake Error ... Failed to update submodules in '.../third_party/ops-tensor'
CMake Error ... Build step for ops-tensor failed: 2
Error: ops prepare build failed.
```

## 根因

commit 210c87f「replace third_party symlinks with cmake-driven git clone for shmem and
tensor_api」将 `tensor_api` 改为 CMake 驱动的 git 子模块拉取：

- `third_party/ops-tensor` 外层仓已存在 → `cmake/third_party/ops-tensor.cmake` 走 offline 分支，
  先执行 `git submodule update --init --recursive --no-fetch`。
- 本地对象库没有子模块 pin 提交 → `--no-fetch` 失败，回退到网络 `git submodule update`，
  但该命令带 `TIMEOUT 20`（ops-tensor.cmake:63）。
- 子模块 `include/tensor_api` 指向 `gitcode.com/cann/asc-devkit.git`，
  分支 `feature/tensor_api_from_9.0.0`，pin 提交 `d03329879e4cc388289b5d3632bbf4159743b225`。
- 本沙箱到 gitcode.com 的网络极慢：手动 `git clone` 该分支 180s 仍超时（EXIT=124）；
  仅能浅克隆 master tip，其中不含 pin 提交，故 offline 缓存无法预置。

结论：纯环境/网络阻塞，与算子代码无关。prepare 阶段在算子编译前即中断，
NDDMA arch35 门禁（VERIFY 步骤 1 的核心结论行）本环境无法执行。

## 影响

- VERIFY 步骤 1（opkernel 构建 / arch35-NDDMA 门禁）：无法执行
- VERIFY 步骤 2（pkg 构建）：无法执行（依赖同一 prepare）
- VERIFY 步骤 3（aclnn 二进制一致性）：无法执行（依赖 pkg）

## 可选解决方案（需具备 gitcode 访问或离线包的环境）

1. 在可访问 gitcode 的环境预置子模块，再拷贝整个 `third_party/ops-tensor`（含
   `.git/modules/include/tensor_api` 对象库 + `include/tensor_api` worktree）到本沙箱。
2. 提高 `cmake/third_party/ops-tensor.cmake:63` 的 `TIMEOUT 20` 并保证网络可达。
3. 提供 asc-devkit `feature/tensor_api_from_9.0.0` @ `d033298...` 的离线 tarball，
   解压到 `third_party/ops-tensor/include/tensor_api`。

## 算子侧状态

算子子树全部文件已创建且内部一致，无待改项；一旦第三方依赖就位即可直接跑 VERIFY 三步。
