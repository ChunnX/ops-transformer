# Catlass vendored source for ChunkKdaFwd

本目录为 ChunkKdaFwd 编译所需的 Catlass/TLA 头文件源码快照，来源为
`https://gitcode.com/cann/catlass`，固定提交：

`83073fedcac3d9df5b83f701b2cc00ec6a94f984`（v1.5.0）

对应上游目录 `include/` 中的 `catlass/` 与 `tla/` 两个命名空间，直接复制自该提交，
未做任何修改，放置位置参考 `attention/block_sparse_attention/op_kernel` 的做法
（`op_kernel/catlass`、`op_kernel/tla`）。ChunkKdaFwd 的 kernel 编译通过
`op_host/CMakeLists.txt` 中的 `CHUNK_KDA_CATLASS_INCLUDE_DIR`（指向 `op_kernel`）
引用本目录，不再依赖公共 `3rdparty/catlass` 子模块。

如需升级，请整体替换 `op_kernel/catlass`、`op_kernel/tla` 下的内容并同步更新上方提交号。
