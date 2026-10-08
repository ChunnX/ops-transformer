# FlashKDA built-in integration

FlashKDA consumes reusable int32 scheduling metadata and supports contiguous
BF16 BNSD, BSND, and packed TND inputs, with `Dk = Dv = 128`. Initial/final state,
A_log, and dt_bias use FP32. The metadata producer is maintained by the arena;
it is not compiled or bundled by this operator. Metadata must match the batch,
value-head count, sequence layout and current stream core count.

The public ACLNN entry is `aclnnFlashKDAGetWorkspaceSize`, followed by
`aclnnFlashKDA`. Its arguments follow `flash_kda_aclnn` in
`op_kernel/flash_kda.py`, with caller-owned output and final-state tensors added
by OpKit. The host layer validates descriptors, maps each layout to logical
BNSD views, uses caller-owned logical output storage and executor-owned scratch.
The kernel clips tail writes to valid rows, so ACLNN outputs do not require
padded backing storage. One Native provider initializes the internal
16x16 identity and launches FlashKDA. Consumers need no Python at runtime.
Eager `flash_kda` selects the input NPU device and delegates to the same
`flash_kda_aclnn` host body. Validation, layout conversion, allocation, and
launch bindings have one implementation. The small
identity kernel remains because the OpKit host subset cannot lower `torch.eye`.

## Build

Install both wheels from `zyh-cannbot-dsl/build_out` in a Python 3.12 environment:

```bash
python3.12 -m venv --system-site-packages /tmp/flash-kda-build-env
/tmp/flash-kda-build-env/bin/python -m pip install --no-deps /path/to/build_out/*.whl
source /path/to/cann/set_env.sh
cmake -S . -B build-flash-kda \
  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON \
  -DASCEND_OP_NAME=flash_kda -DASCEND_COMPUTE_UNIT=ascend950 \
  -DCANNBOTDSL_PYTHON_EXECUTABLE=/tmp/flash-kda-build-env/bin/python \
  -DCANNBOTDSL_NNOPBASE_INCLUDE_DIR=/path/to/opbase/include/nnopbase \
  -DCANNBOTDSL_NNOPBASE_LIBRARY=/path/to/libnnopbase.so
cmake --build build-flash-kda -j8
cmake --build build-flash-kda --target package
```

The nnopbase headers must contain `opdev/direct_invoke_task.h`, and the matching
library must export `op::AddDirectInvokeTask`. This replaces the previous
External Task / frozen OperatorBundle bridge. A DSL source checkout and
`cannbot_framework` are no longer build dependencies. The stage cache includes
operator source, builder, wheel RECORD hashes, target and SDK identity.

Built-in artifacts:

- `opp/built-in/op_impl/cannbotdsl/<binary-key>/provider.so` and its metadata.
- `opp/built-in/op_api/lib/libopapi_cannbot_transformer.so`.
- `x86_64-linux/include/aclnnop/aclnn_flash_kda.h` (or the target processor).

Mixed AscendC/OpKit packages retain the mainline AscendC flow and attach the
helper DSO to the normal opapi library. An OpKit-only package skips AscendC
preparation and unrelated third-party builds. Unsupported explicit selections
fail at configuration; ALL selections skip OpKit on unsupported targets.

## Validation

```bash
python -m pytest attention/flash_kda/test/test_aclnn_adapter.py -q
FLASH_KDA_METADATA_SOURCE=/path/to/arena/samples/flash_kda_metadata/flash_kda_metadata.py \
  python -m pytest attention/flash_kda/test/test_flash_kda.py -q -s
```

The adapter tests cover layout/stride bindings, tail output slices, workspace
shapes and invalid-layout rejection. Precision tests require torch_npu, the
matching CANN runtime, an Ascend950 device and the arena metadata producer.

The matching DirectInvoke nnopbase must also be available at runtime. Linking
against it during the build does not replace the SDK's installed library.
For an isolated validation directory, select it with `LD_PRELOAD` and set
`FLASH_KDA_NNOPBASE_LIBRARY`; a product SDK should install the matching opbase.
The installed DSO integration tests additionally take
`FLASH_KDA_ACLNN_LIBRARY=/absolute/path/opp/built-in/op_api/lib/libopapi_cannbot_transformer.so`.

## Framework delta against upstream

`torch_extension/setup.py` now follows upstream, including operator locations
and exported filtering. The obsolete legacy `build_cannbotdsl` command and its
Python environment hook are removed. The bridge detects `@aclnn` sources and
uses OpKit's generated HostAPI and operator modules instead of frozen bundles.
The root CMake and `config.cmake` retain the separation between selected DSL
and AscendC operators, allowing a DSL-only package to skip AscendC preparation.
Mixed packages retain AscendC and attach the helper DSO in `symbol.cmake`.
`package.cmake` omits the unrelated torch-extension wheel for a DSL-only build;
OpsTransformer.xml delivers Native providers and the ACLNN DSO. The installer
only creates the TBE init file when that directory exists. Obsolete custom
vendor handling introduced by the previous bridge has been removed.
