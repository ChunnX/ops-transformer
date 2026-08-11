# AllToAllV GroupedMatMul SHMEM PTA

This directory contains a standalone torch bridge and a deterministic multi-process PTA driver for the Ascend950 SHMEM `aclnnAlltoAllvGroupedMatMulV2` interface. It does not create an HCCL process group and does not call op-plugin's HCCL V1 wrapper.

## Prerequisites

- A Linux Ascend950 host with matching CANN, torch, and torch_npu versions.
- The migrated custom operator package installed, including the current V2 header and `libcust_opapi.so`.
- A CANN SHMEM SDK using either the source layout or the HCCL-integrated layout.
- One worker process and one unique NPU device for every SHMEM rank.

Set the build environment after sourcing the normal CANN environment script:

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
export CUST_OPAPI_LIBRARY=/path/to/libcust_opapi.so

# Choose exactly one SHMEM layout.
export SHMEM_HOME_PATH=/path/to/source-shmem-sdk
# export SHMEM_HOME=/path/to/hccl-integrated-shmem

# Set this only when the V2 header is not discoverable next to the library
# or below ASCEND_HOME_PATH/include.
export CUST_OPAPI_INCLUDE_DIR=/path/to/op_api/include
export PYTHON_BIN=python3
```

The selected Python must import both `torch` and `torch_npu`. The custom op-api library must export the ABI declared by `aclnn_allto_allv_grouped_mat_mul_v2.h` in this branch.

## Build

From this directory:

```bash
bash build_pta.sh
```

The default output is:

```text
build/pta/liballto_allv_grouped_mat_mul_shmem_pta.so
```

Use `BUILD_DIR` to change the output directory or `BUILD_JOBS` to change parallelism. `build_pta.sh` fails before compilation when a required include directory or library cannot be found.

## Run

The default validation uses two ranks, local devices 0 and 1, two local experts, BF16 `K=32`, `N=64`, and both balanced and uneven routes:

```bash
bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case all
```

The rendezvous defaults to `tcp://127.0.0.1:27101`. Override it through either form:

```bash
export SHMEM_IP_PORT=tcp://127.0.0.1:27102
bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case all

bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case all \
    --ip-port tcp://127.0.0.1:27103
```

Topology variants use the same script:

```bash
bash run_pta_test.sh --world-size 1 --device-ids 0 --case all
bash run_pta_test.sh --world-size 4 --device-ids 0,1,2,3 --case all
bash run_pta_test.sh --world-size 8 --device-ids 0,1,2,3,4,5,6,7 --case all
```

Run a single route pattern or change the smoke shape with:

```bash
bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case uneven \
    --local-experts 2 --k 32 --n 64 --tokens-per-route 2
```

The driver keeps every source rank's input row count identical. Do not manually change route generation so that symmetric input allocation sizes differ across ranks.

## Result interpretation

Each worker prints `rank/case/phase` markers for initialization, input creation, launch, verification, barriers, and finalization. The parent prints one JSON result per rank. A passing run ends with:

```text
PASS SHMEM PTA world_size=2 cases=all
```

Each case result contains the output shape, expected receive rows, maximum absolute error, maximum and mean relative error, normalized mean bias, and mismatched element count. Success requires:

- `output_shape == [sum(recv_counts), N]`;
- expert-major/source-rank-minor output ordering;
- every BF16 element within the source verifier threshold (`2^-7` for `K<2048`, otherwise `2^-6`);
- normalized mean bias no larger than `2^-7`;
- zero worker failures and zero nonzero process exit codes.

Any dynamic-library load error, parameter error, SHMEM/aclnn error, non-finite result, precision mismatch, missing worker result, or timeout returns nonzero.

## Troubleshooting

- `Missing PTA library`: run `build_pta.sh` or set `PTA_LIBRARY` to the absolute `.so` path.
- Undefined V2 symbol while loading: confirm `CUST_OPAPI_LIBRARY` belongs to this branch's installed operator package and is visible through its RPATH or `LD_LIBRARY_PATH`.
- V2 header not found: set `CUST_OPAPI_INCLUDE_DIR` to the directory containing the `aclnnop` subdirectory.
- SHMEM include/library not found: verify the selected `SHMEM_HOME_PATH` or `SHMEM_HOME` layout; do not set both to unrelated SDKs.
- `aclshmem_malloc` or `aclshmem_calloc` failure: keep `--heap-bytes` at least 1004 MiB and verify all ranks use identical input byte sizes and allocation order.
- Collective timeout: inspect the last `rank/case/phase` line for every worker. Increase `PTA_TIMEOUT_SECONDS` only after ruling out rank, device, rendezvous, ABI, and allocation mismatches.
- Port collision: choose another loopback rendezvous with `--ip-port` or `SHMEM_IP_PORT`.

The wrapper bounds the whole run with GNU `timeout`. The default is 600 seconds and can be changed explicitly:

```bash
PTA_TIMEOUT_SECONDS=900 bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case all
```

## Offline verification limits

On a workstation without matching CANN, torch_npu, SHMEM, and Ascend950 devices, only Python syntax, pure route self-checks, shell syntax, and repository static contracts can be validated. Those checks do not prove that the bridge links, loads, launches, or passes multi-card precision. Perform the Build and Run commands above on the target machine before reporting hardware success.
