# Lightning Attention

## Overview

Lightning Attention Forward and Backward pass operators

## Hardware support

| Chip version | Support |
| ------------ | :-----: |
| Atlas A3     |    X    |
| Atlas A2     |    √    |

## Structure

```
|-- experimental
|   |-- attention                            
|   |   |-- CMakeLists.txt                   
|   |   |-- lightning-attention       
|   |       |-- common                                    //tiling
|   |       |-- op_host                   
|   |       |-- op_kernel
|   |           |-- lightning_attn_cmm128_fwd_kernel.h    //forward pass operator  
|   |           |-- lightning_attn_cmm128_bwd_kernel.h    //backward pass operator               
|   |       |-- tests                    
|   |       |-- CMakeLists.txt                       
|   |       |-- torch_interface.cpp                       //pytorch registration                     

```

## Parameters

The solution consists of two kernels for forward and backward passes. The forward kernel is called LaForwardCMM128 and takes the following arguments:

| Argument   |  Type   |                          Description                          |
| ---------- | :-----: | :-----------------------------------------------------------: |
| query      | GM_ADDR |        Global memory address of the Query (Q) matrix.         |
| key        | GM_ADDR |         Global memory address of the Key (K) matrix.          |
| value      | GM_ADDR |        Global memory address of the Value (V) matrix.         |
| slopeRate  | GM_ADDR |          Global memory address of the slope matrix.           |
| kvStateOut | GM_ADDR | Output address for inter-chunk KV stats (saved for training). |
| resultOut  | GM_ADDR |      Global memory address for the forward pass output.       |
| workspace  | GM_ADDR |     Global memory address for temporary scratchpad space.     |
| tiling     | GM_ADDR |     Global memory address for tiling configuration data.      |

The backward kernel is called LaBackwardCMM128 with the following arguments

| Argument     |  Type   |                       Description                        |
| ------------ | :-----: | :------------------------------------------------------: |
| gradResult   | GM_ADDR |    Global memory address of the output gradient (dO).    |
| query        | GM_ADDR |        Global memory address of the Query matrix.        |
| key          | GM_ADDR |         Global memory address of the Key matrix.         |
| value        | GM_ADDR |        Global memory address of the Value matrix.        |
| slopeRate    | GM_ADDR |        Global memory address of the slope matrix.        |
| kvState      | GM_ADDR | Global memory address of the KV state from forward pass. |
| gradQueryOut | GM_ADDR |    Output address for the calculated Query gradient.     |
| gradKeyOut   | GM_ADDR |     Output address for the calculated Key gradient.      |
| gradValueOut | GM_ADDR |    Output address for the calculated Value gradient.     |
| workspace    | GM_ADDR |  Global memory address for temporary scratchpad space.   |
| tiling       | GM_ADDR |   Global memory address for tiling configuration data.   |

A tiling structure, LoCmm128TilingData, is used for both kernels and has the following data entities

| Field      |  Type   |           Description            |
| ---------- | :-----: | :------------------------------: |
| batchSize  | int64_t |     Total number of batches.     |
| seqLen     | int64_t |         Sequence length.         |
| numHeads   | int64_t |    Number of attention heads.    |
| dimSize    | int64_t |    Hidden dimension per head.    |
| blockSize  | int64_t |     Chunk size for sequence.     |
| coreNum    | int64_t |  Number of AI Cores to be used.  |
| isTraining |  bool   | Flag to enable KV-cache dumping. |

## Restrictions on parameters

1. query, key, value must have the same data type and shapes and be non empty.
2. batchSize, seqLen, numHeads, dimSize, blockSize, coreNum must be greater than 0.
3. blockSize must be a multiple of 64.
4. numHeads must be not greater than coreNum, such fact is used in a generator of slope matrix.
5. isTraining equals 0 or 1.

## Build and test

The setup is verified in CANN docker container `quay.io/ascend/cann:8.5.2-910b-ubuntu22.04-py3.11`

``` shell
# install dependencies
apt update 
apt install gawk
pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
pip install torch-npu==2.9.0
pip install pybind11 pytest pyyaml numpy==1.26.2 setuptools decorator einops scipy attrs psutil absl-py cloudpickle ml-dtypes tornado

# build as a part of npu_ops_transformer_ext
cd ops-transformer/experimental/npu_ops_transformer_ext
pip install --no-build-isolation -e .

# run tests
cd ops-transformer/experimental/attention/lightning-attention/tests/python
TASK_QUEUE_ENABLE=0 pytest
```

## Performance and accuracy

Here we compare on Ascend 910B1 the inference and training time of 2 solutions:

1. Lightning Attention Forward and Backward pass operators
2. baseline in Python.

The examined tensors' shape is the following: batchSize = 1, numHeads = 16,  seqLen = 4096, dimSize = 128.

<table border="1">
  <tr>
    <td rowspan="2">Kernel</td>
    <td rowspan="2">Kernel inference time, us</td>
    <td rowspan="2"> Kernel training time (forward + backward), us</td>
    <td colspan="7" align="center">Linear Pangu-1.9B inference benchmarks</td>
  </tr>
  <tr>
    <td>basic_en</td>
    <td>basic_zh</td>
    <td>LAMBADA</td>
    <td>WPLC</td>
    <td>MMLU</td>
    <td>C-Eval</td>
    <td>BigBench</td>
  </tr>
  <tr>
    <td>Baseline, Python</td>
    <td>18320</td>
    <td>46804</td>
    <td>0.392</td>
    <td>0.438</td>
    <td>0.573</td>
    <td>0.1289</td>
    <td>0.257</td>
    <td>0.282</td>
    <td>0.289</td>
  </tr>
   <tr>
    <td>LA kernel, AscendC</td>
    <td>150</td>
    <td>424</td>
    <td>0.390</td>
    <td>0.435</td>
    <td>0.576</td>
    <td>0.1263</td>
    <td>0.256</td>
    <td>0.280</td>
    <td>0.291</td>
  </tr>
</table>

The AscendC kernel inference time is less than Pytorch baseline by 122 times, test benchmarks degradation is less than 1%.
The AscendC kernel training time is less than Pytorch baseline by 110 times.
