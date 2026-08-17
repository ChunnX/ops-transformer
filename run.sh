#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
rm -rf /root/ascend/log/*

echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern

export PYTHONPATH=$PYTHONPATH:/home/vllm
export PYTHONPATH=$PYTHONPATH:/home/vllm-ascend-official

nic_name="enp189s0f0"
local_ip="172.28.1.10"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_VERSION=0.20.2

source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

export MMC_LOCAL_CONFIG_PATH=/home/lsy/layerwise/mmc-local.conf

vllm serve /mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --served-model-name deepseek_v4 \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --gpu-memory-utilization 0.85 \
    --quantization ascend \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/lsy/log_p.log 2>&1
