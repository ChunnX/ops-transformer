import os
import random
from typing import Union, List
import numpy as np
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig

from fd_case_specs import FD_CASE_SPECS

HEAD_DIM = 128
BLOCK_SHAPE_X = 1
BLOCK_SHAPE_Y = 128
PAGED_BLOCK_SIZE = 128
MAX_TOPK = 256
# Keep in sync with sparse_attention_score_metadata.h
METADATA_TOTAL_SIZE = 1024


@GENERATOR_REGISTRY.register("generate_generic_block_sparse_attention")
class GenericBlockSparseAttentionGenerator(CaseGenerator):
    """A5 / chip950 non-quant GBSA cases.

    Fixed (tiling/kernel):
      layoutQ=TND, layoutKv=PAGED_BBND, maskType=1, blockShape=[1,128], D=128
      quantType=0, isPackedGQA=1, returnSoftmaxlse=0|1
      softmaxPrecision=1 only (align BSA 950 innerPrecise=4 / low-prec path)
      topK <= 256, winLeft/winRight=-1, dstTypeMax=0.0

    Randomized (equal seqlens across batch — recoverable from shapes):
      B, q_seqlen, kv_seqlen, Nkv, groupSize, topK, physical block headroom
    """

    def __init__(self, config):
        super().__init__(config)
        self.dtype = "fp16"
        self.query_range = None
        self.test_mode = os.environ.get("TEST_MODE", "normal").lower()
        if self.test_mode not in ("normal", "fd"):
            raise ValueError(
                f"TEST_MODE must be normal or fd, got {self.test_mode!r}"
            )

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0 and input_case.name == "query":
            self.dtype = input_case.dtype
            self.query_range = input_case.range_values

        if index in (1, 2) and input_case.name in ("key", "value"):
            input_case.dtype = self.dtype
            input_case.range_values = self.query_range

        return input_case

    def _sample_shape(self):
        r = random.random()
        if r < 0.2:
            batch = 1
            q_seqlen = random.randint(1, 8)
            max_blocks_cap = random.randint(1, 4)
            kv_heads = 1
            group_size = random.choice([4, 8, 16])
        elif r < 0.55:
            batch = random.randint(1, 8)
            q_seqlen = random.randint(1, 64)
            max_blocks_cap = random.randint(1, 16)
            kv_heads = random.randint(1, 8)
            group_size = random.choice([4, 8, 16, 32])
        elif r < 0.85:
            batch = random.randint(1, 4)
            q_seqlen = random.randint(32, 128)
            max_blocks_cap = random.randint(8, 32)
            kv_heads = random.randint(1, 8)
            group_size = random.choice([4, 8, 16, 32])
        else:
            batch = random.randint(1, 2)
            q_seqlen = random.randint(64, 256)
            max_blocks_cap = random.randint(16, 48)
            kv_heads = random.randint(1, 4)
            group_size = random.choice([8, 16, 32])

        num_heads = kv_heads * group_size
        # Keep kv block-aligned so recover_batch_seqlens (maxBlocks*128) matches.
        # (metadata is INT32[1024] protocol buffer and can no longer encode kv_seqlen.)
        max_blocks = max_blocks_cap
        kv_seqlen = max(max_blocks * PAGED_BLOCK_SIZE, q_seqlen)
        max_blocks = (kv_seqlen + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
        top_k = random.randint(1, min(MAX_TOPK, max_blocks))
        extra = random.randint(0, max(1, batch * 2))
        num_physical = batch * max_blocks + extra
        return {
            "batch": batch,
            "q_seqlen": q_seqlen,
            "kv_seqlen": kv_seqlen,
            "num_heads": num_heads,
            "kv_heads": kv_heads,
            "top_k": top_k,
            "max_blocks": max_blocks,
            "num_physical": num_physical,
            "return_lse": random.choice([0, 1]),
        }

    def _fd_shape(self):
        """Cycle deterministic templates; every generated Case executes FD."""

        # CaseGenerator increments ``self.index`` immediately before calling
        # after_case_config(), so the current zero-based Case ID is index - 1.
        spec_index = (self.index - 1) % len(FD_CASE_SPECS)
        spec = FD_CASE_SPECS[spec_index]
        batch = 1
        max_blocks = int(spec.get("max_blocks", 16))
        kv_heads = int(spec["kv_heads"])
        return {
            "batch": batch,
            "q_seqlen": int(spec["q_storage"]),
            "kv_seqlen": max_blocks * PAGED_BLOCK_SIZE,
            "num_heads": kv_heads * int(spec["group_size"]),
            "kv_heads": kv_heads,
            "top_k": int(spec["top_k"]),
            "max_blocks": max_blocks,
            # Extra physical headroom is a generator-only marker recovered by
            # apply_init_tensors.  blockTable never maps to these spare pages.
            "num_physical": batch * max_blocks + spec_index + 1,
            "dtype": str(spec["dtype"]),
            "return_lse": 0,
        }

    def _normal_shape(self):
        """Keep random regression plus four deterministic FD-gate controls."""

        case_id = self.index - 1
        controls = (
            # topK below the Metadata FD lower bound.
            (1, 1, 1, 16, 1, 0),
            (1, 1, 16, 8, 11, 0),
            # Metadata may schedule FD, but public LSE forces Kernel fallback.
            (1, 1, 16, 16, 16, 1),
            # 9 base tasks violate totalTaskNum * 10 < 28 * 3.
            (1, 9, 16, 8, 16, 0),
        )
        if case_id >= len(controls):
            return self._sample_shape()
        batch, q_seqlen, max_blocks, group_size, top_k, return_lse = controls[case_id]
        kv_heads = 1
        return {
            "batch": batch,
            "q_seqlen": q_seqlen,
            "kv_seqlen": max_blocks * PAGED_BLOCK_SIZE,
            "num_heads": kv_heads * group_size,
            "kv_heads": kv_heads,
            "top_k": top_k,
            "max_blocks": max_blocks,
            "num_physical": batch * max_blocks,
            "return_lse": return_lse,
        }

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        p = self._fd_shape() if self.test_mode == "fd" else self._normal_shape()
        batch = p["batch"]
        q_seqlen = p["q_seqlen"]
        kv_seqlen = p["kv_seqlen"]
        num_heads = p["num_heads"]
        kv_heads = p["kv_heads"]
        top_k = p["top_k"]
        max_blocks = p["max_blocks"]
        num_physical = p["num_physical"]
        total_q_tokens = batch * q_seqlen
        total_q_blocks = total_q_tokens
        case_dtype = p.get("dtype", self.dtype)

        case_config.inputs[0].shape = [total_q_tokens, num_heads, HEAD_DIM]
        case_config.inputs[0].dtype = case_dtype
        case_config.inputs[28].shape = [total_q_tokens, num_heads, HEAD_DIM]
        case_config.inputs[28].dtype = case_dtype
        case_config.inputs[29].shape = [total_q_tokens, num_heads, 1]
        case_config.inputs[29].dtype = "fp32"

        case_config.inputs[1].shape = [num_physical, PAGED_BLOCK_SIZE, kv_heads, HEAD_DIM]
        case_config.inputs[1].dtype = case_dtype
        case_config.inputs[2].shape = [num_physical, PAGED_BLOCK_SIZE, kv_heads, HEAD_DIM]
        case_config.inputs[2].dtype = case_dtype

        case_config.inputs[3].shape = [kv_heads, total_q_blocks, top_k]
        case_config.inputs[3].dtype = "int32"
        case_config.inputs[4].shape = [kv_heads, total_q_blocks]
        case_config.inputs[4].dtype = "int32"

        # Real AICPU metadata protocol buffer (filled in apply_init_tensors).
        case_config.inputs[5].shape = [METADATA_TOTAL_SIZE]
        case_config.inputs[5].dtype = "int32"
        case_config.inputs[6].shape = [1, 1]
        case_config.inputs[7].shape = [1]
        case_config.inputs[8].shape = [1]
        case_config.inputs[9].shape = [1]
        case_config.inputs[10].shape = [1]
        case_config.inputs[13].shape = [batch]
        case_config.inputs[14].shape = [batch]

        case_config.inputs[11].shape = [batch + 1]
        case_config.inputs[11].dtype = "int64"
        case_config.inputs[12].shape = [batch + 1]
        case_config.inputs[12].dtype = "int64"

        case_config.inputs[15].shape = [batch, max_blocks]
        case_config.inputs[15].dtype = "int32"

        case_config.inputs[16][0].range_values = BLOCK_SHAPE_X
        case_config.inputs[16][1].range_values = BLOCK_SHAPE_Y
        case_config.inputs[17].range_values = 1
        case_config.inputs[18].range_values = "TND"
        case_config.inputs[19].range_values = "PAGED_BBND"
        case_config.inputs[20].range_values = 1.0 / np.sqrt(HEAD_DIM)
        case_config.inputs[21].range_values = 1
        case_config.inputs[22].range_values = 0
        case_config.inputs[23].range_values = 0.0  # dstTypeMax
        case_config.inputs[24].range_values = 1  # softmaxPrecision: A5 only supports 1
        case_config.inputs[25].range_values = -1  # winLeft
        case_config.inputs[26].range_values = -1  # winRight
        case_config.inputs[27].range_values = p["return_lse"]  # returnSoftmaxlse

        return case_config
