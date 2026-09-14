"""ATK case generator for BlockSparseAttention V3 HIF8 full quantization."""

import copy
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


LAYOUTS = ("BNSD", "BSND", "TND")
Q_BOUNDARIES = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129,
                191, 255, 256, 257, 511, 512, 1023, 2048, 3071)
KV_BOUNDARIES = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129,
                 255, 256, 257, 511, 512, 1023, 2048, 4095, 8191, 9999)
BLOCK_CHOICES = tuple(range(16, 257, 16))


@GENERATOR_REGISTRY.register("generate_block_sparse_attention_hif8")
class BlockSparseAttentionHif8Generator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _expand(config_list, values):
        if not config_list:
            return
        template = config_list[0]
        config_list.clear()
        for value in values:
            item = copy.deepcopy(template)
            item.range_values = [int(value)]
            config_list.append(item)

    @staticmethod
    def _pick_seq(boundaries, upper):
        if random.random() < 0.8:
            return random.choice(boundaries)
        return random.randint(1, upper)

    def _generate_params(self):
        self.layout = random.choice(LAYOUTS)
        self.batch = random.randint(1, 5)
        self.kv_heads = random.randint(1, 8)
        self.group = random.randint(1, 4)
        self.q_heads = self.kv_heads * self.group
        self.dim = 128
        self.block_x = random.choice(BLOCK_CHOICES)
        self.block_y = random.choice(BLOCK_CHOICES)
        self.q_storage = self._pick_seq(Q_BOUNDARIES, 4095)
        self.kv_storage = self._pick_seq(KV_BOUNDARIES, 9999)

        # TND is intrinsically varlen.  For BNSD/BSND, about 30% of cases
        # carry padding; with 250 cases this gives roughly 50 padded cases.
        padded = self.layout != "TND" and random.random() < 0.30
        self.q_lengths, self.kv_lengths = [], []
        for _ in range(self.batch):
            if self.layout == "TND" or padded:
                q_low = max(1, self.q_storage // 2)
                kv_low = max(1, self.kv_storage // 2)
                self.q_lengths.append(random.randint(q_low, self.q_storage))
                self.kv_lengths.append(random.randint(kv_low, self.kv_storage))
            else:
                self.q_lengths.append(self.q_storage)
                self.kv_lengths.append(self.kv_storage)
        self.tq, self.tkv = sum(self.q_lengths), sum(self.kv_lengths)
        self.out_dtype = random.choice(("bf16", "fp16"))
        self.softmax_lse_flag = random.choice((0, 1))

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        self._generate_params()
        inputs = {
            item[0].name if isinstance(item, list) and item else item.name: item
            for item in case_config.inputs
        }

        q, k, v = inputs["query"], inputs["key"], inputs["value"]
        q.dtype = k.dtype = v.dtype = "uint8"
        if self.layout == "TND":
            q.shape = [self.tq, self.q_heads, self.dim]
            k.shape = v.shape = [self.tkv, self.kv_heads, self.dim]
            out_shape = [self.tq, self.q_heads, self.dim]
            lse_shape = [self.tq, self.q_heads, 1]
            mask_q, mask_kv = max(self.q_lengths), max(self.kv_lengths)
        elif self.layout == "BNSD":
            q.shape = [self.batch, self.q_heads, self.q_storage, self.dim]
            k.shape = v.shape = [self.batch, self.kv_heads, self.kv_storage, self.dim]
            out_shape = [self.batch, self.q_heads, self.q_storage, self.dim]
            lse_shape = [self.batch, self.q_heads, self.q_storage, 1]
            mask_q, mask_kv = self.q_storage, self.kv_storage
        else:
            q.shape = [self.batch, self.q_storage, self.q_heads, self.dim]
            k.shape = v.shape = [self.batch, self.kv_storage, self.kv_heads, self.dim]
            out_shape = [self.batch, self.q_storage, self.q_heads, self.dim]
            lse_shape = [self.batch, self.q_storage, self.q_heads, 1]
            mask_q, mask_kv = self.q_storage, self.kv_storage

        inputs["qInputLayout"].range_values = [self.layout]
        inputs["kvInputLayout"].range_values = [self.layout]
        inputs["attentionOut"].dtype = self.out_dtype
        inputs["attentionOut"].shape = out_shape
        inputs["softmaxLse"].dtype = "fp32"
        inputs["softmaxLse"].shape = lse_shape
        if not self.softmax_lse_flag:
            inputs["softmaxLse"].range_values = ["None"]

        cq = (mask_q + self.block_x - 1) // self.block_x
        ck = (mask_kv + self.block_y - 1) // self.block_y
        sparse_mask = inputs["blockSparseMask"]
        sparse_mask.dtype = "int8"
        sparse_mask.shape = [self.batch, self.q_heads, cq, ck]
        sparse_mask.range_values = [0, 1]

        inputs["attenMask"].range_values = ["None"]
        inputs["blockTable"].range_values = ["None"]
        self._expand(inputs["blockShape"], [self.block_x, self.block_y])
        self._expand(inputs["actualSeqLengths"], self.q_lengths)
        self._expand(inputs["actualSeqLengthsKv"], self.kv_lengths)

        for name in ("qDequantScale", "kDequantScale", "vDequantScale", "pQuantScale"):
            inputs[name].dtype = "fp32"
            inputs[name].shape = [1]
            inputs[name].range_values = [1.0]

        inputs["numKeyValueHeads"].range_values = [self.kv_heads]
        inputs["maskType"].range_values = [0]
        inputs["scaleValue"].range_values = [1.0 / (self.dim ** 0.5)]
        inputs["innerPrecise"].range_values = [0]
        inputs["blockSize"].range_values = [0]
        inputs["preTokens"].range_values = [2147483647]
        inputs["nextTokens"].range_values = [2147483647]
        inputs["softmaxLseFlag"].range_values = [self.softmax_lse_flag]
        inputs["quantMode"].range_values = [5]
        inputs["dstTypeMax"].range_values = [0.0]
        return case_config
