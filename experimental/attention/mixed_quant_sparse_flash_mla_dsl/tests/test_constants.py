"""Test-side MQSMLA contract constants; independent of the installed kernel."""

from enum import IntEnum

D = 512
N1 = 64
D_ROPE = 64
GROUP_SIZE_ORI = 32
GROUP_SIZE_CMP = 16
NUM_GROUPS_ORI = D // GROUP_SIZE_ORI
NUM_GROUPS_CMP = D // GROUP_SIZE_CMP
TILE_N = 128  # CPU online softmax staging size.
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
KV_SCALE_BYTES_ORI = NUM_GROUPS_ORI * 2
KV_SCALE_BYTES_CMP = NUM_GROUPS_CMP * 2
KV_PAGE_PADDING_BYTES = 64  # Both KV sides use the same fixed axis-0 gap.

# Metadata layout (int32[1024]), mirroring the wheel's
# `mixed_quant_sparse_flash_mla_metadata`: FA[36][9] (324) ‖ FD[72][8] (576) ‖
# fdUsedVecNum word (index 900); selective FD also uses words 901..950.
MQSMLA_METADATA_TOTAL_SIZE = 1024
AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
# FA[36][9](324) + FD[72][8](576) = 900 words; index 900 holds fdUsedVecNum.
# Selective FD readiness flags occupy 901..948; async mode is word 950.
# Uniform AICPU plans leave the tail undefined.
FD_USED_VEC_NUM_WORD = AIC_CORE_MAX_NUM * FA_METADATA_SIZE + AIV_CORE_MAX_NUM * FD_METADATA_SIZE
FD_ASYNC_MODE_WORD = FD_USED_VEC_NUM_WORD + 50
FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5


class QuantMode(IntEnum):
    CONTIGUOUS = 1


class Scenario(IntEnum):
    ORI_SPARSE = 0
    ORI_CMP_SPARSE = 1
    CMP_SPARSE = 2
