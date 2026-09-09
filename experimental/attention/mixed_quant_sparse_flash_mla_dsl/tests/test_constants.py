"""Test-side MQSMLA contract constants; independent of the installed kernel."""

from enum import IntEnum

D = 512
N1 = 64
D_ROPE = 64
GROUP_SIZE_WIN = 32
GROUP_SIZE_CMP = 16
NUM_GROUPS_WIN = D // GROUP_SIZE_WIN
NUM_GROUPS_CMP = D // GROUP_SIZE_CMP
TILE_N = 128  # CPU online softmax staging size.
KV_ROW_BYTES_WIN = 544
KV_ROW_BYTES_CMP = 320
KV_SCALE_BYTES_WIN = NUM_GROUPS_WIN * 2
KV_SCALE_BYTES_CMP = NUM_GROUPS_CMP * 2
KV_PAGE_PADDING_BYTES = 64  # Both KV sides use the same fixed axis-0 gap.


class QuantMode(IntEnum):
    CONTIGUOUS = 1


class Scenario(IntEnum):
    ORI_SPARSE = 0
    ORI_CMP_SPARSE = 1
