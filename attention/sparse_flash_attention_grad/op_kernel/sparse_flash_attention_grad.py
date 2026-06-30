import pypto_pro.runtime as fe
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import tilingkey_field
from dataclasses import dataclass


@dataclass
class OpTiling:
    b: int
    n2: int
    g: int
    s1: int
    s2: int
    d: int
    d1: int


class SfagTilingKey:
    InputDType = tilingkey_field(bits=3, values=[0, 1])
    IsTnd = tilingkey_field(bits=1, values=[0, 1])
    GTemplateNum = tilingkey_field(bits=8, values=[128])
    S2TemplateNum = tilingkey_field(bits=8, values=[128])
    DTemplateNum = tilingkey_field(bits=12, values=[512, 576])
    IsRope = tilingkey_field(bits=1, values=[0, 1])
    Deterministic = tilingkey_field(bits=1, values=[0, 1])

    def is_valid(self, key):
        (InputDType, IsTnd, GTemplateNum, S2TemplateNum,
         DTemplateNum, IsRope, Deterministic) = key
        if GTemplateNum != 128: return False
        if S2TemplateNum != 128: return False
        return True


@fe.jit(auto_mutex=True, tiling_key=SfagTilingKey, timeout=300)
def sparse_flash_attention_grad(
    query: pl.Ptr[pl.UINT8],
    key: pl.Ptr[pl.UINT8],
    sparse_indices: pl.Ptr[pl.UINT8],
    d_out: pl.Ptr[pl.UINT8],
    out: pl.Ptr[pl.UINT8],
    softmax_max: pl.Ptr[pl.UINT8],
    softmax_sum: pl.Ptr[pl.UINT8],
    value: pl.Ptr[pl.UINT8],
    actual_seq_qlen: pl.Ptr[pl.UINT8],
    actual_seq_kvlen: pl.Ptr[pl.UINT8],
    query_rope: pl.Ptr[pl.UINT8],
    key_rope: pl.Ptr[pl.UINT8],
    dq: pl.Ptr[pl.UINT8],
    dk: pl.Ptr[pl.UINT8],
    dv: pl.Ptr[pl.UINT8],
    dq_rope: pl.Ptr[pl.UINT8],
    dk_rope: pl.Ptr[pl.UINT8],
    workspace: pl.Ptr[pl.UINT8],
    tiling: OpTiling,
):
    TS = 128
    TD = 128
    TKV = 128
    Q_F16 = TS * TD * 2
    KT_F16 = TKV * TD * 2
    QK_HALF_F32 = TKV * TS * 4

    MA0 = 0
    MA1 = Q_F16 * 2
    LA0 = 0
    RA0 = 0
    CA0 = 0
    VA0 = 0
    VA1 = QK_HALF_F32

    tensor_q = pl.make_tensor(query, [tiling.b, tiling.s1, tiling.n2, tiling.d],
                              [tiling.s1 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1], dtype=pl.FP16)
    tensor_k = pl.make_tensor(key, [tiling.b, tiling.s2, tiling.n2, tiling.d],
                              [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1], dtype=pl.FP16)
    tensor_dq = pl.make_tensor(dq, [tiling.b, tiling.s1, tiling.n2, tiling.d],
                               [tiling.s1 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1], dtype=pl.FP16)

    with pl.section_cube():
        q_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat,
                             blayout=1, slayout=2),
            addrs=MA0, mutex_ids=[0, 1])
        k_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat),
            addrs=MA1, mutex_ids=[2, 3])
        left = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left),
            addrs=LA0, mutex_ids=[4, 5])
        right = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right),
            addrs=RA0, mutex_ids=[6, 7])
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc),
            addrs=CA0, mutex_ids=[8, 9])

        pl.load_tile(q_l1.next(), tensor_q, [0, 0, 0, 0], layout="dn", tile_dims=[1, 3])
        pl.load_tile(k_l1.next(), tensor_k, [0, 0, 0, 0], tile_dims=[1, 3])
        pl.move(left.next(), q_l1.next())
        pl.move(right.next(), k_l1.next())
        pl.matmul(acc.next(), left.next(), right.next())

    with pl.section_vector():
        p_vec = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[VA0, VA1], mutex_ids=[0, 1])
        o_vec = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
            addrs=VA0, mutex_ids=[2])

        p0 = p_vec.next()
        p1 = p_vec.next()
        o0 = o_vec.next()
        pl.add(o0, p0, p1)
        pl.store_tile(tensor_dq, o0, [0, 0, 0, 0], tile_dims=[1, 3])
