import math

import numpy as np


HEAD_DIM = 512
PACKED_BYTES = HEAD_DIM // 2
SLOT_BYTES = PACKED_BYTES + 2
CENTROIDS = np.array(
    [
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
    ],
    dtype=np.float32,
)


def _hadamard(order):
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(order)


def test_compact_slot_layout():
    assert PACKED_BYTES == 256
    assert SLOT_BYTES == 258
    assert (1024 - SLOT_BYTES) / 1024 == 0.748046875


def test_signed_nibble_centroid_order():
    physical_codes = np.arange(16, dtype=np.int16)
    signed_codes = np.where(physical_codes < 8, physical_codes, physical_codes - 16)
    gather_indices = signed_codes + 8
    np.testing.assert_array_equal(gather_indices, np.r_[8:16, 0:8])
    assert np.all(np.diff(CENTROIDS[gather_indices]) > 0)


def test_corrected_scale_preserves_quantized_row_norm():
    rng = np.random.default_rng(0)
    latent = rng.standard_normal(HEAD_DIM).astype(np.float32)
    unit = latent / np.linalg.norm(latent)
    ascending = np.sort(CENTROIDS)
    boundaries = (ascending[:-1] + ascending[1:]) * np.float32(0.5)
    ascending_codes = np.sum(unit[:, None] >= boundaries[None, :], axis=1)
    selected = ascending[ascending_codes]
    corrected_scale = np.linalg.norm(latent) / np.linalg.norm(selected)
    reconstructed = selected * corrected_scale
    np.testing.assert_allclose(np.linalg.norm(reconstructed), np.linalg.norm(latent), rtol=2e-7)


def test_shared_kv_attention_is_hadamard_basis_invariant():
    rng = np.random.default_rng(1)
    transform = _hadamard(HEAD_DIM)
    signs = np.where(np.arange(HEAD_DIM) % 3 == 0, -1.0, 1.0)
    transform = signs[:, None] * transform
    query = rng.standard_normal((3, HEAD_DIM))
    shared_kv = rng.standard_normal((7, HEAD_DIM))
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def attention(q, kv):
        scores = q @ kv.T * scale
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return probabilities @ kv

    reference = attention(query, shared_kv)
    transformed = attention(query @ transform, shared_kv @ transform) @ transform.T
    np.testing.assert_allclose(transformed, reference, rtol=1e-10, atol=1e-10)
