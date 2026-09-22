// Baked TurboQuant int4 codebook used by the fused KV dequantization path.
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_TQ4_CENTROIDS_H_
#define KV_QUANT_SPARSE_FLASH_ATTENTION_TQ4_CENTROIDS_H_

constexpr int TQ4_N_CENT = 16;

// The int4 unpack operation produces signed nibbles in [-8, 7]. The kernel
// adds eight before Gather, so this table is indexed by signed_nibble + 8.
__aicore__ inline void Tq4LoadCentSigned(float (&centroids)[TQ4_N_CENT])
{
    centroids[0] = 0.0054729f;  centroids[1] = 0.0168041f;
    centroids[2] = 0.0285761f;  centroids[3] = 0.0410862f;
    centroids[4] = 0.0549298f;  centroids[5] = 0.0710182f;
    centroids[6] = 0.0911537f;  centroids[7] = 0.1203780f;
    centroids[8] = -0.1209128f;  centroids[9] = -0.0911112f;
    centroids[10] = -0.0711246f; centroids[11] = -0.0551360f;
    centroids[12] = -0.0413207f; centroids[13] = -0.0287497f;
    centroids[14] = -0.0170049f; centroids[15] = -0.0056868f;
}

#endif
