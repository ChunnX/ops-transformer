#pragma once
void score_normalize_launch(uint8_t *topk_logits_gather, int32_t ROWS, int32_t per_count, int32_t blockDim, void *stream, int32_t dtype);