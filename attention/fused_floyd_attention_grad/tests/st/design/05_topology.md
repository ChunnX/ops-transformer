# Factor Dependency Topology

## Solve Order

**level_0_intermediate (pure anchor intermediate):**
  - `_h.value_range`  [domain: [[1, 1], [1, 4], [4, 16], [16, 64], [64, 256]]]
  - `_n.value_range`  [domain: [[16, 16], [16, 256], [256, 4096], [4096, 65536], [65536, 1048576]]]
  - `_b.value_range`  [domain: [[1, 1], [1, 8], [8, 64], [64, 512], [512, 2048]]]
  - `_k.value_range`  [domain: [[128, 128], [128, 1024], [1024, 16384], [16384, 131072], [131072, 1048576]]]
  - `_d.value`  [domain: [32, 64, 128]]
  - `_h.value`  [domain: []]
  - `_m.value_range`  [domain: [[128, 128], [128, 1024], [1024, 16384], [16384, 131072], [131072, 1048576]]]

**level_0_anchors (anchor factors):**
  - `attenMaskOptional.dtype`  [domain: ['bool', 'uint8']]
  - `value1.exist`  [domain: [True]]
  - `query.format`  [domain: ['ND']]
  - `softmaxMax.exist`  [domain: [True]]
  - `value2.dimensions`  [domain: [5]]
  - `dy.exist`  [domain: [True]]
  - `dqOut.exist`  [domain: [True]]
  - `softmaxMax.dtype`  [domain: ['float32']]
  - `query.dtype`  [domain: ['float16', 'bfloat16']]
  - `dk2Out.dimensions`  [domain: [5]]
  - `attentionIn.dimensions`  [domain: [5]]
  - `dv2Out.exist`  [domain: [True]]
  - `key2.dimensions`  [domain: [5]]
  - `scaleValue.value_range_float64`  [domain: [[0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10], [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01], [-2, -1], [-10, -2], [-1000, -10], [-1, 1], [-0.01, 0.01], [-100, 100], [0, 0], [1.0, 1.0], [0.0884, 0.1768], [3.4028235e+38, 3.4028235e+38], [-3.4028235e+38, -3.4028235e+38], [1.7976931348623157e+308, 1.7976931348623157e+308], [-1.7976931348623157e+308, -1.7976931348623157e+308], [2.2250738585072014e-308, 2.2250738585072014e-308], [-2.2250738585072014e-308, -2.2250738585072014e-308], ['-0', '-0'], ['+0', '+0'], ['-0', '+0']]]
  - `scaleValue.dtype`  [domain: ['float64']]
  - `dk1Out.exist`  [domain: [True]]
  - `key2.format`  [domain: ['ND']]
  - `dk2Out.format`  [domain: ['ND']]
  - `dv1Out.format`  [domain: ['ND']]
  - `dv2Out.format`  [domain: ['ND']]
  - `dv1Out.exist`  [domain: [True]]
  - `dqOut.format`  [domain: ['ND']]
  - `dy.format`  [domain: ['ND']]
  - `softmaxSum.format`  [domain: ['ND']]
  - `attenMaskOptional.dimensions`  [domain: [5]]
  - `attenMaskOptional.value_range_bool`  [domain: [[0, 0], [1, 1], [0, 1]]]
  - `softmaxSum.dimensions`  [domain: [5]]
  - `dk1Out.format`  [domain: ['ND']]
  - `query.dimensions`  [domain: [5]]
  - `dy.dimensions`  [domain: [5]]
  - `softmaxSum.dtype`  [domain: ['float32']]
  - `softmaxMax.dimensions`  [domain: [5]]
  - `scaleValue.exist`  [domain: [True]]
  - `key1.dimensions`  [domain: [5]]
  - `attentionIn.exist`  [domain: [True]]
  - `value1.format`  [domain: ['ND']]
  - `dv1Out.dimensions`  [domain: [5]]
  - `dv2Out.dimensions`  [domain: [5]]
  - `query.exist`  [domain: [True]]
  - `key1.format`  [domain: ['ND']]
  - `value2.format`  [domain: ['ND']]
  - `value2.exist`  [domain: [True]]
  - `dk2Out.exist`  [domain: [True]]
  - `attentionIn.format`  [domain: ['ND']]
  - `attenMaskOptional.format`  [domain: ['ND']]
  - `key2.exist`  [domain: [True]]
  - `attenMaskOptional.value_range_uint8`  [domain: [[0, 0], [1, 1], [0, 1]]]
  - `attenMaskOptional.exist`  [domain: [True, False]]
  - `value1.dimensions`  [domain: [5]]
  - `softmaxSum.exist`  [domain: [True]]
  - `key1.exist`  [domain: [True]]
  - `dqOut.dimensions`  [domain: [5]]
  - `softmaxMax.format`  [domain: ['ND']]
  - `dk1Out.dimensions`  [domain: [5]]

**level_1:**
  - `attenMaskOptional.value_range`  <- [attenMaskOptional.dtype]  (built-in: IMPL-RANGE)
  - `softmaxMax.value_range`  <- [softmaxMax.dtype]  (built-in: IMPL-RANGE)
  - `key1.dtype`  <- [query.dtype]  (@solves: solve_key1_dtype)
  - `value1.dtype`  <- [query.dtype]  (@solves: solve_value1_dtype)
  - `key2.dtype`  <- [query.dtype]  (@solves: solve_key2_dtype)
  - `value2.dtype`  <- [query.dtype]  (@solves: solve_value2_dtype)
  - `dy.dtype`  <- [query.dtype]  (@solves: solve_dy_dtype)
  - `attentionIn.dtype`  <- [query.dtype]  (@solves: solve_attentionIn_dtype)
  - `dqOut.dtype`  <- [query.dtype]  (@solves: solve_dqOut_dtype)
  - `dk1Out.dtype`  <- [query.dtype]  (@solves: solve_dk1Out_dtype)
  - `dv1Out.dtype`  <- [query.dtype]  (@solves: solve_dv1Out_dtype)
  - `dk2Out.dtype`  <- [query.dtype]  (@solves: solve_dk2Out_dtype)
  - `dv2Out.dtype`  <- [query.dtype]  (@solves: solve_dv2Out_dtype)
  - `query.value_range`  <- [query.dtype]  (built-in: IMPL-RANGE)
  - `scaleValue.value_range`  <- [scaleValue.dtype]  (built-in: IMPL-RANGE)
  - `softmaxSum.value_range`  <- [softmaxSum.dtype]  (built-in: IMPL-RANGE)
  - `_b.value`  <- [_b.value_range, _h.value, _d.value]  (@solves: solve_b_value)

**level_2:**
  - `key1.value_range`  <- [key1.dtype]  (built-in: IMPL-RANGE)
  - `value1.value_range`  <- [value1.dtype]  (built-in: IMPL-RANGE)
  - `key2.value_range`  <- [key2.dtype]  (built-in: IMPL-RANGE)
  - `value2.value_range`  <- [value2.dtype]  (built-in: IMPL-RANGE)
  - `dy.value_range`  <- [dy.dtype]  (built-in: IMPL-RANGE)
  - `attentionIn.value_range`  <- [attentionIn.dtype]  (built-in: IMPL-RANGE)
  - `scaleValue.value`  <- [scaleValue.value_range, scaleValue.dtype]  (built-in: IMPL-SCALAR-VALUE)
  - `_n.value`  <- [_n.value_range, _b.value, _h.value, _d.value]  (@solves: solve_n_value)

**level_3:**
  - `_m.value`  <- [_m.value_range, _b.value, _h.value, _n.value, _d.value]  (@solves: solve_m_value)

**level_4:**
  - `_k.value`  <- [_k.value_range, _b.value, _h.value, _n.value, _m.value, _d.value]  (@solves: solve_k_value)
  - `query.shape`  <- [_b.value, _h.value, _n.value, _m.value, _d.value]  (@solves: solve_query_shape)

**level_5:**
  - `key1.shape`  <- [_b.value, _h.value, _n.value, _k.value, _d.value]  (@solves: solve_key1_shape)
  - `key2.shape`  <- [_b.value, _h.value, _k.value, _m.value, _d.value]  (@solves: solve_key2_shape)
  - `attenMaskOptional.shape`  <- [attenMaskOptional.exist, _b.value, _n.value, _k.value]  (@solves: solve_attenMask_shape)
  - `dy.shape`  <- [query.shape]  (@solves: solve_dy_shape)
  - `attentionIn.shape`  <- [query.shape]  (@solves: solve_attentionIn_shape)
  - `softmaxMax.shape`  <- [query.shape]  (@solves: solve_softmaxMax_shape)
  - `dqOut.shape`  <- [query.shape]  (@solves: solve_dqOut_shape)

**level_6:**
  - `value1.shape`  <- [key1.shape]  (@solves: solve_value1_shape)
  - `dk1Out.shape`  <- [key1.shape]  (@solves: solve_dk1Out_shape)
  - `value2.shape`  <- [key2.shape]  (@solves: solve_value2_shape)
  - `dk2Out.shape`  <- [key2.shape]  (@solves: solve_dk2Out_shape)
  - `softmaxSum.shape`  <- [softmaxMax.shape]  (@solves: solve_softmaxSum_shape)

**level_7:**
  - `dv1Out.shape`  <- [value1.shape]  (@solves: solve_dv1Out_shape)
  - `dv2Out.shape`  <- [value2.shape]  (@solves: solve_dv2Out_shape)

## Factor Coverage

- Total factors: 103 (including 11 intermediate)
- Anchor factors: 61
- Explicit constraint coverage: 30
- Built-in derivation coverage: 12
- Uncovered factors: 0