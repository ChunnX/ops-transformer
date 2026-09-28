# Factor Dependency Topology

## Solve Order

**level_0_intermediate (pure anchor intermediate):**
  - `_h_in.value_range`  [domain: [[1, 1], [1, 16], [16, 256], [256, 4096]]]
  - `_n.value`  [domain: []]
  - `_c.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]
  - `_w_in.value`  [domain: []]
  - `_c.value`  [domain: []]
  - `_w_in.value_range`  [domain: [[1, 1], [1, 16], [16, 256], [256, 4096]]]
  - `_out_w.value`  [domain: []]
  - `_h_in.value`  [domain: []]
  - `_out_h.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]
  - `_out_h.value`  [domain: []]
  - `_out_w.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]
  - `_n.value_range`  [domain: [[0, 0], [1, 1], [1, 8], [8, 128]]]

**level_0_anchors (anchor factors):**
  - `out.exist`  [domain: [True]]
  - `interpolationMode.exist`  [domain: [True]]
  - `alignCorners.value`  [domain: [True, False]]
  - `paddingMode.value`  [domain: [0, 1, 2]]
  - `out.format`  [domain: ['ND']]
  - `input.format`  [domain: ['ND']]
  - `grid.format`  [domain: ['ND']]
  - `interpolationMode.value`  [domain: [0, 1, 2]]
  - `paddingMode.dtype`  [domain: ['int64']]
  - `interpolationMode.dtype`  [domain: ['int64']]
  - `alignCorners.exist`  [domain: [True]]
  - `input.dimensions`  [domain: [4]]
  - `paddingMode.exist`  [domain: [True]]
  - `input.exist`  [domain: [True]]
  - `grid.exist`  [domain: [True]]
  - `out.dimensions`  [domain: [4]]
  - `grid.dimensions`  [domain: [4]]
  - `alignCorners.dtype`  [domain: ['bool']]

**level_1:**
  - `input.dtype`  <- [interpolationMode.value]  (@solves: solve_input_dtype)
  - `input.shape`  <- [input.dimensions, _n.value, _c.value, _h_in.value, _w_in.value]  (@solves: solve_input_shape)

**level_2:**
  - `grid.dtype`  <- [input.dtype]  (@solves: solve_grid_dtype)
  - `out.dtype`  <- [input.dtype]  (@solves: solve_out_dtype)
  - `input.value_range`  <- [input.dtype]  (built-in: IMPL-RANGE)
  - `grid.shape`  <- [grid.dimensions, input.shape, _out_h.value, _out_w.value]  (@solves: solve_grid_shape)

**level_3:**
  - `grid.value_range`  <- [grid.dtype]  (built-in: IMPL-RANGE)
  - `out.shape`  <- [input.shape, grid.shape]  (@solves: solve_out_shape)

## Factor Coverage

- Total factors: 38 (including 12 intermediate)
- Anchor factors: 30
- Explicit constraint coverage: 6
- Built-in derivation coverage: 2
- Uncovered factors: 0