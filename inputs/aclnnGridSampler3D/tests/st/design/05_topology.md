# Factor Dependency Topology

## Solve Order

**level_0_intermediate (pure anchor intermediate):**
  - `_out_w.value`  [domain: []]
  - `_out_d.value`  [domain: []]
  - `_h_in.value_range`  [domain: [[1, 1], [1, 16], [16, 256]]]
  - `_w_in.value_range`  [domain: [[1, 1], [1, 16], [16, 256]]]
  - `_c.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]
  - `_out_d.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]
  - `_out_w.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]
  - `_h_in.value`  [domain: []]
  - `_n.value`  [domain: []]
  - `_n.value_range`  [domain: [[0, 0], [1, 1], [1, 8], [8, 128]]]
  - `_w_in.value`  [domain: []]
  - `_out_h.value`  [domain: []]
  - `_d_in.value`  [domain: []]
  - `_c.value`  [domain: []]
  - `_d_in.value_range`  [domain: [[1, 1], [1, 16], [16, 256]]]
  - `_out_h.value_range`  [domain: [[0, 0], [1, 1], [1, 16], [16, 256]]]

**level_0_anchors (anchor factors):**
  - `paddingMode.value`  [domain: [0, 1, 2]]
  - `out.dimensions`  [domain: [5]]
  - `alignCorners.value`  [domain: [True, False]]
  - `interpolationMode.exist`  [domain: [True]]
  - `grid.dimensions`  [domain: [5]]
  - `input.format`  [domain: ['NCDHW', 'NDHWC', 'ND']]
  - `paddingMode.exist`  [domain: [True]]
  - `paddingMode.dtype`  [domain: ['int64']]
  - `interpolationMode.dtype`  [domain: ['int64']]
  - `alignCorners.exist`  [domain: [True]]
  - `out.exist`  [domain: [True]]
  - `grid.format`  [domain: ['ND']]
  - `alignCorners.dtype`  [domain: ['bool']]
  - `grid.exist`  [domain: [True]]
  - `input.exist`  [domain: [True]]

**level_1:**
  - `input.dtype`  <- [input.format]  (@solves: solve_input_dtype)
  - `interpolationMode.value`  <- [input.format]  (@solves: solve_interpolation_mode)
  - `out.format`  <- [input.format]  (@solves: solve_out_format)
  - `input.dimensions`  <- [input.format]  (@solves: solve_input_dimensions)

**level_2:**
  - `grid.dtype`  <- [input.dtype]  (@solves: solve_grid_dtype)
  - `out.dtype`  <- [input.dtype]  (@solves: solve_out_dtype)
  - `input.value_range`  <- [input.dtype]  (built-in: IMPL-RANGE)
  - `input.shape`  <- [input.dimensions, input.format, _n.value, _c.value, _d_in.value, _h_in.value, _w_in.value]  (@solves: solve_input_shape)

**level_3:**
  - `grid.value_range`  <- [grid.dtype]  (built-in: IMPL-RANGE)
  - `grid.shape`  <- [grid.dimensions, input.shape, _out_d.value, _out_h.value, _out_w.value]  (@solves: solve_grid_shape)

**level_4:**
  - `out.shape`  <- [input.shape, grid.shape]  (@solves: solve_out_shape)

## Factor Coverage

- Total factors: 42 (including 16 intermediate)
- Anchor factors: 31
- Explicit constraint coverage: 9
- Built-in derivation coverage: 2
- Uncovered factors: 0