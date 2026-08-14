import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  UsageBar,
  useCanvasState,
  useHostTheme,
  type Color,
  type TextProps,
} from "cursor/canvas";

/* ------------------------------------------------------------------ */
/* 共用小组件                                                          */
/* ------------------------------------------------------------------ */

type MemSeg = {
  id: string;
  label: string;
  kb: number;
  color: Color;
};

function Note({ children }: TextProps) {
  return (
    <Text size="small" tone="tertiary">
      {children}
    </Text>
  );
}

function Legend({ segments }: { segments: MemSeg[] }) {
  const theme = useHostTheme();
  return (
    <Row gap={14} wrap>
      {segments.map((s) => (
        <div
          key={s.id}
          style={{ display: "flex", alignItems: "center", gap: 6 }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: 2,
              background: theme.category[s.color],
              flexShrink: 0,
            }}
          />
          <Text size="small" tone="secondary">
            {s.label}
          </Text>
          <Text size="small" tone="tertiary">
            {s.kb} KB
          </Text>
        </div>
      ))}
    </Row>
  );
}

function MemoryMap({
  segments,
  total,
  name,
}: {
  segments: MemSeg[];
  total: number;
  name: string;
}) {
  const used = segments.reduce((a, s) => a + s.kb, 0);
  return (
    <Stack gap={10}>
      <UsageBar
        total={total}
        topLeftLabel={`${name} 占用 ${((used / total) * 100).toFixed(1)}%`}
        topRightLabel={`${used} KB / ${total} KB（剩余 ${total - used} KB）`}
        segments={segments.map((s) => ({
          id: s.id,
          value: s.kb,
          color: s.color,
        }))}
      />
      <Legend segments={segments} />
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/* 数据：内存分配                                                       */
/* ------------------------------------------------------------------ */

const L1_FP8: MemSeg[] = [
  { id: "p", label: "P 数据 (8×32K)", kb: 256, color: "purple" },
  { id: "ps", label: "P-scale (8×1.25K)", kb: 10, color: "blue" },
  { id: "q", label: "Q 数据 (2×16K)", kb: 32, color: "green" },
  { id: "qs", label: "Q-scale (2×0.5K + 对齐)", kb: 2, color: "cyan" },
  { id: "kv", label: "KV 共享池 (4×32K)", kb: 128, color: "orange" },
  { id: "kvs", label: "KV-scale (4×2K)", kb: 8, color: "yellow" },
  { id: "gm", label: "localGlobalMax (4×0.5K)", kb: 2, color: "pink" },
];

const L1_FP4: MemSeg[] = [
  { id: "p", label: "P 数据 (20×16K)", kb: 320, color: "purple" },
  { id: "ps", label: "P-scale (20×1.25K)", kb: 25, color: "blue" },
  { id: "q", label: "Q 数据 (2×8K)", kb: 16, color: "green" },
  { id: "qs", label: "Q-scale (2×0.5K)", kb: 1, color: "cyan" },
  { id: "kv", label: "KV 共享池 (4×32K)", kb: 128, color: "orange" },
  { id: "kvs", label: "KV-scale (4×2K)", kb: 8, color: "yellow" },
  { id: "gm", label: "localGlobalMax (4×0.5K)", kb: 2, color: "pink" },
];

const L0A_FP8: MemSeg[] = [
  { id: "k", label: "K 双缓冲 (2×8K，k(D) 切 2 段)", kb: 16, color: "green" },
  {
    id: "v",
    label: "Vᵀ 双缓冲 (2×18K，k(S2) 切 2 段)",
    kb: 36,
    color: "purple",
  },
];

const L0B_FP8: MemSeg[] = [
  { id: "q", label: "Q 常驻 (2 段各 8K)", kb: 16, color: "orange" },
  { id: "p", label: "Pᵀ 双缓冲 (2×16K)", kb: 32, color: "blue" },
];

const L0C_FP8: MemSeg[] = [
  { id: "qk", label: "QK (2×64K)", kb: 128, color: "orange" },
  { id: "pv", label: "PV (1×72K = 144×128×4B)", kb: 72, color: "purple" },
];

const UB_FP8: MemSeg[] = [
  { id: "s", label: "S = mm1Res，fp16 (2×64K 交错)", kb: 128, color: "purple" },
  { id: "otmp", label: "oTmp = mm2Res，fp32", kb: 32, color: "orange" },
  { id: "rs", label: "local/globalRowSum", kb: 0.5, color: "red" },
  { id: "p", label: "P 区 (2×16K，各半个基块)", kb: 32, color: "blue" },
  { id: "o", label: "attentionOut", kb: 32, color: "green" },
  { id: "gmax", label: "localGroupMax (4×2K)", kb: 8, color: "yellow" },
  { id: "misc", label: "peer/softmax/update/index", kb: 3.5, color: "gray" },
];

const L1_TABLE: string[][] = [
  [
    "P 数据",
    "0",
    "32 KB",
    "8",
    "256 KB",
    "128×256×1B，槽位 = loop % 8",
  ],
  [
    "P-scale",
    "256 KB",
    "1280 B",
    "8",
    "10 KB",
    "128×2×(256/64+1)，与 fp4 逐字节相同",
  ],
  ["Q 数据", "266 KB", "16 KB", "2", "32 KB", "128×128×1B，每 task 只搬一次"],
  ["Q-scale", "298 KB", "512 B", "2", "1 KB", "对齐后到 300 KB"],
  [
    "KV 共享池",
    "300 KB",
    "32 KB",
    "4",
    "128 KB",
    "QK 存 K(256×128)，PV 存 V(128×256)",
  ],
  ["KV-scale", "428 KB", "2 KB", "4", "8 KB", "与 fp4 相同"],
  ["localGlobalMax", "436 KB", "512 B", "4", "2 KB", "两个 AIV 各一行"],
  ["空闲", "438 KB", "—", "—", "74 KB", "调试余量"],
];

const UB_TABLE: string[][] = [
  ["S (mm1Res)", "0", "128 KB", "fp16，2 槽各 64K = 256(kvs)×128(qs)，按 256B 交错"],
  ["oTmp (mm2Res)", "128 KB", "32 KB", "fp32，不变"],
  ["local/globalRowSum", "160 KB", "512 B", "不变"],
  [
    "P",
    "160 KB+512",
    "32 KB",
    "2 槽各 16K，每槽半个基块 = 128(kvs)×128(qs)×1B",
  ],
  ["oTrans（复用 P 区）", "160 KB+512", "18 KB", "不变"],
  ["pScale（复用 P 区）", "178 KB+512", "10 KB", "8×1280B，不变"],
  ["attentionOut", "192 KB+512", "32 KB", "不变"],
  ["peerGlobalMax / softmaxMax", "224 KB+512", "1.25 KB", "不变"],
  [
    "localGroupMax",
    "225 KB+768",
    "8 KB",
    "原 20×1K → 8×1K；每 AIV 在飞 = Dpv/2 = 4 个 2K 槽",
  ],
  [
    "localGlobalMax / updateScale / index",
    "233 KB+768 起",
    "2.25 KB",
    "整体前移 12 KB",
  ],
];

const DELTA_TABLE: string[][] = [
  ["组大小 G", "16 基块 (4096×128)", "4 基块 (1024×128)", "需求给定"],
  ["PV 延迟 Dpv", "20", "8", "下界 G+3 = 7，取 G+4 留 1 拍余量且为偶数"],
  ["P 环 / 预取缓存", "20 / 21", "8 / 9", "环大小恒等于 Dpv"],
  ["P 单槽", "16 KB", "32 KB", "元素从 0.5B 变 1B"],
  ["P-scale 单槽", "1280 B", "1280 B", "MX 组恒 32 元素，scale 分形与位宽无关"],
  ["KV 单槽", "32 KB", "32 KB", "PV 切 k 后 rowsum seed 只要 18 KB"],
  [
    "L0A = K + Vᵀ",
    "2×8K + 2×18K = 52K",
    "2×8K + 2×18K = 52K（余 12K）",
    "槽几何不变：QK 沿 k(D) 切 64、PV 沿 k(S2) 切 128，正好抵掉位宽翻倍",
  ],
  [
    "L0B = Q + Pᵀ",
    "2×8K + 2×18K = 52K",
    "2×8K + 2×16K = 48K（余 16K）",
    "同上；Pᵀ = 128×128×1B 正好 16K",
  ],
  ["UB P 区", "2 槽各 1 个整基块", "2 槽各半个基块", "半块双缓冲，保住重叠又不超 UB"],
  ["UB localGroupMax", "10×2K = 20 KB", "4×2K = 8 KB", "= Dpv/2"],
  ["L0C", "QK 2×64K + PV 72K", "不变", "fp32 累加与位宽无关"],
];

const DTYPE_TABLE: string[][] = [
  ["q / k / v（GM）", "DT_FLOAT8_E4M3FN"],
  ["q/k/v descale（GM）", "DT_FLOAT8_E8M0，MX 组 32：Q/K 沿 D，V 沿 S2"],
  ["S（UB）", "fp16"],
  ["P（UB→L1）", "fp8_e4m3fn_t"],
  ["P-scale（UB→L1）", "fp8_e8m0_t"],
  ["L0C 累加", "fp32，Fixpipe 走 NO_QUANT"],
  ["rowsum seed", "数据 0x38（e4m3 的 1.0）× scale 0x7F（e8m0 的 1.0）= 1.0"],
];

const RISKS: { title: string; body: string; tone: "danger" | "warning" }[] = [
  {
    title: "fp8 转置 LoadData 的 dstStride 奇偶约束",
    body:
      "matmul.h 里 fp8 的转置分支会把 mStep 向上取偶，并在 ifTranspose && (dstStride & 1) 时改走分段下发路径。Pᵀ 的 dstStride = 128/16 = 8（偶，安全）；Vᵀ 按公式本应是 8，但 BSA 的 loader 为了 rowsum 的 16 行 pad 写成 ceil(D/16)+1 = 9（奇），会命中分段路径——第一期就要用最小用例验证。兜底：pad 改成 2 个分形（dstStride = 10），PV 的 m 从 144 padding 到 160，代价是 L0A Vᵀ 槽 18K→20K（合计 56K）、L0C PV 槽 72K→80K（合计 208K），三处都还有余量。",
    tone: "danger",
  },
  {
    title: "mmad 条数翻倍",
    body:
      "每个基块从 3 条（2 条 QK + 1 条 PV）变成 6 条，每条 k 更短。L0 侧因为切 k 反而留出了余量（L0A 12 KB、L0B 16 KB），但 cube 的指令发射开销与短 k 的流水效率需实测；若 QK 的 k=64 不可接受，退路是 K 退回单缓冲、QK 不切 k（L0A = 1×16K + 2×18K = 52 KB 也放得下）。",
    tone: "warning",
  },
  {
    title: "同步频率变成 4 倍",
    body:
      "组首/组尾的跨核 flag 与 fixpipe 从每 16 拍一次变成每 4 拍一次，可能取代 MTE2 成为新瓶颈，需实测确认。",
    tone: "warning",
  },
  {
    title: "kvs 尾块补零要重推",
    body:
      "fp4 的「两个 plane 各补一次」在 1 字节/元素下退化为一次连续 Fill；非对齐分支需要重新推导。",
    tone: "warning",
  },
];

const NEW_FILES: string[] = [
  "arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h",
  "gemm/block/block_mmad_qk_arch35_mxfp8.hpp（含沿 k(D) 切 2 段）",
  "gemm/block/block_mmad_pv_arch35_mxfp8.hpp（含沿 k(S2) 切 2 段）",
  "gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp",
  "epilogue/block/block_epilogue_online_softmax_..._mxfp8.hpp",
  "epilogue/block/block_epilogue_compute_pscale_arch35_mxfp8.hpp",
  "epilogue/block/block_epilogue_rescale_o_..._mxfp8.hpp",
  "epilogue/block/mxfp8_vf/*（128 列分段 softmax + e4m3 cast/gather）",
  "tests/mxfp8_pytest/*",
];

const EDIT_FILES: string[] = [
  "arch35/kernel_utils.hpp（新增 MXFP8Kernel，TILE_GROUP_N = 4）",
  "gemm/block/block_mmad_arch35_utils.hpp（新增 MXFP8 命名空间 + static_assert）",
  "epilogue/block/block_epilogue_arch35_utils.hpp（新增 MXFP8 命名空间）",
  "gemm/bsa_gemm_dispatch_policy.hpp、gemm/block/bsa_block_mmad.hpp",
  "epilogue/bsa_epilogue_dispatch_policy.hpp、epilogue/block/bsa_block_epilogue.hpp",
  "block_sparse_attention_tilingkey.h、block_sparse_attention.cpp",
  "block_sparse_attention_kernel_interface.cpp",
  "op_host/block_sparse_attention_tiling.{h,cpp} 与算子 def 的 dtype 列表",
];

/* ------------------------------------------------------------------ */
/* 流水图                                                              */
/* ------------------------------------------------------------------ */

const G = 4; // 一组的基块数
const DPV = 8; // PV 延迟（单独的 D 指 embed 维 128，这里不复用）
const PSCALE_DELAY = 3;
const LOOPS = 20;

function PipelineTimeline() {
  const theme = useHostTheme();

  const cellW = 38;
  const cellH = 24;
  const rowGap = 6;
  const labelW = 152;
  const headerH = 30;
  const rows = 4;
  const width = labelW + LOOPS * cellW + 8;
  const height = headerH + rows * (cellH + rowGap) + 26;

  const rowLabels = [
    "基块归属 AIV",
    "写 P 槽（loop % 8）",
    "pScale 就绪（组尾 +3）",
    "PV 读 P 槽（延迟 8）",
  ];

  const rowY = (r: number) => headerH + r * (cellH + rowGap);
  const colX = (i: number) => labelW + i * cellW;

  // pScale 就绪的 loop：组尾 loop = g*G + G - 1，再 +3
  const pscaleReady = new Map<number, number>();
  for (let g = 0; g * G + G - 1 + PSCALE_DELAY < LOOPS; g++) {
    pscaleReady.set(g * G + G - 1 + PSCALE_DELAY, g);
  }

  const groupFill = (g: number) =>
    g % 2 === 0 ? theme.fill.tertiary : theme.fill.secondary;

  return (
    <Stack gap={8}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: "block", maxWidth: width }}
      >
        {/* 组边界与组标签 */}
        {Array.from({ length: Math.ceil(LOOPS / G) }).map((_, g) => (
          <g key={`grp-${g}`}>
            <rect
              x={colX(g * G)}
              y={headerH - 18}
              width={cellW * G - 2}
              height={14}
              rx={2}
              fill={groupFill(g)}
            />
            <text
              x={colX(g * G) + (cellW * G) / 2 - 1}
              y={headerH - 7}
              textAnchor="middle"
              fontSize={10}
              fill={theme.text.secondary}
            >
              {`组 ${g}`}
            </text>
            <line
              x1={colX(g * G) - 1}
              y1={headerH - 20}
              x2={colX(g * G) - 1}
              y2={rowY(rows - 1) + cellH + 4}
              stroke={theme.stroke.secondary}
              strokeWidth={1}
            />
          </g>
        ))}

        {/* 行标签 */}
        {rowLabels.map((lbl, r) => (
          <text
            key={`lbl-${r}`}
            x={labelW - 10}
            y={rowY(r) + cellH / 2 + 4}
            textAnchor="end"
            fontSize={11}
            fill={theme.text.secondary}
          >
            {lbl}
          </text>
        ))}

        {/* loop 轴 */}
        {Array.from({ length: LOOPS }).map((_, i) => (
          <text
            key={`ax-${i}`}
            x={colX(i) + cellW / 2 - 1}
            y={rowY(rows - 1) + cellH + 18}
            textAnchor="middle"
            fontSize={10}
            fill={theme.text.quaternary}
          >
            {i}
          </text>
        ))}

        {/* 行 0：AIV 归属 */}
        {Array.from({ length: LOOPS }).map((_, i) => (
          <g key={`aiv-${i}`}>
            <rect
              x={colX(i)}
              y={rowY(0)}
              width={cellW - 2}
              height={cellH}
              rx={2}
              fill={theme.fill.quaternary}
              stroke={theme.stroke.tertiary}
            />
            <text
              x={colX(i) + cellW / 2 - 1}
              y={rowY(0) + cellH / 2 + 4}
              textAnchor="middle"
              fontSize={11}
              fill={
                i % 2 === 0 ? theme.category.blue : theme.category.orange
              }
            >
              {i % 2 === 0 ? "V0" : "V1"}
            </text>
          </g>
        ))}

        {/* 行 1：写 P 槽 */}
        {Array.from({ length: LOOPS }).map((_, i) => (
          <g key={`wr-${i}`}>
            <rect
              x={colX(i)}
              y={rowY(1)}
              width={cellW - 2}
              height={cellH}
              rx={2}
              fill={groupFill(Math.floor(i / G))}
              stroke={theme.stroke.tertiary}
            />
            <text
              x={colX(i) + cellW / 2 - 1}
              y={rowY(1) + cellH / 2 + 4}
              textAnchor="middle"
              fontSize={11}
              fill={theme.text.primary}
            >
              {`#${i % DPV}`}
            </text>
          </g>
        ))}

        {/* 行 2：pScale 就绪 */}
        {Array.from({ length: LOOPS }).map((_, i) => {
          const g = pscaleReady.get(i);
          if (g === undefined) return null;
          return (
            <g key={`ps-${i}`}>
              <rect
                x={colX(i)}
                y={rowY(2)}
                width={cellW - 2}
                height={cellH}
                rx={2}
                fill={theme.category.green}
                opacity={0.85}
              />
              <text
                x={colX(i) + cellW / 2 - 1}
                y={rowY(2) + cellH / 2 + 4}
                textAnchor="middle"
                fontSize={10}
                fill={theme.text.onAccent}
              >
                {`g${g}`}
              </text>
            </g>
          );
        })}

        {/* 行 3：PV 读 P 槽 */}
        {Array.from({ length: LOOPS }).map((_, i) => {
          if (i < DPV) return null;
          const tile = i - DPV;
          return (
            <g key={`pv-${i}`}>
              <rect
                x={colX(i)}
                y={rowY(3)}
                width={cellW - 2}
                height={cellH}
                rx={2}
                fill={groupFill(Math.floor(tile / G))}
                stroke={theme.category.purple}
              />
              <text
                x={colX(i) + cellW / 2 - 1}
                y={rowY(3) + cellH / 2 + 4}
                textAnchor="middle"
                fontSize={11}
                fill={theme.text.primary}
              >
                {`#${tile % DPV}`}
              </text>
            </g>
          );
        })}

        {/* 关键箭头：组 0 的 pScale 就绪(loop 6) → PV 读组 0 首块(loop 8) */}
        <line
          x1={colX(6) + cellW / 2 - 1}
          y1={rowY(2) + cellH}
          x2={colX(8) + cellW / 2 - 1}
          y2={rowY(3)}
          stroke={theme.category.green}
          strokeWidth={1.5}
          strokeDasharray="3 2"
        />
      </svg>

      <Row gap={16} wrap>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: 2,
              background: theme.fill.tertiary,
            }}
          />
          <Text size="small" tone="secondary">
            偶数组
          </Text>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: 2,
              background: theme.fill.secondary,
            }}
          />
          <Text size="small" tone="secondary">
            奇数组
          </Text>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: 2,
              background: theme.category.green,
            }}
          />
          <Text size="small" tone="secondary">
            该组 pScale 就绪
          </Text>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: 2,
              border: `1px solid ${theme.category.purple}`,
            }}
          />
          <Text size="small" tone="secondary">
            PV 消费（紫边）
          </Text>
        </div>
      </Row>
      <Note>
        横轴：主循环 loop（= 全局基块序号）。`#n` 是 L1 的 P 槽号 = loop mod
        8。虚线是唯一的关键约束：组 0 的 pScale 在 loop 6 就绪，PV 在 loop 8
        才读组 0 的首块，留了 1 拍余量。来源：当前分支 MXFP4 实现 +
        本设计（G=4, Dpv=8, DELAY_PSCALE_N=3）。
      </Note>
    </Stack>
  );
}

function GroupShape() {
  const theme = useHostTheme();
  return (
    <Stack gap={8}>
      <Row gap={6} align="center" wrap>
        {[0, 1, 2, 3].map((i) => (
          <div
            key={`bt-${i}`}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              padding: "10px 14px",
              borderRadius: 4,
              border: `1px solid ${theme.stroke.secondary}`,
              background: theme.fill.quaternary,
            }}
          >
            <Text size="small" weight="semibold">
              基块 {i}
            </Text>
            <Text size="small" tone="tertiary">
              256×128
            </Text>
            <Text
              size="small"
              style={{
                color:
                  i % 2 === 0
                    ? theme.category.blue
                    : theme.category.orange,
              }}
            >
              {i % 2 === 0 ? "AIV0" : "AIV1"}
            </Text>
          </div>
        ))}
        <Text tone="tertiary">=</Text>
        <div
          style={{
            padding: "10px 16px",
            borderRadius: 4,
            border: `1px solid ${theme.accent.primary}`,
            background: theme.fill.tertiary,
          }}
        >
          <Text size="small" weight="semibold">
            一组 1024(kvs)×128(qs)
          </Text>
        </div>
      </Row>
      <Note>
        每个基块由两条 128 列的 QK mmad 拼成，整块交给一个 AIV
        做 softmax；一组里每个 AIV 拿 2 块。MXFP4 是 16 块一组（4096×128），
        每个 AIV 拿 8 块。
      </Note>
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/* 各分区                                                              */
/* ------------------------------------------------------------------ */

function MemorySection() {
  return (
    <Stack gap={22}>
      <Stack gap={10}>
        <H2>L1（512 KB）</H2>
        <Text tone="secondary">
          分配规则一句话：
          <Text weight="semibold">数据槽字节数翻倍，scale 槽字节数不变</Text>
          。后者的依据是
          `matmul.h` 里 fp8 的 `LoadData2DMxParams.yStep = (k+63)&gt;&gt;5&gt;&gt;1`
          与 fp4 的 `(singleK+63)/32/2` 是同一个公式——scale 平面的分形几何与元素位宽无关，
          只有数据侧的 C0 从 64 元素变 32 元素。
        </Text>
        <MemoryMap segments={L1_FP8} total={512} name="MXFP8 L1" />
        <Divider />
        <H3>对照：MXFP4 的 L1</H3>
        <MemoryMap segments={L1_FP4} total={512} name="MXFP4 L1" />
      </Stack>

      <Stack gap={10}>
        <H3>L1 明细</H3>
        <Table
          headers={["区", "偏移", "单槽", "槽数", "合计", "说明"]}
          rows={L1_TABLE}
          columnAlign={["left", "right", "right", "right", "right", "left"]}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <H2>L0（A/B 各 64 KB，C 256 KB）</H2>
        <Text tone="secondary">
          fp8 下真正的瓶颈是 L0 而不是 L1，而且操作数角色是固定的：现有 loader 是
          K→L0A、Q→L0B，L0C 的 m 方向才是 kvs、n 方向才是 qs，正好对上 UB 里 S =
          256(kvs)×128(qs)。
          <Text weight="semibold">把 Q 挪到 L0A 会让 L0C 出 Sᵀ</Text>
          ，下游整套 UB 布局都要推翻，所以 L0A 必须同时装 K 与 Vᵀ。照搬双缓冲要
          L0A 104 KB / L0B 96 KB；只让 PV 切 k 也还是 L0A 68 KB &gt; 64 KB。
        </Text>
        <Text tone="secondary">
          决策是{" "}
          <Text weight="semibold">
            QK 沿 k(D) 切 2 段各 64、PV 沿 k(S2) 切 2 段各 128，同一基块的两段累加进同一块
            L0C
          </Text>
          （首段沿用原有 initC 语义，第二段恒 false）。结果是 L0A/L0B
          的偏移布局与 MXFP4 完全一致，只有槽内几何变了。
        </Text>
        <Grid columns={1} gap={16}>
          <MemoryMap segments={L0A_FP8} total={64} name="L0A" />
          <MemoryMap segments={L0B_FP8} total={64} name="L0B" />
          <MemoryMap segments={L0C_FP8} total={256} name="L0C" />
        </Grid>
        <Callout tone="info" title="切 k 换来的三件事">
          K / Vᵀ / Pᵀ 全部保住双缓冲，MTE1 与 MMAD 可重叠；rowsum seed 从 36 KB
          降到 18 KB——seed 是借一个 KV 槽整槽填 1.0 再搬进 L0A 的，槽必须装得下
          seed 镜像，18 KB ≤ 32 KB 才让 L1 的 KV 槽维持 32 KB（否则要涨到 40
          KB）；k=64 仍是 MX scale 组（32 元素）的整数倍，scale 侧不用特殊处理。代价是每个基块的
          mmad 从 3 条变 6 条。
        </Callout>
        <Note>
          scale 平面不占 L0 预算：MXFP4 的 L0A_PV_BUF_SIZE = 18432 恰好等于 Vᵀ
          数据 144×256×0.5B、L0A_QK_BUF_SIZE = 8192 恰好等于 K 子块
          128×128×0.5B，说明 mx LoadData 的 scale 落在 cube 的专用 scale
          存储里，不在 L0A/L0B 地址空间。
        </Note>
      </Stack>

      <Stack gap={10}>
        <H2>UB（256 KB）</H2>
        <Text tone="secondary">
          fp8 下一个基块的 P 从 16 KB 变 32 KB。照搬「两块整基块交错」就是 P 区 64
          KB，即使 localGroupMax 从 20 KB 省到 8 KB，总量仍是 268 KB 超 12
          KB。决策是{" "}
          <Text weight="semibold">
            P 区仍 32 KB，两块各放半个基块（128 列 kvs）
          </Text>
          ，VF 按 128 列分两次算，算后半时前半正在拷 L1。
        </Text>
        <MemoryMap segments={UB_FP8} total={256} name="MXFP8 UB" />
        <Table
          headers={["区", "偏移", "大小", "说明"]}
          rows={UB_TABLE}
          columnAlign={["left", "right", "right", "left"]}
          striped
        />
        <Note>
          可行性依据：softmax 量化 P 时用的指数基准是 localGlobalMax（上一组交换得到），
          不是本基块自己的 max，因此按 128 列切分不改变数值；localGroupMax
          是跑动最大值，分两次更新即可。CopyPUBToPL1 已有「高低半区分别搬运」的分支可复用。
        </Note>
      </Stack>
    </Stack>
  );
}

function PipelineSection() {
  return (
    <Stack gap={22}>
      <Stack gap={10}>
        <H2>一组的形状</H2>
        <GroupShape />
      </Stack>

      <Stack gap={10}>
        <H2>P 槽在主循环上的生产–消费时序（G=4, Dpv=8）</H2>
        <PipelineTimeline />
      </Stack>

      <Stack gap={10}>
        <H2>延迟下界怎么来的</H2>
        <Text tone="secondary">
          组 g 覆盖基块 [gG, gG+G-1]。该组最后一个基块在 loop `gG+G-1` 出
          S；再过 `DELAY_PSCALE_N = 3` 拍，gmax 的 UB→L1 搬运与 pScale
          计算完成。PV 在 loop L 处理的是基块 L-Dpv，它处理组 g 首块的时刻必须严格晚于
          pScale 就绪，于是 <Text weight="semibold">Dpv ≥ G + 3</Text>。
        </Text>
        <Grid columns={3} gap={16}>
          <Stat value="7" label="G=4 时的下界 G+3" />
          <Stat value="8" label="实际取值 G+4" tone="info" />
          <Stat value="19 → 20" label="MXFP4 同口径（G=16）" />
        </Grid>
        <Note>
          取 8 而不是 7 的两个理由：与 MXFP4 一样留 1 拍余量；8 是偶数，两个 AIV
          各自静态分到 4 个 P 槽。环大小恰好等于 Dpv——槽 loop%8 在 loop+8
          被 PV 读，而主循环里 PV 块排在 softmax 写入之前，所以 8 个槽刚好不冲突。
        </Note>
      </Stack>

      <Stack gap={10}>
        <H2>常量改动</H2>
        <Table
          headers={["项目", "MXFP4", "MXFP8", "原因"]}
          rows={DELTA_TABLE}
          striped
        />
        <Callout tone="warning" title="组首/组尾事件频率 ×4">
          `isUpdatePScale` 与 `isTileGoupFirstTile` 的判据不变，只是模数从 16 变
          4。于是 PV 置 initC、fixpipe L0C→UB、跨核 pScale / C2_V2
          同步都从每 16 拍一次变成每 4 拍一次。
        </Callout>
      </Stack>
    </Stack>
  );
}

function DecisionSection() {
  const theme = useHostTheme();
  return (
    <Stack gap={22}>
      <Stack gap={10}>
        <H2>数据类型与量化链路</H2>
        <Table headers={["位置", "类型"]} rows={DTYPE_TABLE} striped />
        <Note>
          只做 OCP 模式，不做 MXFP4 的 CX 模式（log2Cx / log2CxCeil 保持缺省）。
          e4m3 有 3 位尾数（相对步长 12.5%），远好于 e2m1 的 1 位；组从 16
          基块缩到 4 基块后，固定指数基准更贴近各基块自身的 max，下溢更少。
        </Note>
      </Stack>

      <Stack gap={12}>
        <H2>风险</H2>
        {RISKS.map((r) => (
          <div key={r.title}>
            <Callout tone={r.tone} title={r.title}>
              {r.body}
            </Callout>
          </div>
        ))}
      </Stack>

      <Stack gap={12}>
        <H2>实施分期</H2>
        <Grid columns={3} gap={14}>
          <Card>
            <CardHeader trailing={<Pill size="sm">一期</Pill>}>
              cube 通路
            </CardHeader>
            <CardBody>
              <Text size="small" tone="secondary">
                常量表、fp8 的 L1→L0 loader、QK/PV 两个 block（都含切
                k）、dispatch policy、tiling key 与分发。pScale 先固定成
                0x7F（=1.0），绕开 group-max 与跨核交换，只验证 fp8
                的搬运与 mmad 数值——正好提前暴露 Vᵀ 转置风险。
              </Text>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm">二期</Pill>}>
              完整流水
            </CardHeader>
            <CardBody>
              <Text size="small" tone="secondary">
                TILE_GROUP_N = 4 的 group-max / pScale / dm 全链路，UB
                半基块双缓冲的 VF，延迟 8/3 的主循环与跨核同步。
              </Text>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm">三期</Pill>}>收尾</CardHeader>
            <CardBody>
              <Text size="small" tone="secondary">
                kvs 尾块补零与非对齐分支、精度对比、性能实测与瓶颈标注。
              </Text>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Stack gap={12}>
        <H2>文件清单</H2>
        <Grid columns={2} gap={16}>
          <Stack gap={8}>
            <H3>新增</H3>
            {NEW_FILES.map((f) => (
              <div key={f}>
                <Text size="small" tone="secondary">
                  <Code>{f}</Code>
                </Text>
              </div>
            ))}
          </Stack>
          <Stack gap={8}>
            <H3>修改</H3>
            {EDIT_FILES.map((f) => (
              <div key={f}>
                <Text size="small" tone="secondary">
                  {f}
                </Text>
              </div>
            ))}
          </Stack>
        </Grid>
      </Stack>

      <Stack gap={10}>
        <H2>验收标准</H2>
        <div
          style={{
            borderLeft: `2px solid ${theme.stroke.secondary}`,
            paddingLeft: 14,
          }}
        >
          <Stack gap={6}>
            <Text size="small" tone="secondary">
              新 tiling key 分支编译通过；L1 / L0 / UB 三张表的 static_assert 全部成立。
            </Text>
            <Text size="small" tone="secondary">
              与 golden 的精度对比达到与 MXFP4 用例同一口径的阈值。
            </Text>
            <Text size="small" tone="secondary">
              覆盖用例：kvs 满块 256、kvs 尾块（32 对齐 / 64 对齐 / 非对齐）、qs =
              128 与 64、kv tile 数小于 8（验证 drain 不死锁）。
            </Text>
            <Text size="small" tone="secondary">
              性能：记录 MXFP8 与 MXFP4 在同 shape 下的耗时比并标注瓶颈单元。
            </Text>
          </Stack>
        </div>
      </Stack>
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/* 顶层                                                                */
/* ------------------------------------------------------------------ */

const TABS = [
  { id: "mem", label: "内存图" },
  { id: "pipe", label: "流水图" },
  { id: "decide", label: "决策 · 风险 · 计划" },
];

export default function BsaMxfp8DesignReport() {
  const [tab, setTab] = useCanvasState("mxfp8-tab", "mem");

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1120 }}>
      <Stack gap={8}>
        <H1>BSA MXFP8 全量化设计</H1>
        <Text tone="secondary">
          在当前分支的 BSA MXFP4 全量化之上扩展 MXFP8：tile 几何不变（qs 128、kvs
          256、D 128），但一组从 16 个基块改成{" "}
          <Text weight="semibold">4 个基块（1024×128）</Text>；Q/K/V/P 全
          e4m3fn、scale 全 e8m0、MX 组 32，只做 OCP 模式。
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="4" label="一组的基块数 G" tone="info" />
        <Stat value="8" label="PV 延迟 Dpv = P 环深度" tone="info" />
        <Stat value="438 / 512 KB" label="L1 占用" />
        <Stat value="52 / 48 KB" label="L0A / L0B 占用（各 64 KB）" />
      </Grid>

      <Row gap={8} wrap>
        {TABS.map((t) => (
          <div key={t.id}>
            <Pill active={tab === t.id} onClick={() => setTab(t.id)}>
              {t.label}
            </Pill>
          </div>
        ))}
      </Row>

      <Divider />

      {tab === "mem" && <MemorySection />}
      {tab === "pipe" && <PipelineSection />}
      {tab === "decide" && <DecisionSection />}
    </Stack>
  );
}
