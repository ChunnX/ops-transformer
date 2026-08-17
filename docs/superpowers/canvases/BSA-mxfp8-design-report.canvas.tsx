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
  useHostTheme,
  type Color,
  type TextProps,
} from "cursor/canvas";

type MemSeg = { id: string; label: string; kb: number; color: Color };

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

const L1: MemSeg[] = [
  { id: "p", label: "P (12×16K)", kb: 192, color: "purple" },
  { id: "ps", label: "P-scale (12×0.75K)", kb: 9, color: "blue" },
  { id: "q", label: "Q (2×16K)", kb: 32, color: "green" },
  { id: "qs", label: "Q-scale", kb: 2, color: "cyan" },
  { id: "kv", label: "KV (4×32K，含 seed 余量)", kb: 128, color: "orange" },
  { id: "kvs", label: "KV-scale (4×2K)", kb: 8, color: "yellow" },
  { id: "gm", label: "localGlobalMax", kb: 2, color: "pink" },
];

const L0A: MemSeg[] = [
  { id: "k", label: "K 双缓冲 (2×8K = 64×128)", kb: 16, color: "green" },
  { id: "v", label: "Vᵀ 双缓冲 (2×18K = 144×128)", kb: 36, color: "purple" },
];

const L0B: MemSeg[] = [
  { id: "q", label: "Q 单缓冲 (1×16K)", kb: 16, color: "orange" },
  { id: "p", label: "Pᵀ 双缓冲 (2×16K)", kb: 32, color: "blue" },
];

const L0C: MemSeg[] = [
  { id: "qk", label: "QK (2×32K = 64×128×4B)", kb: 64, color: "orange" },
  { id: "pv", label: "PV (1×72K = 144×128×4B)", kb: 72, color: "purple" },
];

const UB: MemSeg[] = [
  { id: "s", label: "S fp16 (2×32K = 128×128)", kb: 64, color: "purple" },
  { id: "otmp", label: "oTmp fp32", kb: 32, color: "orange" },
  { id: "p", label: "P 整基块 (2×16K)", kb: 32, color: "blue" },
  { id: "o", label: "attentionOut", kb: 32, color: "green" },
  { id: "gmax", label: "localGroupMax (12×1K)", kb: 12, color: "yellow" },
  { id: "ps", label: "pScale 等杂项", kb: 13, color: "gray" },
];

const GEO: string[][] = [
  ["QK 半面 → L0C", "128×128", "64×128"],
  ["Softmax 基块", "256×128（两轮拼）", "128×128（两轮 64 拼）"],
  ["组 G / 覆盖", "4 / 1024×128", "8 / 1024×128（同量级）"],
  ["kvBaseTile", "256", "128"],
  ["Dpv / P 环", "8", "12"],
  ["基块 mmad 条数", "6（双切 k）", "3（默认不切 k）"],
  ["UB P", "半基块双缓冲", "整基块双缓冲"],
];

const CONST_TABLE: string[][] = [
  ["TILE_GROUP_N", "16", "4", "8"],
  ["PV_RE_DELAY_N", "20", "8", "12"],
  ["DELAY_PSCALE_N", "3", "3", "3"],
  ["PRELOAD_TASK_CACHE", "21", "9", "13"],
  ["L1_P_BUF_CNT", "20", "8", "12"],
  ["KVS_BASE_SIZE", "256", "256", "128"],
  ["L0C_QK 单槽", "64 KB", "64 KB", "32 KB"],
];

const L1_TABLE: string[][] = [
  ["P", "0", "16 KB", "12", "192 KB", "128×128×1B，loop % 12"],
  ["P-scale", "192 KB", "768 B", "12", "≈9 KB", "128×2×(128/64+1)"],
  ["Q", "≈201 KB", "16 KB", "2", "32 KB", "128×128×1B"],
  ["KV", "—", "32 KB", "4", "128 KB", "数据 16KB；余量给 18KB seed"],
  ["空闲", "—", "—", "—", "≈140 KB", "多于 0814 的 74 KB"],
];

const UB_TABLE: string[][] = [
  ["S", "64 KB", "2×32 KB fp16，128×128"],
  ["oTmp", "32 KB", "fp32"],
  ["P", "32 KB", "2×16 KB 整基块（不再半块）"],
  ["attentionOut", "32 KB", "不变"],
  ["localGroupMax", "12 KB", "Dpv=12"],
  ["合计", "≈185 KB", "余 ≈70 KB"],
];

const RISKS: { title: string; body: string; tone: "danger" | "warning" | "info" }[] =
  [
    {
      title: "Vᵀ dstStride 奇（rowsum pad）",
      body: "dstStride=9 仍可能走 fp8 分段转置。兜底 pad 到 160：L0A 56KB、L0C 208KB 仍可。",
      tone: "danger",
    },
    {
      title: "基块更碎 × Dpv 更深",
      body: "同覆盖 1024 S2 时 loop 约为 0814 的 2 倍，且 Dpv=12。Cube/同步是否打平「少切 k」收益需实测。",
      tone: "warning",
    },
    {
      title: "Q L0B 单缓冲",
      body: "默认省 L0B 顶满。若 Q 搬移成瓶颈，改 Q 沿 D 切 2×64 双缓冲（QK 变 4 条 mmad）。",
      tone: "warning",
    },
    {
      title: "Softmax VF NZ 密度",
      body: "qs128 VF 仍复用脚手架；KvsBase 已随 128 实例化，e4m3 打包密度需 golden（kvs 64/128/尾块）确认。",
      tone: "warning",
    },
  ];

const CHANGELOG: string[][] = [
  [
    "kernel_utils.hpp",
    "TILE_GROUP_N",
    "4 → 8",
  ],
  [
    "block_mmad_arch35_utils.hpp",
    "L1/L0/S2",
    "P 环 12×16K；QK_HALF=64；去 k-split；L0B_Q 单缓；L0C QK 32K×2 / PV@64K",
  ],
  [
    "block_epilogue_arch35_utils.hpp",
    "UB",
    "KVS=128；S 2×32K；整块 P；pScale/localGroupMax×12；偏移重排",
  ],
  [
    "block_mmad_qk_arch35_mxfp8.hpp",
    "QK",
    "n 向 2×64 拼基块；D 一次打满；Q→L0B[0]",
  ],
  [
    "block_mmad_pv_arch35_mxfp8.hpp",
    "PV",
    "去 S2 k-split；单次 mmad；P 槽 %12",
  ],
  [
    "kernel_arch35_mxfp8_full_quant.h",
    "流水",
    "PV_RE_DELAY_N 8→12（PRELOAD=13）",
  ],
  [
    "kernel_interface.cpp",
    "L0TileShapeQK",
    "<128,128,128> → <128,64,128>",
  ],
  [
    "softmax_*_mxfp8.hpp",
    "UB→L1",
    "整基块 16KB NZ 视图（UB_P_NZ_M=64）",
  ],
  [
    "block_sparse_attention_tiling.cpp",
    "Host",
    "kvBaseTile=128；blockShapeY 按 128 倍数校验",
  ],
];

const G = 8;
const DPV = 12;
const PSCALE_DELAY = 3;
const LOOPS = 24;

function PipelineTimeline() {
  const theme = useHostTheme();
  const cellW = 28;
  const cellH = 22;
  const rowGap = 5;
  const labelW = 168;
  const headerH = 28;
  const rows = 4;
  const width = labelW + LOOPS * cellW + 8;
  const height = headerH + rows * (cellH + rowGap) + 24;
  const rowLabels = [
    "基块归属 AIV",
    "写 P 槽 (loop%12)",
    "pScale 就绪 (组尾+3)",
    "PV 读 P (延迟 12)",
  ];
  const rowY = (r: number) => headerH + r * (cellH + rowGap);
  const colX = (i: number) => labelW + i * cellW;
  const groupFill = (g: number) =>
    g % 2 === 0 ? theme.fill.tertiary : theme.fill.secondary;

  return (
    <Stack gap={8}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: "block", maxWidth: width }}
      >
        {Array.from({ length: Math.ceil(LOOPS / G) }).map((_, g) => (
          <g key={`grp-${g}`}>
            <rect
              x={colX(g * G)}
              y={headerH - 16}
              width={cellW * G - 2}
              height={12}
              rx={2}
              fill={groupFill(g)}
            />
            <text
              x={colX(g * G) + (cellW * G) / 2}
              y={headerH - 6}
              textAnchor="middle"
              fontSize={9}
              fill={theme.text.secondary}
            >
              {`组${g}`}
            </text>
            <line
              x1={colX(g * G) - 1}
              y1={headerH - 18}
              x2={colX(g * G) - 1}
              y2={rowY(rows - 1) + cellH + 2}
              stroke={theme.stroke.secondary}
              strokeWidth={1}
            />
          </g>
        ))}
        {rowLabels.map((lbl, r) => (
          <text
            key={`lbl-${r}`}
            x={labelW - 8}
            y={rowY(r) + cellH / 2 + 3}
            textAnchor="end"
            fontSize={10}
            fill={theme.text.secondary}
          >
            {lbl}
          </text>
        ))}
        {Array.from({ length: LOOPS }).map((_, i) => (
          <text
            key={`ax-${i}`}
            x={colX(i) + cellW / 2}
            y={rowY(rows - 1) + cellH + 16}
            textAnchor="middle"
            fontSize={8}
            fill={theme.text.quaternary}
          >
            {i}
          </text>
        ))}
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
              x={colX(i) + cellW / 2}
              y={rowY(0) + cellH / 2 + 3}
              textAnchor="middle"
              fontSize={10}
              fill={i % 2 === 0 ? theme.category.blue : theme.category.orange}
            >
              {i % 2 === 0 ? "V0" : "V1"}
            </text>
          </g>
        ))}
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
              x={colX(i) + cellW / 2}
              y={rowY(1) + cellH / 2 + 3}
              textAnchor="middle"
              fontSize={10}
              fill={theme.text.primary}
            >
              {i % DPV}
            </text>
          </g>
        ))}
        {Array.from({ length: LOOPS }).map((_, i) => {
          const isReady = (i - PSCALE_DELAY + 1) % G === 0 && i >= PSCALE_DELAY;
          return (
            <g key={`ps-${i}`}>
              <rect
                x={colX(i)}
                y={rowY(2)}
                width={cellW - 2}
                height={cellH}
                rx={2}
                fill={isReady ? theme.category.green : theme.fill.quaternary}
                stroke={theme.stroke.tertiary}
              />
              {isReady ? (
                <text
                  x={colX(i) + cellW / 2}
                  y={rowY(2) + cellH / 2 + 3}
                  textAnchor="middle"
                  fontSize={9}
                  fill={theme.text.primary}
                >
                  ✓
                </text>
              ) : null}
            </g>
          );
        })}
        {Array.from({ length: LOOPS }).map((_, i) => {
          const active = i >= DPV;
          const src = active ? (i - DPV) % DPV : -1;
          return (
            <g key={`pv-${i}`}>
              <rect
                x={colX(i)}
                y={rowY(3)}
                width={cellW - 2}
                height={cellH}
                rx={2}
                fill={active ? theme.category.purple : theme.fill.quaternary}
                stroke={theme.stroke.tertiary}
                opacity={active ? 1 : 0.5}
              />
              {active ? (
                <text
                  x={colX(i) + cellW / 2}
                  y={rowY(3) + cellH / 2 + 3}
                  textAnchor="middle"
                  fontSize={10}
                  fill={theme.text.primary}
                >
                  {src}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <Note>
        约束：组 0 的 pScale 在 loop 10 就绪（组尾 loop7 +3）；PV 在 loop 12
        才读组首槽 0。Dpv=12 ≥ G+3。
      </Note>
    </Stack>
  );
}

export default function BsaMxfp8Kvs128DesignReport() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>BSA MXFP8 设计报告（kvs128 修订）</H1>
        <Row gap={8} wrap>
          <Pill tone="info">2026-08-17</Pill>
          <Pill tone="success">几何已落地 · mxfp8_0817</Pill>
          <Pill tone="warning">取代 0814 几何</Pill>
          <Pill>OCP · e4m3 / e8m0</Pill>
          <Pill>低精 softmax (S=fp16)</Pill>
        </Row>
        <Text>
          相对 0814：QK 半面改为 64×128，基块 128×128，组 ×8（覆盖仍
          1024），Dpv=12。换来整块 P、默认不切 k、更松的 UB/L0C。分支
          mxfp8_0817 上常量 / Cube / Host 几何迁移已完成。
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="64×128" label="QK 半面" tone="info" />
        <Stat value="128×128×8" label="基块 × 组" tone="info" />
        <Stat value="12" label="Dpv = P 环" />
        <Stat value="3" label="基块 mmad 条数" tone="success" />
      </Grid>

      <Divider />

      <H2>相对 0814 几何</H2>
      <Table
        headers={["项", "0814 稿", "本修订"]}
        rows={GEO}
        rowTone={GEO.map((_, i) => (i % 2 === 0 ? "neutral" : undefined))}
      />

      <H2>流水常量</H2>
      <Table
        headers={["常量", "MXFP4", "0814", "本修订"]}
        rows={CONST_TABLE}
      />
      <Callout tone="info" title="Dpv 下界">
        同拍先 PV 后 pScale：Dpv ≥ G+3 → G=8 时 ≥11，取 12（偶 + 余量）。
      </Callout>

      <H2>流水时间线</H2>
      <PipelineTimeline />

      <Divider />

      <H2>L1（512 KB）</H2>
      <MemoryMap segments={L1} total={512} name="L1" />
      <Table
        headers={["区", "偏移", "单槽", "槽数", "合计", "说明"]}
        rows={L1_TABLE}
      />

      <H2>L0</H2>
      <Grid columns={3} gap={12}>
        <Card>
          <CardHeader>L0A · 52/64 KB</CardHeader>
          <CardBody>
            <MemoryMap segments={L0A} total={64} name="L0A" />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>L0B · 48/64 KB</CardHeader>
          <CardBody>
            <MemoryMap segments={L0B} total={64} name="L0B" />
            <Note>Q 单缓冲；可选再切 D 双缓。</Note>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>L0C · 136/256 KB</CardHeader>
          <CardBody>
            <MemoryMap segments={L0C} total={256} name="L0C" />
          </CardBody>
        </Card>
      </Grid>
      <Callout tone="neutral" title="默认不切 k">
        K→L0A / Q→L0B 不变。基块 S2=128 使 Vᵀ/Pᵀ 全宽可装；QK 两轮只沿
        kvs 拼基块，不再沿 D/S2 切段累加（除非启用 Q 切 D 退路）。
      </Callout>

      <H2>UB（256 KB）</H2>
      <MemoryMap segments={UB} total={256} name="UB" />
      <Table headers={["区", "大小", "说明"]} rows={UB_TABLE} />

      <Divider />

      <H2>风险</H2>
      <Stack gap={10}>
        {RISKS.map((r) => (
          <div key={r.title}>
            <Callout tone={r.tone} title={r.title}>
              {r.body}
            </Callout>
          </div>
        ))}
      </Stack>

      <H2>实施分期</H2>
      <H2>实施分期</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader trailing={<Pill tone="success">已完成</Pill>}>
            1 · 常量 + tiling
          </CardHeader>
          <CardBody>
            <Text size="small">
              kvBaseTile=128，G=8，Dpv=12，三张内存表与 static_assert。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="success">已完成</Pill>}>
            2 · Cube
          </CardHeader>
          <CardBody>
            <Text size="small">
              QK 64×128×2、PV k=128、推荐 L0；去默认双切 k。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="success">已完成</Pill>}>
            3 · Vec + 流水
          </CardHeader>
          <CardBody>
            <Text size="small">
              整块 P / delay 12；VF NZ 密度待板上 golden。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="warning">待做</Pill>}>
            4 · 收尾
          </CardHeader>
          <CardBody>
            <Text size="small">
              尾块、精度、性能；按需启用 Q 切 D。
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H2>修改点说明（相对 0814 脚手架）</H2>
      <Callout tone="info" title="落地范围">
        9 个源文件；几何与流水常量已切到本修订。Softmax VF
        本体未重写，仅随 KVS_BASE=128 实例化并修正 UB→L1 16KB 布局。
      </Callout>
      <Table
        headers={["文件", "模块", "改动摘要"]}
        rows={CHANGELOG}
        rowTone={CHANGELOG.map((_, i) =>
          i % 2 === 0 ? "neutral" : undefined,
        )}
      />

      <Divider />
      <Stack gap={4}>
        <H3>规格文件</H3>
        <Code>
          docs/superpowers/specs/2026-08-17-bsa-mxfp8-design.md
        </Code>
        <Note>
          §14 为与代码同步的修改点说明；0814 规格视为被本修订取代。
        </Note>
      </Stack>
    </Stack>
  );
}
