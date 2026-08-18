import {
  BarChart,
  Button,
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
  TodoListCard,
  UsageBar,
  useCanvasState,
  useHostTheme,
  type Color,
  type TextProps,
} from "cursor/canvas";

type ViewId = "precision" | "asbuilt" | "geometry";
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
                  R
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
        组 0 的 pScale 在 loop 10 就绪（组尾 loop7 +3）；PV 在 loop 12
        才读组首槽 0。Dpv=12 ≥ G+3。板上未改这条流水。
      </Note>
    </Stack>
  );
}

function PrecisionView() {
  return (
    <Stack gap={16}>
      <H2>官方套件满足率（dump-off）</H2>
      <BarChart
        categories={[
          "kvs128",
          "kvs64",
          "kvs160",
          "kvs256",
          "GQA192",
          "sparse",
        ]}
        series={[
          {
            name: "fulfill (%)",
            data: [99.7437, 99.7925, 99.9512, 99.9878, 99.855, 99.6368],
            tone: "success",
          },
        ]}
        beginAtZero={false}
        yMin={99.4}
        yMax={100.05}
        valueSuffix="%"
        showValues
        referenceLines={[
          { value: 99.5, label: "门槛 99.5%", tone: "warning" },
        ]}
        height={220}
      />
      <Note>
        Y 轴：NPU vs golden 满足率（%）。门槛 99.5%。Source: mxfp8_pytest ·
        dump-off · 2026-08-18 · NPU 3 · cann_bsa_0817。判据 rtol=1/128、atol=1e-4、maxRE &lt; 10。
      </Note>

      <Table
        headers={["case", "shape", "fulfill", "结果"]}
        columnAlign={["left", "left", "right", "left"]}
        rows={[
          ["qs128 / kvs128 dense", "b1 n1 s128", "99.74%", "过"],
          ["qs128 / kvs64 dense", "b1 n1 s64", "99.79%", "过"],
          ["qs128 / kvs160 tail", "b1 n1 s160", "99.95%", "过"],
          ["qs128 / kvs256 dense", "b1 n1 s256", "99.99%", "过"],
          ["GQA qs128 / kvs192", "n1=4 n2=1 s192", "99.86%", "过"],
          ["sparse qs128 / kvs256", "n1=2 n2=2 s256", "99.64%", "过（隔离复跑）"],
          ["qs64 / kvs128", "blockShapeX=64", "—", "tiling 失败"],
        ]}
        rowTone={[
          "success",
          "success",
          "success",
          "success",
          "success",
          "success",
          "danger",
        ]}
      />

      <Grid columns={2} gap={12}>
        <Callout tone="warning" title="qs64 未进精度">
          host CheckBlockShapeQuantConstraint 要求 float8 全量化
          blockShapeX==128。不是精度失败。qs64 VF 文件在仓库里，但 tiling
          拦在前面。
        </Callout>
        <Callout tone="info" title="同进程污染">
          全量一次跑时 qs64 的 tiling inner error 会把后续 sparse
          带崩。sparse 必须另起进程；官方 6 个 qs128 case 都过线。
        </Callout>
      </Grid>

      <H3>套件外（不纳入验收）</H3>
      <Table
        headers={["case", "fulfill（dump 开启时）", "说明"]}
        rows={[
          ["s2=32 单 tile", "≈98.91%", "kvs32 pad 修过后仍低于 99.5%"],
          ["s2=96 = 64+32", "≈99.15%", "含 kvs32 尾，未再 dump-off 复测"],
        ]}
        rowTone={["warning", "warning"]}
      />

      <TodoListCard
        defaultExpanded
        todos={[
          {
            id: "qs64",
            content: "qs64：放开 blockShapeX==128，接 qs64 VF",
            status: "pending",
          },
          {
            id: "s32",
            content: "套件外 s2=32 / s2=96 未达 99.5%",
            status: "pending",
          },
          {
            id: "perf",
            content: "性能 vs MXFP4；Q L0B 单缓冲是否瓶颈",
            status: "pending",
          },
        ]}
      />
    </Stack>
  );
}

function AsBuiltView() {
  return (
    <Stack gap={16}>
      <H2>原稿 vs 板上落地</H2>
      <Text>
        §1–§6 几何表仍然成立。下面是原稿没写死、板上用 dump
        钉死的约束；改这些会回退已过线的 qs128 套件。
      </Text>
      <Table
        headers={["项", "0817 原稿", "板上落地"]}
        rows={[
          ["QK / 基块 / G / Dpv", "64×128 / 128×128 / 8 / 12", "一致"],
          ["kvBaseTile", "128", "一致"],
          [
            "多 tile softmax",
            "未写死",
            "TILE_GROUP=8 + last-tile Vec2（同 mxfp4）",
          ],
          ["P snap", "未写 −2 vs −8", "NEG_TWO_VALE (−2)，同 mxfp4"],
          [
            "Cube dtype",
            "未写 LoadData/Mmad 一致性",
            "二者都必须是 mx_fp8_e4m3_t",
          ],
          [
            "P NZ",
            "两路 ZERO/ONE",
            "mxfp4 even/odd + VL128 + FP8 C0=32 K-outer",
          ],
          [
            "peer gmax",
            "组尾经 L1 交换",
            "asc_copy_l12ub_sync；UB 偏移 160KB+512",
          ],
          ["pScale 槽", "每基块 1 槽", "kvs128 = 1 slot；不可套 mxfp4 ×2"],
          [
            "kvs32 pad",
            "未写",
            "groupmax pad 满 4 组；PadPS2 128 行 × 4 C0",
          ],
          ["qs64", "验收覆盖", "host 仍要求 blockShapeX==128"],
        ]}
        rowTone={[
          "success",
          "success",
          "info",
          "info",
          "warning",
          "warning",
          "warning",
          "info",
          "warning",
          "danger",
        ]}
      />

      <H2>板上修过的根因</H2>
      <Stack gap={10}>
        <Callout tone="danger" title="1 · QK/PV Mmad 类型">
          LoadData 已是 mx，Mmad 仍用 fp8_e4m3fn_t → S 溢出 inf → 全
          NaN。FIA 同路径要求 Mmad 也是 mx_fp8_e4m3_t；LoadData srcStride=yStep。
        </Callout>
        <Callout tone="warning" title="2 · P store 平面">
          99.74% 路径：连续 4 行 load + 4-way Gather；even/odd j 的
          VL128 偏移；K-outer C0=32。CopyP 用 64 行 TLA 填 128 行 NZ；LoadP
          srcStride=8。
        </Callout>
        <Callout tone="warning" title="3 · 两 tile peer gmax">
          kvs256 曾 ~1.4%：两 AIV 独立 softmax。AIC 上 GetSubBlockIdx 恒
          0。必须 L1 row0→AIV0、row1→AIV1。AIV DataCopy L1→UB
          无效；cube GetValue dump 曾 507015。修完 99.99%。
        </Callout>
        <Callout tone="info" title="4 · kvs160 行 NaN">
          99.17%→99.95%。kvs32 VF 只 pad 1 个 groupmax，computePscale
          固定读 4 组，残留 Inf 变成 e8m0=255。PadPS2 曾按 64 行 2
          平面写，应对 128 行 × 4 个 FP8 C0。
        </Callout>
      </Stack>

      <H2>实施分期（as-built）</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader trailing={<Pill size="sm" active>已完成</Pill>}>
            1–5 · 几何 / Cube / Vec / VF / 调用链
          </CardHeader>
          <CardBody>
            <Text size="small">
              kvBaseTile=128，G=8，Dpv=12，quantMode=4，e4m3 NZ 与 pScale
              768B 已由 qs128 套件确认。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm" active>已完成</Pill>}>
            6 · qs128 精度
          </CardHeader>
          <CardBody>
            <Text size="small">
              dense 64/128/160/256、GQA 192、sparse 256 均 ≥99.5%。dump
              已关。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">待做</Pill>}>
            7 · qs64 / 短 kvs / 性能
          </CardHeader>
          <CardBody>
            <Text size="small">
              blockShapeX、套件外 s2=32/96、MXFP4 性能对比、可选 Q 切 D。
            </Text>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function GeometryView() {
  return (
    <Stack gap={16}>
      <H2>相对 0814 几何（未改）</H2>
      <Table
        headers={["项", "0814 稿", "0817 / as-built"]}
        rows={[
          ["QK 半面 → L0C", "128×128", "64×128"],
          ["Softmax 基块", "256×128（两轮拼）", "128×128（两轮 64 拼）"],
          ["组 G / 覆盖", "4 / 1024×128", "8 / 1024×128"],
          ["kvBaseTile", "256", "128"],
          ["Dpv / P 环", "8", "12"],
          ["基块 mmad 条数", "6（双切 k）", "3（默认不切 k）"],
          ["UB P", "半基块双缓冲", "整基块双缓冲"],
        ]}
      />

      <H2>流水常量</H2>
      <Table
        headers={["常量", "MXFP4", "0814", "0817 as-built"]}
        rows={[
          ["TILE_GROUP_N", "16", "4", "8"],
          ["PV_RE_DELAY_N", "20", "8", "12"],
          ["DELAY_PSCALE_N", "3", "3", "3"],
          ["PRELOAD_TASK_CACHE", "21", "9", "13"],
          ["L1_P_BUF_CNT", "20", "8", "12"],
          ["KVS_BASE_SIZE", "256", "256", "128"],
          ["L0C_QK 单槽", "64 KB", "64 KB", "32 KB"],
          ["peer gmax UB 偏移", "224KB+512", "—", "160KB+512"],
        ]}
      />
      <Callout tone="info" title="Dpv 下界">
        同拍先 PV 后 pScale：Dpv ≥ G+3 → G=8 时 ≥11，取 12。板上流水未改。
      </Callout>

      <H2>流水时间线</H2>
      <PipelineTimeline />

      <Divider />

      <H2>L1（512 KB）</H2>
      <MemoryMap segments={L1} total={512} name="L1" />
      <Table
        headers={["区", "偏移", "单槽", "槽数", "合计", "说明"]}
        rows={[
          ["P", "0", "16 KB", "12", "192 KB", "128×128×1B，loop % 12"],
          ["P-scale", "192 KB", "768 B", "12", "≈9 KB", "板上分形已确认"],
          ["Q", "≈201 KB", "16 KB", "2", "32 KB", "128×128×1B"],
          ["KV", "—", "32 KB", "4", "128 KB", "数据 16KB；余量给 18KB seed"],
          ["空闲", "—", "—", "—", "≈140 KB", "多于 0814 的 74 KB"],
        ]}
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
            <Note>Q 单缓冲；性能若成瓶颈再切 D。</Note>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>L0C · 136/256 KB</CardHeader>
          <CardBody>
            <MemoryMap segments={L0C} total={256} name="L0C" />
          </CardBody>
        </Card>
      </Grid>

      <H2>UB（256 KB）</H2>
      <MemoryMap segments={UB} total={256} name="UB" />
      <Table
        headers={["区", "大小", "说明"]}
        rows={[
          ["S", "64 KB", "2×32 KB fp16，128×128"],
          ["oTmp", "32 KB", "fp32"],
          ["P", "32 KB", "2×16 KB 整基块"],
          ["attentionOut", "32 KB", "不变"],
          ["localGroupMax", "12 KB", "Dpv=12"],
          ["peer gmax", "160KB+512", "不是 mxfp4 的 224KB+512"],
          ["合计", "≈185 KB", "余 ≈70 KB"],
        ]}
      />
    </Stack>
  );
}

export default function BsaMxfp8AsBuiltReport() {
  const [view, setView] = useCanvasState<ViewId>("view", "precision");

  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>BSA MXFP8 as-built（精度反向分析）</H1>
        <Row gap={8} wrap>
          <Pill active>2026-08-18 板上修订</Pill>
          <Pill>quantMode=4 · e4m3 / e8m0</Pill>
          <Pill>kvs128 · G=8 · Dpv=12</Pill>
        </Row>
        <Text>
          0817 几何（QK 半面 64×128、基块 128×128）没有改。dump-off
          全量后，官方 qs128 套件 6/6 过 99.5%；qs64 仍被 host tiling
          拦住。下面按实测把原稿补成 as-built。
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="6 / 6" label="qs128 套件过线" tone="success" />
        <Stat value="99.64%" label="最低 fulfill（sparse）" tone="success" />
        <Stat value="99.5%" label="门槛" />
        <Stat value="qs64" label="未进比对（tiling）" tone="warning" />
      </Grid>

      <Row gap={8} wrap>
        <Button
          variant={view === "precision" ? "primary" : "ghost"}
          onClick={() => setView("precision")}
        >
          精度
        </Button>
        <Button
          variant={view === "asbuilt" ? "primary" : "ghost"}
          onClick={() => setView("asbuilt")}
        >
          原稿到落地
        </Button>
        <Button
          variant={view === "geometry" ? "primary" : "ghost"}
          onClick={() => setView("geometry")}
        >
          几何与内存
        </Button>
      </Row>

      {view === "precision" ? <PrecisionView /> : null}
      {view === "asbuilt" ? <AsBuiltView /> : null}
      {view === "geometry" ? <GeometryView /> : null}

      <Divider />
      <Stack gap={4}>
        <H3>规格</H3>
        <Code>
          docs/superpowers/specs/2026-08-17-bsa-mxfp8-design.md · §15
        </Code>
        <Note>
          几何表仍是 §1–§6；板上约束与精度表在 §15。0814 规格视为被取代。
        </Note>
      </Stack>
    </Stack>
  );
}
