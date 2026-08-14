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
/* 共用小组件                                                           */
/* ------------------------------------------------------------------ */

type MemSeg = {
  id: string;
  label: string;
  kb: number;
  color: Color;
};

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
      <UsageMapBar segments={segments} total={total} name={name} used={used} />
      <Legend segments={segments} />
    </Stack>
  );
}

function UsageMapBar({
  segments,
  total,
  name,
  used,
}: {
  segments: MemSeg[];
  total: number;
  name: string;
  used: number;
}) {
  return (
    <UsageBar
      total={total}
      topLeftLabel={`${name} 占用 ${((used / total) * 100).toFixed(1)}%`}
      topRightLabel={`${used} KB / ${total} KB（剩余 ${total - used} KB）`}
      segments={segments.map((s) => ({ id: s.id, value: s.kb, color: s.color }))}
    />
  );
}

function Note({ children }: TextProps) {
  return (
    <Text size="small" tone="tertiary">
      {children}
    </Text>
  );
}

/* ------------------------------------------------------------------ */
/* 数据                                                                */
/* ------------------------------------------------------------------ */

const L1_SEGS: MemSeg[] = [
  { id: "p", label: "P 数据 (20×16K)", kb: 320, color: "purple" },
  { id: "ps", label: "P-scale (20×1.25K)", kb: 25, color: "blue" },
  { id: "q", label: "Q 数据 (2×8K)", kb: 16, color: "green" },
  { id: "qs", label: "Q-scale (2×0.5K)", kb: 1, color: "cyan" },
  { id: "kv", label: "KV 共享池 (4×32K)", kb: 128, color: "orange" },
  { id: "kvs", label: "KV-scale (4×2K)", kb: 8, color: "yellow" },
  { id: "gm", label: "localGlobalMax (4×0.5K)", kb: 2, color: "pink" },
];

const UB_SEGS: MemSeg[] = [
  { id: "s", label: "S = MM1 结果 (2 槽交织)", kb: 128, color: "purple" },
  { id: "otmp", label: "OTmp = MM2 结果", kb: 32, color: "orange" },
  { id: "rs", label: "local/globalRowSum", kb: 0.5, color: "red" },
  { id: "p", label: "P 区（复用: oTrans + pScale）", kb: 32, color: "blue" },
  { id: "o", label: "attentionOut", kb: 32, color: "green" },
  { id: "max", label: "groupMax / gmax / dm / index", kb: 23.5, color: "yellow" },
];

const L1_TABLE: string[][] = [
  ["P 数据", "0", "320 KB", "16 KB", "20", "AIV 写入的 fp4 概率矩阵 (zN)，槽位 = loop % 20"],
  ["P-scale", "320 KB", "25 KB", "1280 B", "20", "P 的 E8M0 微缩放，128×2×(256/64+1)"],
  ["Q 数据", "345 KB", "16 KB", "8 KB", "2", "128×D fp4，每 task 只搬一次"],
  ["Q-scale", "361 KB", "1 KB", "512 B", "2", "128×D/32 E8M0"],
  ["KV 共享池", "362 KB", "128 KB", "32 KB", "4", "QK 存 K / PV 存 V，同一地址复用"],
  ["KV-scale", "490 KB", "8 KB", "2 KB", "4", "512×D/32 E8M0"],
  ["localGlobalMax", "498 KB", "2 KB", "512 B", "4", "2×128 half：两个 AIV 子核各一行"],
  ["（空闲）", "500 KB", "12 KB", "—", "—", "预留余量"],
];

const UB_TABLE: string[][] = [
  ["S (mm1Res)", "0", "128 KB", "64 KB×2", "2", "half，256(kvs)×256(qs×2)；双槽按列交织，槽间距 256 B"],
  ["OTmp (mm2Res)", "128 KB", "32 KB", "32 KB", "1", "float，(128+1)×qsAlign64，最后一行是 rowsum"],
  ["localRowSum", "160 KB", "256 B", "256 B", "1", "本 tile-group 的 P 行和（cube 算出）"],
  ["globalRowSum", "160 KB+256", "256 B", "256 B", "1", "跨 tile 累积行和，常驻"],
  ["P (vec1Res)", "160 KB+512", "32 KB", "16 KB×2", "2", "uint8 伪装的 fp4，双槽按列交织（槽间距 256 B）"],
  ["└ oTrans（复用 P）", "160 KB+512", "18 KB", "18 KB", "1", "最终输出转置暂存，与 P 同址"],
  ["└ pScale（复用 P）", "178 KB+512", "10 KB", "1280 B", "8", "P 的 E8M0 scale，落在 P 区尾部"],
  ["attentionOut", "192 KB+512", "32 KB", "32 KB", "1", "float 累积 O，末尾复用为 b16 输出"],
  ["peerGlobalMax", "224 KB+512", "1 KB", "256 B", "4", "从 L1 取回的对端子核 gmax"],
  ["softmaxMax", "225 KB+512", "256 B", "256 B", "1", "上一轮 gmax（算 dm 用），常驻"],
  ["localGroupMax", "225 KB+768", "20 KB", "1 KB", "20", "每 32 行一个 group 的 max 环形缓冲"],
  ["localGlobalMax", "245 KB+768", "1 KB", "256 B", "4", "本子核运行 gmax"],
  ["updateScale (dm)", "246 KB+768", "1 KB", "256 B", "4", "float 重标定因子 2^Δgmax"],
  ["Index", "247 KB+768", "256 B", "256 B", "1", "ND→NZ 的 Gather 索引表，常驻"],
  ["（空闲）", "248 KB", "8 KB", "—", "—", "预留余量"],
];

const L0_TABLE: string[][] = [
  ["L0A", "QK", "0", "8 KB", "2", "K 分片（左矩阵）"],
  ["L0A", "PV", "16 KB", "18 KB", "2", "Vᵀ + 16 行 rowsum seed"],
  ["L0B", "QK", "0", "8 KB", "2", "Q 分片（右矩阵）"],
  ["L0B", "PV", "16 KB", "18 KB", "2", "Pᵀ"],
  ["L0C", "QK", "0", "64 KB", "2", "Sᵀ = K·Qᵀ，float"],
  ["L0C", "PV", "128 KB", "72 KB", "1", "Oᵀ 144×qsAlign64，float"],
];

const CROSS_CORE: string[][] = [
  ["9 / 10", "V1 ↔ C1", "MM1 fixpipe 写完 UB S ↔ AIV softmax 读完，双槽各一个 flag"],
  ["0", "V2 → C2", "AIV rescale-O 用完 OTmp，放行 cube 下一次 MM2 fixpipe"],
  ["1", "C2 → V2", "MM2 fixpipe 写完 OTmp，通知 AIV 做 rescale"],
  ["2 / 3", "PScale → C2", "P-scale 写入 L1 完成，MM2 才能读（tile-group 首 tile 等）"],
  ["6 / 7", "gmax UB→L1", "AIV 把本核 gmax 写进 L1，cube 才能搬"],
  ["4 / 5", "gmax L1→UB", "cube 把对端 gmax 搬进 UB，AIV 才能算 pscale"],
  ["8", "gmax 回环", "四缓冲用满一轮后的回卷保护"],
];

const AIV_EVENTS: string[][] = [
  ["0 / 1", "MTE3_V / V_MTE3", "P 双缓冲：softmax 写 ↔ DataCopy 搬去 L1"],
  ["2–5", "MTE3_V / V_MTE3", "gmax UB→L1 四缓冲（+tileMaxIdx）"],
  ["6", "MTE3_V / V_MTE3", "attentionOut 缓冲（SYNC_ATTN_BUF_FLAG）"],
  ["7", "—", "SYNC_INIT_OUTPUT（输出初始化）"],
];

const AIC_EVENTS: string[][] = [
  ["2 / 3", "MTE1_MTE2 ↔ MTE2_MTE1", "QK：Q 的 L1 双缓冲"],
  ["4–7", "MTE1_MTE2 ↔ MTE2_MTE1", "QK/PV 共享：KV 池四缓冲（角色无关）"],
  ["2 / 3", "M_MTE1 ↔ MTE1_M", "QK：L0A/L0B 双缓冲"],
  ["2 / 3", "FIX_M ↔ M_FIX", "QK：L0C 双缓冲"],
  ["4 / 5", "M_MTE1 ↔ MTE1_M", "PV：L0A/L0B 双缓冲（数值复用，HardEvent 类型隔离）"],
  ["4", "FIX_M ↔ M_FIX", "PV：L0C 单缓冲"],
];

const PIPE_STAGES: string[][] = [
  [
    "当前 tile",
    "0",
    "MM1：K·Qᵀ → UB S",
    "online softmax + fp4 量化 → L1 P；gmax → L1",
    "每 tile",
  ],
  [
    "tile − 3",
    "DELAY_PSCALE_N = 3",
    "gmax L1 → UB（取对端子核）",
    "computePScale：算 E8M0 P-scale + dm → L1",
    "tile-group 边界",
  ],
  [
    "tile − 20",
    "PV_RE_DELAY_N = 20",
    "MM2：Vᵀ·P → UB OTmp（含 rowsum）",
    "rescale-O：O = O·dm + PV；末 tile 归一化写 GM",
    "tile-group 边界",
  ],
];

const IO_TABLE: string[][] = [
  ["query / key / value", "DT_FLOAT4_E2M1", "TND [T,N,D] / BNSD [B,N,S,D] / BSND [B,S,N,D]", "D ∈ {64,128}，设备侧按 fp4x2_e2m1_t 打包"],
  ["qDequantScale", "DT_FLOAT8_E8M0", "BNSD [B,N1,S1,D/64,2]", "沿 D 分组，组大小 64"],
  ["kDequantScale", "DT_FLOAT8_E8M0", "BNSD [B,N2,S2,D/64,2]", "沿 D 分组，组大小 64"],
  ["vDequantScale", "DT_FLOAT8_E8M0", "BNSD [B,N2,⌈S2/64⌉,D,2]", "沿 S 分组（因为 PV 的归约轴是 S）"],
  ["blockSparseMask", "DT_INT8", "[B,N1,⌈S1/bX⌉,⌈S2/bY⌉]", "块级稀疏图，必选"],
  ["attentionOut", "FP16 / BF16", "同 query", "反量化后的高精度输出"],
  ["softmaxLse", "—", "—", "MXFP4 路径不支持，host 直接拦截"],
];

const HOST_FIELDS: string[][] = [
  ["quantMode", "2 = OCP，3 = CX", "选择 MXQuantMode 模板参数"],
  ["log2Cx", "CX: log2(dstTypeMax)，默认 log2(6)=2.585", "kernel 取负后作为 NEG_LOG2_CX 传给 softmax"],
  ["log2CxCeil", "CX: ⌈log2(dstTypeMax)⌉ = 3", "传给 computePScale 做 scale 校正"],
  ["scaleValue", "attr，通常 1/√D", "softmax 前的缩放"],
  ["qBaseTile", "min(blockShapeX, 128) → 128", "M 方向基本块"],
  ["kvBaseTile", "256（D ≤ 128 且 innerPrecise=4）", "N/K 方向基本块，决定 KVS_BASE_SIZE"],
  ["totalTaskNum", "Σ_b qSTileNum(b) × N1", "一个 task = (batch, qHead, qSTile)"],
  ["coreTaskNum", "⌊totalTaskNum / aicNum⌋", "余数摊给前若干核 +1"],
  ["selectIdxSize", "B×N1×xBlkAligned×yBlkAligned×4 B", "workspace 内 sparseIdx"],
  ["selectNumIdxSize", "B×N1×xBlkAligned×4 B", "workspace 内 sparseCount"],
];

const CONSTRAINTS: string[][] = [
  ["芯片", "仅 Ascend 950 / arch35（__CCE_AICORE__ == 310）"],
  ["D", "64 或 128"],
  ["innerPrecise", "必须为 4（LOW_HIGH_MIXED），否则 kvBaseTile 拿不到 256"],
  ["blockShapeX", "必须是 64 的倍数（MXFP4 专有校验）"],
  ["blockShapeY", "必须是 128 的倍数"],
  ["dstTypeMax", "CX 模式下为 0（取默认）或落在 [6.0, 12.0]"],
  ["不支持", "softmaxLse、attenMask、blockTable、pQuantScale 一律拦截"],
];

const TILING_KEYS: string[][] = [
  ["9050000030400032", "TND", "OCP", "BF16"],
  ["9050000030400033", "TND", "OCP", "FP16"],
  ["9050000050400033", "BNSD", "OCP", "BF16"],
  ["9050000060400034", "BSND", "OCP", "BF16"],
  ["9050000030400042", "TND", "CX", "BF16"],
  ["9050000030400043", "TND", "CX", "FP16"],
];

/* ------------------------------------------------------------------ */
/* 各分页                                                              */
/* ------------------------------------------------------------------ */

function Overview() {
  const theme = useHostTheme();
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={12}>
        <Stat value="500 / 512 KB" label="L1 已分配" />
        <Stat value="248 / 256 KB" label="UB 已分配" />
        <Stat value="20 tiles" label="PV 流水延迟" tone="info" />
        <Stat value="32" label="MX 微缩放组大小" />
      </Grid>

      <Stack gap={8}>
        <H2>这条路径在做什么</H2>
        <Text>
          BSA 的 MXFP4 是<Text weight="semibold">全量化（full quant）</Text>
          注意力：不仅 Q/K/V 以 fp4_e2m1 + E8M0 微缩放存放，中间的概率矩阵 P
          也被重新量化成 fp4 再喂回 cube。因此 MM1 和 MM2
          两次矩阵乘都跑在硬件原生的 MX 通路上，反量化由{" "}
          <Code>LoadData</Code> 在 L1→L0 阶段完成，fixpipe 不做 quant
          （<Code>ScaleGranularity::NO_QUANT</Code>）。
        </Text>
        <Text>
          代价是需要一整套"缩放因子的簿记"：分组最大值、跨子核全局最大值、P
          的 per-group scale、以及在线 softmax 的重标定因子 dm。这套簿记正是
          MXFP4 相比 BF16 路径多出来的全部复杂度，也是流水被拆成三条延迟游标的原因。
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>数据流</H2>
        <Row gap={0} align="stretch" wrap={false} style={{ overflowX: "auto" }}>
          {[
            { t: "GM", s: "Q/K/V fp4 + E8M0" },
            { t: "L1", s: "稀疏 gather 后的 K/V" },
            { t: "L0", s: "LoadData 带 MX 反量化" },
            { t: "MMAD", s: "Sᵀ = K·Qᵀ" },
            { t: "UB", s: "S (half)" },
            { t: "AIV", s: "group-max softmax → fp4 P" },
            { t: "L1", s: "P (zN) + P-scale" },
            { t: "MMAD", s: "Oᵀ = Vᵀ·P (M=144)" },
            { t: "AIV", s: "O·dm + PV → 归一化" },
            { t: "GM", s: "attentionOut b16" },
          ].map((n, i) => (
            <div
              key={n.t + i}
              style={{ display: "flex", alignItems: "center" }}
            >
              <Stack
                gap={2}
                style={{
                  padding: "8px 10px",
                  border: `1px solid ${theme.stroke.tertiary}`,
                  borderRadius: 4,
                  background: theme.fill.quaternary,
                  minWidth: 108,
                }}
              >
                <Text size="small" weight="semibold">
                  {n.t}
                </Text>
                <Text size="small" tone="tertiary">
                  {n.s}
                </Text>
              </Stack>
              {i < 9 && (
                <Text size="small" tone="quaternary" style={{ padding: "0 6px" }}>
                  →
                </Text>
              )}
            </div>
          ))}
        </Row>
        <Note>
          实现文件：<Code>op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp4_full_quant.h</Code>
        </Note>
      </Stack>

      <Stack gap={10}>
        <H2>算子输入输出</H2>
        <Table
          headers={["张量", "dtype", "shape（以 BNSD 为例）", "说明"]}
          rows={IO_TABLE}
        />
        <Note>
          三种 layout 的 scale 形状不同：TND 少一维 batch（V-scale 为 [BTkv,N2,D,2]，
          BTkv = Σ_b ⌈S2_b/64⌉）；BSND 把 N 与 S 调换。校验见{" "}
          <Code>ValidateMxfp4DequantScale</Code>。
        </Note>
      </Stack>
    </Stack>
  );
}

function Pipeline() {
  const theme = useHostTheme();
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H2>一个 while 循环，三个延迟游标</H2>
        <Text>
          整个 kernel 只有一层主循环（<Code>coreTileLoop</Code>），AIC 与 AIV
          共用同一份循环骨架，靠 <Code>ASCEND_IS_AIC</Code>{" "}
          分叉。循环里同时推进三个逻辑阶段，各自有独立的 task/tile 游标与
          "已完成 tile 数"累加器，这样后级阶段可以落后前级若干个 tile
          而不必阻塞前级。
        </Text>
      </Stack>

      <Table
        headers={["阶段", "延迟", "AIC（cube）做什么", "AIV（vector）做什么", "触发频率"]}
        rows={PIPE_STAGES}
      />

      <Callout tone="info" title="为什么是 20 和 3">
        <Text>
          <Code>PV_RE_DELAY_N = 20</Code> 与 L1 的 P
          槽位数 20 一一对应：MM2 落后 20 个 tile，正好等于 P
          环形缓冲绕一圈的长度，因此 AIV 写 P 时永远不会追上还没被 MM2
          读走的槽位，无需额外反压。<Code>DELAY_PSCALE_N = 3</Code>
          则只要覆盖 "gmax 写 L1 → cube 搬回 UB → 算 pscale" 这条往返链路的深度。
          <Code>PRELOAD_TASK_CACHE_SIZE = 21</Code> 即 20 + 1，用来缓存 TaskInfo/TileInfo。
        </Text>
      </Callout>

      <Stack gap={10}>
        <H2>tile-group：16 个 tile 一组</H2>
        <Text>
          <Code>TILE_GROUP_N = 16</Code>。只有在 tile-group 边界（或 task
          最后一个 tile）才会触发 <Code>isUpdatePScale</Code>，也就是说 MM2、
          pscale 更新、O 的重标定都是每 16 个 KV tile 才做一次，而 MM1 + softmax
          每个 tile 都做。这把"重"的归约与重标定摊薄了 16 倍。
        </Text>
        <Row gap={3} wrap>
          {Array.from({ length: 34 }, (_, i) => {
            const isGroupEnd = (i + 1) % 16 === 0;
            return (
              <div
                key={i}
                title={`tile ${i}`}
                style={{
                  width: 20,
                  height: 26,
                  borderRadius: 3,
                  border: `1px solid ${
                    isGroupEnd ? theme.accent.primary : theme.stroke.tertiary
                  }`,
                  background: isGroupEnd
                    ? theme.accent.primary
                    : theme.fill.quaternary,
                }}
              />
            );
          })}
        </Row>
        <Note>
          浅格 = 只做 MM1 + softmax；高亮格（第 16、32…个 tile）= 额外触发 MM2 +
          computePScale + rescale-O。
        </Note>
      </Stack>

      <Stack gap={10}>
        <H2>双 AIV 子核如何分工</H2>
        <Text>
          一个 AIC 配两个 AIV 子核。<Code>subBlockIdx = curTileInfo.loop % 2</Code>
          ，即相邻两个 KV tile 的 softmax 交替派给两个子核，cube 的 fixpipe
          也按这个 subBlockIdx 决定写哪半个 UB。两个子核各自维护自己的
          group max，因此在算 P-scale 之前必须先通过 L1
          交换、取 max，得到真正的全局最大值 —— 这就是{" "}
          <Code>localGlobalMax → L1 → peerGlobalMax</Code> 往返链路存在的原因。
        </Text>
        <Text>
          最终输出阶段换成另一种切分：按 Q 行切一半（
          <Code>splitMSizePerCore = 64</Code>），两个子核各写 64 行到 GM。
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>核间同步 flag（CrossCoreSetFlag / WaitFlag，SYNC_MODE_4）</H2>
        <Table headers={["flag ID", "方向", "含义"]} rows={CROSS_CORE} />
        <Note>
          成对出现的 <Code>flag</Code> 与 <Code>flag + 16</Code>{" "}
          用于覆盖两个 AIV 子核；AIC 侧两次 wait，AIV 侧各自 set 一次。
        </Note>
      </Stack>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>AIV 核内 event（HardEvent）</CardHeader>
          <CardBody style={{ padding: 0 }}>
            <Table
              framed={false}
              headers={["ID", "类型", "保护对象"]}
              rows={AIV_EVENTS}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>AIC 核内 event（HardEvent）</CardHeader>
          <CardBody style={{ padding: 0 }}>
            <Table
              framed={false}
              headers={["ID", "类型", "保护对象"]}
              rows={AIC_EVENTS}
            />
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="neutral" title="event ID 数值复用是安全的">
        <Text>
          QK 的 <Code>KV_EVENT0/1</Code>（值 4、5）与 PV 的 L0AB event
          数值相同，但前者是 <Code>MTE1_MTE2</Code> 类型、后者是{" "}
          <Code>M_MTE1</Code> 类型。硬件按 (HardEvent 类型, ID)
          二元组独立存储 flag，因此不冲突。这也是 <Code>block_mmad_arch35_utils.hpp</Code>{" "}
          里那段注释的结论。
        </Text>
      </Callout>

      <Stack gap={10}>
        <H2>缓冲轮转索引一览</H2>
        <Table
          headers={["资源", "槽数", "索引表达式", "切换时机"]}
          rows={[
            ["UB S", "2", "loop / 2 % 2", "每两个 tile（两子核共享一槽）"],
            ["UB P", "2", "loop / 2 % 2", "同上"],
            ["UB localGroupMax", "20", "loop / 2 * 2 % 20", "环形，容纳 16-tile group"],
            ["UB pScale", "8", "firstLoop / secondLoop 计算", "tile-group 内分两段"],
            ["L1 P / P-scale", "20", "loop % 20", "每 tile"],
            ["L1 KV 池", "4", "kvBufId % 4", "QK 与 PV 共用同一游标（引用传递）"],
            ["L1 Q", "2", "qBufId % 2", "每 task"],
            ["gmax（L1/UB 四缓冲）", "4", "tileMaxIdx = (tileMaxIdx+1) % 4", "tile-group 首 tile"],
            ["dm (updateScale)", "4", "updateScaleIdx = (idx+1) % 4", "tile-group 首 tile"],
          ]}
        />
      </Stack>
    </Stack>
  );
}

function Memory() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>L1（512 KB）</H2>
        <Text>
          常量定义在 <Code>Gemm::Block::MXFP4</Code> 命名空间（
          <Code>block_mmad_arch35_utils.hpp</Code>）。所有区域首尾相接、完全连续，
          分配到 500 KB 后留 12 KB 余量。
        </Text>
        <MemoryMap segments={L1_SEGS} total={512} name="L1" />
        <Table
          headers={["区域", "起始偏移", "总大小", "单槽", "槽数", "用途"]}
          rows={L1_TABLE}
          columnAlign={["left", "right", "right", "right", "right", "left"]}
        />
      </Stack>

      <Callout tone="info" title="L1 的主要开销是 P 的 20 路环形缓冲">
        <Text>
          320 KB —— 也就是 L1 的 62% —— 全给了 P。这是流水延迟直接换来的内存代价：
          MM2 落后 20 个 tile，就必须留住 20 份 P 不被覆盖。KV
          池则反过来做了激进的复用：QK 阶段装 K、PV 阶段装 V，共用同一批地址和同一批
          event（4 路），因为二者在时间上被 20 个 tile 的延迟天然隔开。
        </Text>
      </Callout>

      <Divider />

      <Stack gap={8}>
        <H2>UB（256 KB）</H2>
        <Text>
          常量定义在 <Code>Epilogue::Block::MXFP4</Code> 命名空间（
          <Code>block_epilogue_arch35_utils.hpp</Code>）。同样完全连续，用到 248 KB。
        </Text>
        <MemoryMap segments={UB_SEGS} total={256} name="UB" />
        <Table
          headers={["区域", "起始偏移", "总大小", "单槽", "槽数", "用途"]}
          rows={UB_TABLE}
          columnAlign={["left", "right", "right", "right", "right", "left"]}
        />
      </Stack>

      <Callout tone="warning" title="两处非直觉的分配，读代码时最容易看错">
        <Stack gap={8}>
          <Text>
            <Text weight="semibold">1）双缓冲不是"隔 64 KB"，而是"隔 256 字节"。</Text>{" "}
            <Code>UB_S_INNER_BUF_OFFSET = 256</Code>，构造时按{" "}
            <Code>OFFSET + 256 * i</Code> 取指针。S 实际是一个 256×256 的 half
            矩阵（行距 512 B），两个 ping-pong 槽是它的左右两半{" "}
            <Text weight="semibold">按列交织</Text>，槽 1 的首地址就在槽 0
            往后 256 B（= 128 个 half = 128 列）。P 区同理。所以这两块的"单槽"在地址上
            不连续，不能按连续 16 KB/64 KB 理解。
          </Text>
          <Text>
            <Text weight="semibold">2）P / oTrans / pScale 三者同址复用。</Text>{" "}
            <Code>UB_P_BUF_OFFSET == UB_O_TRANS_BUF_OFFSET == 160 KB + 512</Code>
            ，而 pScale 起点 178 KB+512 正好是 oTrans（18 KB）的末尾，落在 P
            区的 32 KB 之内。三者靠 event 做时间互斥：
            <Code>computePScale</Code> 与 <Code>processOut</Code> 都会先{" "}
            <Code>WaitFlag(MTE3_V)</Code> 拿走{" "}
            <Code>SYNC_VEC1_RES_BUF0/1</Code> 两个 P 槽的令牌，用完再 set 回去，
            期间 softmax 无法写 P。
          </Text>
        </Stack>
      </Callout>

      <Divider />

      <Stack gap={8}>
        <H2>L0A / L0B / L0C</H2>
        <Text>
          QK 与 PV 的 L0 区域刻意错开：L0A/L0B 里 QK 用 [0, 16 KB)、PV 用 [16 KB, 52
          KB)；L0C 里 QK 用 [0, 128 KB)、PV 用 [128 KB, 200 KB)。互不重叠，
          于是两条流水可以真正并行而不需要互等。
        </Text>
        <Table
          headers={["Buffer", "阶段", "偏移", "单槽", "槽数", "内容"]}
          rows={L0_TABLE}
          columnAlign={["left", "left", "right", "right", "right", "left"]}
        />
        <Note>
          PV 的 L0C 只有 1 槽（72 KB = 144 × 128 × 4 B），因为 MM2 每 16 个 tile
          才做一次，没有连续背靠背发射的压力。
        </Note>
      </Stack>
    </Stack>
  );
}

function CubeSide() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H2>MM1：为什么算的是 Sᵀ = K·Qᵀ</H2>
        <Text>
          MXFP4 的 QK block 被硬特化为{" "}
          <Code>MmadAtlasA5BsaQK&lt;true&gt;</Code>（<Code>transposedMm1 = true</Code>
          ），K 进 L0A（左）、Q 进 L0B（右），MMAD 天然得到{" "}
          <Code>[kvs, qs]</Code> 形状的 Sᵀ。三个理由叠在一起：
        </Text>
        <Stack gap={6}>
          <Text>
            · <Text weight="semibold">softmax 的归约方向</Text>：MX
            量化组是沿 KV 方向每 32 个元素一组，Sᵀ 让"32 行一组"在 UB
            里是连续的 32 行 × 128 列，VF 可以直接按行 LoadAlign。
          </Text>
          <Text>
            · <Text weight="semibold">双子核切分</Text>：fixpipe 按{" "}
            <Code>subBlockIdx</Code> 切 Q 列，正好落在 Sᵀ 的列方向。
          </Text>
          <Text>
            · <Text weight="semibold">下游 P 的布局</Text>：P 以 zN 存进 L1，
            PV 再以 transpose 方式 L1→L0B，整条链路的转置只在 LoadData
            里发生一次，不需要额外的搬移。
          </Text>
        </Stack>
      </Stack>

      <Card>
        <CardHeader>MM1 单 tile 时序（AIC）</CardHeader>
        <CardBody>
          <Stack gap={6}>
            <Text size="small">
              1. <Code>CrossCoreWaitFlag(V1_C1[mm1ResBufId])</Code> —— 等 AIV
              腾出 UB S 槽
            </Text>
            <Text size="small">
              2. 若 task 首 tile：<Code>WaitFlag(MTE1_MTE2, qEvent)</Code> → MTE2
              搬 Q + Q-scale（<Code>Nd2Nz</Code>，scale 走{" "}
              <Code>CopyGmToL1MxScaleDn2NzA5</Code>）
            </Text>
            <Text size="small">
              3. <Code>WaitFlag(MTE1_MTE2, kEvent)</Code> → MTE2 稀疏 gather K +
              K-scale（<Code>SparseKFusedBaseTileL1FullLoad</Code>，数据与 scale
              在同一循环内融合搬运）
            </Text>
            <Text size="small">
              4. 对每个 n 子块：MTE1 <Code>LoadData(dst, src, scale, ...)</Code>{" "}
              把 Q→L0B、K→L0A，<Text weight="semibold">MX 反量化在此完成</Text>
            </Text>
            <Text size="small">
              5. MMAD <Code>K·Qᵀ → L0C</Code>（<Code>initC = true</Code>，每个 n
              子块独立）
            </Text>
            <Text size="small">
              6. FIXPIPE <Code>L0C → UB S</Code>，<Code>NO_QUANT</Code>、按{" "}
              <Code>SPLIT_N</Code> 分给两个 AIV
            </Text>
            <Text size="small">
              7. <Code>CrossCoreSetFlag(V1_C1[...])</Code>，KV 游标 +1
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Stack gap={8}>
        <H2>MM2：借 MMAD 顺手算出行和</H2>
        <Text>
          在线 softmax 需要 P 的行和。MXFP4 没有单独用 vector 去 reduce，而是把它
          折进 MM2 的 MMAD 里：
        </Text>
        <Table
          headers={["常量", "值", "含义"]}
          rows={[
            ["ROW_SUM_NUM", "128", "真正的 embed 行数（Oᵀ 的行）"],
            ["ROW_SUM_PAD_NUM", "16", "追加的 pad 行"],
            ["PV_MMAD_M_DIM", "144", "MMAD 的 M = 128 + 16"],
            ["SEED_V_DATA_FILL", "0x6666", "fp4 打包的 4.0"],
            ["SEED_V_SCALE_FILL", "0x7D7D", "E8M0 打包的 0.25"],
          ]}
        />
        <Text>
          <Code>InitL0BufferForReduceSum()</Code> 把 L0A 尾部 16 行填成 fp4 4.0
          配 scale 0.25，反量化后等于 <Text weight="semibold">1.0</Text>。于是{" "}
          <Code>Vᵀ·P</Code> 的第 129 行就是{" "}
          <Code>Σ_kv P[kv, q]</Code>，也就是行和。UB 侧对应的 tla 形状正是{" "}
          <Code>(ROW_SUM_NUM + 1, qsActBaseTileAlign64)</Code>，行距固定 64。
          seed 会在 <Code>isLastSecondKvsTile</Code> 时重新填一次，供下一个
          tile-group 使用。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>稀疏 gather：把不连续的 KV 块拼成一个 base tile</H2>
        <Text>
          <Code>kvBaseTile = 256</Code> 而 <Code>blockShapeY = 128</Code>，
          所以一个 base tile 通常横跨 2 个稀疏块，且起点可能落在块中间。核心几行：
        </Text>
        <Stack gap={5}>
          <Text size="small">
            <Code>firstYBlockIdx = (gatheredKvSTileIdx * kvSBaseTile) / blockShapeY</Code>
          </Text>
          <Text size="small">
            <Code>oriYBlockIdx = gSparseBlockIdx.GetValue(firstYBlockIdx)</Code> ——
            压缩序号 → 原始序号
          </Text>
          <Text size="small">
            <Code>yBlockInnerStart = gatheredKvSTileIdx * kvSBaseTile − firstYBlockIdx * blockShapeY</Code>
          </Text>
          <Text size="small">
            <Code>oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart</Code>{" "}
            —— 真实 GM 行偏移
          </Text>
        </Stack>
        <Text>
          循环里会向前探一格，若相邻的 <Code>oriYBlockIdx</Code> 在 GM
          上连续就合并成一次 <Code>Nd2Nz</Code>，减少 MTE2 指令数。
          K-scale 的偏移同步换算为{" "}
          <Code>oriStartOffset * kvHeadMul * (scaleK / 2)</Code>；V-scale
          因为沿 S 分组，换算为 <Code>oriStartOffset / 32</Code>。
        </Text>
      </Stack>

      <Callout tone="neutral" title="BSND / TND 的 stride 修正">
        <Text>
          头内 token stride 在 BNSD 下是 D（头内连续），BSND/TND 下是 N×D。
          代码用 <Code>kvHeadMul</Code> / <Code>qHeadMul</Code>（BNSD 取 1，
          否则取 head 数）乘进 layout 的 extent，让 col/row stride
          自动正确；extent 高估对按真实 extent 取子块的{" "}
          <Code>GetTile</Code> 无害。这是注释里记录的一个已修缺陷。
        </Text>
      </Callout>
    </Stack>
  );
}

function VectorSide() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H2>不是经典的在线 softmax</H2>
        <Text>
          BF16 路径按行求 max、在 FP32 域做 exp 与 rowsum。MXFP4
          路径把这套换成了<Text weight="semibold">与 MX 量化组对齐的分组最大值</Text>
          ，全程 FP16，且不在 softmax 里求和：
        </Text>
        <Table
          headers={["步骤", "BF16 路径", "MXFP4 路径"]}
          rows={[
            ["最大值粒度", "每行（FP32 ReduceMax）", "每 32 行一组，log₂ 域取整后的 FP16"],
            ["exp 精度", "FP32（FusedExpSub）", "FP16 原生 Exp"],
            ["行和位置", "softmax 内 ReduceSum", "挪到 cube MM2 的 pad 行"],
            ["P 输出", "FP16/BF16/FP8 稠密", "fp4_e2m1 + per-group E8M0 + NZ 重排"],
            ["O 重标定", "softmax epilogue 内", "独立的 rescale-O，延迟 20 tile"],
          ]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>分组最大值的数学形式</H2>
        <Text>
          先把 S 缩放到 log₂ 域：<Code>M = S × scaleValue × (1/ln2)</Code>，
          再按量化模式取整：
        </Text>
        <Grid columns={2} gap={16}>
          <Card>
            <CardHeader>OCP 模式（quantMode = 2）</CardHeader>
            <CardBody>
              <Stack gap={6}>
                <Text size="small">
                  <Code>M' = floor(M)</Code>
                </Text>
                <Text size="small">
                  <Code>M_exp = (M' − 2) × ln2</Code>
                </Text>
                <Note>
                  减 2 是为把 exp 结果压进 fp4_e2m1 的可表示范围（最大 6.0）。
                  pscale 侧对应 <Code>gmax + (−125)</Code>。
                </Note>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>CX 模式（quantMode = 3）</CardHeader>
            <CardBody>
              <Stack gap={6}>
                <Text size="small">
                  <Code>M' = ceil(M − log2Cx)</Code>
                </Text>
                <Text size="small">
                  <Code>M_exp = M' × ln2</Code>
                </Text>
                <Note>
                  <Code>log2Cx</Code> 由 host 预算（默认 log2(6) = 2.585）。
                  pscale 侧对应 <Code>gmax + (−127) + log2CxCeil</Code>。
                </Note>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
        <Text>
          随后 <Code>P = exp(S × scale − M_exp)</Code>，
          经 FP16 → BF16 → <Code>float4_e2m1x2_t</Code> 两级 Cast 落成 fp4，
          最后用预置索引表做一次 <Code>Gather</Code> 完成 ND→NZ 重排，
          分两次 <Code>StoreAlign</Code>（偏移 0 与 +16256）写出高低半区。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>P-scale 与重标定因子 dm</H2>
        <Text>
          <Code>computePscale</Code>（延迟 3 tile）做两件事：
        </Text>
        <Stack gap={6}>
          <Text>
            · <Text weight="semibold">合并全局最大值</Text>：
            <Code>gmax = max(peerMax₀, peerMax₁, gmax_old)</Code>
          </Text>
          <Text>
            · <Text weight="semibold">算 dm</Text>：
            <Code>dm = 2^(gmax_old − gmax)</Code>。实现上不用 exp，而是把
            <Code>(Δ + 127)</Code> 截断到 ≥ 0 后 <Code>ShiftLeft(23)</Code>{" "}
            直接拼出 FP32 的指数位 —— 一条整数移位代替一次超越函数。
          </Text>
          <Text>
            · <Text weight="semibold">算 E8M0 scale</Text>：
            <Code>pscale = saturate_uint8(localGroupMax − gmax_adj)</Code>，
            以 <Code>SatMode::SAT</Code> 截断（负数饱和到 0），
            再按 stride 5 的 <Code>DATA_BLOCK_COPY</Code> 写成 cube 需要的 NZ 分形。
          </Text>
        </Stack>
      </Stack>

      <Stack gap={8}>
        <H2>rescale-O 与最终输出</H2>
        <Text>
          <Code>processUpdate</Code>（每个 tile-group 一次）在 FP32 域推进累积：
        </Text>
        <Stack gap={5}>
          <Text size="small">
            首个 group：<Code>O = PV</Code>，<Code>globalRowSum = localRowSum</Code>
          </Text>
          <Text size="small">
            后续 group：<Code>globalRowSum = globalRowSum × dm + localRowSum</Code>，
            <Code>O = O × dm + PV</Code>
          </Text>
        </Stack>
        <Text>
          最后一个 KV tile 时 <Code>processOut</Code> 求{" "}
          <Code>1 / globalRowSum</Code>（一次 Div，非逐元素倒数），
          乘到 O 上、Cast 成 b16、交织成 ND 布局写进{" "}
          <Code>oTransUBTensor</Code>（复用 P 空间），再拷回{" "}
          <Code>oUBTensor</Code> 并由 <Code>copyUbToGmO</Code> 写 GM。
          两个子核各负责 64 行。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>VF 编程模型</H2>
        <Text>
          <Code>mxfp4_vf/</Code> 下的 12 个文件都是{" "}
          <Code>__simd_vf__</Code> + <Code>AscendC::MicroAPI</Code>
          风格的向量核函数：直接操作 <Code>__ubuf__</Code> 裸指针与{" "}
          <Code>RegTensor</Code> 寄存器，由一层 <Code>__aicore__</Code>{" "}
          包装函数从 <Code>LocalTensor::GetPhyAddr()</Code> 取地址后调用。
        </Text>
        <Text>
          一次 <Code>LoadAlign</Code> 处理 128 个 half（256 B，正好一个 UB
          分形行）；掩码用 <Code>CreateMask&lt;uint16_t, MaskPattern::ALL&gt;</Code>
          ，尾块用 <Code>UpdateMask</Code> 动态收窄。分块常量固定为{" "}
          <Code>ROWS_PER_GROUP = 32</Code>、<Code>ROW_SUB_LOOP = 4</Code>、
          <Code>ITER_PER_GROUP = 8</Code>，内层循环按奇偶子块拆成两段，
          以便把下一组的 max 预取与当前组的 Sub+Exp 交错，隐藏依赖延迟。
        </Text>
        <Note>
          之所以有 8 个 softmax 变体（qs128/qs64 × kvs256/kvs32/kvs32_multi/align），
          是因为 VF 的循环边界必须编译期常量，尾块形状只能靠特化穷举。
        </Note>
      </Stack>
    </Stack>
  );
}

function HostSide() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H2>Tiling 中与 MXFP4 相关的字段</H2>
        <Table
          headers={["字段", "取值", "作用"]}
          rows={HOST_FIELDS}
          columnAlign={["left", "left", "left"]}
        />
        <Note>
          <Code>log2Cx</Code> / <Code>log2CxCeil</Code> 只在 CX 模式下被赋值 ——
          device 侧不能调 host 数学库，所以 <Code>std::log2</Code> /{" "}
          <Code>std::ceil</Code> 必须在 host 算完再传。OCP 模式下这两个字段保持 0，
          量化行为由模板参数 <Code>MXQuantMode::OCP</Code> 在编译期决定。
        </Note>
      </Stack>

      <Stack gap={8}>
        <H2>Tiling key 编码</H2>
        <Text>
          十进制位域拼接：<Code>905</Code>（BSA on 950）+ KV layout（TND
          +3e7 / BNSD +5e7 / BSND +6e7）+ <Code>innerPrecise=4</Code>（+4e5）+
          量化与输出类型（OCP BF16 +30 / OCP FP16 +31 / CX BF16 +40 / CX FP16 +41）+
          Q layout（TND +2 / BNSD +3 / BSND +4）。共 12 个 MXFP4 实例
          = 3 layout × 2 量化模式 × 2 输出 dtype。
        </Text>
        <Table
          headers={["tilingKey", "layout", "量化模式", "输出 dtype"]}
          rows={TILING_KEYS}
        />
        <Note>
          分发在 <Code>block_sparse_attention.cpp</Code> 里用{" "}
          <Code>#elif TILING_KEY_VAR == ...</Code> 静态展开，
          最终落到 <Code>BsaInferInterfaceMXFP4FullQuant&lt;fp4x2_e2m1_t, half,
          float, bfloat16_t, ...&gt;</Code>。
        </Note>
      </Stack>

      <Stack gap={8}>
        <H2>模板实参与具体元素类型</H2>
        <Table
          headers={["角色", "类型"]}
          rows={[
            ["Q / K / V / P", "fp4x2_e2m1_t"]
            ,["S（MM1 输出、softmax 输入）", "half"],
            ["OTmp / dm / 累加器", "float"],
            ["O（输出）", "half 或 bfloat16_t"],
            ["各类 scale（含 P-scale）", "uint8_t（E8M0 伪装）"],
            ["groupMax / gmax", "half"],
            ["sparseIdx / sparseCount", "int32_t"],
            ["MM1 L0C→UB 模式", "CopyL0CToUBMode::SPLIT_N"],
            ["MM2 L0C→UB 模式", "CopyL0CToUBMode::SPLIT_M"],
            ["scale 粒度", "ScaleGranularity::PER_GROUP"],
            ["L1 tile（QK 与 PV）", "128 × 128 × 128"],
          ]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>任务划分与 workspace</H2>
        <Text>
          一个 task = <Code>(batch, qHead, qSTile)</Code>。device 侧由{" "}
          <Code>taskIdx = coreTaskId × coreNum + coreIdx</Code> 反推，
          batch 扫描状态保存在 <Code>BatchOffsetInfo</Code> 里跨调用续算，
          避免每次从 batch 0 重扫。
        </Text>
        <Text>
          每个 task 的 KV tile 数由稀疏图决定：读{" "}
          <Code>sparseCount</Code> 得到保留块数 <Code>yBlockNumRsvd</Code>，
          读最后一个 <Code>sparseIdx</Code> 判断尾块实际长度，
          得 <Code>gatheredKvSeqlen = (yBlockNumRsvd−1) × blockShapeY + lastYBlockSize</Code>
          ，再除以 <Code>kvBaseTile</Code> 得 <Code>taskTileNum</Code>。
        </Text>
        <Text>
          workspace 布局为 <Code>[sparseIdx | sparseCount]</Code>，
          由主循环之前的 AIV <Code>Mask2Idx</Code> epilogue 一次性生成，
          之后 <Code>SyncAll</Code> 同步。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>约束与拦截</H2>
        <Table headers={["项", "约束"]} rows={CONSTRAINTS} />
      </Stack>
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/* 顶层                                                                */
/* ------------------------------------------------------------------ */

const TABS = [
  { id: "overview", label: "概览" },
  { id: "pipeline", label: "流水" },
  { id: "memory", label: "内存分配" },
  { id: "cube", label: "Cube / AIC" },
  { id: "vector", label: "Vector / AIV" },
  { id: "host", label: "主机侧" },
] as const;

export default function BsaMxfp4DesignReport() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState<string>("bsa-mxfp4-tab", "overview");

  return (
    <Stack gap={18} style={{ padding: 24, maxWidth: 1120 }}>
      <Stack gap={4}>
        <H1>BSA MXFP4 全量化实现设计报告</H1>
        <Text tone="secondary">
          Block Sparse Attention · Ascend 950 / arch35 · 源码路径{" "}
          <Code>attention/block_sparse_attention</Code>
        </Text>
      </Stack>

      <Row gap={6} wrap>
        {TABS.map((t) => (
          <div key={t.id}>
            <Pill active={tab === t.id} onClick={() => setTab(t.id)}>
              {t.label}
            </Pill>
          </div>
        ))}
      </Row>

      <div
        style={{
          borderTop: `1px solid ${theme.stroke.tertiary}`,
          paddingTop: 18,
        }}
      >
        {tab === "overview" && <Overview />}
        {tab === "pipeline" && <Pipeline />}
        {tab === "memory" && <Memory />}
        {tab === "cube" && <CubeSide />}
        {tab === "vector" && <VectorSide />}
        {tab === "host" && <HostSide />}
      </div>

      <Divider />
      <Stack gap={4}>
        <H3>本报告的事实来源</H3>
        <Note>
          内存偏移与常量逐条核对自 <Code>block_mmad_arch35_utils.hpp</Code>（L1/L0）
          与 <Code>block_epilogue_arch35_utils.hpp</Code>（UB）；流水与同步来自{" "}
          <Code>block_sparse_attention_kernel_arch35_mxfp4_full_quant.h</Code>{" "}
          主循环；epilogue 行为来自 <Code>block_epilogue_*_mxfp4.hpp</Code> 及{" "}
          <Code>mxfp4_vf/</Code>；host 侧来自{" "}
          <Code>block_sparse_attention_tiling.cpp</Code> 与{" "}
          <Code>block_sparse_attention_tilingkey.h</Code>。
        </Note>
      </Stack>
    </Stack>
  );
}
