import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  TodoListCard,
  UsageBar,
} from "cursor/canvas";

const SEQ = [
  8422, 26352, 49408, 72000, 75348, 75600, 82800, 108000, 118800, 162000, 186300,
  243000, 1048576,
];

const FP16_K50_US = [
  91.534, 578.979, 1861.883, 3960.684, 4386.56, 4463.482, 5312.947, 8823.85,
  10667.026, 19746.324, 26132.281, 46245.613, 1447270.625,
];
const FP8_K50_US = [
  92.881, 565.916, 1813.64, 3914.84, 4308.322, 4343.253, 5164.533, 8614.619,
  10400.663, 19098.504, 24893.822, 42380.852, 904409.188,
];
const FP16_K20_US = [
  52.948, 254.5, 781.4, 1650.23, 1799.321, 1824.178, 2194.106, 3593.201,
  4303.782, 8000.616, 10419.663, 18383.674, 574314.688,
];
const FP8_K20_US = [
  53.871, 250.131, 760.179, 1631.39, 1770.817, 1772.221, 2136.266, 3522.208,
  4192.913, 7750.817, 10382.028, 17226.359, 382334.781,
];

const RATIO_K50 = SEQ.map((_, i) =>
  Number((FP16_K50_US[i] / FP8_K50_US[i]).toFixed(2)),
);
const RATIO_K20 = SEQ.map((_, i) =>
  Number((FP16_K20_US[i] / FP8_K20_US[i]).toFixed(2)),
);

const SEQ_NO_1M = SEQ.slice(0, -1);

function fmtMs(us: number): string {
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(1)} µs`;
}

export default function BsaMxfp8VsFp16Analysis() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>BSA MXFP8 vs FP16：理论、实测、下一步</H1>
        <Text tone="secondary">
          Ascend950PR PCIE · 28 Cube @ 1650 MHz · B=1 N=1 D=128 BNSD
          block=128×128 · innerPrecise=4 · cann_bsa_0817 · msprof
          BlockSparseAttention warmup=5 launch=10
        </Text>
        <Row gap={8} wrap>
          <Pill>Cube FP16 378 TFLOPS</Pill>
          <Pill tone="info">Cube MXFP8 757 TFLOPS = 2.00×</Pill>
          <Pill>HBM ~1.4 TB/s · L2 112 MB</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat label="理论 Cube 加速" value="2.00×" tone="info" />
        <Stat label="S=243k keep50 实测" value="1.09×" tone="warning" />
        <Stat label="S=1M keep50 实测" value="1.60×" />
        <Stat label="MXFP8 cube 占用" value="0.50" tone="danger" />
      </Grid>

      <Callout tone="warning" title="结论：中等 S 没吃到 FP8 算力；大 S 主要吃到带宽">
        同一套 QK+PV FLOP。MXFP8 Cube 峰值是 FP16 的 2 倍、KV 字节是一半。S≤243k
        时 Duration 几乎打平（1.03–1.09×），因为 MXFP8 Cube 一半时间在等 Vector /
        scalar（cube 0.50 vs FP16 0.93）。S=1M 时 FP16 撞上 KV 重读的 HBM
        墙，MXFP8 字节减半，加速拉到 1.50–1.60×，接近带宽上限而不是算力上限。
      </Callout>

      <H2>1. 理论 Roofline</H2>
      <Text tone="secondary">
        FLOP = 4 · S · S_kv · D，S_kv = keep · S。KV 重读字节 ≈ 2 · keep · S² · D ·
        bpe / 128。AI = 256 / bpe → FP16 128 FLOP/B，MXFP8 256 FLOP/B。PCIE
        ridge ≈ peak / 1.4 TB/s：FP16 270 FLOP/B，MXFP8 541 FLOP/B。若 KV
        每次都从 HBM 来，两者都偏带宽；实测中等 S 被 L2 挡住了大部分重读。
      </Text>

      <Table
        headers={[
          "工况",
          "FLOP",
          "FP16 Cube 下限",
          "MXFP8 Cube 下限",
          "FP16 HBM 下限",
          "MXFP8 HBM 下限",
          "FP16 实测",
          "MXFP8 实测",
        ]}
        rows={[
          [
            "S=243k keep50",
            "15.1 TFLOP",
            "40.0 ms",
            "20.0 ms",
            "84 ms（L2 掩盖）",
            "42 ms",
            "46.2 ms",
            "42.4 ms",
          ],
          [
            "S=243k keep20",
            "6.0 TFLOP",
            "16.0 ms",
            "8.0 ms",
            "34 ms",
            "17 ms",
            "18.4 ms",
            "17.2 ms",
          ],
          [
            "S=1M keep50",
            "281 TFLOP",
            "745 ms",
            "372 ms",
            "1570 ms",
            "785 ms",
            "1447 ms",
            "904 ms",
          ],
          [
            "S=1M keep20",
            "113 TFLOP",
            "298 ms",
            "149 ms",
            "628 ms",
            "314 ms",
            "574 ms",
            "382 ms",
          ],
        ]}
        columnAlign={[
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
        ]}
        striped
        stickyHeader
      />
      <Text tone="secondary">
        Source: npu-arch 950PR PCIE Cube 规格 + 本轮 msprof Task Duration。HBM
        1.4 TB/s 为规格估计值。S=243k FP16 实测 46 ms 远低于 HBM 下限 84
        ms，说明 L2（112 MB）吞掉了大部分 KV 重读；S=1M FP16 1447 ms 接近 1570
        ms HBM 下限。
      </Text>

      <H2>2. 实测 Duration 与加速比</H2>
      <LineChart
        height={260}
        categories={SEQ_NO_1M.map(String)}
        series={[
          {
            name: "FP16 keep50 (ms)",
            data: FP16_K50_US.slice(0, -1).map((u) => u / 1000),
            tone: "neutral",
          },
          {
            name: "MXFP8 keep50 (ms)",
            data: FP8_K50_US.slice(0, -1).map((u) => u / 1000),
            tone: "info",
          },
        ]}
        valueSuffix=" ms"
        beginAtZero
      />
      <Text tone="secondary">
        Task Duration vs qs=kvs，不含 S=1M。Y: ms。Source: bsa_msprof_perf_fp16 /
        mxfp8_longseq。
      </Text>

      <BarChart
        height={240}
        categories={SEQ.map((s) => (s >= 1e6 ? "1M" : String(s)))}
        series={[
          { name: "keep50 FP16/MXFP8", data: RATIO_K50, tone: "info" },
          { name: "keep20 FP16/MXFP8", data: RATIO_K20, tone: "neutral" },
        ]}
        referenceLines={[
          { value: 2.0, label: "Cube 理论 2.0×", tone: "success" },
          { value: 1.0, label: "打平", tone: "warning" },
        ]}
        yMax={2.2}
        beginAtZero
        valueSuffix="×"
      />
      <Text tone="secondary">
        加速比 = T_FP16 / T_MXFP8。S≤243k 贴在 1.0–1.1；仅 S=1M 升到 1.50–1.60，仍低于
        Cube 2.0×。
      </Text>

      <Table
        headers={["S", "FP16 k50", "MXFP8 k50", "加速", "FP16 k20", "MXFP8 k20", "加速"]}
        rows={SEQ.map((s, i) => [
          String(s),
          fmtMs(FP16_K50_US[i]),
          fmtMs(FP8_K50_US[i]),
          RATIO_K50[i].toFixed(2),
          fmtMs(FP16_K20_US[i]),
          fmtMs(FP8_K20_US[i]),
          RATIO_K20[i].toFixed(2),
        ])}
        columnAlign={[
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
        ]}
        striped
        stickyHeader
      />

      <H2>3. 流水：为什么理论 2× 没出现</H2>
      <Text tone="secondary">
        PipeUtilization 对 S=243k 与 S=1M 几乎不变。Bound 类型要从 Duration 对
        Roofline 的位置读，不能只看占用率。
      </Text>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill>46.2 ms · cube-bound</Pill>}>
            FP16 S=243k keep50
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <UsageBar
                total={1}
                topLeftLabel="AIC cube 占用 0.93"
                topRightLabel="接近峰值"
                segments={[{ id: "cube", value: 0.933, color: "green" }]}
              />
              <Text>cube 0.93 · scalar 0.62 · mte2 0.87 · fixpipe 0.87</Text>
              <Text>AIV vec 0.88 · scalar 0.13</Text>
              <Text tone="secondary">
                实测 / Cube 下限 = 46.2 / 40.0 = 1.16×。Cube 喂饱，softmax 叠在
                mmad 上。
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="warning">42.4 ms · AIV 卡住 Cube</Pill>}>
            MXFP8 S=243k keep50
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <UsageBar
                total={1}
                topLeftLabel="AIC cube 占用 0.50"
                topRightLabel="一半在等"
                segments={[{ id: "cube", value: 0.504, color: "orange" }]}
              />
              <Text>cube 0.50 · scalar 0.90 · mte2 0.75 · fixpipe 0.42</Text>
              <Text>AIV vec 0.84 · scalar 0.38</Text>
              <Text tone="secondary">
                实测 / Cube 下限 = 42.4 / 20.0 = 2.12×。scalar 0.90 + AIV
                scalar 0.38：AIC 在等 softmax / P 量化 / pScale / 组同步。
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill tone="warning">1447 ms · HBM</Pill>}>
            FP16 S=1M keep50
          </CardHeader>
          <CardBody>
            <Text>cube 仍 0.93，mte2 升到 0.94</Text>
            <Text tone="secondary">
              243k→1M 工作量 ×18.6，时间 ×31.3。1447 ms ≈ HBM 下限 1570
              ms。L2 装不下 768 MB QKV，KV 重读回 HBM。keep20/keep50 = 0.40，符合稀疏度。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="info">904 ms · HBM + 仍等 AIV</Pill>}>
            MXFP8 S=1M keep50
          </CardHeader>
          <CardBody>
            <Text>cube 仍 0.50，mte2 0.76</Text>
            <Text tone="secondary">
              时间 ×21.3，更接近 S²。904 ms 相对 HBM 下限 785 ms 只高
              15%；相对 Cube 下限 372 ms 高 2.4×。再填 Cube 也很难低于 ~785
              ms，除非提高 L2 对 KV 的复用。keep20 382 ms 已贴 314–392 ms
              带宽带。
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <H3>设计里已经标出的气泡</H3>
      <Table
        headers={["来源", "现状", "对流水的影响"]}
        rows={[
          [
            "Q L0B",
            "单缓冲 16 KB",
            "Q 与 mmad 重叠弱（设计 §10.3 退路未启用）",
          ],
          [
            "QK L0 tile",
            "64×128 两轮拼 128",
            "每基块 2 次 FixPipe；fixpipe 0.42 vs FP16 0.87",
          ],
          [
            "Softmax / P",
            "S 为 fp16 128×128；P 再量化 e4m3 + e8m0",
            "比 FP16 多一截 AIV；AIC scalar 0.90",
          ],
          [
            "pScale",
            "单槽 768 B，环 12",
            "远小于 16 KB 搬运门槛，MTE2 效率差",
          ],
          [
            "Dpv=12 / G=8",
            "PV 滞后 12 步；每 8 基块一组",
            "组同步比 MXFP4 G=16 更密；基块 128 比 MXFP4 256 多一倍 loop",
          ],
        ]}
        striped
      />

      <H2>4. 下一步修改方案（按收益排序）</H2>
      <Callout title="不要先改大 S 的 L2 切分">
        S=243k 是产品主战场且加速比最差（~1.09×）。这里 Cube 有约 2×
        头寸：若占用从 0.50 提到 0.80–0.86，Duration 可从 42 ms 落到 ~23–25
        ms，相对 FP16 接近理论 2×。S=1M 的 HBM 地板约 785
        ms，流水优化只剩约 13%。
      </Callout>

      <TodoListCard
        defaultExpanded
        todos={[
          {
            id: "p0-overlap",
            status: "pending",
            content:
              "P0 把 VF softmax / P-quant / CopyPScale 与下一拍 QK Cube 重叠。目标：AIC scalar 0.90→<0.70，cube 0.50→≥0.75。验收：S=243k keep50 Duration ≤28 ms。",
          },
          {
            id: "p0-q-db",
            status: "pending",
            content:
              "P0 启用设计已写的 Q 切 D 双缓冲（Q 槽 8KB×2，L0B 仍 48KB）。现在 L0B Q 单缓是明确气泡。精度套件回归后再看 cube 占用。",
          },
          {
            id: "p1-qk128",
            status: "pending",
            content:
              "P1 尝试 QK 一次 128×128（L0C：64KB×2 QK + 72KB PV = 200KB < 256KB），去掉 64 半面两轮。预期减少 FixPipe 次数、缩短 S 半块等待。",
          },
          {
            id: "p1-scale",
            status: "pending",
            content:
              "P1 e8m0 / pScale 合并搬运（768B → 凑满 ≥16KB 或随 KV 同包）。对照 scale_coalescing：小 scale copy 会把 MTE2 打散。",
          },
          {
            id: "p1-aiv",
            status: "pending",
            content:
              "P1 压 AIV scalar 0.38（FP16 仅 0.13）：融合 softmax+e4m3 pack VF，减少组首/组尾分支。G=8 组屏障是否可改到 16 看 L1（P 环加长）。",
          },
          {
            id: "p2-hbm",
            status: "pending",
            content:
              "P2 仅针对 S≳1M：Q tile 按 L2 112MB 分块，让一组 Q 复用同一段 KV。MXFP8 S=1M keep50 地板 ~785 ms，现在 904 ms。",
          },
        ]}
      />

      <H3>建议落地顺序与预期</H3>
      <Table
        headers={["轮次", "改动", "主看指标", "S=243k keep50 预期", "S=1M keep50 预期"]}
        rows={[
          [
            "0 基线",
            "当前 mxfp8_0817",
            "Duration / cube / scalar",
            "42.4 ms · cube 0.50",
            "904 ms · 贴 HBM",
          ],
          [
            "1",
            "Q 双缓 + softmax/P 与 QK 重叠",
            "cube ≥0.75 · scalar ↓",
            "28–32 ms（保守）",
            "870–900 ms（小）",
          ],
          [
            "2",
            "QK 128 一次打满 + scale 合并",
            "fixpipe ↑ · mte2 更整",
            "23–28 ms（冲 2× vs FP16）",
            "仍受 785 ms 限制",
          ],
          [
            "3",
            "L2 Q-blocking（可选）",
            "1M mte2 / Duration",
            "几乎不变",
            "挑战 800 ms 附近",
          ],
        ]}
        striped
      />

      <Callout tone="info" title="验证约定">
        每轮只改一类气泡。精度：官方 qs128 + 跨组 s2 子集，rtol=1/128。性能：同矩阵
        S∈{"{"}8422, 243000, 1048576{"}"} × keep 0.5/0.2，msprof
        kernel-name=BlockSparseAttention，过滤 Op Name。不要覆盖 cann_bsa_0817
        以外的 opp。
      </Callout>

      <Divider />
      <Text tone="secondary">
        原始 Duration 画布：BSA-FP16-vs-MXFP-msprof。设计：docs/superpowers/specs/2026-08-17-bsa-mxfp8-design.md
        §5 L0、§10 风险。采集目录：tests/fp16_pytest/bsa_msprof_perf_fp16_longseq、
        tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_longseq。
      </Text>
    </Stack>
  );
}
