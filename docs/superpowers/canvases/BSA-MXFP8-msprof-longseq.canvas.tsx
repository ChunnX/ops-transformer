import {
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
} from "cursor/canvas";

const SEQ = [
  8422, 26352, 49408, 72000, 75348, 75600, 82800, 108000, 118800, 162000, 186300,
  243000, 1048576,
];
const KEEP50_US = [
  92.881, 565.916, 1813.64, 3914.84, 4308.322, 4343.253, 5164.533, 8614.619,
  10400.663, 19098.504, 24893.822, 42380.852, 904409.188,
];
const KEEP20_US = [
  53.871, 250.131, 760.179, 1631.39, 1770.817, 1772.221, 2136.266, 3522.208,
  4192.913, 7750.817, 10382.028, 17226.359, 382334.781,
];

const SEQ_NO_1M = SEQ.slice(0, -1);
const KEEP50_MS = KEEP50_US.map((u) => u / 1000);
const KEEP20_MS = KEEP20_US.map((u) => u / 1000);

function fmtUs(us: number): string {
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(2)} µs`;
}

const tableRows = SEQ.map((s, i) => {
  const r = KEEP20_US[i] / KEEP50_US[i];
  return [
    String(s),
    fmtUs(KEEP50_US[i]),
    fmtUs(KEEP20_US[i]),
    r.toFixed(3),
    (KEEP50_US[i] / (s / 1e6) ** 2 / 1e3).toFixed(0),
  ];
});

export default function BsaMxfp8MsprofLongseq() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>BSA MXFP8 OCP msprof 长序列</H1>
        <Text tone="secondary">
          Source: msprof op · warm-up=5 · launch-count=10 · NPU 1 Ascend950PR ·
          2026-08-20 10:43–10:52 · cann_bsa_0817 · kernel BlockSparseAttention
          mix_aic
        </Text>
        <Row gap={8} wrap>
          <Pill>B=1 N=1 D=128</Pill>
          <Pill>BNSD bf16 out</Pill>
          <Pill>quantMode=4 OCP</Pill>
          <Pill>block 128×128</Pill>
          <Pill tone="success">26/26 OK</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat
          label="S=243k keep 50%"
          value="42.4 ms"
          tone="info"
        />
        <Stat
          label="S=243k keep 20%"
          value="17.2 ms"
        />
        <Stat
          label="S=1,048,576 keep 50%"
          value="904 ms"
          tone="warning"
        />
        <Stat
          label="keep20/keep50 (target 0.40)"
          value="~0.41"
        />
      </Grid>

      <Callout tone="warning" title="判定：有优化空间，大 S 偏 SCALAR / MTE2 / FixPipe">
        全点 Current Freq=Rated=1650 MHz，无降频。Block Dim 恒为 28（N=1 核填充不足）。
        S≥243k：aic_cube_ratio≈0.50，aic_scalar_ratio≈0.89，aic_mte2_ratio≈0.75，aic_fixpipe_ratio≈0.42。
        AIV 列在本批 CSV 为 NA（只采到 mix_aic）。Duration 随 S²、随 keep 近似线性，计算路径是对的。
      </Callout>

      <H2>Task Duration vs S（不含 1M，避免压扁）</H2>
      <Text tone="secondary">
        Y: Task Duration (ms)，mean of launch-count=10。X: qs=kvs。keep 50% =
        稀疏度 50% 保留 50%；keep 20% = 稀疏度 80% 保留 20%。
      </Text>
      <LineChart
        height={280}
        categories={SEQ_NO_1M.map(String)}
        series={[
          { name: "keep 50% (ms)", data: KEEP50_MS.slice(0, -1), tone: "info" },
          { name: "keep 20% (ms)", data: KEEP20_MS.slice(0, -1), tone: "success" },
        ]}
        valueSuffix=" ms"
        beginAtZero
      />
      <Text tone="secondary">
        S=1,048,576 未画在上图：keep50=904.4 ms，keep20=382.3 ms。
      </Text>

      <H2>全点 Duration</H2>
      <Table
        headers={[
          "S (qs=kvs)",
          "keep 50%",
          "keep 20%",
          "20/50",
          "keep50 / (S/1e6)² (ms)",
        ]}
        rows={tableRows}
        columnAlign={["right", "right", "right", "right", "right"]}
        striped
        stickyHeader
      />
      <Text tone="secondary">
        末列在 S≥50k 约 720–760，接近二次方；S=8422 固定开销占比高所以偏大。
      </Text>

      <H2>大 shape 流水（S=243000 keep 50%）</H2>
      <Grid columns={4} gap={12}>
        <Card>
          <CardHeader>Cube</CardHeader>
          <CardBody>
            <Text weight="semibold">0.50</Text>
            <Text tone="secondary">aic_cube_ratio · 21.2 ms</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Scalar</CardHeader>
          <CardBody>
            <Text weight="semibold">0.90</Text>
            <Text tone="secondary">严重 · aic_scalar_ratio</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>MTE2</CardHeader>
          <CardBody>
            <Text weight="semibold">0.75</Text>
            <Text tone="secondary">搬运偏高 · 64 GB/s active</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>FixPipe</CardHeader>
          <CardBody>
            <Text weight="semibold">0.42</Text>
            <Text tone="secondary">严重 · 阈值 0.15</Text>
          </CardBody>
        </Card>
      </Grid>
      <Text tone="secondary">
        各 ratio 可重叠（同拍多流水），不能加总成 100%。S=1M keep50 比例几乎相同（cube 0.50 /
        scalar 0.89 / mte2 0.76 / fixpipe 0.42）。
      </Text>

      <Divider />
      <H3>采集条件</H3>
      <Text>
        安装 cann_bsa_0817；layout BNSD；innerPrecise=4；e4m3 + e8m0=127 造数（perf
        不做 OCP golden）；mask 按行 Bernoulli keep，空行补 1 块。产物目录
        tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_longseq/。
      </Text>
    </Stack>
  );
}
