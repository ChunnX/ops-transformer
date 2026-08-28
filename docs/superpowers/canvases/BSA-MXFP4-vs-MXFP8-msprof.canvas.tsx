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

const FP8_K50 = [
  92.881, 565.916, 1813.64, 3914.84, 4308.322, 4343.253, 5164.533, 8614.619,
  10400.663, 19098.504, 24893.822, 42380.852, 904409.188,
];
const FP8_K20 = [
  53.871, 250.131, 760.179, 1631.39, 1770.817, 1772.221, 2136.266, 3522.208,
  4192.913, 7750.817, 10382.028, 17226.359, 382334.781,
];
const FP4_K50 = [
  54.097, 337.72, 979.013, 2093.799, 2254.264, 2282.954, 3072.484, 4594.16,
  5505.824, 10159.83, 13290.346, 22562.986, 501442.281,
];
const FP4_K20 = [
  34.853, 150.678, 451.169, 935.255, 985.548, 992.168, 1191.706, 1989.237,
  2396.432, 4382.198, 5707.664, 9661.971, 395209.875,
];

const SEQ_NO_1M = SEQ.slice(0, -1);

function fmt(us: number): string {
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(1)} µs`;
}

const rows = SEQ.map((s, i) => [
  String(s),
  fmt(FP4_K50[i]),
  fmt(FP8_K50[i]),
  (FP8_K50[i] / FP4_K50[i]).toFixed(2),
  fmt(FP4_K20[i]),
  fmt(FP8_K20[i]),
  (FP8_K20[i] / FP4_K20[i]).toFixed(2),
]);

export default function BsaMxfp4VsMxfp8Msprof() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>BSA MXFP4 vs MXFP8 同用例 msprof</H1>
        <Text tone="secondary">
          同一矩阵：B=1 N=1 D=128 BNSD bf16 block=128×128，keep 50%/20%。warm-up=5
          launch-count=10。MXFP8 NPU1 · MXFP4 NPU7 · 均 1650 MHz · cann_bsa_0817
        </Text>
        <Row gap={8} wrap>
          <Pill>MXFP4 quantMode=2</Pill>
          <Pill>MXFP8 quantMode=4</Pill>
          <Pill tone="success">各 26/26 OK</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat label="S=243k keep50 MXFP4" value="22.6 ms" tone="info" />
        <Stat label="S=243k keep50 MXFP8" value="42.4 ms" />
        <Stat label="MXFP8 / MXFP4 (keep50 大S)" value="~1.88×" />
        <Stat label="S=1M keep50 MXFP4" value="501 ms" tone="warning" />
      </Grid>

      <Callout tone="info" title="MXFP4 更快约 1.7–1.9×（大 S keep 50%）">
        同几何下 MXFP8 更慢符合字节量（e4m3 vs packed e2m1）。两边 Block Dim 都是 28。
        MXFP4 大 S：cube≈0.48，scalar≈0.98，mte2≈0.75，fixpipe≈0.60，Scalar/FixPipe 比 MXFP8 更重。
      </Callout>
      <Callout tone="warning" title="S=1M keep 20% 是离群点">
        MXFP4 keep20/keep50 在 S≤243k 约 0.43，但 S=1M 变成 395/501≈0.79；该点 MXFP8 甚至略快于 MXFP4（382 vs 395 ms）。不要用这一格外推稀疏加速比。
      </Callout>

      <H2>Task Duration vs S（不含 1M）keep 50%</H2>
      <Text tone="secondary">Y: Task Duration (ms)。X: qs=kvs。</Text>
      <LineChart
        height={280}
        categories={SEQ_NO_1M.map(String)}
        series={[
          { name: "MXFP4 keep50 (ms)", data: FP4_K50.slice(0, -1).map((u) => u / 1000), tone: "success" },
          { name: "MXFP8 keep50 (ms)", data: FP8_K50.slice(0, -1).map((u) => u / 1000), tone: "info" },
        ]}
        valueSuffix=" ms"
        beginAtZero
      />

      <H2>keep 20%（不含 1M）</H2>
      <LineChart
        height={240}
        categories={SEQ_NO_1M.map(String)}
        series={[
          { name: "MXFP4 keep20 (ms)", data: FP4_K20.slice(0, -1).map((u) => u / 1000), tone: "success" },
          { name: "MXFP8 keep20 (ms)", data: FP8_K20.slice(0, -1).map((u) => u / 1000), tone: "info" },
        ]}
        valueSuffix=" ms"
        beginAtZero
      />

      <H2>全点对照</H2>
      <Table
        headers={[
          "S",
          "FP4 keep50",
          "FP8 keep50",
          "FP8/FP4",
          "FP4 keep20",
          "FP8 keep20",
          "FP8/FP4",
        ]}
        rows={rows}
        columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
        striped
        stickyHeader
      />

      <H2>S=243000 keep 50% 流水</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>MXFP4</CardHeader>
          <CardBody>
            <Text>cube 0.48 · scalar 0.98 · mte2 0.75 · fixpipe 0.60</Text>
            <Text tone="secondary">22.6 ms · Block Dim 28</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>MXFP8</CardHeader>
          <CardBody>
            <Text>cube 0.50 · scalar 0.90 · mte2 0.75 · fixpipe 0.42</Text>
            <Text tone="secondary">42.4 ms · Block Dim 28</Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />
      <H3>采集</H3>
      <Text>
        MXFP4：tests/mxfp4_pytest/bsa_msprof_perf_mxfp4_longseq/ · NPU 7 ·
        quantMode=2 packed e2m1。MXFP8：mxfp8_pytest 同名目录 · NPU 1 ·
        quantMode=4 e4m3。两卡同 950PR 1650 MHz。
      </Text>
    </Stack>
  );
}
