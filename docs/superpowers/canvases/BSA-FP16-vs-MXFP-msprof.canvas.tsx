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

const FP16_K50 = [
  91.534, 578.979, 1861.883, 3960.684, 4386.56, 4463.482, 5312.947, 8823.85,
  10667.026, 19746.324, 26132.281, 46245.613, 1447270.625,
];
const FP16_K20 = [
  52.948, 254.5, 781.4, 1650.23, 1799.321, 1824.178, 2194.106, 3593.201,
  4303.782, 8000.616, 10419.663, 18383.674, 574314.688,
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
  fmt(FP16_K50[i]),
  fmt(FP8_K50[i]),
  fmt(FP4_K50[i]),
  (FP16_K50[i] / FP8_K50[i]).toFixed(2),
  fmt(FP16_K20[i]),
  fmt(FP8_K20[i]),
  fmt(FP4_K20[i]),
  (FP16_K20[i] / FP8_K20[i]).toFixed(2),
]);

export default function BsaFp16VsMxfpMsprof() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>BSA FP16 vs MXFP8 vs MXFP4 同用例 msprof</H1>
        <Text tone="secondary">
          同一矩阵：B=1 N=1 D=128 BNSD block=128×128，keep 50%/20%。warm-up=5
          launch-count=10。FP16 走 quantMode=0，QKV/out 均为 float16。cann_bsa_0817 ·
          950PR 1650 MHz
        </Text>
        <Row gap={8} wrap>
          <Pill>FP16 quantMode=0 · NPU 4</Pill>
          <Pill>MXFP8 quantMode=4 · NPU 1</Pill>
          <Pill>MXFP4 quantMode=2 · NPU 7</Pill>
          <Pill tone="success">各 26/26 OK</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat label="S=243k keep50 FP16" value="46.2 ms" />
        <Stat label="S=243k keep50 MXFP8" value="42.4 ms" tone="info" />
        <Stat label="S=243k keep50 MXFP4" value="22.6 ms" tone="success" />
        <Stat label="S=1M keep50 FP16" value="1.45 s" tone="warning" />
      </Grid>

      <Callout tone="info" title="大 S 上 FP16 慢于 MXFP8 约 1.5–1.6×">
        S≤243k keep50 时 FP16 与 MXFP8 几乎打平（约 1.03–1.09×）。S=1M keep50
        拉开到 1447 / 904 ≈ 1.60×；keep20 为 574 / 382 ≈ 1.50×。MXFP4 仍最快，S=1M
        keep50 上 FP16/MXFP4 ≈ 2.89×。
      </Callout>
      <Callout title="FP16 稀疏比干净">
        keep20/keep50 全程约 0.40，S=1M 也是 574/1447≈0.40。MXFP4 在 S=1M keep20
        曾到 0.79 离群；FP16 没有这个点。
      </Callout>

      <H2>Task Duration vs S（不含 1M）keep 50%</H2>
      <Text tone="secondary">Y: Task Duration (ms)。X: qs=kvs。</Text>
      <LineChart
        height={280}
        categories={SEQ_NO_1M.map(String)}
        series={[
          { name: "FP16 keep50 (ms)", data: FP16_K50.slice(0, -1).map((u) => u / 1000), tone: "neutral" },
          { name: "MXFP8 keep50 (ms)", data: FP8_K50.slice(0, -1).map((u) => u / 1000), tone: "info" },
          { name: "MXFP4 keep50 (ms)", data: FP4_K50.slice(0, -1).map((u) => u / 1000), tone: "success" },
        ]}
        valueSuffix=" ms"
        beginAtZero
      />

      <H2>keep 20%（不含 1M）</H2>
      <LineChart
        height={240}
        categories={SEQ_NO_1M.map(String)}
        series={[
          { name: "FP16 keep20 (ms)", data: FP16_K20.slice(0, -1).map((u) => u / 1000), tone: "neutral" },
          { name: "MXFP8 keep20 (ms)", data: FP8_K20.slice(0, -1).map((u) => u / 1000), tone: "info" },
          { name: "MXFP4 keep20 (ms)", data: FP4_K20.slice(0, -1).map((u) => u / 1000), tone: "success" },
        ]}
        valueSuffix=" ms"
        beginAtZero
      />

      <H2>全点对照</H2>
      <Table
        headers={[
          "S",
          "FP16 k50",
          "FP8 k50",
          "FP4 k50",
          "FP16/FP8",
          "FP16 k20",
          "FP8 k20",
          "FP4 k20",
          "FP16/FP8",
        ]}
        rows={rows}
        columnAlign={[
          "right",
          "right",
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

      <H2>S=243000 keep 50% 流水</H2>
      <Grid columns={3} gap={16}>
        <Card>
          <CardHeader trailing={<Pill>46.2 ms</Pill>}>FP16</CardHeader>
          <CardBody>
            <Text>cube 0.93 · scalar 0.62 · mte2 0.87 · fixpipe 0.87</Text>
            <Text tone="secondary">Block Dim 28 · cube 更满</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="info">42.4 ms</Pill>}>MXFP8</CardHeader>
          <CardBody>
            <Text>cube 0.50 · scalar 0.90 · mte2 0.75 · fixpipe 0.42</Text>
            <Text tone="secondary">Block Dim 28</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="success">22.6 ms</Pill>}>MXFP4</CardHeader>
          <CardBody>
            <Text>cube 0.48 · scalar 0.98 · mte2 0.75 · fixpipe 0.60</Text>
            <Text tone="secondary">Block Dim 28</Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />
      <H3>采集</H3>
      <Text>
        FP16：tests/fp16_pytest/bsa_msprof_perf_fp16_longseq/ · NPU 4 ·
        quantMode=0。MXFP8 / MXFP4 为先前同矩阵扫点。msprof --kernel-name=BlockSparseAttention
        --warm-up=5 --launch-count=10。
      </Text>
    </Stack>
  );
}
