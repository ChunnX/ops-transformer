import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const CAPTION =
  "S=118800, B=1 N=1 D=128, keep=1.0（FIA=dense）。msprof warm-up=5 launch-count=10。PipeUtilization 核均：AIC 28 核、AIV 56 核。2026-08-20。AIC MTE3 time≈0 时 active_bw 为 msprof 伪值，表中记为 —。";

const AIC_HEADERS = ["字段", "BSA MXFP8", "FIA MXFP8", "BSA MXFP4"];
const AIC_ROWS: string[][] = [
  ["aic_time(us)", "20053.9", "13818.6", "9538.4"],
  ["aic_total_cycles", "33088984", "22800753", "15738346"],
  ["aic_cube_time(us)", "10151.9", "9552.9", "5073.7"],
  ["aic_cube_ratio", "0.506", "0.691", "0.532"],
  ["aic_scalar_time(us)", "18018.9", "8548.6", "9421.2"],
  ["aic_scalar_ratio", "0.899", "0.619", "0.988"],
  ["aic_mte1_time(us)", "9453.9", "6883.3", "5209.8"],
  ["aic_mte1_ratio", "0.471", "0.498", "0.546"],
  ["aic_mte1_active_bw(GB/s)", "311.3", "298.8", "293.6"],
  ["aic_mte2_time(us)", "15368.4", "10629.8", "7916.5"],
  ["aic_mte2_ratio", "0.766", "0.769", "0.830"],
  ["aic_mte2_active_bw(GB/s)", "63.1", "91.2", "63.1"],
  ["aic_mte3_time(us)", "0.001", "0.001", "0.001"],
  ["aic_mte3_ratio", "0.000", "0.000", "0.000"],
  ["aic_mte3_active_bw(GB/s)", "—", "—", "—"],
  ["aic_fixpipe_time(us)", "8543.7", "10823.1", "6711.6"],
  ["aic_fixpipe_ratio", "0.426", "0.783", "0.704"],
  ["aic_fixpipe_active_bw(GB/s)", "137.9", "217.2", "149.1"],
  ["aic_icache_miss_rate", "0", "0", "0"],
];

const AIV_ROWS: string[][] = [
  ["aiv_time(us)", "20053.9", "13818.9", "9538.3"],
  ["aiv_total_cycles", "33088890", "22801262", "15738236"],
  ["aiv_vec_time(us)", "16819.6", "12161.1", "6689.7"],
  ["aiv_vec_ratio", "0.839", "0.880", "0.701"],
  ["aiv_scalar_time(us)", "7747.1", "6598.1", "3442.3"],
  ["aiv_scalar_ratio", "0.386", "0.477", "0.361"],
  ["aiv_mte2_time(us)", "6.58", "1.44", "6.60"],
  ["aiv_mte2_ratio", "0.0003", "0.0001", "0.0007"],
  ["aiv_mte2_active_bw(GB/s)", "106.3", "29.3", "76.0"],
  ["aiv_mte3_time(us)", "1897.5", "3444.0", "942.2"],
  ["aiv_mte3_ratio", "0.095", "0.249", "0.099"],
  ["aiv_icache_miss_rate", "1.2e-5", "7.4e-5", "4.7e-5"],
];

const ALIGN: Array<"left" | "right"> = ["left", "right", "right", "right"];

export default function S118800PipeCompare() {
  return (
    <Stack gap={24}>
      <Stack gap={6}>
        <H1>S=118800 流水占比</H1>
        <Text tone="secondary" size="small">
          {CAPTION}
        </Text>
      </Stack>

      <Grid columns={3} gap={12}>
        <Stat label="BSA MXFP8 keep1.0" value="20.58 ms" tone="warning" />
        <Stat label="FIA MXFP8 dense" value="14.60 ms" tone="info" />
        <Stat label="BSA MXFP4 keep1.0" value="9.78 ms" tone="success" />
      </Grid>

      <Callout tone="warning" title="BSA MXFP8 相对 FIA：Cube 吃不满">
        同一 dense 算量下 FIA 快 29%（14.6 vs 20.6 ms）。FIA cube 占用 0.69，BSA
        MXFP8 只有 0.51，AIC scalar 0.90 — Cube 在等 softmax / 同步。两边 AIC
        MTE2 ratio 都约 0.77；FIA 的 MTE2 活跃带宽 91 GB/s，BSA MXFP8/MXFP4 约 63
        GB/s。
      </Callout>

      <H2>主要 ratio 对照</H2>
      <BarChart
        height={280}
        yMax={1}
        categories={[
          "Cube",
          "AIC scalar",
          "MTE1",
          "MTE2",
          "Fixpipe",
          "AIV vec",
          "AIV scalar",
          "AIV MTE3",
        ]}
        series={[
          {
            name: "BSA MXFP8 keep1.0",
            data: [0.506, 0.899, 0.471, 0.766, 0.426, 0.839, 0.386, 0.095],
            tone: "warning",
          },
          {
            name: "FIA MXFP8 dense",
            data: [0.691, 0.619, 0.498, 0.769, 0.783, 0.88, 0.477, 0.249],
            tone: "info",
          },
          {
            name: "BSA MXFP4 keep1.0",
            data: [0.532, 0.988, 0.546, 0.83, 0.704, 0.701, 0.361, 0.099],
            tone: "success",
          },
        ]}
      />
      <Text tone="secondary" size="small">
        纵轴为 ratio（0–1）。来源：PipeUtilization.csv 核均。完整字段见下方 AIC /
        AIV 表。
      </Text>

      <H2>AIC 完整 PipeUtilization（核均）</H2>
      <Card>
        <CardHeader trailing={<Text size="small">28 Cube 核</Text>}>
          Cube Core
        </CardHeader>
        <CardBody>
          <Table
            stickyHeader
            striped
            headers={AIC_HEADERS}
            rows={AIC_ROWS}
            columnAlign={ALIGN}
          />
        </CardBody>
      </Card>

      <H2>AIV 完整 PipeUtilization（核均）</H2>
      <Card>
        <CardHeader trailing={<Text size="small">56 Vector 核</Text>}>
          Vector Core
        </CardHeader>
        <CardBody>
          <Table
            stickyHeader
            striped
            headers={AIC_HEADERS}
            rows={AIV_ROWS}
            columnAlign={ALIGN}
          />
        </CardBody>
      </Card>

      <H2>BSA MXFP4 vs MXFP8（同 keep=1.0）</H2>
      <Callout tone="success" title="墙钟约 2.10×，Cube 有效时间正好 2.00×">
        MXFP4 9.78 ms vs MXFP8 20.58 ms。Cube 有效时间：MXFP4 5074 µs，MXFP8
        10152 µs，比值 2.00，对上峰值 1514/757 TFLOPS。两边 cube ratio 都只有
        ~0.5；MXFP4 AIC scalar 0.99。FIA fixpipe ratio 0.78、AIV MTE3 0.25，高于
        两条 BSA 路径。
      </Callout>

      <Callout tone="neutral" title="MXFP8 fixpipe 占比低：绝对时间并不少">
        ratio = fixpipe_time / aic_time。MXFP8 fixpipe 绝对时间 8.54 ms，高于
        MXFP4 的 6.71 ms；占比 0.43 vs 0.70 是因为 aic_time 被拉到 20.05 ms（MXFP4
        9.54 ms）。两边 aic_time ≈ aiv_time。AIV vec：MXFP8 16.8 ms vs MXFP4 6.7
        ms（基块 128 vs 256，softmax 次数约 2×）。MXFP8 每基块两次 64×128
        即时 fixpipe（P0 延后一拍已回退）；MXFP4 走 EmitFixpipeMm1 延后一拍，短时间轴上
        重叠更好。看占比前应先看 aic_fixpipe_time。
      </Callout>
    </Stack>
  );
}
