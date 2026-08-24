#!/usr/bin/env python3
"""Save MegaMoe outputs and build a self-contained precision error report."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_TOLERANCE = 0.001
MAX_HEATMAP_ROWS = 1024
MAX_HEATMAP_COLS = 4096
MAX_HEATMAP_CELLS = 131072
TOP_ERROR_COUNT = 100


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            # numpy does not support bfloat16. Float32 is also what the existing
            # batch comparison uses.
            return value.detach().cpu().to(torch.float32).numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _as_matrix(value):
    array = _to_numpy(value)
    original_shape = list(array.shape)
    if array.ndim == 0:
        matrix = array.reshape(1, 1)
    elif array.ndim == 1:
        matrix = array.reshape(1, -1)
    else:
        matrix = array.reshape(-1, array.shape[-1])
    return matrix.astype(np.float32, copy=False), original_shape


def _format_number(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return f"{value:.9g}"


def _json_number(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def _error_arrays(expected, actual, tolerance):
    expected64 = expected.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    finite = np.isfinite(expected64) & np.isfinite(actual64)
    abs_error = np.full(expected.shape, np.inf, dtype=np.float64)
    abs_error[finite] = np.abs(actual64[finite] - expected64[finite])
    denominator = np.maximum(np.abs(actual64), np.abs(expected64)) + 1.0e-10
    with np.errstate(invalid="ignore", divide="ignore"):
        relative_error = abs_error / denominator

    # Match toolss_pro.cal_relative_diff_np: differences below tolerance are
    # treated as absolute errors; larger differences use symmetric relative error.
    score = np.where(abs_error < tolerance, abs_error, relative_error)
    equal_special = (
        (np.isnan(expected64) & np.isnan(actual64))
        | (np.isposinf(expected64) & np.isposinf(actual64))
        | (np.isneginf(expected64) & np.isneginf(actual64))
    )
    score[equal_special] = 0.0
    abs_error[equal_special] = 0.0
    relative_error[equal_special] = 0.0
    mismatched_special = ~finite & ~equal_special
    score[mismatched_special] = np.inf
    abs_error[mismatched_special] = np.inf
    relative_error[mismatched_special] = np.inf
    failure = score > tolerance
    return abs_error, relative_error, score, failure


def _heatmap_steps(rows, cols):
    """Keep small/skinny matrices cell-exact while bounding HTML size."""
    row_step = max(1, math.ceil(rows / MAX_HEATMAP_ROWS))
    col_step = max(1, math.ceil(cols / MAX_HEATMAP_COLS))
    while math.ceil(rows / row_step) * math.ceil(cols / col_step) > MAX_HEATMAP_CELLS:
        bin_rows = math.ceil(rows / row_step)
        bin_cols = math.ceil(cols / col_step)
        if bin_cols >= bin_rows:
            col_step += 1
        else:
            row_step += 1
    return row_step, col_step


def _comparison_data(
    name, expected, actual, tolerance, baseline=None, zero_reference=None
):
    rows, cols = expected.shape
    abs_error, relative_error, score, failure = _error_arrays(
        expected, actual, tolerance
    )
    baseline_abs_error = None
    if baseline is not None:
        expected32 = expected.astype(np.float32, copy=False)
        actual_abs_for_location = np.abs(
            actual.astype(np.float32, copy=False) - expected32
        )
        baseline_abs_error = np.abs(
            baseline.astype(np.float32, copy=False) - expected32
        )
        # This is exactly the location criterion used by
        # toolss_pro.data_compare_triangle.
        failure = baseline_abs_error < actual_abs_for_location
        # Use MegaMoe's own cal_relative_diff_np-style error as color strength.
        # A ratio against Cascade would explode wherever Cascade error is zero,
        # making harmless numerical noise look like the worst region.
        score[~failure] = 0.0
    row_step, col_step = _heatmap_steps(rows, cols)
    bin_rows = math.ceil(rows / row_step)
    bin_cols = math.ceil(cols / col_step)
    bin_count = np.zeros((bin_rows, bin_cols), dtype=np.int64)
    bin_score = np.zeros((bin_rows, bin_cols), dtype=np.float64)
    worst_row = np.full((bin_rows, bin_cols), -1, dtype=np.int64)
    worst_col = np.full((bin_rows, bin_cols), -1, dtype=np.int64)

    nonzero_reference = baseline if baseline is not None else zero_reference
    zero_dropout = (
        np.isfinite(expected)
        & np.isfinite(actual)
        & (actual == 0)
        & (expected != 0)
    )
    if nonzero_reference is not None:
        zero_dropout &= np.isfinite(nonzero_reference) & (nonzero_reference != 0)
    zero_drop_count = np.zeros((bin_rows, bin_cols), dtype=np.int64)
    zero_drop_row = np.full((bin_rows, bin_cols), -1, dtype=np.int64)
    zero_drop_col = np.full((bin_rows, bin_cols), -1, dtype=np.int64)
    zero_rows, zero_cols = np.nonzero(zero_dropout)
    if zero_rows.size:
        zero_bins = (zero_rows // row_step) * bin_cols + zero_cols // col_step
        np.add.at(zero_drop_count.ravel(), zero_bins, 1)
        zero_severity = np.abs(expected[zero_rows, zero_cols])
        zero_order = np.lexsort((-zero_severity, zero_bins))
        sorted_zero_bins = zero_bins[zero_order]
        zero_first = np.r_[True, sorted_zero_bins[1:] != sorted_zero_bins[:-1]]
        zero_selected = zero_order[zero_first]
        selected_zero_bins = zero_bins[zero_selected]
        zero_drop_row.ravel()[selected_zero_bins] = zero_rows[zero_selected]
        zero_drop_col.ravel()[selected_zero_bins] = zero_cols[zero_selected]

    error_rows, error_cols = np.nonzero(failure)
    if error_rows.size:
        error_scores = score[error_rows, error_cols]
        flat_bins = (error_rows // row_step) * bin_cols + error_cols // col_step
        np.add.at(bin_count.ravel(), flat_bins, 1)
        np.maximum.at(bin_score.ravel(), flat_bins, error_scores)

        # Sort by bin and then descending score; the first item in each bin is
        # its exact worst element, used by the hover tooltip.
        order = np.lexsort((-error_scores, flat_bins))
        sorted_bins = flat_bins[order]
        first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
        selected = order[first]
        selected_bins = flat_bins[selected]
        worst_row.ravel()[selected_bins] = error_rows[selected]
        worst_col.ravel()[selected_bins] = error_cols[selected]

        top_size = min(TOP_ERROR_COUNT, error_scores.size)
        top_indices = np.argpartition(error_scores, -top_size)[-top_size:]
        top_indices = top_indices[np.argsort(error_scores[top_indices])[::-1]]
    else:
        top_indices = np.empty(0, dtype=np.int64)

    worst_expected = np.full((bin_rows, bin_cols), np.nan, dtype=np.float64)
    worst_actual = np.full((bin_rows, bin_cols), np.nan, dtype=np.float64)
    worst_baseline = np.full((bin_rows, bin_cols), np.nan, dtype=np.float64)
    valid_worst = worst_row >= 0
    worst_expected[valid_worst] = expected[
        worst_row[valid_worst], worst_col[valid_worst]
    ]
    worst_actual[valid_worst] = actual[
        worst_row[valid_worst], worst_col[valid_worst]
    ]
    if baseline is not None:
        worst_baseline[valid_worst] = baseline[
            worst_row[valid_worst], worst_col[valid_worst]
        ]

    top_errors = []
    for index in top_indices:
        row = int(error_rows[index])
        col = int(error_cols[index])
        top_errors.append(
            {
                "row": row,
                "col": col,
                "expected": _format_number(expected[row, col]),
                "actual": _format_number(actual[row, col]),
                "cascade": (
                    _format_number(baseline[row, col])
                    if baseline is not None else None
                ),
                "absolute": _format_number(abs_error[row, col]),
                "cascadeAbsolute": (
                    _format_number(baseline_abs_error[row, col])
                    if baseline_abs_error is not None else None
                ),
                "relative": _format_number(relative_error[row, col]),
                "score": _format_number(score[row, col]),
            }
        )

    error_count = int(failure.sum())
    return {
        "name": name,
        "mode": "baseline_regression" if baseline is not None else "threshold",
        "rows": rows,
        "cols": cols,
        "total": int(expected.size),
        "errorCount": error_count,
        "errorRatio": error_count / expected.size if expected.size else 0.0,
        "zeroDropoutCount": int(zero_dropout.sum()),
        "zeroDropoutRatio": (
            float(zero_dropout.sum()) / expected.size if expected.size else 0.0
        ),
        "pointLabel": "退化点" if baseline is not None else "异常元素",
        "criterion": (
            "abs(MegaMoe-Golden) > abs(Cascade-Golden)"
            if baseline is not None
            else f"cal_relative_diff_np > {tolerance}"
        ),
        "maxAbsolute": _json_number(abs_error.max()) if abs_error.size else None,
        "maxRelative": _json_number(relative_error.max()) if relative_error.size else None,
        "rowStep": row_step,
        "colStep": col_step,
        "binRows": bin_rows,
        "binCols": bin_cols,
        "counts": bin_count.ravel().tolist(),
        "scores": [_json_number(value) for value in bin_score.ravel()],
        "worstRows": worst_row.ravel().tolist(),
        "worstCols": worst_col.ravel().tolist(),
        "worstExpected": [
            None if row < 0 else _format_number(value)
            for row, value in zip(worst_row.ravel(), worst_expected.ravel())
        ],
        "worstActual": [
            None if row < 0 else _format_number(value)
            for row, value in zip(worst_row.ravel(), worst_actual.ravel())
        ],
        "worstBaseline": [
            None if row < 0 or baseline is None else _format_number(value)
            for row, value in zip(worst_row.ravel(), worst_baseline.ravel())
        ],
        "zeroDropCounts": zero_drop_count.ravel().tolist(),
        "zeroDropRows": zero_drop_row.ravel().tolist(),
        "zeroDropCols": zero_drop_col.ravel().tolist(),
        "zeroDropExpected": [
            None if row < 0 else _format_number(expected[row, col])
            for row, col in zip(zero_drop_row.ravel(), zero_drop_col.ravel())
        ],
        "zeroDropReference": [
            None
            if row < 0 or nonzero_reference is None
            else _format_number(nonzero_reference[row, col])
            for row, col in zip(zero_drop_row.ravel(), zero_drop_col.ravel())
        ],
        "topErrors": top_errors,
    }


def _result_scalar(result, name):
    value = getattr(result, name)
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _script_comparison(actual_result, baseline_result, script_result, script_rules=None):
    if actual_result is None or baseline_result is None:
        return None
    script_rules = script_rules or {}
    max_ratio = float(script_rules.get("max_rtol_ratio", 5.0))
    avg_ratio = float(script_rules.get("avg_rtol_ratio", 1.5))
    rmse_ratio = float(script_rules.get("rmse_ratio", 1.5))
    error_ratio = float(script_rules.get("error_diff_ratio", 2.0))
    small_threshold = int(script_rules.get("small_num_total_atol", 100000))
    total_small = _result_scalar(actual_result, "total_small_num")
    rules = (
        ("大值最大相对误差", "diff_big_max", max_ratio, True),
        ("大值平均相对误差", "diff_big_avg", avg_ratio, True),
        ("大值相对误差总和", "diff_big_sum", error_ratio, True),
        ("小值错误数", "err_small_num", error_ratio, total_small >= small_threshold),
        ("RMSE", "diff_rmse", rmse_ratio, True),
    )
    metrics = []
    for label, attribute, multiplier, enabled in rules:
        actual = _result_scalar(actual_result, attribute)
        baseline = _result_scalar(baseline_result, attribute)
        limit = baseline * multiplier
        metrics.append({
            "label": label,
            "actual": _json_number(actual),
            "baseline": _json_number(baseline),
            "multiplier": multiplier,
            "limit": _json_number(limit),
            "enabled": enabled,
            "failed": bool(enabled and actual > limit),
        })
    return {
        "status": str(script_result).upper(),
        "totalSmall": int(total_small),
        "smallMetricEnabled": total_small >= small_threshold,
        "smallMetricThreshold": small_threshold,
        "metrics": metrics,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MegaMoe 精度诊断</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#131a2c;--muted:#91a0bc;--line:#28324a;--ok:#35c88a;--bad:#ff5b6e}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#edf2ff;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:24px}
h1{font-size:24px;margin:0 0 6px}.sub{color:var(--muted);margin-bottom:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.card,.view{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.label{color:var(--muted);font-size:12px}.value{font-size:20px;margin-top:3px}.view{margin-top:16px}.view h2{margin:0 0 10px;font-size:18px}.status{font-weight:700}.pass{color:var(--ok)}.fail{color:var(--bad)}
.canvas-wrap{position:relative;overflow:auto;border:1px solid var(--line);background:#080c17;margin-top:12px}canvas{display:block;width:100%;height:min(55vh,620px);image-rendering:pixelated}.tip{display:none;position:fixed;z-index:5;pointer-events:none;background:#050812;border:1px solid #53617e;border-radius:7px;padding:9px;white-space:pre;box-shadow:0 8px 30px #0009}
.hint{color:var(--muted);font-size:12px;margin-top:7px}.toolbar{display:flex;align-items:center;gap:7px;margin-top:12px;flex-wrap:wrap}.view-range{color:var(--muted);font-variant-numeric:tabular-nums;margin-left:4px}table{width:100%;border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}.table-wrap{overflow:auto;max-height:420px}button{background:#222c45;color:#edf2ff;border:1px solid #3b496b;border-radius:6px;padding:6px 10px;cursor:pointer}canvas{cursor:grab}canvas.dragging{cursor:grabbing}
</style></head><body><main><h1>MegaMoe 精度诊断报告</h1><div class="sub" id="meta"></div><div class="cards" id="summary"></div><section class="view" id="rules" hidden></section><div id="views"></div></main><div class="tip" id="tip"></div>
<script>const REPORT=__REPORT_DATA__;
const fmt=v=>v===null?'N/A':(typeof v==='number'?v.toExponential(5):v); const pct=v=>(v*100).toFixed(6)+'%';
document.getElementById('meta').textContent=`case=${REPORT.caseName} · rank=${REPORT.rank} · shape=${JSON.stringify(REPORT.originalShape)} · ${REPORT.createdAt}`;
const script=REPORT.scriptComparison;document.getElementById('summary').innerHTML=script?`<div class="card"><div class="label">原脚本 checkResult 综合判定</div><div class="value status ${script.status==='PASS'?'pass':'fail'}">${script.status}</div><div>此结果与 CSV 的 output_x 判定一致</div></div>`:'';
if(script){const rules=document.getElementById('rules');rules.hidden=false;rules.innerHTML=`<h2>原脚本综合判定指标</h2><div class="hint">actual 为 MegaMoe 指标，baseline 为 Cascade 指标；只有 actual 超过 baseline × 倍率时才导致 FAIL。小值错误数仅在小值总数不少于 ${script.smallMetricThreshold} 时参与判定。</div><table><thead><tr><th>指标</th><th>MegaMoe actual</th><th>Cascade baseline</th><th>倍率</th><th>上限</th><th>结果</th></tr></thead><tbody>${script.metrics.map(m=>`<tr><td>${m.label}${m.enabled?'':'（不参与）'}</td><td>${fmt(m.actual)}</td><td>${fmt(m.baseline)}</td><td>${m.multiplier}</td><td>${fmt(m.limit)}</td><td class="status ${m.failed?'fail':'pass'}">${m.enabled?(m.failed?'FAIL':'PASS'):'N/A'}</td></tr>`).join('')}</tbody></table>`}
const views=document.getElementById('views'),tip=document.getElementById('tip');
for(const c of REPORT.comparisons){
const regression=c.mode==='baseline_regression';
const hint=regression?`原脚本 data_compare_triangle 定位口径：${c.criterion}。综合结果以顶部 checkResult 为准；PASS 时用黄色，FAIL 时用红色。`:`独立固定阈值口径：${c.criterion}。此紫色视图不参与顶部原脚本 PASS/FAIL，只用于观察 MegaMoe 相对 Golden 的绝对分布。`;
const tableHead=regression?`<tr><th>坐标 [row,col]</th><th>Golden</th><th>Cascade</th><th>MegaMoe</th><th>Cascade 绝对误差</th><th>MegaMoe 绝对误差</th><th>MegaMoe 判定误差</th></tr>`:`<tr><th>坐标 [row,col]</th><th>Golden</th><th>MegaMoe</th><th>绝对误差</th><th>相对误差</th><th>判定值</th></tr>`;
const tableRows=c.topErrors.map(e=>regression?`<tr><td>[${e.row}, ${e.col}]</td><td>${e.expected}</td><td>${e.cascade}</td><td>${e.actual}</td><td>${e.cascadeAbsolute}</td><td>${e.absolute}</td><td>${e.score}</td></tr>`:`<tr><td>[${e.row}, ${e.col}]</td><td>${e.expected}</td><td>${e.actual}</td><td>${e.absolute}</td><td>${e.relative}</td><td>${e.score}</td></tr>`).join('');
const section=document.createElement('section');section.className='view';section.innerHTML=`<h2>${c.name}</h2><div class="cards"><div><span class="label">矩阵</span><br>${c.rows} × ${c.cols}</div><div><span class="label">${c.pointLabel}</span><br>${c.errorCount.toLocaleString()} (${pct(c.errorRatio)})</div><div><span class="label">疑似输出清零</span><br><span style="color:#00e5ff">${c.zeroDropoutCount.toLocaleString()} (${pct(c.zeroDropoutRatio)})</span></div><div><span class="label">MegaMoe 最大绝对误差</span><br>${fmt(c.maxAbsolute)}</div><div><span class="label">MegaMoe 最大相对误差</span><br>${fmt(c.maxRelative)}</div></div><div class="toolbar"><button class="zoom-in">放大</button><button class="zoom-out">缩小</button><button class="zoom-reset">重置</button><span class="view-range"></span></div><div class="canvas-wrap"><canvas></canvas></div><div class="hint">${hint} ${regression?'颜色强度取 MegaMoe 自身判定误差，不使用容易被零分母放大的退化倍数。':''} <span style="color:#00e5ff">亮青色</span>表示 MegaMoe 为 0、Golden 非零，且存在 Cascade 数据时 Cascade 也非零的疑似清零点。鼠标滚轮缩放、按住拖拽平移、双击重置；悬浮可查看原始值。</div><button class="table-toggle">显示/隐藏 Top ${c.topErrors.length} ${c.pointLabel}</button><div class="table-wrap" hidden><table><thead>${tableHead}</thead><tbody>${tableRows}</tbody></table></div>`;views.appendChild(section);
const canvas=section.querySelector('canvas'),ctx=canvas.getContext('2d'),dpr=devicePixelRatio||1,rangeLabel=section.querySelector('.view-range');
const view={x0:0,y0:0,x1:c.binCols,y1:c.binRows};
function updateRange(){const rs=Math.max(0,Math.floor(view.y0)*c.rowStep),re=Math.min(c.rows-1,Math.ceil(view.y1)*c.rowStep-1),cs=Math.max(0,Math.floor(view.x0)*c.colStep),ce=Math.min(c.cols-1,Math.ceil(view.x1)*c.colStep-1);rangeLabel.textContent=`当前范围：row ${rs}..${re}，col ${cs}..${ce}`}
function draw(){const rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.floor(rect.width*dpr));canvas.height=Math.max(280,Math.floor(rect.height*dpr));ctx.clearRect(0,0,canvas.width,canvas.height);const spanX=view.x1-view.x0,spanY=view.y1-view.y0,cw=canvas.width/spanX,ch=canvas.height/spanY;let max=0;for(const s of c.scores)if(Number.isFinite(s))max=Math.max(max,s);const logMax=Math.log10(1+Math.max(max,REPORT.tolerance,Number.EPSILON)),overallFail=script&&script.status!=='PASS';const r0=Math.max(0,Math.floor(view.y0)),r1=Math.min(c.binRows,Math.ceil(view.y1)),c0=Math.max(0,Math.floor(view.x0)),c1=Math.min(c.binCols,Math.ceil(view.x1));for(let r=r0;r<r1;r++)for(let col=c0;col<c1;col++){const i=r*c.binCols+col;if(c.zeroDropCounts[i]){ctx.fillStyle='#00e5ff'}else if(!c.counts[i]){ctx.fillStyle='#10172a'}else{const q=Number.isFinite(c.scores[i])?Math.min(1,Math.log10(1+c.scores[i])/logMax):1;ctx.fillStyle=!regression?`hsl(${285-25*q} 82% ${30+28*q}%)`:(overallFail?`hsl(${34-34*q} 95% ${35+24*q}%)`:`hsl(${52-12*q} 90% ${28+24*q}%)`)}ctx.fillRect((col-view.x0)*cw,(r-view.y0)*ch,Math.ceil(cw),Math.ceil(ch))}updateRange()}
function setView(x0,y0,width,height){width=Math.min(c.binCols,Math.max(1,width));height=Math.min(c.binRows,Math.max(1,height));x0=Math.min(c.binCols-width,Math.max(0,x0));y0=Math.min(c.binRows-height,Math.max(0,y0));view.x0=x0;view.y0=y0;view.x1=x0+width;view.y1=y0+height;draw()}
function zoom(factor,fx=.5,fy=.5){const oldW=view.x1-view.x0,oldH=view.y1-view.y0,newW=oldW*factor,newH=oldH*factor,cx=view.x0+fx*oldW,cy=view.y0+fy*oldH;setView(cx-fx*newW,cy-fy*newH,newW,newH)}
function reset(){setView(0,0,c.binCols,c.binRows)}
draw();new ResizeObserver(draw).observe(canvas);section.querySelector('.zoom-in').onclick=()=>zoom(.5);section.querySelector('.zoom-out').onclick=()=>zoom(2);section.querySelector('.zoom-reset').onclick=reset;canvas.ondblclick=reset;
canvas.addEventListener('wheel',e=>{e.preventDefault();const rect=canvas.getBoundingClientRect(),fx=(e.clientX-rect.left)/rect.width,fy=(e.clientY-rect.top)/rect.height;zoom(e.deltaY<0?.75:1.333333,fx,fy)},{passive:false});
let dragging=false,lastX=0,lastY=0;canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});canvas.addEventListener('pointermove',e=>{if(!dragging)return;const rect=canvas.getBoundingClientRect(),dx=-(e.clientX-lastX)/rect.width*(view.x1-view.x0),dy=-(e.clientY-lastY)/rect.height*(view.y1-view.y0);lastX=e.clientX;lastY=e.clientY;setView(view.x0+dx,view.y0+dy,view.x1-view.x0,view.y1-view.y0)});canvas.addEventListener('pointerup',e=>{dragging=false;canvas.releasePointerCapture(e.pointerId);canvas.classList.remove('dragging')});canvas.addEventListener('pointercancel',()=>{dragging=false;canvas.classList.remove('dragging')});
canvas.addEventListener('mousemove',e=>{if(dragging){tip.style.display='none';return}const rect=canvas.getBoundingClientRect(),binX=view.x0+(e.clientX-rect.left)/rect.width*(view.x1-view.x0),binY=view.y0+(e.clientY-rect.top)/rect.height*(view.y1-view.y0),col=Math.min(c.binCols-1,Math.max(0,Math.floor(binX))),r=Math.min(c.binRows-1,Math.max(0,Math.floor(binY))),i=r*c.binCols+col,rs=r*c.rowStep,re=Math.min(c.rows-1,rs+c.rowStep-1),cs=col*c.colStep,ce=Math.min(c.cols-1,cs+c.colStep-1);let text=`区域 row ${rs}..${re}, col ${cs}..${ce}\n${c.pointLabel}数: ${c.counts[i]}`;if(c.counts[i])text+=regression?`\n最严重坐标: [${c.worstRows[i]}, ${c.worstCols[i]}]\nGolden: ${c.worstExpected[i]}\nCascade: ${c.worstBaseline[i]}\nMegaMoe: ${c.worstActual[i]}\nMegaMoe 判定误差: ${c.scores[i]}`:`\n最严重坐标: [${c.worstRows[i]}, ${c.worstCols[i]}]\nGolden: ${c.worstExpected[i]}\nMegaMoe: ${c.worstActual[i]}\n判定值: ${c.scores[i]}`;if(c.zeroDropCounts[i])text+=`\n[疑似输出清零] 区域内 ${c.zeroDropCounts[i]} 个\n样本坐标: [${c.zeroDropRows[i]}, ${c.zeroDropCols[i]}]\nGolden: ${c.zeroDropExpected[i]}\nCascade/参考: ${c.zeroDropReference[i]}\nMegaMoe: 0`;tip.textContent=text;tip.style.display='block';tip.style.left=Math.min(innerWidth-tip.offsetWidth-8,e.clientX+14)+'px';tip.style.top=Math.min(innerHeight-tip.offsetHeight-8,e.clientY+14)+'px'});canvas.addEventListener('mouseleave',()=>tip.style.display='none');section.querySelector('.table-toggle').onclick=()=>{const x=section.querySelector('.table-wrap');x.hidden=!x.hidden}}
</script></body></html>"""


def generate_precision_report(
    expected,
    actual,
    output_dir,
    case_name="unknown",
    rank=0,
    cascade=None,
    tolerance=DEFAULT_TOLERANCE,
    threshold_source="precision_report.py:DEFAULT_TOLERANCE",
    actual_result=None,
    baseline_result=None,
    script_result="UNKNOWN",
    script_rules=None,
):
    """Persist arrays and return the generated self-contained HTML path."""
    expected_matrix, original_shape = _as_matrix(expected)
    actual_matrix, actual_shape = _as_matrix(actual)
    if expected_matrix.shape != actual_matrix.shape:
        raise ValueError(
            f"expected shape {expected_matrix.shape} != actual shape {actual_matrix.shape}"
        )
    arrays = {"expected": expected_matrix, "actual": actual_matrix}
    comparisons = []
    if cascade is not None:
        cascade_matrix, cascade_shape = _as_matrix(cascade)
        if cascade_matrix.shape != expected_matrix.shape:
            raise ValueError(
                f"cascade shape {cascade_matrix.shape} != expected shape {expected_matrix.shape}"
            )
        arrays["cascade"] = cascade_matrix
        comparisons.append(
            _comparison_data(
                "MegaMoe 相对 Cascade 的逐元素退化位置",
                expected_matrix,
                actual_matrix,
                tolerance,
                baseline=cascade_matrix,
            ),
        )
    else:
        cascade_shape = None
    comparisons.append(
        _comparison_data(
            "MegaMoe vs Golden 的固定阈值异常位置",
            expected_matrix,
            actual_matrix,
            tolerance,
            zero_reference=cascade_matrix if cascade is not None else None,
        )
    )

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    npz_path = report_dir / "precision_data.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "caseName": str(case_name),
        "rank": int(rank),
        "createdAt": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "tolerance": float(tolerance),
        "thresholdSource": str(threshold_source),
        "originalShape": original_shape,
        "actualShape": actual_shape,
        "cascadeShape": cascade_shape,
        "dataFile": npz_path.name,
        "scriptComparison": _script_comparison(
            actual_result, baseline_result, script_result, script_rules
        ),
        "comparisons": comparisons,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    safe_json = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    html_path = report_dir / "precision_report.html"
    html_path.write_text(
        HTML_TEMPLATE.replace("__REPORT_DATA__", safe_json), encoding="utf-8"
    )
    return html_path


def main():
    parser = argparse.ArgumentParser(description="从 precision_data.npz 生成精度诊断 HTML")
    parser.add_argument("input", help="包含 expected/actual，可选 cascade 的 .npz 文件")
    parser.add_argument("-o", "--output-dir", default=None, help="报告输出目录")
    parser.add_argument("--case-name", default="offline_report")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    data = np.load(args.input)
    required = {"expected", "actual"}
    if not required.issubset(data.files):
        parser.error(f"npz 必须包含 {sorted(required)}，实际为 {data.files}")
    output_dir = args.output_dir or str(Path(args.input).with_suffix("")) + "_report"
    html_path = generate_precision_report(
        data["expected"], data["actual"], output_dir,
        case_name=args.case_name, rank=args.rank,
        cascade=data["cascade"] if "cascade" in data.files else None,
        tolerance=args.tolerance,
    )
    print(html_path)


if __name__ == "__main__":
    main()
