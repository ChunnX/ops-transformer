import os

import torch

SEP = "━" * 80


def analyze_failure(
    cpu_out: dict,
    npu_out: dict,
    compare_result: dict,
    case_dir: str = None,
    verbose: bool = False,
) -> str:
    """失败分析: 打印失败点位 + Max-RE + 预览, 返回分析报告文本"""
    lines = [SEP, "  精度失败分析报告", SEP]

    for key, result in compare_result.items():
        if not isinstance(result, dict):
            continue
        if result.get("passed", True):
            continue
        if key not in cpu_out or key not in npu_out:
            lines.append(f"\n  [WARN] key '{key}' 不在 cpu_out/npu_out 中, 跳过")
            continue

        cpu_t = cpu_out[key].float()
        npu_t = npu_out[key].float()

        if cpu_t.shape != npu_t.shape:
            lines.append(f"\n  [ERROR] key='{key}' shape不匹配: cpu={tuple(cpu_t.shape)} npu={tuple(npu_t.shape)}")
            continue

        diff = torch.abs(cpu_t - npu_t)
        ref_abs = torch.abs(cpu_t)
        rel_err = diff / (ref_abs + 1e-6)

        rtol = result.get("rtol", 0.005)
        atol = result.get("atol", 0.000025)
        fail_mask = ~torch.isclose(cpu_t, npu_t, rtol=rtol, atol=atol)

        fail_cnt = int(fail_mask.sum().item())
        total = cpu_t.numel()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        lines.append(f"\n  ── key='{key}' ──")
        lines.append(f"  Shape: {tuple(cpu_t.shape)}  FailElems: {fail_cnt}/{total} ({fail_cnt / total * 100:.4f}%)")
        lines.append(f"  MaxAbsErr: {max_abs:.8f}  MeanAbsErr: {mean_abs:.8f}")
        lines.append(f"  Threshold: atol={atol}  rtol={rtol}")

        cpu_flat = cpu_t.reshape(-1)
        npu_flat = npu_t.reshape(-1)
        diff_flat = diff.reshape(-1)
        rel_flat = rel_err.reshape(-1)
        mask_flat = fail_mask.reshape(-1)

        fail_idxs = mask_flat.nonzero(as_tuple=False).squeeze(1)

        if len(fail_idxs) > 0:
            show_idxs = fail_idxs if verbose else fail_idxs[:10]
            label = "全量" if verbose else f"前{len(show_idxs)}"
            lines.append(f"\n  [失败点位 {label}]")
            lines.append(f"  {'idx':>10}  {'cpu':>14}  {'npu':>14}  {'absErr':>12}  {'relErr':>10}")
            lines.append(f"  {'─' * 10}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 10}")
            for i in show_idxs.tolist():
                lines.append(
                    f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}"
                    f"  {diff_flat[i].item():>12.8f}  {rel_flat[i].item():>10.6f}"
                )
            if not verbose and fail_cnt > 10:
                lines.append(f"  ... 共 {fail_cnt} 个失败点 (使用 --verbose-diff 查看全量)")

            lines.append(f"\n  [Max-RE 前3] (绝对误差最大)")
            lines.append(f"  {'idx':>10}  {'cpu':>14}  {'npu':>14}  {'absErr':>12}  {'relErr':>10}")
            lines.append(f"  {'─' * 10}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 10}")
            top_k = min(3, total)
            _, top_abs_idxs = torch.topk(diff_flat, top_k)
            for i in top_abs_idxs.tolist():
                lines.append(
                    f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}"
                    f"  {diff_flat[i].item():>12.8f}  {rel_flat[i].item():>10.6f}"
                )

        lines.append(f"\n  [前10后10预览]")
        lines.append(f"  {'idx':>10}  {'cpu':>14}  {'npu':>14}")
        lines.append(f"  {'─' * 10}  {'─' * 14}  {'─' * 14}")
        n = min(10, total)
        for i in range(n):
            lines.append(f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}")
        if total > 20:
            lines.append(f"  {'...':>10}")
            for i in range(total - 10, total):
                lines.append(f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}")
        elif total > 10:
            for i in range(10, total):
                lines.append(f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}")

    lines.append(f"\n{SEP}")
    report = "\n".join(lines)

    if case_dir is not None:
        os.makedirs(case_dir, exist_ok=True)
        report_file = os.path.join(case_dir, "fail_analysis.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def analyze_dimensions(cpu_out: dict, npu_out: dict, compare_result: dict) -> str:
    # TODO: 维度分析 (移植非量化FA analyze_fail_distribution: batch/head/seq分维统计)
    ...


def analyze_continuity(cpu_out: dict, npu_out: dict, compare_result: dict) -> str:
    # TODO: 连续性分析 (失败点位连段检测+模式判定)
    ...


def visualize_failure(cpu_out: dict, npu_out: dict, compare_result: dict, save_dir: str = None) -> str:
    # TODO: 可视化 (热力图)
    ...
