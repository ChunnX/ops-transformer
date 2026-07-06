import itertools
import importlib


def _val_str(v):
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    return str(v)


def _param_segment(name, value):
    s = _val_str(value)
    return f"{name}_{s}" if name[-1].isdigit() else f"{name}{s}"


def load_cases(op_name: str, marker: str = "ci") -> tuple[list[dict], list[str]]:
    mod = importlib.import_module(f"ops.{op_name}.paramset")
    param_names = mod.PARAM_NAMES
    defaults = mod.DEFAULTS
    case_sets = mod.CASE_SETS
    skip_cases = mod.SKIP_CASES

    case_set = case_sets.get(marker, {})
    skip_set = skip_cases.get(marker, set())

    cases = []
    case_ids = []
    for case_name, case_dict in case_set.items():
        expanded = dict(case_dict)
        for key, default_vals in defaults.items():
            if key not in expanded:
                expanded[key] = default_vals
        param_values = [expanded[n] for n in param_names]
        for combo in itertools.product(*param_values):
            case = {"name": case_name}
            for key, val in zip(param_names, combo):
                case[key] = val
            param_str = "_".join(
                _param_segment(n, v) for n, v in zip(param_names, combo)
            )
            case_id = f"{case_name}_{param_str}"
            if case_id in skip_set or case_name in skip_set:
                continue
            cases.append(case)
            case_ids.append(case_id)
    return cases, case_ids


def resolve_case_ids(pattern: str, all_case_ids: list[str]) -> list[str]:
    if pattern == "all":
        return sorted(all_case_ids)
    ids = [x.strip() for x in pattern.split(",")]
    result = []
    missing = []
    for cid in ids:
        matches = [
            k
            for k in all_case_ids
            if k == cid or k.startswith(f"{cid}_") or k.endswith(f"_{cid}")
        ]
        if matches:
            result.extend(matches)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] 以下 case 不存在: {missing}")
    return list(dict.fromkeys(result))
