__input__ = {
    "aclnn": {
        "aclnnMoeDistributeDispatch": "moe_distribute_dispatch_inputs",
        "aclnnMoeDistributeDispatchV2": "moe_distribute_dispatch_inputs",
        "aclnnMoeDistributeDispatchV3": "moe_distribute_dispatch_inputs",
        "aclnnMoeDistributeDispatchV4": "moe_distribute_dispatch_inputs",
    }
}

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_case_data_module():
    return importlib.import_module("mc2.common.tests.assets.impl.moe_case_data")


def _parse_remark(remark):
    return {
        item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
        for item in (remark or "").split(",")
        if "=" in item
    }


def moe_distribute_dispatch_inputs(
    x,
    expert_ids,
    scales=None,
    x_active_mask=None,
    expert_scales=None,
    elastic_info=None,
    performance_info=None,
    **kwargs,
):
    remark = _parse_remark(kwargs.get("remark", ""))
    patched_ep_rank_id = int(kwargs.pop("_ttkPatchedEpRankId", -1))
    patched_tp_rank_id = int(kwargs.pop("_ttkPatchedTpRankId", -1))
    original_top_k = int(kwargs.pop("_moeDeepepOriginalTopK", 0))
    attrs = {
        "globalBs": int(kwargs.pop("globalBs", remark.get("global_bs", x.shape[0]))),
        "epWorldSize": int(kwargs.get("epWorldSize", kwargs.get("ep_world_size", 1))),
        "tpWorldSize": int(kwargs.get("tpWorldSize", kwargs.get("tp_world_size", 1))),
        "moeExpertNum": int(kwargs.get("moeExpertNum", kwargs.get("moe_expert_num", 1))),
        "sharedExpertNum": int(kwargs.get("sharedExpertNum", kwargs.get("shared_expert_num", 0))),
        "sharedExpertRankNum": int(
            kwargs.get("sharedExpertRankNum", kwargs.get("shared_expert_rank_num", 0))
        ),
        "rankNumPerSharedExpert": int(
            kwargs.get("rankNumPerSharedExpert", kwargs.get("rank_num_per_shared_expert", 0))
        ),
        "localMoeExpertNum": int(kwargs.get("localMoeExpertNum", 0)),
        "epRankId": int(
            patched_ep_rank_id
            if patched_ep_rank_id >= 0
            else kwargs.get("epRankId", kwargs.get("ep_rank_id", 0))
        ),
        "tpRankId": int(
            patched_tp_rank_id
            if patched_tp_rank_id >= 0
            else kwargs.get("tpRankId", kwargs.get("tp_rank_id", 0))
        ),
    }
    module = _load_case_data_module()
    config = module.MoeCaseConfig(
        global_bs=attrs["globalBs"],
        hidden_size=int(x.shape[1]),
        top_k=(
            original_top_k
            if original_top_k > 0
            else expert_ids.shape[1]
            if expert_ids.dim() > 1
            else 1
        ),
        seed=int(remark.get("seed", 1)),
        ep_world_size=max(attrs["epWorldSize"], 1),
        tp_world_size=max(attrs["tpWorldSize"], 1),
        moe_expert_num=max(attrs["moeExpertNum"], 1),
        local_moe_expert_num_override=attrs["localMoeExpertNum"],
        shared_expert_num=attrs["sharedExpertNum"],
        shared_expert_rank_num=(
            attrs["sharedExpertRankNum"]
            or attrs["sharedExpertNum"] * attrs["rankNumPerSharedExpert"]
        ),
        const_expert_num=int(kwargs.get("constExpertNum", 0)),
        zero_expert_num=int(kwargs.get("zeroExpertNum", 0)),
        copy_expert_num=int(kwargs.get("copyExpertNum", 0)),
        rank_num_per_shared_expert=(
            attrs["rankNumPerSharedExpert"]
            or attrs["sharedExpertRankNum"] // max(attrs["sharedExpertNum"], 1)
        ),
        input_dtype=str(x.dtype).replace("torch.", ""),
        active_mask_dim=int(kwargs.get("xActiveMaskDim", 0)),
        is_mask=int(kwargs.get("xActiveMaskDim", 0)) > 0,
        is_variable_bs=bool(int(kwargs.get("_moeVariableBs", 0))),
        is_quant=(
            int(kwargs.get("quantMode", 0)) > 0
        ),
        is_dispatch_scales=bool(int(kwargs.get("isDispatchScales", 0))),
        is_topk_balance=kwargs.get("commAlg") == "fullmesh_v2",
        is_x_random=int(kwargs.get("isXRandom", 0)) == 1,
        is_combine_x=int(kwargs.get("isCombineX", 0)) == 1,
        is_shared_expert_x=bool(int(kwargs.get("isSharedExpertX", 0))),
        is_deepep=(
            remark.get("isDeepep") == "1"
            or int(kwargs.get("_moeDeepep", 0) or 0) == 1
        ),
        is_elastic=int(kwargs.get("isElastic", 0)) == 1,
        elastic_shared_rank_num=int(kwargs.get("shareBrokenCardNum", 0)),
        elastic_moe_rank_num=int(kwargs.get("moeBrokenCardNum", 0)),
        comm_quant_mode=int(kwargs.get("commQuantMode", 0)),
        is_expand=int(kwargs.get("isExpand", 0)) == 1,
    )
    is_deepep = config.is_deepep
    context = kwargs.get("_ttk_context")
    if context is not None:
        context.attributes["isElastic"] = int(config.is_elastic)
        context.attributes["isExpand"] = int(config.is_expand)
        context.attributes["_moeDeepepOriginalTopK"] = config.top_k
    generated = module.rank_case(
        module.load_or_generate_case(config),
        config,
        patched_ep_rank_id
        if patched_ep_rank_id >= 0
        else attrs["epRankId"],
        patched_tp_rank_id
        if patched_tp_rank_id >= 0
        else attrs["tpRankId"],
    )
    x = generated["x"]
    expert_ids = generated["expert_ids"]
    x_active_mask = generated.get("x_active_mask_input", generated["x_active_mask"])
    scales = generated["scales"]
    expert_scales = generated["expert_scales"]
    elastic_info = None

    if kwargs.get("api_name") == "aclnnMoeDistributeDispatchV4":
        return x, expert_ids, scales, x_active_mask, expert_scales, elastic_info, performance_info
    if kwargs.get("api_name") == "aclnnMoeDistributeDispatchV2":
        return x, expert_ids, scales, x_active_mask, expert_scales
    if is_deepep:
        if context is not None:
            context.attributes["_moeCaseData"] = generated
            context.attributes["_moeDeepepExpertIdsExpand"] = (
                generated["expert_ids_expand"]
                if generated.get("expert_ids_expand") is not None
                else expert_ids
            )
            context.attributes["_moeElasticAvailableRanks"] = config.available_ranks
            context.attributes["_moeDeepepElasticInfo"] = generated["elastic_info"]
        return x, expert_ids, None, x_active_mask, None, None
    return x, expert_ids, scales, x_active_mask, expert_scales, elastic_info
