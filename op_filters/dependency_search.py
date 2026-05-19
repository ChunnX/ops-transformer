# -*- coding: utf-8 -*-
"""
反向依赖搜索模块
"""

import logging
from typing import Dict, List, Set


def build_reverse_dependency_graph(dependency_data: Dict, logger: logging.Logger) -> Dict[str, Set[str]]:
    """
    构建反向依赖图。

    反向依赖：如果算子 A 的 dependencies 中包含算子 B，则在反向依赖图中，
    B -> A（即 B 的变更会影响 A）。

    Args:
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        反向依赖图字典，key 为算子名称，value 为依赖该算子的算子集合
    """
    reverse_graph: Dict[str, Set[str]] = {}
    
    for op_name in dependency_data.keys():
        reverse_graph[op_name] = set()
    
    for op_name, op_info in dependency_data.items():
        dependencies = op_info.get("dependencies", [])
        for dep_op in dependencies:
            if dep_op in reverse_graph:
                reverse_graph[dep_op].add(op_name)
            else:
                logger.warning(
                    f"算子 {op_name} 依赖的算子 {dep_op} 不在依赖关系中，将被忽略"
                )
    
    logger.info(f"构建反向依赖图完成，共 {len(reverse_graph)} 个算子")
    return reverse_graph


def find_dependent_ops(
    changed_ops: Set[str],
    reverse_graph: Dict[str, Set[str]],
    logger: logging.Logger
) -> Set[str]:
    """
    根据反向依赖图，找到所有受影响的算子。

    只从初始变更的算子（changed_ops）传播一次，不继续传播被纳入测试的算子。
    这样可以避免被间接纳入测试的算子进一步传播影响。

    Args:
        changed_ops: 直接变更的算子集合
        reverse_graph: 反向依赖图
        logger: 日志记录器

    Returns:
        所有需要测试的算子集合（包括 changed_ops 自身及其直接依赖者）
    """
    ops_to_run = set(changed_ops)
    visited = set(changed_ops)
    
    expansion_log: List[str] = []
    
    for current_op in changed_ops:
        if current_op not in reverse_graph:
            logger.warning(f"算子 {current_op} 不在依赖关系中，跳过")
            continue
        
        dependents = reverse_graph.get(current_op, set())
        new_dependents = dependents - visited
        
        if new_dependents:
            expansion_log.append(
                f"从 {current_op} 扩展到: {sorted(new_dependents)}"
            )
        
        for dep_op in new_dependents:
            visited.add(dep_op)
            ops_to_run.add(dep_op)
    
    for log_entry in expansion_log:
        logger.info(f"反向依赖搜索: {log_entry}")
    
    logger.info(f"反向依赖搜索完成，共 {len(ops_to_run)} 个算子需要测试")
    return ops_to_run