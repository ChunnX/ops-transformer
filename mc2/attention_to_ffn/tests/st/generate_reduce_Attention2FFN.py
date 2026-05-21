import random
import sys
import os
import torch
from typing import Union, List
# 添加 atk 的路径到 Python 路径
atk_path = "/home/bin/anaconda3/envs/ffn_atk/lib/python3.8/site-packages/atk"
if atk_path not in sys.path:
    sys.path.append(atk_path)
# 现在导入
import case_generator
from case_generator.generator.generate_types import GENERATOR_REGISTRY
from case_generator.generator.base_generator import CaseGenerator
from configs.case_config import InputCaseConfig, CaseConfig
# yaml中 generate 字段需修改为自定义文件中使用的注册器的名称
@GENERATOR_REGISTRY.register("reduce_Attention2FFN")
class FFNToAttentionGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        # 初始化相关参数
        self.L = 1
        self.mircoBatchNum = 1
        self.X = 1
        self.BS = 8
        self.H = 1024
        self.HS = self.H + 128
        self.K = 8
        self.worldsize = 16
        self.moeExpertNum = 8
        self.localmoeNum = 1
        self.sharedExpertNum = 1
        self.rankNumPerSharedExpert = 1
        self.moeRankNum = int(self.moeExpertNum / self.localmoeNum)
        self.sharedRankNum = self.sharedExpertNum * self.rankNumPerSharedExpert
        self.ffnRankNum = self.moeRankNum + self.sharedRankNum
        self.attnRankNum = self.worldsize - self.ffnRankNum
        self.totalExpertNum = self.moeExpertNum + self.sharedExpertNum # 每个token对应的专家数量
        self.expertNumPerToken = self.K + self.sharedExpertNum # 每个token对应的专家数量
        self.M = self.ffnRankNum * 2 + 1
        self.IsQuant = 0
        self.MaxGlobalBS = self.BS * self.attnRankNum
    #after_case_config接口主要用于在整个yaml文件进行用例生成后，对于输入（如输入quantmode（后面的输入）的值对输入scale（前面的输入）的shape会带来影响）
    
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        self.IsQuant = (case_config.inputs[13].range_values > 0)
        
        self.BS = case_config.inputs[25].shape[0]
        self.H =  case_config.inputs[26].shape[0]
        self.K = case_config.inputs[27].shape[0]
        self.HS = self.H + 128
        self.expertNumPerToken = self.K + self.sharedExpertNum # 每个token对应的专家数量
        self.MaxGlobalBS = self.BS * self.attnRankNum
        #修正 x 的维度
        case_config.inputs[0].shape[0] = self.X 
        case_config.inputs[0].shape[1] = self.BS
        case_config.inputs[0].shape[2] = self.H
        #修正 expertIds 的维度
        case_config.inputs[4].shape[0] = self.X 
        case_config.inputs[4].shape[1] = self.BS
        case_config.inputs[4].shape[2] = self.K
        #修正 expertRankTable 的维度
        case_config.inputs[5].shape[0] = self.L
        case_config.inputs[5].shape[1] = self.totalExpertNum
        case_config.inputs[5].shape[2] = self.M
        #修正 scalesOptional 的维度
        case_config.inputs[6].shape[0] = self.L
        case_config.inputs[6].shape[1] = self.totalExpertNum
        case_config.inputs[6].shape[2] = self.H
        #修正 activeMaskOptional 的维度
        case_config.inputs[7].shape[0] = self.X
        case_config.inputs[7].shape[1] = self.BS
        #修正 ffnTokenInfoTableShape 的value
        case_config.inputs[9][0].range_values = self.attnRankNum
        case_config.inputs[9][1].range_values = self.mircoBatchNum
        case_config.inputs[9][2].range_values = 1 + 1 + self.BS * self.expertNumPerToken
        #修正 ffnTokenDataShape 的value
        case_config.inputs[10][0].range_values = self.attnRankNum
        case_config.inputs[10][1].range_values = self.mircoBatchNum
        case_config.inputs[10][2].range_values = self.BS
        case_config.inputs[10][3].range_values = self.expertNumPerToken
        case_config.inputs[10][4].range_values = self.HS if self.IsQuant else self.H
        #修正 attnTokenInfoTableShape 的value
        case_config.inputs[11][0].range_values = self.mircoBatchNum
        case_config.inputs[11][1].range_values = self.BS
        case_config.inputs[11][2].range_values = self.expertNumPerToken
        #修正 attnTokenInfoTableShape 的value
        #修正 x_world 的维度
        case_config.inputs[18].shape[0] = self.attnRankNum
        case_config.inputs[18].shape[1] = self.X 
        case_config.inputs[18].shape[2] = self.BS
        case_config.inputs[18].shape[3] = self.H
        #修正 expertIds_world 的维度
        case_config.inputs[19].shape[0] = self.attnRankNum
        case_config.inputs[19].shape[1] = self.X 
        case_config.inputs[19].shape[2] = self.BS
        case_config.inputs[19].shape[3] = self.K
        
        #修正 activeMaskOptional_world 的维度
        case_config.inputs[20].shape[0] = self.attnRankNum
        case_config.inputs[20].shape[1] = self.X 
        case_config.inputs[20].shape[2] = self.BS
        return case_config