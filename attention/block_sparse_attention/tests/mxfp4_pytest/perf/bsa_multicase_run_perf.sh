#!/bin/bash
# ============================================================
# BSA 多Case性能对比脚本
#   - 对比维度: mxfp4 vs bf16(或half)
#   - 每个case分别用 mxfp4 和 bf16 跑一遍, 输出到两个根目录
#   - 由 bsa_gen_summry_data.py 汇总两个目录的Task Duration做对比
#   - 支持指定 qseqlen / kvseqlen / sparsity_ratio / dtype 列表遍历
# ============================================================

# ---------- 1. 环境加载(路径请按本机实际修改) ----------
source /home/w00947241/Ascendc_CMC_0717/install/cann/set_env.sh

source /home/w00947241/lwg/custom/c_0612/vendors/custom_transformer/bin/set_env.bash
export PYTHONPATH=/home/w00947241/lwg/0709_1/ops-transformer_0703/torch_extension:$PYTHONPATH


# ---------- 2. 输出根目录(路径请按本机实际修改) ----------
# ROOT_DIR1 存 mxfp4 结果, ROOT_DIR2 存 bf16/half 结果(对照组)
ROOT_DIR1="./msprof_cache/mxfp4_dir"
ROOT_DIR2="./msprof_cache/bf16_dir"

# ---------- 3. 多Case参数列表(全部可配) ----------
batchList=(1)
qseqlenList=(40960)
kvseqlenList=(40960)
headsList=(8)
kvheadList=(2)
embeddingList=(128)                # head_dim, mxfp4 仅支持 64/128
sBlockXList=(128)                   # S矩阵Q方向分块
sBlockYList=(128)                   # S矩阵KV方向分块
sparsityRatioList=(1)              # 舍弃占比, 0.3 表示保留 70%
innerPrecList=(4)                  # 芯片950仅支持 innerPrec=4
qLayoutList=("BNSD")
kvLayoutList=("BNSD")              # kv_layout, 如需遍历可改为 ("BNSD" "BSND")
isVariedLenList=(0)
isVariedSparsityList=(0)
isActSeqlenAvalList=(1)
device=0

# dtype 对照组: 第一个为"自研", 第二个为"对照"
# mxfp4 vs bf16 -> 会在 ROOT_DIR1 跑 mxfp4, 在 ROOT_DIR2 跑 bf16
DTYPE_MXFP4="mxfp4"
DTYPE_BASE="half"                  # 也可改为 "half"

# ---------- 4. 日志初始化 ----------
LOG_FILE="bsa_perf_${DTYPE_MXFP4}_vs_${DTYPE_BASE}.log"
rm -f $LOG_FILE
rm -f bsa_resultData.csv

# 清空结果目录(若存在), 再重建
rm -rf "$ROOT_DIR1" "$ROOT_DIR2"
mkdir -p "$ROOT_DIR1" "$ROOT_DIR2"

# ---------- 5. 主循环 ----------
for batch in ${batchList[@]}
do
    for qSeqlen in ${qseqlenList[@]}
    do
        for kvSeqlen in ${kvseqlenList[@]}
        do
            for numHeads in ${headsList[@]}
            do
                for kvHeads in ${kvheadList[@]}
                do
                    for embedding in ${embeddingList[@]}
                    do
                        for sBlockX in ${sBlockXList[@]}
                        do
                            for sBlockY in ${sBlockYList[@]}
                            do
                                for sparsityRatio in ${sparsityRatioList[@]}
                                do
                                    for innerPrec in ${innerPrecList[@]}
                                    do
                                        for qLayout in ${qLayoutList[@]}
                                        do
                                            for kvLayout in ${kvLayoutList[@]}
                                            do
                                                for isVariedLen in ${isVariedLenList[@]}
                                                do
                                                    for isVariedSparsity in ${isVariedSparsityList[@]}
                                                    do
                                                        for isActSeqlenAval in ${isActSeqlenAvalList[@]}
                                                        do
                                                            # GQA 约束: numHeads 必须是 kvHeads 的整数倍
                                                            if [ $((numHeads % kvHeads)) -ne 0 ]; then
                                                                continue
                                                            fi
                                                            # qSeqlen 不能大于 kvSeqlen
                                                            if [ "$qSeqlen" -gt "$kvSeqlen" ]; then
                                                                continue
                                                            fi
                                                            # mxfp4 仅支持 head_dim 64/128
                                                            if [ "$embedding" != "64" ] && [ "$embedding" != "128" ]; then
                                                                continue
                                                            fi

                                                            # 生成 Case 名称
                                                            CASE_NAME="case_B${batch}_qS${qSeqlen}_kvS${kvSeqlen}_qN${numHeads}_kvN${kvHeads}_D${embedding}_sbX${sBlockX}_sbY${sBlockY}_sp${sparsityRatio}_innerP${innerPrec}_qL${qLayout}_kvL${kvLayout}_isVL${isVariedLen}_isVS${isVariedSparsity}_isActS${isActSeqlenAval}_dev${device}"

                                                            CASE_DIR1="$ROOT_DIR1/$CASE_NAME"
                                                            CASE_DIR2="$ROOT_DIR2/$CASE_NAME"
                                                            mkdir -p $CASE_DIR1 $CASE_DIR2

                                                            CASE_INFO="Case: B=$batch qS=$qSeqlen kvS=$kvSeqlen qN=$numHeads kvN=$kvHeads D=$embedding sbX=$sBlockX sbY=$sBlockY sparsity=$sparsityRatio innerP=$innerPrec qL=$qLayout kvL=$kvLayout isVL=$isVariedLen isVS=$isVariedSparsity isActS=$isActSeqlenAval dev=$device"
                                                            echo $CASE_NAME
                                                            echo $CASE_INFO
                                                            echo $CASE_INFO >> $LOG_FILE

                                                            # ---------- 环境1: 跑 mxfp4 ----------
                                                            echo "--------- mxfp4_跑case开始 -------"
                                                            echo "--------- mxfp4_跑case开始 -------" >> $LOG_FILE
                                                            msprof op --warm-up=20 --application="python3 invoke_npu_bsa_perf_only_ignore_accu.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $embedding $sBlockX $sBlockY $sparsityRatio $innerPrec $qLayout $kvLayout $isVariedLen $isVariedSparsity $DTYPE_MXFP4 $isActSeqlenAval $device" --output=$CASE_DIR1 >> $LOG_FILE
                                                            echo "mxfp4 finished" | tee -a $LOG_FILE

                                                            # ---------- 环境2: 跑 bf16/half(对照组) ----------
                                                            echo "--------- ${DTYPE_BASE}_跑case开始 -------"
                                                            echo "--------- ${DTYPE_BASE}_跑case开始 -------" >> $LOG_FILE
                                                            msprof op --warm-up=20 --application="python3 invoke_npu_bsa_perf_only_ignore_accu.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $embedding $sBlockX $sBlockY $sparsityRatio $innerPrec $qLayout $kvLayout $isVariedLen $isVariedSparsity $DTYPE_BASE $isActSeqlenAval $device" --output=$CASE_DIR2 >> $LOG_FILE
                                                            echo "${DTYPE_BASE} finished" | tee -a $LOG_FILE

                                                            echo -e "Case End!!!! \n"
                                                            echo -e "Case End!!!! \n" >> $LOG_FILE
                                                            echo ""
                                                            echo "" >> $LOG_FILE
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# ---------- 6. 汇总数据 ----------
echo "================ 开始生成汇总数据 ================"
echo "================ 开始生成汇总数据 ================" >> $LOG_FILE
BSA_MXFP4_DIR="$ROOT_DIR1" BSA_BASE_DIR="$ROOT_DIR2" python3 bsa_gen_summry_data.py >> $LOG_FILE 2>&1
echo "================ 汇总完成 ================"
echo "================ 汇总完成 ================" >> $LOG_FILE
