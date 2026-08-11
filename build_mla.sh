#!/bin/bash 
 # Build script for MLA D512 non-quant tilingKeys 
 # Generates 16 tilingKeys from the MLA ASCENDC_TPL_ARGS_SEL entry: 
 #   InOutLayoutType=BNSD, Config=9(576/512), PseMode=NONE, QuantMode=NoQuant, 
 #   HasAttenMask={false,true}, HasRope=true, KvLayoutType={NO_PA,BBH,BNBD,NZ}, 
 #   IsFd={false,true}, EmptyTensor=false, EnableKVPrefix=false, 
 #   EnableS1OutSplit=false, IsReconstructTemp=true 
 
 
 
 
 # python3 attention/fused_infer_attention_score/tests/pytest/test_mla_d512_runner.py --case_files mla_d512_noquant_gen 
 set -e 
 
 
 DEFAULT_INSTALL_PATH="${MLA_INSTALL_PATH:-${ASCEND_HOME_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/custom}}" 
 INSTALL_PATH="$DEFAULT_INSTALL_PATH" 
 
 
 usage() { 
     echo "Usage: bash build_mla.sh [--install-path <path>]" 
     echo "  --install-path   Custom install path (default: $DEFAULT_INSTALL_PATH)" 
     exit 1 
 } 
 
 
 while [[ $# -gt 0 ]]; do 
     case "$1" in 
         --install-path) 
             INSTALL_PATH="$2" 
             shift 2 
             ;; 
         -h|--help) 
             usage 
             ;; 
         *) 
             echo "Unknown option: $1" 
             usage 
             ;; 
     esac 
 done 
 
 
 MLA_TILING_KEYS=( 
     34760558848 
     34760558849 
     34760558850 
     34760558852 
     34894776576 
     34894776577 
     34894776578 
     34894776580 
     35297429760 
     35297429761 
     35297429762 
     35297429764 
     35431647488 
     35431647489 
     35431647490 
     35431647492 
     35834300672 
     35834300673 
     35834300674 
     35834300676 
     35968518400 
     35968518401 
     35968518402 
     35968518404 
     36371171584 
     36371171585 
     36371171586 
     36371171588 
     36505389312 
     36505389313 
     36505389314 
     36505389316 
     36908042496 
     36908042497 
     36908042498 
     36908042500 
     37042260224 
     37042260225 
     37042260226 
     37042260228 
     37444913408 
     37444913409 
     37444913410 
     37444913412 
     37579131136 
     37579131137 
     37579131138 
     37579131140 
     37981784320 
     37981784321 
     37981784322 
     37981784324 
     38116002048 
     38116002049 
     38116002050 
     38116002052 
     38518655232 
     38518655233 
     38518655234 
     38518655236 
     38652872960 
     38652872961 
     38652872962 
     38652872964 
 ) 
 
 
 SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" 
 cd "$SCRIPT_DIR" 
 
 
 KEYS_STR=$(IFS=\;; echo "${MLA_TILING_KEYS[*]}") 
 
 
 echo "============================================" 
 echo "Building MLA D512 non-quant (${#MLA_TILING_KEYS[@]} tilingKeys)" 
 echo "============================================" 
 
 
 rm -rf build build_out 
 
 
 bash build.sh --ops=fused_infer_attention_score --soc=ascend950 --tiling_key="${KEYS_STR}" --pkg -j20 
 
 
 echo "============================================" 
 echo "Build complete. Custom package at: build/_CPack_Packages/" 
 echo "============================================" 
 
 
 # Install the custom package 
 PKG_FILE=$(ls build/*.run build_out/*.run 2>/dev/null | head -1) 
 if [ -z "$PKG_FILE" ]; then 
     echo "ERROR: Custom package not found in build/ or build_out/" 
     exit 1 
 fi 
 
 
 echo "============================================" 
 echo "Installing: $PKG_FILE" 
 echo "  install-path: $INSTALL_PATH" 
 echo "============================================" 
 bash "$PKG_FILE" --install-path="$INSTALL_PATH"