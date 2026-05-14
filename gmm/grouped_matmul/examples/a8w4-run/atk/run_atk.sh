source /data/huangjunzhe/Ascend.GDN/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=/data/huangjunzhe/Ascend.GDN/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
atk node --backend pyaclnn --devices 7 node --backend cpu task -c GroupedMatmul_A8W4.json -p GroupedMatmul_A8W4_multi.py --task accuracy