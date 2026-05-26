multi_stream.py ——测试主程序
使用方法详见python multi_stream.py --help
eg:
python multi_stream.py --op1 alltoall_allgather_bmm --op2 all_gather_base_mm --dynamic

run_test_multi_stream.sh ——用例批跑脚本
使用方法详见run_test_multi_stream.sh --help
eg:
  1. 全量运行所有用例：
     ./run_test_multi_stream.sh
  2. 单独重跑指定OP1+OP2组合：
     ./run_test_multi_stream.sh --op1 <OP1值> --op2 <OP2值>
  3. 批量重跑指定report.log中的失败用例：
     ./run_test_multi_stream.sh --restart_fail <report.log路径>
  4. 查看帮助：
     ./run_test_multi_stream.sh --help