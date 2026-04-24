import json

data = {'cpu_0': [{'filename': 'output_5.pt', 'result': True, 'error_info': '', 'new_benchmark_indicate': {'Actual_最大相对误差': 0.0, 'Actual_平均相对误差': 0.0, 'Actual_均方根误差': 0.0, 'Actual_误差均衡性': 0.0, 'Actual_小值域数错数量': 0, 'Benchmark_最大相对误差': 0.0, 'Benchmark_平均相对误差': 0.0, 'Benchmark_均方根误差': 0.0, 'Benchmark_误差均衡性': 0.0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.0, '平均相对误差比例': 0.0, '均方根误差比例': 0.0, '误差均衡性差值': 0.0, '小值域数错误差比例': 0.0}}, {'filename': 'output_3.pt', 'result': True, 'error_info': '', 'new_benchmark_indicate': {'Actual_最大相对误差': 0, 'Actual_平均相对误差': 0, 'Actual_均方根误差': 0, 'Actual_误差均衡性': 0, 'Actual_小值域数错数量': 0, 'Benchmark_最大相对误差': 0, 'Benchmark_平均相对误差': 0, 'Benchmark_均方根误差': 0, 'Benchmark_误差均衡性': 0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.0, '平均相对误差比例': 0.0, '均方根误差比例': 0.0, '误差均衡性差值': 0, '小值域数错误差比例': 0.0}}, {'filename': 'output_7.pt', 'result': False, 'error_info': 'ERROR: 小值域数错误差比例(33.0)超过阈值(2.0)，\n', 'new_benchmark_indicate': {'Actual_最大相对误差': 6.864238457637839e-07, 'Actual_平均相对误差': 8.083310376605368e-08, 'Actual_均方根误差': 4.335365844099215e-08, 'Actual_误差均衡性': -4.7037684858253215e-09, 'Actual_小值域数错数量': 33, 'Benchmark_最大相对误差': 0.0, 'Benchmark_平均相对误差': 0.0, 'Benchmark_均方根误差': 0.0, 'Benchmark_误差均衡性': 0.0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.011246368288993835, '平均相对误差比例': 0.0013243695721030235, '均方根误差比例': 0.0007103063398972154, '误差均衡性差值': 4.7037684858253215e-09, '小值域数错误差比例': 33.0}}, {'filename': 'output_1.pt', 'result': True, 'error_info': '', 'new_benchmark_indicate': {'Actual_最大相对误差': 0.0, 'Actual_平均相对误差': 0.0, 'Actual_均方根误差': 0.0, 'Actual_误差均衡性': 0.0, 'Actual_小值域数错数量': 0, 'Benchmark_最大相对误差': 0.0, 'Benchmark_平均相对误差': 0.0, 'Benchmark_均方根误差': 0.0, 'Benchmark_误差均衡性': 0.0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.0, '平均相对误差比例': 0.0, '均方根误差比例': 0.0, '误差均衡性差值': 0.0, '小值域数错误差比例': 0.0}}, {'filename': 'output_2.pt', 'result': True, 'error_info': '', 'new_benchmark_indicate': {'Actual_最大相对误差': 5.05812636220071e-07, 'Actual_平均相对误差': 1.445027209001637e-07, 'Actual_均方根误差': 4.8545050645998344e-08, 'Actual_误差均衡性': 1.7695128917694092e-08, 'Actual_小值域数错数量': 1, 'Benchmark_最大相对误差': 0.0, 'Benchmark_平均相对误差': 0.0, 'Benchmark_均方根误差': 0.0, 'Benchmark_误差均衡性': 0.0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.008287234231829643, '平均相对误差比例': 0.002367532579228282, '均方根误差比例': 0.0007953621097840369, '误差均衡性差值': 1.7695128917694092e-08, '小值域数错误差比例': 1.0}}, {'filename': 'output_4.pt', 'result': False, 'error_info': 'ERROR: 最大相对误差比例(22.077959060668945)超过阈值(5.0)，\nERROR: 平均相对误差比例(5.329309463500977)超过阈值(1.5)，\nERROR: 均方根误差比例(474.4055480957031)超过阈值(1.5)，\n', 'new_benchmark_indicate': {'Actual_最大相对误差': 0.0013475316809490323, 'Actual_平均相对误差': 0.0003252752358093858, 'Actual_均方根误差': 0.02895541675388813, 'Actual_误差均衡性': 6.828702316852286e-05, 'Actual_小值域数错数量': 0, 'Benchmark_最大相对误差': 0.0, 'Benchmark_平均相对误差': 0.0, 'Benchmark_均方根误差': 0.0, 'Benchmark_误差均衡性': 0.0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 22.077959060668945, '平均相对误差比例': 5.329309463500977, '均方根误差比例': 474.4055480957031, '误差均衡性差值': 6.828702316852286e-05, '小值域数错误差比例': 0.0}}, {'filename': 'output_0.pt', 'result': True, 'error_info': '', 'new_benchmark_indicate': {'Actual_最大相对误差': 0, 'Actual_平均相对误差': 0, 'Actual_均方根误差': 0, 'Actual_误差均衡性': 0, 'Actual_小值域数错数量': 0, 'Benchmark_最大相对误差': 0, 'Benchmark_平均相对误差': 0, 'Benchmark_均方根误差': 0, 'Benchmark_误差均衡性': 0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.0, '平均相对误差比例': 0.0, '均方根误差比例': 0.0, '误差均衡性差值': 0, '小值域数错误差比例': 0.0}}, {'filename': 'output_6.pt', 'result': True, 'error_info': '', 'new_benchmark_indicate': {'Actual_最大相对误差': 1.4475384659817792e-06, 'Actual_平均相对误差': 9.498825761511398e-07, 'Actual_均方根误差': 9.629541182221146e-07, 'Actual_误差均衡性': 9.097942665903247e-07, 'Actual_小值域数错数量': 0, 'Benchmark_最大相对误差': 0.0, 'Benchmark_平均相对误差': 0.0, 'Benchmark_均方根误差': 0.0, 'Benchmark_误差均衡性': 0.0, 'Benchmark_小值域数错数量': 1, '最大相对误差比例': 0.02371647022664547, '平均相对误差比例': 0.015562876127660275, '均方根误差比例': 0.015777040272951126, '误差均衡性差值': 9.097942665903247e-07, '小值域数错误差比例': 0.0}}]}

def format_test_result(data):
    print("=" * 100)
    print("测试结果汇总")
    print("=" * 100)
    
    for device, results in data.items():
        print(f"\n设备: {device}")
        print("-" * 100)
        
        passed = sum(1 for r in results if r['result'])
        failed = sum(1 for r in results if not r['result'])
        print(f"总计: {len(results)} 个测试 | 通过: {passed} | 失败: {failed}")
        print("-" * 100)
        
        for item in results:
            status = "✓ 通过" if item['result'] else "✗ 失败"
            print(f"\n文件: {item['filename']:<20} 状态: {status}")
            
            if item['error_info']:
                print(f"错误信息:\n{item['error_info']}")
            
            indicate = item['new_benchmark_indicate']
            print(f"  最大相对误差: {indicate['Actual_最大相对误差']:.6e}  (比例: {indicate['最大相对误差比例']:.4f})")
            print(f"  平均相对误差: {indicate['Actual_平均相对误差']:.6e}  (比例: {indicate['平均相对误差比例']:.4f})")
            print(f"  均方根误差:   {indicate['Actual_均方根误差']:.6e}  (比例: {indicate['均方根误差比例']:.4f})")
            print(f"  误差均衡性:   {indicate['Actual_误差均衡性']:.6e}")
            print(f"  小值域数错数量: {indicate['Actual_小值域数错数量']}  (比例: {indicate['小值域数错误差比例']:.4f})")
    
    print("\n" + "=" * 100)

format_test_result(data)