import os
import yaml
import subprocess
from pathlib import Path
import re
import sys
from datetime import datetime
import threading

class TaskRunner:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.tasks_dir = self.project_root / "tasks"
        self.base_config_path = self.project_root / "base_config.yaml"
        self.batch_compile_path = self.project_root / "framework" / "batch_compile.py"
        self.log_file = self.project_root / f"batch_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def find_operator_dir(self, operator_name):
        for level_dir in self.tasks_dir.iterdir():
            if level_dir.is_dir():
                for category_dir in level_dir.iterdir():
                    if category_dir.is_dir():
                        operator_dir = category_dir / operator_name
                        if operator_dir.exists() and operator_dir.is_dir():
                            return operator_dir
        return None
    
    def get_test_case_count(self, operator_dir):
        test_cases_path = operator_dir / "validation" / "test_cases.csv"
        if not test_cases_path.exists():
            self.log(f"Warning: {test_cases_path} does not exist, using default n_case=1")
            return 1
        
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return len(lines) - 1
    
    def update_n_case_in_config(self, n_case):
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['n_case'] = n_case
        
        with open(self.base_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self.log(f"Updated n_case in base_config.yaml to {n_case}")
    
    def get_answer_dirs(self, operator_dir):
        answer_dir = operator_dir / "answer"
        if not answer_dir.exists() or not answer_dir.is_dir():
            return []
        
        answer_dirs = []
        for item in sorted(answer_dir.iterdir()):
            if item.is_dir() and item.name.isdigit():
                answer_dirs.append(int(item.name))
        
        return sorted(answer_dirs)
    
    def update_batch_compile_question_src(self, answer_num):
        with open(self.batch_compile_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 153:
            raise RuntimeError(f"batch_compile.py has only {len(lines)} lines, cannot update line 153")
        
        new_line = f'            question_src = f"{{task_obj.problem_def_full_path}}/answer/{answer_num}"\n'
        lines[152] = new_line
        
        with open(self.batch_compile_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        self.log(f"Updated batch_compile.py line 153 to use answer/{answer_num}")
    
    def _stream_output(self, pipe, stream_name):
        for line in iter(pipe.readline, ''):
            if line:
                self.log(f"[{stream_name}] {line.rstrip()}")
    
    def run_test_command(self, operator_name):
        cmd = f"python run_multi_test.py -task_name {operator_name} -stages compile precision"
        self.log(f"Running command: {cmd}")
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1
            )
            
            stdout_thread = threading.Thread(target=self._stream_output, args=(process.stdout, 'STDOUT'))
            stderr_thread = threading.Thread(target=self._stream_output, args=(process.stderr, 'STDERR'))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            
            process.wait()
            
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            return_code = process.returncode
            self.log(f"Command return code: {return_code}")
            
            return return_code == 0
        except Exception as e:
            self.log(f"Error running command: {str(e)}")
            return False
    
    def process_operator(self, operator_name):
        self.log(f"\n{'='*60}")
        self.log(f"Processing operator: {operator_name}")
        self.log(f"{'='*60}")
        
        operator_dir = self.find_operator_dir(operator_name)
        if not operator_dir:
            self.log(f"Error: Operator directory not found for {operator_name}")
            return False
        
        self.log(f"Found operator directory: {operator_dir}")
        
        n_case = self.get_test_case_count(operator_dir)
        self.log(f"Test case count: {n_case}")
        
        self.update_n_case_in_config(n_case)
        
        answer_dirs = self.get_answer_dirs(operator_dir)
        self.log(f"Answer directories found: {answer_dirs}")
        
        if not answer_dirs:
            self.log("No answer directories found, skipping execution")
            return True
        
        success_count = 0
        for answer_num in answer_dirs:
            self.log(f"\n--- Processing answer/{answer_num} start---")
            try:
                self.update_batch_compile_question_src(answer_num)
                if self.run_test_command(operator_name):
                    success_count += 1
            except Exception as e:
                self.log(f"Error processing answer/{answer_num}: {str(e)}")
            self.log(f"\n--- Processing answer/{answer_num} end---")
        
        self.log(f"\nSummary for {operator_name}: {success_count}/{len(answer_dirs)} successful")
        return success_count == len(answer_dirs)
    
    def process_operators(self, operator_list):
        self.log(f"Starting batch processing of {len(operator_list)} operators")
        self.log(f"Log file: {self.log_file}")
        
        results = {}
        for operator_name in operator_list:
            success = self.process_operator(operator_name)
            results[operator_name] = success
        
        self.log(f"\n{'='*60}")
        self.log("FINAL SUMMARY")
        self.log(f"{'='*60}")
        for operator_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            self.log(f"{operator_name}: {status}")
        
        return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_run_tasks.py <operator_name1> <operator_name2> ...")
        print("Example: python batch_run_tasks.py Neg Gemm Mla")
        sys.exit(1)
    
    operator_list = sys.argv[1:]
    
    project_root = Path(__file__).parent
    runner = TaskRunner(project_root)
    
    results = runner.process_operators(operator_list)
    
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)

if __name__ == "__main__":
    main()
